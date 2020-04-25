from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from utils import create_op_dir, save_results
from config import SEEDS
from mnist.smart_coord import craft_first_5_target, craft_last_5_target
from mnist.dataloaders import mnist_combined_train_loader, mnist_combined_test_loader
from mnist_cifar10.smart_coord import craft_mnist_target, craft_cifar10_target
from mnist_cifar10.dataloaders import (
    mnist_cifar10_single_channel_train_loader,
    mnist_cifar10_single_channel_test_loader,
    mnist_cifar10_3_channel_train_loader,
    mnist_cifar10_3_channel_test_loader,
)
from archs.lenet5 import LeNet5, LeNet5Halfed
from archs.resnet import ResNet18
from archs.pan import PAN, AgnosticPAN, compute_agnostic_stats


def train(args, pan, model, device, train_loader, target_create_fn, optimizer, epoch):
    model.eval()
    pan.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits, feature = model(data, out_feature=True)
        if args.pan_type == "feature":
            output = pan(feature)
        elif args.pan_type == "logits":
            output = pan(logits)
        elif args.pan_type == "agnostic_feature":
            output = pan(compute_agnostic_stats(feature))
        elif args.pan_type == "agnostic_logits":
            output = pan(compute_agnostic_stats(logits))
        else:
            raise NotImplementedError("Not an eligible pan type.")

        pan_target = target_create_fn(target).to(device)
        loss = F.cross_entropy(output, pan_target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(args, pan, model, device, test_loader, target_create_fn):
    model.eval()
    pan.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits, feature = model(data, out_feature=True)

            if args.pan_type == "feature":
                output = pan(feature)
            elif args.pan_type == "logits":
                output = pan(logits)
            elif args.pan_type == "agnostic_feature":
                output = pan(compute_agnostic_stats(feature))
            elif args.pan_type == "agnostic_logits":
                output = pan(compute_agnostic_stats(logits))
            else:
                raise NotImplementedError("Not an eligible pan type.")

            pan_target = target_create_fn(target).to(device)
            test_loss += F.cross_entropy(
                output, pan_target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(pan_target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100.0 * correct / len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), acc,
        )
    )
    return test_loss, acc


def train_model(
    pan, model, device, train_loader, test_loader, target_create_fn, config_args
):
    pan_model = pan.to(device)
    model = model.to(device)
    optimizer = optim.SGD(
        pan_model.parameters(), lr=config_args.lr, momentum=config_args.momentum
    )

    for epoch in range(1, config_args.epochs + 1):
        train(
            config_args,
            pan_model,
            model,
            device,
            train_loader,
            target_create_fn,
            optimizer,
            epoch,
        )
        test_loss, acc = test(
            config_args, pan_model, model, device, test_loader, target_create_fn
        )
    return pan_model, test_loss, acc


def train_pan(args):

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize arguments based on dataset chosen
    if args.dataset == "disjoint_mnist":
        train_loaders = [
            mnist_combined_train_loader(args.batch_size),
            mnist_combined_train_loader(args.batch_size),
        ]
        test_loaders = [
            mnist_combined_test_loader(args.test_batch_size),
            mnist_combined_test_loader(args.test_batch_size),
        ]
        args.d1 = "first5_mnist"
        args.d2 = "last5_mnist"
        args.m1_input_channel = 1
        args.m2_input_channel = 1
        args.output_size = 5
        target_create_fns = [craft_first_5_target, craft_last_5_target]
    elif args.dataset == "mnist_cifar10":
        train_loaders = [
            mnist_cifar10_single_channel_train_loader(args.batch_size),
            mnist_cifar10_3_channel_train_loader(args.batch_size),
        ]
        test_loaders = [
            mnist_cifar10_single_channel_test_loader(args.test_batch_size),
            mnist_cifar10_3_channel_test_loader(args.test_batch_size),
        ]
        args.d1 = "mnist"
        args.d2 = "cifar10"
        args.m1_input_channel = 1
        args.m2_input_channel = 3
        args.output_size = 10
        target_create_fns = [craft_mnist_target, craft_cifar10_target]

    # Initialize models based on architecture chosen
    if args.arch == "lenet5":
        arch = LeNet5
        feature_size = 120
    elif args.arch == "lenet5_halfed":
        arch = LeNet5Halfed
        feature_size = 60
    elif args.arch == "resnet18":
        arch = ResNet18
        feature_size = 512

    # Initialize PAN based on its type
    if args.pan_type == "feature":
        pan_input_size = feature_size
        pan_arch = PAN
    elif args.pan_type == "logits":
        pan_input_size = args.output_size
        pan_arch = PAN
    elif args.pan_type == "agnostic_feature":
        pan_input_size = 3
        pan_arch = AgnosticPAN
    elif args.pan_type == "agnostic_logits":
        pan_input_size = 3
        pan_arch = AgnosticPAN

    # Create the directory for saving if it does not exist
    create_op_dir(args.output_dir)

    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.arch}")
    print(f"PAN type: {args.pan_type}")
    pan1_results = []
    pan2_results = []

    for i in range(len(args.seeds)):
        print(f"Iteration {i}, Seed {args.seeds[i]}")

        np.random.seed(args.seeds[i])
        torch.manual_seed(args.seeds[i])

        # Load models
        model1 = arch(
            input_channel=args.m1_input_channel, output_size=args.output_size
        ).to(device)
        model1.load_state_dict(
            torch.load(
                args.model_dir + f"{args.d1}_{args.arch}_{args.seeds[i]}",
                map_location=torch.device("cpu"),
            )
        )
        pan1, pan1_test_loss, pan1_acc = train_model(
            pan=pan_arch(input_size=pan_input_size).to(device),
            model=model1,
            device=device,
            train_loader=train_loaders[0],
            test_loader=test_loaders[0],
            target_create_fn=target_create_fns[0],
            config_args=args,
        )

        model2 = arch(
            input_channel=args.m2_input_channel, output_size=args.output_size
        ).to(device)
        model2.load_state_dict(
            torch.load(
                args.model_dir + f"{args.d2}_{args.arch}_{args.seeds[i]}",
                map_location=torch.device("cpu"),
            )
        )
        pan2, pan2_test_loss, pan2_acc = train_model(
            pan=pan_arch(input_size=pan_input_size).to(device),
            model=model2,
            device=device,
            train_loader=train_loaders[1],
            test_loader=test_loaders[1],
            target_create_fn=target_create_fns[1],
            config_args=args,
        )

        # Save the pan model
        torch.save(
            pan1.state_dict(),
            args.output_dir
            + f"pan_{args.pan_type}_{args.dataset}({args.d1})_{args.arch}_{args.seeds[i]}",
        )
        torch.save(
            pan2.state_dict(),
            args.output_dir
            + f"pan_{args.pan_type}_{args.dataset}({args.d2})_{args.arch}_{args.seeds[i]}",
        )

        # save the results in list first
        pan1_results.append(
            {
                "iteration": i,
                "seed": args.seeds[i],
                "loss": pan1_test_loss,
                "acc": pan1_acc,
            }
        )
        pan2_results.append(
            {
                "iteration": i,
                "seed": args.seeds[i],
                "loss": pan2_test_loss,
                "acc": pan2_acc,
            }
        )

    # Save all the results
    if args.save_results:
        save_results(
            f"pan_{args.pan_type}_{args.dataset}({args.d1})_{args.arch}",
            pan1_results,
            args.results_dir,
        )
        save_results(
            f"pan_{args.pan_type}_{args.dataset}({args.d2})_{args.arch}",
            pan2_results,
            args.results_dir,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="disjoint_mnist",
        choices=["disjoint_mnist", "mnist_cifar10"],
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="lenet5",
        choices=["lenet5", "lenet5_halfed", "resnet18"],
    )
    parser.add_argument(
        "--pan_type",
        type=str,
        default="feature",
        choices=["feature", "logits", "agnostic_logits"],
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--test_batch_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--no_cuda", type=bool, default=False)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_results", type=bool, default=True)
    parser.add_argument("--results_dir", type=str, default="./results/pan/")
    parser.add_argument("--model_dir", type=str, default="./cache/models/")
    parser.add_argument("--output_dir", type=str, default="./cache/models/pan/")

    args = parser.parse_args()
    args.seeds = SEEDS

    train_pan(args)
