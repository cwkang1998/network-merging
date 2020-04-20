from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from utils import create_op_dir, save_results
from config import SEEDS
from mnist.dataloaders import (
    mnist_first5_train_loader,
    mnist_first5_test_loader,
    mnist_last5_train_loader,
    mnist_last5_test_loader,
)
from mnist_cifar10.dataloaders import (
    mnist_train_loader,
    mnist_test_loader,
    cifar10_train_loader,
    cifar10_test_loader,
)
from archs.lenet5 import LeNet5, LeNet5Halfed
from archs.resnet import ResNet18

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
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


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100.0 * correct / len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), acc,
        )
    )
    return test_loss, acc


def train_model(model, device, train_loader, test_loader, config_args):
    model = model.to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=config_args.lr, momentum=config_args.momentum
    )

    for epoch in range(1, config_args.epochs + 1):
        train(config_args, model, device, train_loader, optimizer, epoch)
        test_loss, acc = test(config_args, model, device, test_loader)
    return model, test_loss, acc


def train_main(args):

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize arguments based on dataset chosen
    if args.dataset == "first5_mnist":
        train_loader = mnist_first5_train_loader(args.batch_size)
        test_loader = mnist_first5_test_loader(args.test_batch_size)
        args.output_size = 5
        args.input_channel = 1
    elif args.dataset == "last5_mnist":
        train_loader = mnist_last5_train_loader(args.batch_size)
        test_loader = mnist_last5_test_loader(args.test_batch_size)
        args.output_size = 5
        args.input_channel = 1
    elif args.dataset == "mnist":
        train_loader = mnist_train_loader(args.batch_size)
        test_loader = mnist_test_loader(args.test_batch_size)
        args.output_size = 10
        args.input_channel = 1
    elif args.dataset == "cifar10":
        train_loader = cifar10_train_loader(args.batch_size)
        test_loader = cifar10_test_loader(args.test_batch_size)
        args.output_size = 10
        args.input_channel = 3

    # Initialize models based on architecture chosen
    if args.arch == "lenet5":
        arch = LeNet5
    elif args.arch == "lenet5_halfed":
        arch = LeNet5Halfed
    elif args.arch == "resnet18":
        arch = ResNet18

    # Create the directory for saving if it does not exist
    create_op_dir(args.output_dir)

    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.arch}")
    results = []

    for i in range(len(args.seeds)):
        print(f"Iteration {i}, Seed {args.seeds[i]}")

        np.random.seed(args.seeds[i])
        torch.manual_seed(args.seeds[i])

        model, test_loss, acc = train_model(
            arch(input_channel=args.input_channel, output_size=args.output_size),
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            config_args=args,
        )

        # Save the model
        torch.save(
            model.state_dict(),
            args.output_dir + f"{args.dataset}_{args.arch}_{args.seeds[i]}",
        )

        # save the results in list first
        results.append(
            {"iteration": i, "seed": args.seeds[i], "loss": test_loss, "acc": acc}
        )

    # Save all the results
    if args.save_results:
        save_results(f"{args.dataset}_{args.arch}", results, args.results_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["first5_mnist", "last5_mnist", "mnist", "cifar10"],
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="lenet5",
        choices=["lenet5", "lenet5_halfed", "resnet18"],
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--test_batch_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--no_cuda", type=bool, default=False)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_results", type=bool, default=True)
    parser.add_argument("--results_dir", type=str, default="./results/source_net/")
    parser.add_argument("--output_dir", type=str, default="./cache/models/")

    args = parser.parse_args()
    args.seeds = SEEDS

    train_main(args)
