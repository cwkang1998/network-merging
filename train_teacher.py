import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from utils.files import create_op_dir
from config import Config
from dataloaders.mnist import (
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

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def train_model(model, device, train_loader, test_loader, config_args):
    model = model.to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=config_args.lr, momentum=config_args.momentum
    )

    for epoch in range(1, config_args.epochs + 1):
        train(config_args, model, device, train_loader, optimizer, epoch)
        test(config_args, model, device, test_loader)
    return model


def main(): 

    args = Config()
    ori_log_interval = args.log_interval

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    create_op_dir(args.output_dir)

    for i in range(len(args.seeds)):
        print(f"Iteration {i}, Seed {args.seeds[i]}")

        np.random.seed(args.seeds[i])
        torch.manual_seed(args.seeds[i])
        args.log_interval = ori_log_interval

        # Lenet5 models
        # print("First 5 MNIST disjointed model")
        # first5_mnist_model = train_model(
        #     LeNet5(padding=0, output_size=5),
        #     device=device,
        #     train_loader=mnist_first5_train_loader,
        #     test_loader=mnist_first5_test_loader,
        #     config_args=args,
        # )

        # print("Last 5 MNIST disjointed model")
        # last5_mnist_model = train_model(
        #     LeNet5(padding=0, output_size=5),
        #     device=device,
        #     train_loader=mnist_last5_train_loader,
        #     test_loader=mnist_last5_test_loader,
        #     config_args=args,
        # )

        print("Complete MNIST Model")
        mnist_model = train_model(
            LeNet5(padding=0),
            device=device,
            train_loader=mnist_train_loader,
            test_loader=mnist_test_loader,
            config_args=args,
        )

        print("Complete CIFAR10 Model")
        args.log_interval = 100
        cifar10_model = train_model(
            LeNet5(input_channel=3, padding=0),
            device=device,
            train_loader=cifar10_train_loader,
            test_loader=cifar10_test_loader,
            config_args=args,
        )

        # torch.save(
        #     first5_mnist_model.state_dict(),
        #     args.output_dir + f"first5_mnist_model_{args.seeds[i]}",
        # )
        # torch.save(
        #     last5_mnist_model.state_dict(),
        #     args.output_dir + f"last5_mnist_model_{args.seeds[i]}",
        # )
        torch.save(
            mnist_model.state_dict(), args.output_dir + f"mnist_model_{args.seeds[i]}",
        )
        torch.save(
            cifar10_model.state_dict(),
            args.output_dir + f"cifar10_model_{args.seeds[i]}",
        )

        # Lenet5 Half
        args.log_interval = ori_log_interval
        # print("First 5 MNIST disjointed model")
        # first5_mnist_halfed_model = train_model(
        #     LeNet5Halfed(padding=0, output_size=5),
        #     device=device,
        #     train_loader=mnist_first5_train_loader,
        #     test_loader=mnist_first5_test_loader,
        #     config_args=args,
        # )

        # print("Last 5 MNIST disjointed model")
        # last5_mnist_halfed_model = train_model(
        #     LeNet5Halfed(padding=0, output_size=5),
        #     device=device,
        #     train_loader=mnist_last5_train_loader,
        #     test_loader=mnist_last5_test_loader,
        #     config_args=args,
        # )

        print("Complete MNIST Model")
        mnist_halfed_model = train_model(
            LeNet5Halfed(padding=0),
            device=device,
            train_loader=mnist_train_loader,
            test_loader=mnist_test_loader,
            config_args=args,
        )

        print("Complete CIFAR10 Model")
        args.log_interval = 100
        cifar10_halfed_model = train_model(
            LeNet5Halfed(input_channel=3, padding=0),
            device=device,
            train_loader=cifar10_train_loader,
            test_loader=cifar10_test_loader,
            config_args=args,
        )

        # torch.save(
        #     first5_mnist_halfed_model.state_dict(),
        #     args.output_dir + f"first5_mnist_halfed_model_{args.seeds[i]}",
        # )
        # torch.save(
        #     last5_mnist_halfed_model.state_dict(),
        #     args.output_dir + f"last5_mnist_halfed_model_{args.seeds[i]}",
        # )
        torch.save(
            mnist_halfed_model.state_dict(),
            args.output_dir + f"mnist_halfed_model_{args.seeds[i]}",
        )
        torch.save(
            cifar10_halfed_model.state_dict(),
            args.output_dir + f"cifar10_halfed_model_{args.seeds[i]}",
        )


if __name__ == "__main__":
    main()
