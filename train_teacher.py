import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from utils import create_op_dir

from config import Config
from dataloaders.mnist_dataloader import (
    first5_train_loader,
    first5_test_loader,
    last5_train_loader,
    last5_test_loader,
)
from models.lenet5 import LeNet5

seed = 1  # Arbitrary seeds


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


def train_model(arch, device, train_loader, test_loader, config_args):
    model = arch().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=config_args.lr, momentum=config_args.momentum
    )

    for epoch in range(1, config_args.epochs + 1):
        train(config_args, model, device, train_loader, optimizer, epoch)
        test(config_args, model, device, test_loader)
    return model


def main():
    # Load the configurations
    args = Config()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    print("First 5 MNIST disjointed model")
    first5_mnist_model = train_model(
        LeNet5,
        device=device,
        train_loader=first5_train_loader,
        test_loader=first5_test_loader,
        config_args=args,
    )

    print("Last 5 MNIST disjointed model")
    last5_mnist_model = train_model(
        LeNet5,
        device=device,
        train_loader=last5_train_loader,
        test_loader=last5_test_loader,
        config_args=args,
    )

    # Save all source models
    create_op_dir(args.output_dir)
    torch.save(first5_mnist_model.state_dict(), args.output_dir + "first5_mnist_model")
    torch.save(last5_mnist_model.state_dict(), args.output_dir + "last5_mnist_model")


if __name__ == "__main__":
    main()
