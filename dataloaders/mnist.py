from torch.utils.data import DataLoader
from torchvision import transforms
from .datasets import DisjointMNIST, MNIST, CIFAR10
from config import Config

args = Config()

mnist_first5_train_loader = DataLoader(
    DisjointMNIST(
        args.data_dir,
        start_idx=0,
        end_idx=5,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.batch_size,
    shuffle=True,
)
mnist_first5_test_loader = DataLoader(
    DisjointMNIST(
        args.data_dir,
        start_idx=0,
        end_idx=5,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.test_batch_size,
    shuffle=True,
)

mnist_last5_train_loader = DataLoader(
    DisjointMNIST(
        args.data_dir,
        start_idx=5,
        end_idx=10,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.batch_size,
    shuffle=True,
)
mnist_last5_test_loader = DataLoader(
    DisjointMNIST(
        args.data_dir,
        start_idx=5,
        end_idx=10,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.test_batch_size,
    shuffle=True,
)

mnist_combined_train_loader = DataLoader(
    MNIST(
        args.data_dir,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.batch_size,
    shuffle=True,
)

mnist_combined_test_loader = DataLoader(
    MNIST(
        args.data_dir,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=args.test_batch_size,
    shuffle=True,
)
