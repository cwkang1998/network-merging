from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from config import Config

args = Config()

mnist_train_dataset = MNIST(
    args.data_dir,
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
mnist_test_dataset = MNIST(
    args.data_dir,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
cifar10_train_dataset = CIFAR10(
    args.data_dir,
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    ),
)
cifar10_test_dataset = CIFAR10(
    args.data_dir,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    ),
)

mnist_train_loader = DataLoader(
    mnist_train_dataset, batch_size=args.mnist_batch_size, shuffle=True,
)

mnist_test_loader = DataLoader(
    mnist_test_dataset, batch_size=args.test_batch_size, shuffle=True,
)

cifar10_train_loader = DataLoader(
    cifar10_train_dataset, batch_size=args.cifar10_batch_size, shuffle=True,
)

cifar10_test_loader = DataLoader(
    cifar10_test_dataset, batch_size=args.test_batch_size, shuffle=True,
)

mnist_cifar10_test_loader = DataLoader(
    ConcatDataset([mnist_test_dataset, cifar10_test_dataset]),
    batch_size=args.test_batch_size,
    shuffle=True,
)
