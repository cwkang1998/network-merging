from torch.utils.data import DataLoader
from config import Config
from .datasets import (
    mnist_train_dataset,
    mnist_test_dataset,
    cifar10_train_dataset,
    cifar10_test_dataset,
    dual_channel_mnist_dataset,
    dual_channel_cifar10_dataset,
)

args = Config()

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

dual_channel_mnist_test_loader = DataLoader(
    dual_channel_mnist_dataset, batch_size=args.test_batch_size, shuffle=True,
)

dual_channel_cifar10_test_loader = DataLoader(
    dual_channel_cifar10_dataset, batch_size=args.test_batch_size, shuffle=True,
)
