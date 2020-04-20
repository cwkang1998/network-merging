from torch.utils.data import DataLoader
from .datasets import (
    mnist_train_dataset,
    mnist_test_dataset,
    cifar10_train_dataset,
    cifar10_test_dataset,
    dual_channel_mnist_test_dataset,
    dual_channel_cifar10_test_dataset,
    mnist_cifar10_single_channel_train_dataset,
    mnist_cifar10_single_channel_test_dataset,
    mnist_cifar10_3_channel_train_dataset,
    mnist_cifar10_3_channel_test_dataset,
)


def mnist_train_loader(batch_size):
    return DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True,)


def mnist_test_loader(batch_size):
    return DataLoader(mnist_test_dataset, batch_size=batch_size, shuffle=True,)


def cifar10_train_loader(batch_size):
    return DataLoader(cifar10_train_dataset, batch_size=batch_size, shuffle=True,)


def cifar10_test_loader(batch_size):
    return DataLoader(cifar10_test_dataset, batch_size=batch_size, shuffle=True,)


# For testing the combined models
def dual_channel_mnist_test_loader(batch_size):
    return DataLoader(
        dual_channel_mnist_test_dataset, batch_size=batch_size, shuffle=True,
    )


def dual_channel_cifar10_test_loader(batch_size):
    return DataLoader(
        dual_channel_cifar10_test_dataset, batch_size=batch_size, shuffle=True,
    )


# For training the PAN models
def mnist_cifar10_single_channel_train_loader(batch_size):
    return DataLoader(
        mnist_cifar10_single_channel_train_dataset, batch_size=batch_size, shuffle=True,
    )


def mnist_cifar10_single_channel_test_loader(batch_size):
    return DataLoader(
        mnist_cifar10_single_channel_test_dataset, batch_size=batch_size, shuffle=False,
    )


def mnist_cifar10_3_channel_train_loader(batch_size):
    return DataLoader(
        mnist_cifar10_3_channel_train_dataset, batch_size=batch_size, shuffle=True,
    )


def mnist_cifar10_3_channel_test_loader(batch_size):
    return DataLoader(
        mnist_cifar10_3_channel_test_dataset, batch_size=batch_size, shuffle=False,
    )
