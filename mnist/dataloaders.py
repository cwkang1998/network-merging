from torch.utils.data import DataLoader
from .datasets import (
    mnist_first5_train_dataset,
    mnist_first5_test_dataset,
    mnist_last5_train_dataset,
    mnist_last5_test_dataset,
    mnist_combined_train_dataset,
    mnist_combined_test_dataset
)


def mnist_first5_train_loader(batch_size):
    return DataLoader(mnist_first5_train_dataset, batch_size=batch_size, shuffle=True,)


def mnist_first5_test_loader(batch_size):
    return DataLoader(mnist_first5_test_dataset, batch_size=batch_size, shuffle=True,)


def mnist_last5_train_loader(batch_size):
    return DataLoader(mnist_last5_train_dataset, batch_size=batch_size, shuffle=True,)


def mnist_last5_test_loader(batch_size):
    return DataLoader(mnist_last5_test_dataset, batch_size=batch_size, shuffle=True,)

def mnist_combined_train_loader(batch_size):
    return DataLoader(mnist_combined_train_dataset, batch_size=batch_size, shuffle=True,)


def mnist_combined_test_loader(batch_size):
    return DataLoader(mnist_combined_test_dataset, batch_size=batch_size, shuffle=True,)
