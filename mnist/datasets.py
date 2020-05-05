from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from config import DATA_DIR


class DisjointMNIST(MNIST):
    def __init__(self, root, start_idx=0, end_idx=10, **kwargs):
        super(DisjointMNIST, self).__init__(root, **kwargs)
        self.start_idx = start_idx
        self.end_idx = end_idx
        classes = [
            "0 - zero",
            "1 - one",
            "2 - two",
            "3 - three",
            "4 - four",
            "5 - five",
            "6 - six",
            "7 - seven",
            "8 - eight",
            "9 - nine",
        ]
        # Slicing begins here
        classes = classes[start_idx:end_idx]
        sliced_labels_idx = [
            i
            for i in range(len(self.targets))
            if self.targets[i] in list(range(self.start_idx, self.end_idx))
        ]
        self.data = self.data[sliced_labels_idx]
        self.targets = self.targets[sliced_labels_idx] - self.start_idx


mnist_first5_train_dataset = DisjointMNIST(
    DATA_DIR,
    start_idx=0,
    end_idx=5,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

mnist_first5_test_dataset = DisjointMNIST(
    DATA_DIR,
    start_idx=0,
    end_idx=5,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

mnist_last5_train_dataset = DisjointMNIST(
    DATA_DIR,
    start_idx=5,
    end_idx=10,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

mnist_last5_test_dataset = DisjointMNIST(
    DATA_DIR,
    start_idx=5,
    end_idx=10,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

mnist_combined_train_dataset = MNIST(
    DATA_DIR,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)


mnist_combined_test_dataset = MNIST(
    DATA_DIR,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)


# Extras here
mnist_6_7_train_dataset = DisjointMNIST(
    DATA_DIR,
    start_idx=6,
    end_idx=8,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

mnist_6_7_test_dataset = DisjointMNIST(
    DATA_DIR,
    start_idx=6,
    end_idx=8,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)


mnist_8_9_train_dataset = DisjointMNIST(
    DATA_DIR,
    start_idx=8,
    end_idx=10,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

mnist_8_9_test_dataset = DisjointMNIST(
    DATA_DIR,
    start_idx=8,
    end_idx=10,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)