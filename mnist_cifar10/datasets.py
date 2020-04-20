from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from config import DATA_DIR

mnist_classes = [
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

cifar10_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class ExtendedMNIST(MNIST):
    """
    MNIST with extended labels for use with other data in concatenation test
    """

    def __init__(self, root, extended_classes=[], **kwargs):
        super(ExtendedMNIST, self).__init__(root, **kwargs)
        self.classes.extend(extended_classes)


class ExtendedCIFAR10(CIFAR10):
    """
    MNIST with extended labels for use with other data in concatenation test
    """

    def __init__(self, root, extended_classes=[], **kwargs):
        super(ExtendedCIFAR10, self).__init__(root, **kwargs)
        extended_class_len = len(extended_classes)
        self.classes = extended_classes.extend(self.classes)
        self.targets = [t + extended_class_len for t in self.targets]


# Rgb transform
transform_rgb = transforms.Lambda(lambda img: img.convert("RGB"))


class DualChannelDatasets(Dataset):
    def __init__(self, dataset, initial_channels):
        self.dataset = dataset
        self.initial_channels = initial_channels
        if self.initial_channels == 1:
            self.transform_func = self.grayscale_to_rgb
        else:
            self.transform_func = self.rgb_to_grayscale

    def __getitem__(self, i):
        d = self.dataset[i % len(self.dataset)]
        d_trans = self.transform_func(d[0])
        if self.initial_channels == 1:
            return (d[0], d_trans, d[1])
        else:
            return (d_trans, d[0], d[1])

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def grayscale_to_rgb(data):
        image = transforms.ToPILImage()(data)
        image = transform_rgb(image)
        image = transforms.ToTensor()(image)
        return image

    @staticmethod
    def rgb_to_grayscale(data):
        image = transforms.ToPILImage()(data)
        image = transforms.Grayscale()(image)
        image = transforms.ToTensor()(image)
        return image


mnist_train_dataset = MNIST(
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
mnist_test_dataset = MNIST(
    DATA_DIR,
    train=False,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)
cifar10_train_dataset = CIFAR10(
    DATA_DIR,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
)
cifar10_test_dataset = CIFAR10(
    DATA_DIR,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
)


# Dataset for testing the mnist_cifar10 combined models
extended_mnist_test_dataset = ExtendedMNIST(
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
    extended_classes=cifar10_classes,
)

extended_cifar10_test_dataset = ExtendedCIFAR10(
    DATA_DIR,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
    extended_classes=mnist_classes,
)

dual_channel_mnist_test_dataset = DualChannelDatasets(
    dataset=extended_mnist_test_dataset, initial_channels=1
)

dual_channel_cifar10_test_dataset = DualChannelDatasets(
    dataset=extended_cifar10_test_dataset, initial_channels=3
)


# Dataset for training PAN purpose
extended_mnist_single_channel_train_dataset = ExtendedMNIST(
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
    extended_classes=cifar10_classes,
)

extended_cifar10_single_channel_train_dataset = ExtendedCIFAR10(
    DATA_DIR,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor()
        ]
    ),
    extended_classes=mnist_classes,
)


extended_mnist_single_channel_test_dataset = ExtendedMNIST(
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
    extended_classes=cifar10_classes,
)

extended_cifar10_single_channel_test_dataset = ExtendedCIFAR10(
    DATA_DIR,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor()
        ]
    ),
    extended_classes=mnist_classes,
)


extended_mnist_3_channel_train_dataset = ExtendedMNIST(
    DATA_DIR,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.ToPILImage(),
            transform_rgb,
            transforms.ToTensor()
        ]
    ),
    extended_classes=cifar10_classes,
)

extended_cifar10_3_channel_train_dataset = ExtendedCIFAR10(
    DATA_DIR,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
    extended_classes=mnist_classes,
)

extended_mnist_3_channel_test_dataset = ExtendedMNIST(
    DATA_DIR,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.ToPILImage(),
            transform_rgb,
            transforms.ToTensor()
        ]
    ),
    extended_classes=cifar10_classes,
)

extended_cifar10_3_channel_test_dataset = ExtendedCIFAR10(
    DATA_DIR,
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
    extended_classes=mnist_classes,
)


# Concat the datasets
mnist_cifar10_single_channel_train_dataset = ConcatDataset(
    [extended_mnist_single_channel_train_dataset, extended_cifar10_single_channel_train_dataset]
)
mnist_cifar10_single_channel_test_dataset = ConcatDataset(
    [extended_mnist_single_channel_test_dataset, extended_cifar10_single_channel_test_dataset]
)
mnist_cifar10_3_channel_train_dataset = ConcatDataset(
    [extended_mnist_3_channel_train_dataset, extended_cifar10_3_channel_train_dataset]
)
mnist_cifar10_3_channel_test_dataset = ConcatDataset(
    [extended_mnist_3_channel_test_dataset, extended_cifar10_3_channel_test_dataset]
)
