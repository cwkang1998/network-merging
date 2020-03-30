from torchvision.datasets.mnist import MNIST


class DisjointMNIST(MNIST):
    resources = [
        (
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        ),
        (
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "d53e105ee54ea40749a09fcbcd1e9432",
        ),
        (
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "9fb629c4189551a2d022fa330f9573f3",
        ),
        (
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
            "ec29112dd5afa0611ce80d1b7f02629c",
        ),
    ]

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
        classes = classes[start_idx:end_idx]
        self.slice_data_and_save()

    def slice_data_and_save(self):
        # The slicing begins here.
        sliced_labels_idx = [
            i
            for i in range(len(self.targets))
            if self.targets[i] in list(range(self.start_idx, self.end_idx))
        ]
        self.data = self.data[sliced_labels_idx] - self.start_idx
        self.targets = self.targets[sliced_labels_idx] - self.start_idx
