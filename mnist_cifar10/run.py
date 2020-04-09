import numpy as np
import torch
from archs.lenet5 import LeNet5
from config import Config
from .dataloaders import (
    dual_channel_mnist_test_loader,
    dual_channel_cifar10_test_loader,
)
from .logits_op import concat_naive

args = Config()
use_cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}


def cifar10_and_mnist_main():
    dataloaders = [dual_channel_mnist_test_loader, dual_channel_cifar10_test_loader]
    for i in range(len(args.seeds)):
        seed = args.seeds[i]
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Iteration: {i+1}, Seed: {seed}")

        # Load models
        mnist_model = LeNet5(input_channel=1).to(device)
        mnist_model.load_state_dict(
            torch.load(args.output_dir + f"mnist_model_{args.seeds[i]}")
        )
        cifar10_model = LeNet5(input_channel=3).to(device)
        cifar10_model.load_state_dict(
            torch.load(args.output_dir + f"cifar10_model_{args.seeds[i]}")
        )

        print("1.1 Naive")
        concat_naive(args, mnist_model, cifar10_model, device, dataloaders)
