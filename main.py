import numpy as np
import torch
from dataloaders.mnist_dataloader import combined_test_loader
from models.lenet5 import LeNet5
from concat.logits_operations.disjointed_mnist import (
    concat_naive,
    concat_overall_ratio,
    concat_ratio,
    concat_std,
    concat_thirdQ,
)

from config import Config

args = Config()

use_cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}


def mnist_main():
    # Load the trained models
    first5_mnist_model = LeNet5().to(device)
    first5_mnist_model.load_state_dict(
        torch.load(args.output_dir + "first5_mnist_model")
    )

    last5_mnist_model = LeNet5().to(device)
    last5_mnist_model.load_state_dict(torch.load(args.output_dir + "last5_mnist_model"))

    print("Method 1: Concatenation with logits operations.")
    print("1.1 Naive")
    concat_naive(
        args, first5_mnist_model, last5_mnist_model, device, combined_test_loader
    )

    print("1.2 Standard Deviation")
    concat_std(
        args, first5_mnist_model, last5_mnist_model, device, combined_test_loader
    )

    print("1.3 Individual Ratio")
    concat_ratio(
        args, first5_mnist_model, last5_mnist_model, device, combined_test_loader
    )

    print("1.4 Combined Ratio")
    concat_overall_ratio(
        args, first5_mnist_model, last5_mnist_model, device, combined_test_loader
    )

    print("1.5 Third Quartile Difference")
    concat_thirdQ(
        args, first5_mnist_model, last5_mnist_model, device, combined_test_loader
    )


if __name__ == "__main__":
    mnist_main()
