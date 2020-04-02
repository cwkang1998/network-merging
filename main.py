import numpy as np
import torch
from dataloaders.mnist import mnist_combined_test_loader
from models.lenet5 import LeNet5
from concat.logits_operations.disjointed_mnist import (
    concat_naive,
    concat_overall_ratio,
    concat_ratio,
    concat_std,
    concat_thirdQ,
)
from concat.ensemble_operations.disjointed_mnist import (
    concat_gauss_avg,
    concat_poisson_avg,
    concat_5_gauss_avg,
    concat_5_gauss_vote,
    concat_5_poisson_avg,
    concat_5_poisson_vote,
    concat_diff_gauss_avg,
    concat_diff_gauss_vote,
    concat_diff_poisson_avg,
    concat_diff_poisson_vote,
    concat_flip_avg,
    concat_flip_vote,
    concat_randcrop_avg,
    concat_randcrop_vote,
)

from config import Config

args = Config()

use_cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

# Load the trained models
first5_mnist_model = LeNet5().to(device)
first5_mnist_model.load_state_dict(torch.load(args.output_dir + "first5_mnist_model"))

last5_mnist_model = LeNet5().to(device)
last5_mnist_model.load_state_dict(torch.load(args.output_dir + "last5_mnist_model"))


def mnist_main():
    print("Method 1: Concatenation with logits operations.")
    print("1.1 Naive")
    concat_naive(
        args, first5_mnist_model, last5_mnist_model, device, mnist_combined_test_loader
    )

    print("1.2 Standard Deviation")
    concat_std(
        args, first5_mnist_model, last5_mnist_model, device, mnist_combined_test_loader
    )

    print("1.3 Individual Ratio")
    concat_ratio(
        args, first5_mnist_model, last5_mnist_model, device, mnist_combined_test_loader
    )

    print("1.4 Combined Ratio")
    concat_overall_ratio(
        args, first5_mnist_model, last5_mnist_model, device, mnist_combined_test_loader
    )

    print("1.5 Third Quartile Difference")
    concat_thirdQ(
        args, first5_mnist_model, last5_mnist_model, device, mnist_combined_test_loader
    )


def ensemble_noise_main():
    """
    Make decision via multiple passes through the same network, 
    with different augmentation each pass, mainly using usual argmax
    """
    print("Gauss Avg")
    concat_gauss_avg(
        args, first5_mnist_model, last5_mnist_model, device, mnist_combined_test_loader
    )

    print("Poisson Avg")
    concat_poisson_avg(
        args, first5_mnist_model, last5_mnist_model, device, mnist_combined_test_loader
    )

    print("5 Gauss Avg")
    concat_5_gauss_avg(
        args, first5_mnist_model, last5_mnist_model, device, mnist_combined_test_loader
    )

    print("5 Gauss Vote")
    concat_5_gauss_vote(
        args, first5_mnist_model, last5_mnist_model, device, mnist_combined_test_loader
    )

    print("5 Poisson Avg")
    concat_5_poisson_avg(
        args, first5_mnist_model, last5_mnist_model, device, mnist_combined_test_loader
    )

    print("5 Poisson Vote")
    concat_5_poisson_vote(
        args, first5_mnist_model, last5_mnist_model, device, mnist_combined_test_loader
    )

    print("Varying Gauss Stds Avg")
    concat_diff_gauss_avg(
        args, first5_mnist_model, last5_mnist_model, device, mnist_combined_test_loader
    )

    print("Varying Gauss Stds Vote")
    concat_diff_gauss_vote(
        args, first5_mnist_model, last5_mnist_model, device, mnist_combined_test_loader
    )

    print("Varying Poisson Rate Avg")
    concat_diff_poisson_avg(
        args, first5_mnist_model, last5_mnist_model, device, mnist_combined_test_loader
    )

    print("Varying Poisson Rate Avg")
    concat_diff_poisson_vote(
        args, first5_mnist_model, last5_mnist_model, device, mnist_combined_test_loader
    )


def ensemble_flip_randcrop_main():
    """
    Make decision via multiple passes through the same network, 
    with different augmentation each pass, mainly using usual argmax
    """
    concat_flip_avg(
        args, first5_mnist_model, last5_mnist_model, device, mnist_combined_test_loader
    )

    concat_flip_vote(
        args, first5_mnist_model, last5_mnist_model, device, mnist_combined_test_loader
    )

    concat_randcrop_avg(
        args, first5_mnist_model, last5_mnist_model, device, mnist_combined_test_loader
    )

    concat_randcrop_vote(
        args, first5_mnist_model, last5_mnist_model, device, mnist_combined_test_loader
    )


if __name__ == "__main__":
    # Statistical methods on logits
    # Operations on logits to make decisions
    # mnist_main()

    # Statistical significant test with list of seeds
    # for i in range(len(args.seeds)):
    #     seed = args.seeds[i]
    #     print(f"Iteration: {i+1}, Seed: {seed}")
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    ensemble_noise_main()

    # ensemble_flip_randcrop_main()
