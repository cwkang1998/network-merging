from argparse import ArgumentParser
import numpy as np
import torch
from utils import save_results
from mnist.dataloaders import (
    mnist_first5_test_loader,
    mnist_last5_test_loader,
    mnist_combined_test_loader,
)
from mnist_cifar10.dataloaders import (
    mnist_test_loader,
    cifar10_test_loader,
    dual_channel_cifar10_test_loader,
    dual_channel_mnist_test_loader,
)
from merging import logits_statistics
from archs.lenet5 import LeNet5, LeNet5Halfed
from config import SEEDS


def main(args):
    # Initialize arguments based on dataset chosen
    if args.dataset == "disjoint_mnist":
        test_loader = mnist_combined_test_loader(args.test_batch_size)
        args.d1 = "first5_mnist"
        args.d2 = "last5_mnist"
        args.m1_input_channel = 1
        args.m2_input_channel = 1
        args.output_size = 5
    elif args.dataset == "mnist_cifar10":
        test_loader = mnist_last5_test_loader(args.test_batch_size)
        args.d1 = "mnist"
        args.d2 = "cifar10"
        args.m1_input_channel = 1
        args.m2_input_channel = 3
        args.output_size = 10

    # Initialize models based on architecture chosen
    if args.arch == "lenet5":
        arch = LeNet5
    elif args.arch == "lenet5_halfed":
        arch = LeNet5Halfed

    # Initialize logits statistics function
    if args.experiment == "logits_statistics":
        experiment = logits_statistics
    elif args.experiment == "multi_pass_augment":
        pass
    elif args.experiment == "smart_coord":
        pass

    # Running the test
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.arch}")
    results = []

    for i in range(len(args.seeds)):
        seed = args.seeds[i]
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"\nIteration: {i+1}, Seed: {seed}")

        # Load models
        model1 = arch(
            input_channel=args.m1_input_channel, output_size=args.output_size
        ).to(args.device)
        model1.load_state_dict(
            torch.load(args.output_dir + f"{args.d1}_{args.arch}_{args.seeds[i]}")
        )
        model2 = arch(
            input_channel=args.m2_input_channel, output_size=args.output_size
        ).to(args.device)
        model2.load_state_dict(
            torch.load(args.output_dir + f"{args.d2}_{args.arch}_{args.seeds[i]}")
        )

        # Running the experiment
        result = experiment(args, model1, model2, device, test_loader)

        # Adding more info to the result to be saved
        for r in result:
            r.update({"iteration": i, "seed": args.seeds[i]})
        results.extend(result)

    # Save the results
    if args.save_results:
        save_results(
            f"{args.dataset}_{args.arch}", results, f"{args.results_dir}{args.experiment}/"
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="disjoint_mnist",
        choices=["disjoint_mnist", "mnist_cifar10"],
    )
    parser.add_argument(
        "--arch", type=str, default="lenet5", choices=["lenet5", "lenet5_halfed"]
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="logits_statistics",
        choices=["logits_statistics", "multi_pass_augment", "smart_coord"],
    )
    parser.add_argument("--test_batch_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--no_cuda", type=bool, default=False)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_results", type=bool, default=True)
    parser.add_argument("--results_dir", type=str, default="./results/merge/")
    parser.add_argument("--output_dir", type=str, default="./cache/models/")

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    args.seeds = SEEDS
    args.device = device
    main(args)
