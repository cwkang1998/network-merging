from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import optim
import torchvision.utils as vutils
from utils import create_op_dir
from config import SEEDS
from archs.lenet5 import LeNet5, LeNet5Halfed
from archs.resnet import ResNet18
from archs.gan import Generator


def train(args, gan, model, device, optimizer, epoch):
    model.eval()
    gan.train()
    for i in range(120):
        optimizer.zero_grad()
        z = torch.randn(args.batch_size, args.latent_dim)
        gen_imgs = gan(z)
        outputs_T, features_T = model(gen_imgs, out_feature=True)
        pred = outputs_T.data.max(1)[1]
        loss_activation = -features_T.abs().mean()
        loss_one_hot = F.cross_entropy(outputs_T, pred)
        softmax_o_T = F.softmax(outputs_T, dim=1).mean(dim=0)
        loss_information_entropy = (softmax_o_T * torch.log(softmax_o_T)).sum()
        loss = (
            loss_one_hot * args.oh
            + loss_information_entropy * args.ie
            + loss_activation * args.a
        )
        loss.backward()
        optimizer.step()
        if i == 1:
            print(
                "[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f]"
                % (
                    epoch,
                    args.epochs,
                    loss_one_hot.item(),
                    loss_information_entropy.item(),
                    loss_activation.item(),
                )
            )


def generate_and_display(args, gan):
    def show_imgs(x, new_fig=True):
        grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
        grid = grid.transpose(0, 2).transpose(0, 1)  # channels as last dimension
        if new_fig:
            plt.figure()
        plt.imshow(grid.numpy())

    noise = torch.randn(64, args.latent_dim)
    imgs = gan(noise)
    show_imgs(imgs)


def train_model(gan, model, device, config_args):
    gan = gan.to(device)
    model = model.to(device)
    optimizer = optim.Adam(gan.parameters(), lr=config_args.lr, weight_decay=5e-4)
    for epoch in range(1, config_args.epochs + 1):
        train(
            config_args, gan, model, device, optimizer, epoch,
        )
        generate_and_display(args, gan)
    return gan


def train_gan(args):

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize arguments based on dataset chosen
    if args.dataset == "disjoint_mnist":
        args.d1 = "first5_mnist"
        args.d2 = "last5_mnist"
        args.m1_input_channel = 1
        args.m2_input_channel = 1
        args.output_size = 5
    elif args.dataset == "mnist_cifar10":
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
    elif args.arch == "resnet18":
        arch = ResNet18

    # Create the directory for saving if it does not exist
    create_op_dir(args.output_dir)

    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.arch}")

    for i in range(len(args.seeds)):
        print(f"Iteration {i}, Seed {args.seeds[i]}")

        np.random.seed(args.seeds[i])
        torch.manual_seed(args.seeds[i])

        # Load models
        model1 = arch(
            input_channel=args.m1_input_channel, output_size=args.output_size
        ).to(device)
        model1.load_state_dict(
            torch.load(
                args.model_dir + f"{args.d1}_{args.arch}_{args.seeds[i]}",
                map_location=torch.device("cpu"),
            )
        )
        gan1 = train_model(
            gan=Generator(
                img_size=32, latent_dim=args.latent_dim, channels=args.m1_input_channel
            ).to(device),
            model=model1,
            device=device,
            config_args=args,
        )

        model2 = arch(
            input_channel=args.m2_input_channel, output_size=args.output_size
        ).to(device)
        model2.load_state_dict(
            torch.load(
                args.model_dir + f"{args.d2}_{args.arch}_{args.seeds[i]}",
                map_location=torch.device("cpu"),
            )
        )
        gan2 = train_model(
            gan=Generator(
                img_size=32, latent_dim=args.latent_dim, channels=args.m2_input_channel
            ).to(device),
            model=model1,
            device=device,
            config_args=args,
        )

        # Save the pan model
        torch.save(
            gan1.state_dict(),
            args.output_dir
            + f"gan_{args.dataset}({args.d1})_{args.arch}_{args.seeds[i]}",
        )
        torch.save(
            gan2.state_dict(),
            args.output_dir
            + f"gan_{args.dataset}({args.d2})_{args.arch}_{args.seeds[i]}",
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
        "--arch",
        type=str,
        default="lenet5",
        choices=["lenet5", "lenet5_halfed", "resnet18"],
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--test_batch_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.2, help="learning rate")
    parser.add_argument(
        "--latent_dim", type=int, default=100, help="dimensionality of the latent space"
    )
    parser.add_argument(
        "--img_size", type=int, default=32, help="size of each image dimension"
    )
    parser.add_argument("--oh", type=float, default=1, help="one hot loss")
    parser.add_argument("--ie", type=float, default=10, help="information entropy loss")
    parser.add_argument("--a", type=float, default=0.1, help="activation loss")
    parser.add_argument("--no_cuda", type=bool, default=False)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_results", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default="./cache/models/")
    parser.add_argument("--output_dir", type=str, default="./cache/models/gan/")

    args = parser.parse_args()
    args.seeds = SEEDS

    train_gan(args)
