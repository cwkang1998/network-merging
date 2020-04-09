import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.augments import (
    apply_gaussian,
    apply_poisson,
    apply_hflip,
    apply_vflip,
    apply_random_crop,
)



def concat_gauss_avg(args, model1, model2, device, test_loader):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs_list = []

        # plt.figure(1)
        # plt.imshow(data[0][0])
        # Original image
        output1 = model1(data)
        output2 = model2(data)
        ori_output = torch.cat([output1, output2], dim=1)
        outputs_list.append(ori_output)

        # Gaussian noise added
        gauss_data = apply_gaussian(device=device, data=data)
        # plt.figure(2)
        # plt.imshow(gauss_data[0][0])
        # plt.show()
        gauss_output1 = model1(gauss_data)
        gauss_output2 = model2(gauss_data)
        gauss_output = torch.cat([gauss_output1, gauss_output2], dim=1)
        outputs_list.append(gauss_output)

        combined_output = torch.mean(torch.stack(outputs_list), dim=0)

        test_loss += F.cross_entropy(
            combined_output, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = combined_output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def concat_5_gauss_avg(args, model1, model2, device, test_loader):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs_list = []

        # Original image
        output1 = model1(data)
        output2 = model2(data)
        ori_output = torch.cat([output1, output2], dim=1)
        outputs_list.append(ori_output)

        # Gaussian noise added
        for i in range(5):
            gauss_data = apply_gaussian(device=device, data=data)
            gauss_output1 = model1(gauss_data)
            gauss_output2 = model2(gauss_data)
            gauss_output = torch.cat([gauss_output1, gauss_output2], dim=1)
            outputs_list.append(gauss_output)

        combined_output = torch.mean(torch.stack(outputs_list), dim=0)

        test_loss += F.cross_entropy(
            combined_output, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = combined_output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def concat_5_gauss_vote(args, model1, model2, device, test_loader):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs_list = []
        pred_list = []

        # Original image
        output1 = model1(data)
        output2 = model2(data)
        ori_output = torch.cat([output1, output2], dim=1)
        outputs_list.append(ori_output)
        pred_list.append(ori_output.argmax(dim=1, keepdim=True))

        # Gaussian noise added
        for i in range(5):
            gauss_data = apply_gaussian(device=device, data=data)
            gauss_output1 = model1(gauss_data)
            gauss_output2 = model2(gauss_data)
            gauss_output = torch.cat([gauss_output1, gauss_output2], dim=1)
            outputs_list.append(gauss_output)
            pred_list.append(gauss_output.argmax(dim=1, keepdim=True))

        combined_output = torch.mean(torch.stack(outputs_list), dim=0)
        pred_list = torch.stack(pred_list)

        test_loss += F.cross_entropy(
            combined_output, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = torch.mode(pred_list, dim=0)[0]
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def concat_diff_gauss_avg(args, model1, model2, device, test_loader):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs_list = []

        # Original image
        output1 = model1(data)
        output2 = model2(data)
        ori_output = torch.cat([output1, output2], dim=1)
        outputs_list.append(ori_output)

        # Gaussian noise added
        stds = [0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
        for s in stds:
            gauss_data = apply_gaussian(device=device, data=data, std=s)
            gauss_output1 = model1(gauss_data)
            gauss_output2 = model2(gauss_data)
            gauss_output = torch.cat([gauss_output1, gauss_output2], dim=1)
            outputs_list.append(gauss_output)

        combined_output = torch.mean(torch.stack(outputs_list), dim=0)

        test_loss += F.cross_entropy(
            combined_output, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = combined_output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def concat_diff_gauss_vote(args, model1, model2, device, test_loader):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs_list = []
        pred_list = []

        # Original image
        output1 = model1(data)
        output2 = model2(data)
        ori_output = torch.cat([output1, output2], dim=1)
        outputs_list.append(ori_output)
        pred_list.append(ori_output.argmax(dim=1, keepdim=True))

        # Gaussian noise added
        stds = [0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
        for s in stds:
            gauss_data = apply_gaussian(device=device, data=data, std=s)
            gauss_output1 = model1(gauss_data)
            gauss_output2 = model2(gauss_data)
            gauss_output = torch.cat([gauss_output1, gauss_output2], dim=1)
            outputs_list.append(gauss_output)
            pred_list.append(gauss_output.argmax(dim=1, keepdim=True))

        combined_output = torch.mean(torch.stack(outputs_list), dim=0)
        pred_list = torch.stack(pred_list)

        test_loss += F.cross_entropy(
            combined_output, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = torch.mode(pred_list, dim=0)[0]
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def concat_poisson_avg(args, model1, model2, device, test_loader):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs_list = []

        # plt.figure(1)
        # plt.imshow(data[0][0])

        # Original image
        output1 = model1(data)
        output2 = model2(data)
        ori_output = torch.cat([output1, output2], dim=1)
        outputs_list.append(ori_output)

        # Poisson noise added
        pois_data = apply_poisson(device=device, data=data, rate=0.5)
        # plt.figure(2)
        # plt.imshow(pois_data[0][0])
        # plt.show()
        pois_output1 = model1(pois_data)
        pois_output2 = model2(pois_data)
        pois_output = torch.cat([pois_output1, pois_output2], dim=1)
        outputs_list.append(pois_output)

        combined_output = torch.mean(torch.stack(outputs_list), dim=0)

        test_loss += F.cross_entropy(
            combined_output, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = combined_output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def concat_5_poisson_avg(args, model1, model2, device, test_loader):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs_list = []

        # Original image
        output1 = model1(data)
        output2 = model2(data)
        ori_output = torch.cat([output1, output2], dim=1)
        outputs_list.append(ori_output)

        # Poisson noise added
        for i in range(5):
            pois_data = apply_poisson(device=device, data=data, rate=0.5)
            pois_output1 = model1(pois_data)
            pois_output2 = model2(pois_data)
            pois_output = torch.cat([pois_output1, pois_output2], dim=1)
            outputs_list.append(pois_output)

        combined_output = torch.mean(torch.stack(outputs_list), dim=0)

        test_loss += F.cross_entropy(
            combined_output, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = combined_output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def concat_5_poisson_vote(args, model1, model2, device, test_loader):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs_list = []
        pred_list = []

        # Original image
        output1 = model1(data)
        output2 = model2(data)
        ori_output = torch.cat([output1, output2], dim=1)
        outputs_list.append(ori_output)
        pred_list.append(ori_output.argmax(dim=1, keepdim=True))

        # Gaussian noise added
        for i in range(5):
            pois_data = apply_poisson(device=device, data=data, rate=0.5)
            pois_output1 = model1(pois_data)
            pois_output2 = model2(pois_data)
            pois_output = torch.cat([pois_output1, pois_output2], dim=1)
            outputs_list.append(pois_output)
            pred_list.append(pois_output.argmax(dim=1, keepdim=True))

        combined_output = torch.mean(torch.stack(outputs_list), dim=0)
        pred_list = torch.stack(pred_list)

        test_loss += F.cross_entropy(
            combined_output, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = torch.mode(pred_list, dim=0)[0]
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def concat_diff_poisson_avg(args, model1, model2, device, test_loader):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs_list = []

        # Original image
        output1 = model1(data)
        output2 = model2(data)
        ori_output = torch.cat([output1, output2], dim=1)
        outputs_list.append(ori_output)

        # Poisson noise added
        rates = [0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
        for r in rates:
            pois_data = apply_poisson(device=device, data=data, rate=r)
            pois_output1 = model1(pois_data)
            pois_output2 = model2(pois_data)
            pois_output = torch.cat([pois_output1, pois_output2], dim=1)
            outputs_list.append(pois_output)

        combined_output = torch.mean(torch.stack(outputs_list), dim=0)

        test_loss += F.cross_entropy(
            combined_output, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = combined_output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def concat_diff_poisson_vote(args, model1, model2, device, test_loader):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs_list = []
        pred_list = []

        # Original image
        output1 = model1(data)
        output2 = model2(data)
        ori_output = torch.cat([output1, output2], dim=1)
        outputs_list.append(ori_output)
        pred_list.append(ori_output.argmax(dim=1, keepdim=True))

        # Gaussian noise added
        rates = [0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
        for r in rates:
            pois_data = apply_poisson(device=device, data=data, rate=r)
            pois_output1 = model1(pois_data)
            pois_output2 = model2(pois_data)
            pois_output = torch.cat([pois_output1, pois_output2], dim=1)
            outputs_list.append(pois_output)
            pred_list.append(pois_output.argmax(dim=1, keepdim=True))

        combined_output = torch.mean(torch.stack(outputs_list), dim=0)
        pred_list = torch.stack(pred_list)

        test_loss += F.cross_entropy(
            combined_output, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = torch.mode(pred_list, dim=0)[0]
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def concat_flip_avg(args, model1, model2, device, test_loader):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs_list = []

        # Original image
        output1 = model1(data)
        output2 = model2(data)
        ori_output = torch.cat([output1, output2], dim=1)
        outputs_list.append(ori_output)

        # Horizontally flipped data
        hflip_data = apply_hflip(data=data)
        hflip_output1 = model1(hflip_data)
        hflip_output2 = model2(hflip_data)
        hflip_output = torch.cat([hflip_output1, hflip_output2], dim=1)
        outputs_list.append(hflip_output)

        # Vertically flipped data
        vflip_data = apply_vflip(data=data)
        vflip_output1 = model1(vflip_data)
        vflip_output2 = model2(vflip_data)
        vflip_output = torch.cat([vflip_output1, vflip_output2], dim=1)
        outputs_list.append(vflip_output)

        combined_output = torch.mean(torch.stack(outputs_list), dim=0)

        test_loss += F.cross_entropy(
            combined_output, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = combined_output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def concat_flip_vote(args, model1, model2, device, test_loader):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs_list = []
        pred_list = []

        # Original image
        output1 = model1(data)
        output2 = model2(data)
        ori_output = torch.cat([output1, output2], dim=1)
        outputs_list.append(ori_output)
        pred_list.append(ori_output.argmax(dim=1, keepdim=True))

        # Horizontally flipped data
        hflip_data = apply_hflip(data=data)
        hflip_output1 = model1(hflip_data)
        hflip_output2 = model2(hflip_data)
        hflip_output = torch.cat([hflip_output1, hflip_output2], dim=1)
        outputs_list.append(hflip_output)
        pred_list.append(hflip_output.argmax(dim=1, keepdim=True))

        # Vertically flipped data
        vflip_data = apply_vflip(data=data)
        vflip_output1 = model1(vflip_data)
        vflip_output2 = model2(vflip_data)
        vflip_output = torch.cat([vflip_output1, vflip_output2], dim=1)
        outputs_list.append(vflip_output)
        pred_list.append(vflip_output.argmax(dim=1, keepdim=True))

        combined_output = torch.mean(torch.stack(outputs_list), dim=0)
        pred_list = torch.stack(pred_list)

        test_loss += F.cross_entropy(
            combined_output, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = torch.mode(pred_list, dim=0)[0]
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def concat_randcrop_avg(args, model1, model2, device, test_loader):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs_list = []

        # Original image
        output1 = model1(data)
        output2 = model2(data)
        ori_output = torch.cat([output1, output2], dim=1)
        outputs_list.append(ori_output)

        # random cropped data
        randcrop_data = apply_random_crop(data=data, size=28)
        randcrop_output1 = model1(randcrop_data)
        randcrop_output2 = model2(randcrop_data)
        randcrop_output = torch.cat([randcrop_output1, randcrop_output2], dim=1)
        outputs_list.append(randcrop_output)

        combined_output = torch.mean(torch.stack(outputs_list), dim=0)

        test_loss += F.cross_entropy(
            combined_output, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = combined_output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def concat_randcrop_vote(args, model1, model2, device, test_loader):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs_list = []
        pred_list = []

        # Original image
        output1 = model1(data)
        output2 = model2(data)
        ori_output = torch.cat([output1, output2], dim=1)
        outputs_list.append(ori_output)
        pred_list.append(ori_output.argmax(dim=1, keepdim=True))

        # random cropped data
        randcrop_data = apply_random_crop(data=data, size=28)
        randcrop_output1 = model1(randcrop_data)
        randcrop_output2 = model2(randcrop_data)
        randcrop_output = torch.cat([randcrop_output1, randcrop_output2], dim=1)
        outputs_list.append(randcrop_output)
        pred_list.append(randcrop_output.argmax(dim=1, keepdim=True))

        combined_output = torch.mean(torch.stack(outputs_list), dim=0)
        pred_list = torch.stack(pred_list)

        test_loss += F.cross_entropy(
            combined_output, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = torch.mode(pred_list, dim=0)[0]
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
