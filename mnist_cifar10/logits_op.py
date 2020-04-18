import torch
import torch.nn.functional as F


def concat_naive(args, model1, model2, device, test_loaders):
    """
    Naive concatenation, taking the lanel with highest confidence as prediction(Max)
    """
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    test_loaders_len = 0
    for test_loader in test_loaders:
        for m1_data, m2_data, target in test_loader:
            m1_data, m2_data, target = m1_data.to(device), m2_data.to(device), target.to(device)
            output1 = model1(m1_data)
            output2 = model2(m2_data)
            combined_output = torch.cat([output1, output2], dim=1)
            test_loss += F.cross_entropy(
                combined_output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = combined_output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loaders_len += len(test_loader.dataset)
    test_loss /= test_loaders_len

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            test_loaders_len,
            100.0 * correct / test_loaders_len,
        )
    )