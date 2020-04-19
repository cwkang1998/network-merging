import torch
import torch.nn.functional as F
from .logits_stats import naive


def mean_agg_test(
    args, model1, model2, aug_list, device, test_loader, include_naive=True
):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs_list = []

        if include_naive:
            outputs_list.append(naive(args, model1, model2, data))

        for aug in aug_list:
            for i in range(aug["iter"]):
                aug_data = aug["func"](device=device, data=data, **aug['kwargs'])
                outputs_list.append(naive(args, model1, model2, aug_data))

        combined_output = torch.mean(torch.stack(outputs_list), dim=0)

        test_loss += F.cross_entropy(
            combined_output, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = combined_output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100.0 * correct / len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), acc
        )
    )
    return test_loss, acc


def voting_agg_test(
    args, model1, model2, aug_list, device, test_loader, include_naive=True
):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs_list = []
        pred_list = []

        if include_naive:
            output = naive(args, model1, model2, data)
            outputs_list.append(output)
            pred_list.append(output.argmax(dim=1, keepdim=True))

        for aug in aug_list:
            for i in range(aug["iter"]):
                aug_data = aug["func"](device=device, data=data, **aug["kwargs"])
                output = naive(args, model1, model2, aug_data)
                outputs_list.append(output)
                pred_list.append(output.argmax(dim=1, keepdim=True))

        combined_output = torch.mean(torch.stack(outputs_list), dim=0)
        pred_list = torch.stack(pred_list)

        test_loss += F.cross_entropy(
            combined_output, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = torch.mode(pred_list, dim=0)[0]
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100.0 * correct / len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), acc
        )
    )
    return test_loss, acc
