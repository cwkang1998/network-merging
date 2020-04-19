import torch
import torch.nn.functional as F
from .logits_stats import naive


def mean_agg_test(
    args, model1, model2, aug_list, device, test_loaders, include_naive=True
):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    test_loaders_len = 0
    for test_loader in test_loaders:
        for m1_data, m2_data, target in test_loader:
            m1_data, m2_data, target = (
                m1_data.to(device),
                m2_data.to(device),
                target.to(device),
            )
            outputs_list = []

            if include_naive:
                outputs_list.append(naive(args, model1, model2, m1_data, m2_data))

            for aug in aug_list:
                for i in range(aug["iter"]):
                    aug_m1_data = aug["func"](
                        device=device, data=m1_data, **aug["kwargs"]
                    )
                    aug_m2_data = aug["func"](
                        device=device, data=m2_data, **aug["kwargs"]
                    )
                    outputs_list.append(
                        naive(args, model1, model2, aug_m1_data, aug_m2_data)
                    )

            combined_output = torch.mean(torch.stack(outputs_list), dim=0)

            test_loss += F.cross_entropy(
                combined_output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = combined_output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loaders_len += len(test_loader.dataset)
    test_loss /= test_loaders_len
    acc = 100.0 * correct / test_loaders_len

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, test_loaders_len, acc
        )
    )
    return test_loss, acc


def voting_agg_test(
    args, model1, model2, aug_list, device, test_loaders, include_naive=True
):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    test_loaders_len = 0
    for test_loader in test_loaders:
        for m1_data, m2_data, target in test_loader:
            m1_data, m2_data, target = (
                m1_data.to(device),
                m2_data.to(device),
                target.to(device),
            )
            outputs_list = []
            pred_list = []

            if include_naive:
                output = naive(args, model1, model2, m1_data, m2_data)
                outputs_list.append(output)
                pred_list.append(output.argmax(dim=1, keepdim=True))

            for aug in aug_list:
                for i in range(aug["iter"]):
                    aug_m1_data = aug["func"](
                        device=device, data=m1_data, **aug["kwargs"]
                    )
                    aug_m2_data = aug["func"](
                        device=device, data=m2_data, **aug["kwargs"]
                    )
                    output = naive(args, model1, model2, aug_m1_data, aug_m2_data)
                    outputs_list.append(output)
                    pred_list.append(output.argmax(dim=1, keepdim=True))

            combined_output = torch.mean(torch.stack(outputs_list), dim=0)
            pred_list = torch.stack(pred_list)

            test_loss += F.cross_entropy(
                combined_output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = torch.mode(pred_list, dim=0)[0]
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loaders_len += len(test_loader.dataset)
    test_loss /= test_loaders_len
    acc = 100.0 * correct / test_loaders_len

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, test_loaders_len, acc
        )
    )
    return test_loss, acc
