import torch
import torch.nn.functional as F


def test(args, model1, model2, model_eval, device, test_loader):
    model1.eval()
    model2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model_eval(args, model1, model2, data)

        test_loss += F.cross_entropy(
            output, target, reduction="sum"
        ).item()  # sum up batch loss
        pred = output.argmax(
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


def naive(args, model1, model2, data):
    output1 = model1(data)
    output2 = model2(data)
    combined_output = torch.cat([output1, output2], dim=1)
    return combined_output


def std(args, model1, model2, data):
    output1 = model1(data)
    output2 = model2(data)

    combined_output = [0] * len(data)
    for i in range(len(combined_output)):
        if output1[i].std() < output2[i].std():
            combined_output[i] = torch.cat(
                [
                    output1[i],
                    torch.Tensor([torch.min(output1[i])] * len(output1[i])).to(
                        args.device
                    ),
                ]
            )
        else:
            combined_output[i] = torch.cat(
                [
                    torch.Tensor([torch.min(output2[i])] * len(output2[i])).to(
                        args.device
                    ),
                    output2[i],
                ]
            )
    combined_output = torch.stack(combined_output, 0)
    return combined_output


def ratio(args, model1, model2, data):
    output1 = model1(data)
    output2 = model2(data)

    def calc_ratio(arr):
        total = torch.sum(arr)
        arr_calc = list(arr)
        for i in range(len(arr_calc)):
            arr_calc[i] = arr_calc[i] / total
        out = torch.stack(arr_calc, 0)
        return out

    o1_ratio = torch.stack(list(map(calc_ratio, output1)), 0)
    o2_ratio = torch.stack(list(map(calc_ratio, output2)), 0)
    combined_output = torch.cat([o1_ratio, o2_ratio], dim=1)
    return combined_output


def overall_ratio(args, model1, model2, data):
    output1 = model1(data)
    output2 = model2(data)
    combined_output = torch.cat([output1, output2], dim=1)

    def calc_ratio(arr):
        total = torch.sum(arr)
        arr_calc = list(arr)
        for i in range(len(arr_calc)):
            arr_calc[i] = arr_calc[i] / total
        out = torch.stack(arr_calc, 0)
        return out

    combined_output = torch.stack(list(map(calc_ratio, combined_output)), 0)
    return combined_output


def thirdQ(args, model1, model2, data):
    output1 = model1(data)
    output2 = model2(data)

    def percentile(t: torch.tensor, q: float):
        k = 1 + round(0.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result

    combined_output = [0] * len(data)
    for i in range(len(combined_output)):
        o1_max = torch.max(output1[i])
        o2_max = torch.max(output2[i])
        o1_q3_diff = o1_max - percentile(output1[i], 0.75)
        o2_q3_diff = o2_max - percentile(output2[i], 0.75)
        if o1_q3_diff > o2_q3_diff:
            combined_output[i] = torch.cat(
                [
                    output1[i],
                    torch.Tensor([torch.min(output1[i])] * len(output1[i])).to(
                        args.device
                    ),
                ]
            )
        else:
            combined_output[i] = torch.cat(
                [
                    torch.Tensor([torch.min(output2[i])] * len(output2[i])).to(
                        args.device
                    ),
                    output2[i],
                ]
            )
    combined_output = torch.stack(combined_output, 0)
    return combined_output
