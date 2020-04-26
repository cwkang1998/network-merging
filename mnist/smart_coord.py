import torch
import torch.nn.functional as F
from archs.pan import compute_agnostic_stats

def craft_first_5_target(target):
    pan_target = [0] * len(target)
    for i in range(len(pan_target)):
        if target[i] < 5:
            pan_target[i] = torch.tensor(1)
        else:
            pan_target[i] = torch.tensor(0)
    pan_target = torch.stack(pan_target, 0)
    return pan_target


def craft_last_5_target(target):
    pan_target = [0] * len(target)
    for i in range(len(pan_target)):
        if target[i] < 5:
            pan_target[i] = torch.tensor(0)
        else:
            pan_target[i] = torch.tensor(1)
    pan_target = torch.stack(pan_target, 0)
    return pan_target


def predict_with_logits_pan(args, model1, model2, pan1, pan2, data):
    """
    Make a prediction with PAN using features of the models.
    Here we take a winner takes all approach, as we have 2 classifier classifying 1 input with
    1 intended label(output). However, theoredically we can also go for a multi-label(multi-output)
    appproach, with multiple network working together to classify one input into multiple class.
    """

    output1 = model1(data)
    output2 = model2(data)
    p1_out = pan1(output1)
    p2_out = pan2(output2)

    # debugging
    p1_count = 0
    p2_count = 0
    p0_count = 0

    # Winner takes all
    combined_output = [0] * len(data)
    for i in range(len(combined_output)):
        if p1_out[i].max(0)[1] == 1 and p2_out[i].max(0)[1] == 0:
            # p1 true and p2 false
            combined_output[i] = torch.cat(
                [
                    output1[i],
                    torch.Tensor([torch.min(output1[i])] * len(output1[i])).to(
                        args.device
                    ),
                ]
            )
            p1_count += 1
        elif p1_out[i].max(0)[1] == 0 and p2_out[i].max(0)[1] == 1:
            # p1 false and p2 true
            combined_output[i] = torch.cat(
                [
                    torch.Tensor([torch.min(output2[i])] * len(output2[i])).to(
                        args.device
                    ),
                    output2[i],
                ]
            )
            p2_count += 1
        else:
            combined_output[i] = torch.cat([output1[i], output2[i]])
            p0_count += 1
    combined_output = torch.stack(combined_output, 0)
    print(p1_count, p2_count, p0_count)
    return combined_output


def predict_with_feature_pan(args, model1, model2, pan1, pan2, data):
    """
    Make a prediction with PAN using features of the models.
    Here we take a winner takes all approach, as we have 2 classifier classifying 1 input with
    1 intended label(output). However, theoredically we can also go for a multi-label(multi-output)
    appproach, with multiple network working together to classify one input into multiple class.
    """

    output1, feature1 = model1(data, out_feature=True)
    output2, feature2 = model2(data, out_feature=True)
    p1_out = pan1(feature1)
    p2_out = pan2(feature2)

    # debugging
    p1_count = 0
    p2_count = 0
    p0_count = 0

    # Winner takes all
    combined_output = [0] * len(data)
    for i in range(len(combined_output)):
        if p1_out[i].max(0)[1] == 1 and p2_out[i].max(0)[1] == 0:
            # p1 true and p2 false
            combined_output[i] = torch.cat(
                [
                    output1[i],
                    torch.Tensor([torch.min(output1[i])] * len(output1[i])).to(
                        args.device
                    ),
                ]
            )
            p1_count += 1
        elif p1_out[i].max(0)[1] == 0 and p2_out[i].max(0)[1] == 1:
            # p1 false and p2 true
            combined_output[i] = torch.cat(
                [
                    torch.Tensor([torch.min(output2[i])] * len(output2[i])).to(
                        args.device
                    ),
                    output2[i],
                ]
            )
            p2_count += 1
        else:
            combined_output[i] = torch.cat([output1[i], output2[i]])
            p0_count += 1
    combined_output = torch.stack(combined_output, 0)
    print(p1_count, p2_count, p0_count)
    return combined_output


def predict_with_agnostic_pan(args, model1, model2, pan1, pan2, data):
    """
    Make a prediction with PAN using agnostic features of the models.
    Here we take a winner takes all approach, as we have 2 classifier classifying 1 input with
    1 intended label(output). However, theoredically we can also go for a multi-label(multi-output)
    appproach, with multiple network working together to classify one input into multiple class.
    """

    output1, feature1 = model1(data, out_feature=True)
    output2, feature2 = model2(data, out_feature=True)

    if args.pan_type == "agnostic_feature":
        stats1 = compute_agnostic_stats(feature1)
        stats2 = compute_agnostic_stats(feature2)
    elif args.pan_type == "agnostic_logits":
        stats1 = compute_agnostic_stats(output1)
        stats2 = compute_agnostic_stats(output2)

    p1_out = pan1(stats1)
    p2_out = pan2(stats2)

    # debugging
    p1_count = 0
    p2_count = 0
    p0_count = 0

    # Winner takes all
    combined_output = [0] * len(data)
    for i in range(len(combined_output)):
        if p1_out[i].max(0)[1] == 1 and p2_out[i].max(0)[1] == 0:
            # p1 true and p2 false
            combined_output[i] = torch.cat(
                [
                    output1[i],
                    torch.Tensor([torch.min(output1[i])] * len(output1[i])).to(
                        args.device
                    ),
                ]
            )
            p1_count += 1
        elif p1_out[i].max(0)[1] == 0 and p2_out[i].max(0)[1] == 1:
            # p1 false and p2 true
            combined_output[i] = torch.cat(
                [
                    torch.Tensor([torch.min(output2[i])] * len(output2[i])).to(
                        args.device
                    ),
                    output2[i],
                ]
            )
            p2_count += 1
        else:
            combined_output[i] = torch.cat([output1[i], output2[i]])
            p0_count += 1
    combined_output = torch.stack(combined_output, 0)
    print(p1_count, p2_count, p0_count)
    return combined_output

def smart_coord_test(args, model1, model2, pan1, pan2, device, test_loader):
    model1.eval()
    model2.eval()
    pan1.eval()
    pan2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        if args.pan_type == "feature":
            output = predict_with_feature_pan(args, model1, model2, pan1, pan2, data)
        elif args.pan_type == "logits":
            output = predict_with_logits_pan(args, model1, model2, pan1, pan2, data)
        elif args.pan_type == "agnostic_feature" or args.pan_type == "agnostic_logits":
            output = predict_with_agnostic_pan(args, model1, model2, pan1, pan2, data)
        else:
            raise NotImplementedError("Not an eligible pan type.")

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
