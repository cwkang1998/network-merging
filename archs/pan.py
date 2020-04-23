import torch
from torch import nn


class PAN(nn.Module):
    def __init__(self, input_size):
        super(PAN, self).__init__()
        self.fc1 = nn.Linear(input_size, 80)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(80, 60)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(60, 2)

    def forward(self, data, out_feature=False):
        output = self.fc1(data)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        return output


class AgnosticPAN(nn.Module):
    def __init__(self, input_size):
        super(AgnosticPAN, self).__init__()
        self.fc1 = nn.Linear(input_size, 20)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(20, 10)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 2)

    def forward(self, data, out_feature=False):
        output = self.fc1(data)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        return output


def compute_agnostic_stats(data):
    mean = torch.mean(data, dim=1)
    prod = torch.prod(data, dim=1)
    std = torch.std(data, dim=1)
    sums = torch.sum(data, dim=1)
    maximum = torch.max(data, dim=1)[0]
    minimum = torch.min(data, dim=1)[0]
    res = torch.stack([mean, prod, std, sums, maximum, minimum], dim=1)
    return res
