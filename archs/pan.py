from torch import nn


class PAN(nn.Module):
    def __init__(self, input_size):
        super(PAN, self).__init__()
        self.fc1 = nn.Linear(input_size, 80)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(80, 60)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(60, 80)

    def forward(self, data, out_feature=False):
        output = self.fc1(data)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        return output
