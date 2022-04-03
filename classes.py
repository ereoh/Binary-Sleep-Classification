import torch.nn as nn
import torch.nn.functional as F


class binarySleepNet(nn.Module):
    def __init__(self):
        super().__init__()
        # each input is 3000 long
        self.conv1 = nn.Conv1d(1, 1, kernel_size=2996)
        # self.b = nn.BatchNorm1d(3000)
        self.r = nn.ReLU()
        # slef.d1 = nn.Dropout(p=0.25)
        # self.conv2 = nn.Conv1d(1, 1, kernel_size=50)
        self.fc1 = nn.Linear(5, 1)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        # x = self.b(x)
        x = self.r(x)
        x = self.fc1(x)
        x = self.logsoftmax(x)
        x = self.r(x)
        return x
