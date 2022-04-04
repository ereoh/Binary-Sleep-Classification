import torch.nn as nn
import torch.nn.functional as F


class binarySleepNet(nn.Module):
    def __init__(self):
        super().__init__()
        # each input is 3000 long

        self.conv1 = nn.Conv1d(1, 1, 2)
        self.mp1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(1, 1, 10)
        self.mp2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(1, 1, 50)
        self.mp3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(1, 1, 100)
        self.mp4 = nn.MaxPool1d(2)

        self.conv5 = nn.Conv1d(1, 1, 100)
        self.mp5 = nn.MaxPool1d(2)

        # self.conv6 = nn.Conv1d(1, 1, 100)
        # self.mp6 = nn.MaxPool1d(2)

        # self.conv7 = nn.Conv1d(1, 1, 1000)
        # self.mp7 = nn.MaxPool1d(4)

        self.r = nn.ReLU()
        self.fc1 = nn.Linear(12, 1)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.r(x)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.r(x)
        x = self.mp2(x)

        x = self.conv3(x)
        x = self.r(x)
        x = self.mp3(x)

        x = self.conv4(x)
        x = self.r(x)
        x = self.mp4(x)

        x = self.conv5(x)
        x = self.r(x)
        x = self.mp5(x)

        x = self.fc1(x)
        x = self.logsoftmax(x)
        x = self.r(x)
        return x
