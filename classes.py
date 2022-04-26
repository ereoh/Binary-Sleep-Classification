import numpy as np
import sklearn
import torch.nn as nn
import torch.nn.functional as F
from binaryClassification import loadBinaryModel

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}

class_to_num_dict = {
    "W":0,
    "N1":1,
    "N2":2,
    "N3":3,
    "REM":4,
    "UNKNOWN":5
}

class binaryHierarchy:

    # classifier order is a list of strings of classifiers order
    # from shallowest to deepest
    def __init__(self, name, classifierOrder, subjectNum, modelTypes):
        self.name = name
        self.order = classifierOrder
        self.subjectNum = subjectNum
        self.models = []
        self.modelTypes = modelTypes
        self.buildModel()

    def buildModel(self):
        for i, s in enumerate(self.order):
            pos = class_to_num_dict[s]
            m = loadBinaryModel(self.modelTypes, self.subjectNum, pos)
            self.models.append(m)

    def predict(self,x):
        numSamples = x.shape[0]
        predictions = []

        classified = False

        for i,sample in enumerate(x):
            # print("on ", i)
            classified = False
            for modelNum, model in enumerate(self.models): # for every binary classifier
                # print("\tmodel", self.order[modelNum])
                ans = model.predict(sample.reshape(1,-1)) # predict class
                # print("\tpredicted", ans)
                if ans: # if is that class
                    # print("adding ", class_to_num_dict[self.order[modelNum]])
                    # exit()
                    predictions.append(class_to_num_dict[self.order[modelNum]]) # save answers
                    classified = True
                    break # stop evaluating lower layers
            if not classified: # sample not classified as any class, set y = -1
                # print("not classified, adding -1")
                predictions.append(-1)
                classified = False

        preds = np.array(predictions)
        # print(preds.shape)
        # print(preds)

        return preds

    def __str__(self):
        string = "Model: " + self.name + " (Subject "+ str(self.subjectNum) + ")"
        for s in self.order:
            string += "\n\t" + s

        return string


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

class binarySleepNetSimple(nn.Module):
    def __init__(self):
        super().__init__()
        # each input is 3000 long

        self.conv1 = nn.Conv1d(1, 1, 2996)
        self.r = nn.ReLU()
        self.fc1 = nn.Linear(5, 1)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.r(x)
        x = self.fc1(x)
        x = self.r(x)
        x - self.logsoftmax(x)
        return x
