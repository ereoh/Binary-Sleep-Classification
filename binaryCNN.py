import torch
import torch.optim as optim
import torch.nn as nn
from torchinfo import summary
import numpy as np
import time
from sklearn.preprocessing import StandardScaler

from classes import binarySleepNet, binarySleepNetSimple
from utility import validSubject
from createDataset import getBinaryDatasetAll

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

label2ann = {
    0: "Sleep stage W",
    1: "Sleep stage NREM 1",
    2: "Sleep stage NREM 2",
    3: "Sleep stage NREM 3",
    4: "Sleep stage REM",
    5: "Unknown/Movement"
}

# each sample is 3000 data points long
width = 3000

# subjects to skip evaluating
skip = [36, 52, 13, 39, 68, 69, 78, 79, 32, 33, 34, 72, 73, 57, 74, 75, 64]

# given train dataset
# return trained cnn model and criterion
def trainCNN(data, model, criterion, optimizer):
    # print(model)
    # unpack data
    xTrain, yTrain, heightTrain = data
    xTrain = torch.from_numpy(np.array(xTrain).reshape(heightTrain, -1, width))
    yTrain = torch.from_numpy(np.array(yTrain).reshape(-1, 1, 1))
    yTrain = yTrain.float()
    # print(xTrain.shape)
    # print(yTrain.shape)
    # exit()

    # zero parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(xTrain)
    # print(outputs.shape)
    # print(outputs)
    # exit()
    loss = criterion(outputs, yTrain)
    loss.backward()
    optimizer.step()

    return model, criterion

'''
    for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    print('Finished Training')
'''

# given test dataset
# return accuracy score
def testCNN(data, model, criterion):
    # unpack data
    xTest, yTest, heightTest = data
    xTest = torch.from_numpy(np.array(xTest).reshape(heightTest, -1, width))
    yTest = torch.from_numpy(np.array(yTest).reshape(-1, 1, 1))
    yTest = yTest.float()
    # print(xTest.shape)
    with torch.no_grad():
        model.eval()
        loss = 0.0
        acc = 0.0
        preds = model(xTest) # what does squeeze do?
        loss = criterion(preds, yTest)
        acc = binaryAcc(preds, yTest, heightTest)

    return acc

    '''

     # VALIDATION
    with torch.no_grad():
        model.eval()
        val_epoch_loss = 0
        val_epoch_acc = 0
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            y_val_pred = model(X_val_batch).squeeze()
            y_val_pred = torch.unsqueeze(y_val_pred, 0)
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = binary_acc(y_val_pred, y_val_batch)
            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()


    def binary_acc(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc
    '''

def binaryAcc(predictions, targets, numTest):
    # print(predictions[0][0][0])
    # print(targets[0][0][0])
    # exit()

    numCorrect = 0.0
    for i,pred in enumerate(predictions):
        # predPrime = max(0, pred[0][0])
        predPrime = pred[0][0]
        if targets[i] == predPrime:
            # print("yay")
            numCorrect += 1
        # exit()

    acc = numCorrect/numTest
    # print(acc)
    # exit()

    return acc


def main():
    start = time.time()

    test = binarySleepNetSimple()
    print(test)
    summary(test, input_size=(1, 1, width))
    # exit()

    N = 83 - len(skip)
    scores = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    datasets = []

    for i in range(83):
        print("on subject", i)
        if validSubject(i):
            # get binary datasets
            datasets = []
            datasetW, dataset1, dataset2, dataset3, datasetREM = getBinaryDatasetAll(i)
            datasets.append(datasetW)
            datasets.append(dataset1)
            datasets.append(dataset2)
            datasets.append(dataset3)
            datasets.append(datasetREM)
            # xTrain, yTrain, xTest, yTest, heightTrain, heightTest = datasetW
            # print(xTest.shape)


            # for each binary dataset
            for i,data in enumerate(datasets):
                # make new model, optimizer/criterion

                # model = binarySleepNet()
                model = binarySleepNetSimple()


                # criterion = nn.BCELoss()
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters()) # choose better one?

                xTrain, yTrain, xTest, yTest, heightTrain, heightTest = data
                ## TODO: standardized inputs ???
                # scaler = StandardScaler()
                # X_train = scaler.fit_transform(xTrain)
                # X_test = scaler.transform(xTest)

                trainData = (xTrain, yTrain, heightTrain)
                testData = (xTest, yTest, heightTest)

                # train models
                model, criterion = trainCNN(trainData, model, criterion, optimizer)
                # print(model)
                # exit()

                # test models
                scores[i] += testCNN(testData, model, criterion)
                # print(scores[i])
                # exit()

    print(scores)
    end = time.time()
    print("\nRuntime:", end-start, "seconds")

if __name__ == "__main__":
    main()
