# Code Written by: Erebus Oh, Kenneth Barkdoll

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from createDataset import getDataFromFile, multiDatasetPersonal, getMulticlassDataset
from utility import testModel, validSubject

import time
import os

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

# each sample is 3000 data points long
width = 3000

# subjects to skip evaluating
skip = [36, 52, 13, 39, 68, 69, 78, 79, 32, 33, 34, 72, 73, 57, 74, 75, 64]

def knn(dataset):
    # print("Using k Nearest Neighbors----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset
    # print("Successfully loaded dataset.")

    # print("Training Models...")
    knn = KNeighborsClassifier(3)
    knn.fit(xTrain, yTrain)

    # print("Testing Models...")
    knnAcc = testModel(knn, xTest, yTest)

    # print("k nearest neighbors: ", knnAcc)

    return knnAcc

def decisionTree(dataset):
    # print("Using Decision Trees----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset
    # print("Successfully loaded dataset.")

    # print("Training Models...")
    # ['29.80', '11.06', '37.65', '7.50', '13.99',
    # dict({0:y0, 1:z0}, {0:y1, 1:z2}, {0:y1, 1:z1},...)
    # where y is the confidence/weight of outcome 0 and z the confidence/weight of outcome 1.
    # classWeights = dict{0:1, 1:0.2980}, {0:1, 1:0.1106}, {0:1, 1:0.3765}, {0:1, 1:0.0750}, {0:1, 1:0.1399})
    #     0: 0.2980,
    #     1: 0.1106,
    #     2: 0.3765,
    #     3: 0.0750,
    #     4: 0.1399
    # }
    dt = DecisionTreeClassifier(max_depth=4, max_leaf_nodes=5)
    dt.fit(xTrain, yTrain)

    # print("Testing Models...")
    dtAcc = testModel(dt, xTest, yTest)

    # print("decision tree: ", dtAcc)

    return dtAcc

def randomForests(dataset):
    # print("Using Random Forests----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset
    # print("Successfully loaded dataset.")

    # print("Training Models...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5)
    rf.fit(xTrain, yTrain)

    # print("Testing Models...")
    rfAcc = testModel(rf, xTest, yTest)

    # print("random forests: ", rfAcc)

    return rfAcc

def randomForestsCM(dataset):
    # print("Using Random Forests----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset
    # print("Successfully loaded dataset.")

    # print("Training Models...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5)
    rf.fit(xTrain, yTrain)

    # print("Testing Models...")
    # rfAcc = testModel(rf, xTest, yTest)
    preds = rf.predict(xTest)

    cm = confusion_matrix(yTest, preds, labels=[0, 1, 2, 3, 4], normalize=None)

    return cm

def testAlgorithms():
    # scores[decision tree, random forests]
    modelNames = ["decision tree", "random forests"]
    scores = np.array([0.0, 0.0])

    N = 83 - len(skip)

    for i in range(83):
        print("on subject", i)
        if validSubject(i):
            dataset = getMulticlassDataset(i)
            print("\tcreated dataset")

            dcAcc = decisionTree(dataset)
            scores[0] += dcAcc
            print("\tdecision tree done")

            scores[1] += randomForests(dataset)
            print("\trandom forests done")
        else:
            print("\tskipped", i)

    for s in range(scores.shape[0]):
        scores[s] /= N
        print(modelNames[s], " score:", scores[s])

def confusionMatrix():
    totalCM = np.zeros((5,5))

    N = 83 - len(skip)

    for i in range(83):
        print("on subject", i)
        if validSubject(i):
            dataset = getMulticlassDataset(i)
            print("\tcreated dataset")

            cm = randomForestsCM(dataset)
            print("\trandom forests done")

            for r in range(5):
                for c in range(5):
                    totalCM[r][c] += cm[r][c]
        else:
            print("\tskipped", i)

    np.set_printoptions(suppress=True)
    print(totalCM)
    np.set_printoptions(suppress=False)

def main():
    start = time.time()

    # testAlgorithms() # run all implement multi-classification algorithms on dataset
    confusionMatrix() # run Random Forests on dataset for confusion matrix

    end = time.time()
    print("Runtime:", end-start, "seconds")

if __name__ == "__main__":
    main()
