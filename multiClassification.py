# Code Written by: Erebus Oh, Kenneth Barkdoll

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs

from createDataset import getDataFromFile, multiDataset
from utility import testModel

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

# each sample is 3000 data points long
width = 3000

def knn():
    print("Using k Nearest Neighbors----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = multiDataset(1, 2)
    print("Successfully loaded dataset.")

    print("Training Models...")
    knn = KNeighborsClassifier()
    knn.fit(xTrain, yTrain)

    print("Testing Models...")
    knnAcc = testModel(knn, xTest, yTest)

    print("k nearest neighbors: ", knnAcc)

def decisionTree():
    print("Using Decision Trees----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = multiDataset(1, 2)
    print("Successfully loaded dataset.")

    print("Training Models...")
    # ['29.80', '11.06', '37.65', '7.50', '13.99',
    classWeights = {
        0: 0.2980,
        1: 0.1106,
        2: 0.3765,
        3: 0.0750,
        4: 0.1399
    }
    dt = DecisionTreeClassifier(max_depth=4, max_leaf_nodes=5, class_weight=classWeights)
    dt.fit(xTrain, yTrain)

    print("Testing Models...")
    dtAcc = testModel(dt, xTest, yTest)

    print("decision tree: ", dtAcc)

def randomForests():
    print("Using Random Forests----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = multiDataset(1, 2)
    print("Successfully loaded dataset.")

    print("Training Models...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5)
    rf.fit(xTrain, yTrain)

    print("Testing Models...")
    rfAcc = testModel(rf, xTest, yTest)

    print("random forests: ", rfAcc)

def main():
    decisionTree()
    randomForests()
    knn()

if __name__ == "__main__":
    main()
