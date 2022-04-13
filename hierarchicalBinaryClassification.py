import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from joblib import dump, load

from createDataset import getMulticlassDataset
from utility import testModel, validSubject, evaluateAcc
from classes import binaryHierarchy

import time
import os

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

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}

# each sample is 3000 data points long
width = 3000

# subjects to skip evaluating
skip = [36, 52, 13, 39, 68, 69, 78, 79, 32, 33, 34, 72, 73, 57, 74, 75, 64]

# given model name, test model on all subjects
def testCustomModel(modelName):
    # load model
    order = []
    if modelName == "intuitive":
        order = ["W", "REM", "N1", "N2", "N3"]
    elif modelName == "size":
        order = ["N2", "W", "REM", "N1", "N3"]
    elif modelName == "accuracy":
        order = ["N3", "N1", "REM", "W", "N2"]
    elif modelName == "confusion":
        order = ["N2", "W", "N3", "REM", "N1"]
    else:
        print("Error: Unkown model name ", modelName)
        exit()

    model = binaryHierarchy(modelName, order, 0)
    print(model)
    print(model.models)

    # predict testing dataset
    N = 83 - len(skip)
    accSum = 0.0

    totalCM = np.zeros((6,6))

    for i in range(83):
        print("on subject", i)
        if validSubject(i):
            model = binaryHierarchy(modelName, order, i)

            xTrain, yTrain, xTest, yTest, heightTrain, heightTest = getMulticlassDataset(i)

            # evaluate accuracy
            preds = model.predict(xTest)
            accSum += evaluateAcc(yTest, preds)

            cm = confusion_matrix(yTest, preds, labels=[0, 1, 2, 3, 4, -1], normalize=None)

            for r in range(6):
                for c in range(6):
                    totalCM[r][c] += cm[r][c]


    np.set_printoptions(suppress=True)
    print(totalCM)
    np.set_printoptions(suppress=False)

    acc = accSum/N


    return acc, totalCM

def main():
    start = time.time()

    allModels = ["intuitive", "size", "accuracy", "confusion"]
    numModels = len(allModels)
    accuracies = np.zeros((numModels,1))
    confusionMatrix = np.zeros((numModels, 6, 6))

    for i,m in enumerate(allModels):
        print("on model", m)
        accuracies[i], confusionMatrix[i] = testCustomModel(m)

    np.set_printoptions(suppress=True)
    for j,acc in enumerate(accuracies):
        print("---")
        print(allModels[j] + ":" + str(acc))
        print(confusionMatrix[i])
    np.set_printoptions(suppress=False)

    end = time.time()
    print("\nRuntime:", end-start, "seconds")

if __name__ == "__main__":
    main()
