import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from joblib import dump, load
from tqdm import tqdm

from createDataset import getMulticlassDataset
from utility import testModel, validSubject, evaluateAcc
from classes import binaryHierarchy
from binaryClassification import loadBinaryModel

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

def checkBinaryModelsExist(modelTypes):
    print("Checking Binary Models are Saved---")
    N = 83 - len(skip)

    for i in tqdm(range(83)):
        # print("on subject", i)
        if validSubject(i):
            for c in range(5):
                saveModelDir = os.getcwd() + "/saved_models/" + modelTypes + "/" + str(i) + "/"
                filename = modelTypes + "_" + class_dict[c] +".joblib"
                isExist = os.path.exists(saveModelDir + filename)
                if not isExist:
                    loadBinaryModel(modelTypes, i, c)
    print("Model Check done.\n")

# given model name, test model on all subjects
def testCustomModel(modelName, modelType, balanced=True):
    # load model
    order = []
    if modelName == "intuitive":
        order = ["W", "REM", "N1", "N2", "N3"]
    elif modelName == "size":
        order = ["N2", "W", "REM", "N1", "N3"]
    elif modelName == "accuracy":
        order = ["N3", "REM", "W", "N1", "N2"]
    elif modelName == "confusion":
        order = ["N1", "REM", "W", "N2", "N3"]
    else:
        print("Error: Unkown model name ", modelName)
        exit()

    model = binaryHierarchy(modelName, order, 0, modelType)
    print(model)
    print(model.models)

    # predict testing dataset
    N = 83 - len(skip)
    accSum = 0.0

    totalCM = np.zeros((6,6))

    for i in tqdm(range(83)):
        # print("on subject", i)
        if validSubject(i):
            model = binaryHierarchy(modelName, order, i, modelType)

            if balanced:
                xTrain, yTrain, xTest, yTest, heightTrain, heightTest = getMulticlassDatasetBalanced(i)
            else:
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

def testAllHierachies(modelTypes):
    checkBinaryModelsExist(modelTypes)

    print("Testing All Hierarchical Binary Classifiers---")

    # allModels = ["intuitive", "size", "accuracy", "confusion"]
    # allModels = ["accuracy"]
    # allModels = ["intuitive", "size", "accuracy"]
    allModels = ["confusion"]
    numModels = len(allModels)
    accuracies = np.zeros((numModels,1))
    confusionMatrix = np.zeros((numModels, 6, 6))

    for i,m in tqdm(enumerate(allModels)):
        print("on model", m)
        accuracies[i], confusionMatrix[i] = testCustomModel(m, modelTypes, balanced=False)

    np.set_printoptions(suppress=True)
    for j,acc in enumerate(accuracies):
        print("---")
        print(allModels[j] + ":" + str(acc))
        # print(confusionMatrix[i])
    np.set_printoptions(suppress=False)

def main():
    start = time.time()

    testAllHierachies("best")

    end = time.time()
    print("\nRuntime:", end-start, "seconds")

if __name__ == "__main__":
    main()
