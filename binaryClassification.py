# Code Written by: Erebus Oh, Kenneth Barkdoll

import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

from createDataset import getDataFromFile, binaryDatasetPersonal, makeAllBinaryDatasets,getBinaryDataset, getBinaryDatasetAll
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

def knn(dataset):
    # print("Using k Nearest Neighbors----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset
    # print("Successfully loaded dataset.")

    # print("Training Models...")
    knn = KNeighborsClassifier(21)
    knn.fit(xTrain, yTrain)

    # print("Testing Models...")
    knnAcc = testModel(knn, xTest, yTest)

    # print("k nearest neighbors: ", knnAcc)

    return knnAcc

def voting(dataset):
    # print("Using Voting Classification----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset
    # print(yTrain.shape)
    # print("Successfully loaded dataset.")

    # print("Training Models...")
    # rbf, poly, linear, LDA
    rbfModel = trainSVMModel("rbf", xTrain, yTrain)
    polyModel = trainSVMModel("poly", xTrain, yTrain)
    ad = AdaBoostClassifier()
    knn = KNeighborsClassifier(23)
    estimators=[("rbf", rbfModel), ("poly", polyModel), ("adaboost", ad), ("knn", knn)]
    weights=[1.5, 1, 1.5, 1]
    vc = VotingClassifier(estimators,voting="hard", weights=weights)

    rbfModel.fit(xTrain, yTrain)
    polyModel.fit(xTrain, yTrain)
    knn.fit(xTrain, yTrain)
    ad.fit(xTrain, yTrain)
    vc.fit(xTrain, yTrain)

    # print("Testing Models...")
    vcAcc = testModel(vc, xTest, yTest)

    # print("Voting Classifier: ", vcAcc)

    return vcAcc

def NB(dataset):
    # print("Using Naive Bayes----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset
    # print(yTrain.shape)
    # print("Successfully loaded dataset.")

    # print("Training Models...")
    nb = GaussianNB()
    nb.fit(xTrain, yTrain)

    # print("Testing Models...")
    nbAcc = testModel(nb, xTest, yTest)

    # print("Naive Bayes: ", nbAcc)

    return nbAcc

def adaBoost(dataset):
    # print("Using AdaBoost----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset
    # print(yTrain.shape)
    # print("Successfully loaded dataset.")

    # print("Training Models...")
    ad = AdaBoostClassifier()
    ad.fit(xTrain, yTrain)

    # print("Testing Models...")
    adAcc = testModel(ad, xTest, yTest)

    # print("ada boost: ", adAcc)

    return adAcc

def logReg(dataset):
    # print("Using Logistic Regression----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset
    # print(yTrain.shape)
    # print("Successfully loaded dataset.")

    # print("Training Models...")
    lr = LogisticRegression(solver="liblinear", max_iter=100)
    lr.fit(xTrain, yTrain)

    # print("Testing Models...")
    lrAcc = testModel(lr, xTest, yTest)

    # print("logistic regression: ", lrAcc)

    return lrAcc

def trainSVMModel(kernel, xTrain, yTrain):
    model = svm.SVC(kernel=kernel, C=1000)
    model.fit(xTrain, yTrain)

    return model

def svms(dataset):
    # print("Using SVM----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset
    # print("Train")
    # print(xTrain.shape)
    # print(yTrain.shape)
    #
    # print("Test")
    # print(xTest.shape)
    # print(yTest.shape)
    # print("Successfully loaded dataset.")

    # print("Training Models...")
    rbfModel = trainSVMModel("rbf", xTrain, yTrain)
    linearModel = trainSVMModel("linear", xTrain, yTrain)
    polyModel = trainSVMModel("poly", xTrain, yTrain)
    # sigmoidModel = trainSVMModel("sigmoid", xTrain, yTrain)

    # print("Testing Models...")
    rbfAcc = testModel(rbfModel, xTest, yTest)
    linearAcc = testModel(linearModel, xTest, yTest)
    polyAcc = testModel(polyModel, xTest, yTest)
    # sigmoidAcc = testModel(sigmoidModel, xTest, yTest)

    # print("rbf model: ", rbfAcc)
    # print("linear model: ", linearAcc)
    # print("poly model:", polyAcc)
    # print("sigmoid model:", sigmoidAcc)

    return rbfAcc, linearAcc, polyAcc

def svmRBF(dataset):
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset
    rbfModel = trainSVMModel("rbf", xTrain, yTrain)
    rbfAcc = testModel(rbfModel, xTest, yTest)

    return rbfAcc

def lda(dataset):
    # print("Using LDA----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset

    # print("Successfully loaded dataset.")

    # print("Training Models...")
    lda = LinearDiscriminantAnalysis(solver='svd', n_components=1)
    lda.fit(xTrain, yTrain)

    # print("Testing Models...")
    ldaAcc = testModel(lda, xTest, yTest)

    # print("lda model: ", ldaAcc)

    return ldaAcc

def testAlgorithms():
    # scores = [rbfAcc, linearAcc, polyAcc, logAcc, nbAcc, dAcc, vcAcc, knnAcc]
    modelNames = ["rbf", "linear","poly", "logistic regression", "naive bayes","ada boost", "voting", "knn"]
    scores = np.zeros((len(modelNames,)))

    N = 83 - len(skip)

    for i in range(83):
        print("on subject", i)
        if validSubject(i):
            dataset = getBinaryDataset(W, i)
            print("\tcreated dataset")

            r, l, p = svms(dataset)
            scores[0] += r
            scores[1] += l
            scores[2] += p
            print("\tsvm done")

            scores[3] += logReg(dataset)
            print("\tlogistic regression done")

            scores[4] += NB(dataset)
            print("\tnaive bayes done")

            scores[5] += adaBoost(dataset)
            print("\tada boost done")

            scores[6] += voting(dataset)
            print("\tvoting done")

            scores[7] += knn(dataset)
            print("\tknn done")
        else:
            print("\tskipped", i)

    for s in range(scores.shape[0]):
        scores[s] /= float(N)
        print(modelNames[s], " score:", scores[s])

def createBinaryClassifiers():
    # Wake, 1, 2, 3, REM
    scores = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    N = 83 - len(skip)
    # 34
    # 70
    for i in range(83):
        print("on subject", i)
        if validSubject(i):
            datasetW, dataset1, dataset2, dataset3, datasetREM = getBinaryDatasetAll(i)

            # insert algorthm to use where
            scores[0] += svmRBF(datasetW)
            print("\tW done")
            scores[1] += svmRBF(dataset1)
            print("\tN1 done")
            scores[2] += svmRBF(dataset2)
            print("\tN2 done")
            scores[3] += svmRBF(dataset3)
            print("\tN3 done")
            scores[4] += svmRBF(datasetREM)
            print("\tREM done")

    for s in range(scores.shape[0]):
        scores[s] /= N
        print(label2ann[s] + ": " + str(scores[s]))

    scoresNP = np.array(scores)
    filename = os.getcwd() + "/results/svm-rbf"
    np.save(filename, scoresNP)


def main():
    start = time.time()

    # testAlgorithms() # run all implemented binary classifier algorithms on the dataset
    createBinaryClassifiers() # run SVM with RBF kernel on dataset

    end = time.time()
    print("\nRuntime:", end-start, "seconds")

if __name__ == "__main__":
    main()
