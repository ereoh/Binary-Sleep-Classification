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

from createDataset import getDataFromFile, binaryDatasetPersonal, makeAllBinaryDatasets
from utility import testModel

import time

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
    # print("Using k Nearest Neighbors----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = binaryDatasetPersonal(W, 1)
    # print("Successfully loaded dataset.")

    # print("Training Models...")
    knn = KNeighborsClassifier(21)
    knn.fit(xTrain, yTrain)

    # print("Testing Models...")
    knnAcc = testModel(knn, xTest, yTest)

    print("k nearest neighbors: ", knnAcc)

def voting():
    # print("Using Voting Classification----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = binaryDatasetPersonal(W, 1)
    # print(yTrain.shape)
    # print("Successfully loaded dataset.")

    # print("Training Models...")
    # rbf, poly, linear, LDA
    rbfModel = trainSVMModel("rbf", xTrain, yTrain)
    linearModel = trainSVMModel("linear", xTrain, yTrain)
    polyModel = trainSVMModel("poly", xTrain, yTrain)
    knn = KNeighborsClassifier(21)
    estimators=[("rbf", rbfModel), ("linear", linearModel), ("poly", polyModel), ("knn", knn)]
    weights=[1, 1, 1.5, 1.5]
    vc = VotingClassifier(estimators,voting="hard", weights=weights)

    rbfModel.fit(xTrain, yTrain)
    linearModel.fit(xTrain, yTrain)
    polyModel.fit(xTrain, yTrain)
    knn.fit(xTrain, yTrain)
    vc.fit(xTrain, yTrain)

    # print("Testing Models...")
    vcAcc = testModel(vc, xTest, yTest)

    print("Voting Classifier: ", vcAcc)

def NB():
    # print("Using Naive Bayes----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = binaryDatasetPersonal(W, 1)
    # print(yTrain.shape)
    # print("Successfully loaded dataset.")

    # print("Training Models...")
    nb = GaussianNB()
    nb.fit(xTrain, yTrain)

    # print("Testing Models...")
    nbAcc = testModel(nb, xTest, yTest)

    print("Naive Bayes: ", nbAcc)

def adaBoost():
    # print("Using AdaBoost----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = binaryDatasetPersonal(W, 1)
    # print(yTrain.shape)
    # print("Successfully loaded dataset.")

    # print("Training Models...")
    ad = AdaBoostClassifier()
    ad.fit(xTrain, yTrain)

    # print("Testing Models...")
    adAcc = testModel(ad, xTest, yTest)

    print("ada boost: ", adAcc)

def logReg():
    # print("Using Logistic Regression----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = binaryDatasetPersonal(W, 1)
    # print(yTrain.shape)
    # print("Successfully loaded dataset.")

    # print("Training Models...")
    lr = LogisticRegression(solver="liblinear", max_iter=100)
    lr.fit(xTrain, yTrain)

    # print("Testing Models...")
    lrAcc = testModel(lr, xTest, yTest)

    print("logistic regression: ", lrAcc)

def trainSVMModel(kernel, xTrain, yTrain):
    model = svm.SVC(kernel=kernel, C=1000)
    model.fit(xTrain, yTrain)

    return model

def svms():
    # print("Using SVM----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = binaryDatasetPersonal(W, 1)
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
    sigmoidModel = trainSVMModel("sigmoid", xTrain, yTrain)

    # print("Testing Models...")
    rbfAcc = testModel(rbfModel, xTest, yTest)
    linearAcc = testModel(linearModel, xTest, yTest)
    polyAcc = testModel(polyModel, xTest, yTest)
    sigmoidAcc = testModel(sigmoidModel, xTest, yTest)

    print("rbf model: ", rbfAcc)
    print("linear model: ", linearAcc)
    print("poly model:", polyAcc)
    print("sigmoid model:", sigmoidAcc)

def ldas():
    # print("Using LDA----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = binaryDatasetPersonal(W, 1)

    # print("Successfully loaded dataset.")

    # print("Training Models...")
    lda = LinearDiscriminantAnalysis(solver='svd', n_components=1)
    lda.fit(xTrain, yTrain)

    # print("Testing Models...")
    ldaAcc = testModel(lda, xTest, yTest)

    print("lda model: ", ldaAcc)

def main():
    start = time.time()
    svms()
    ldas()
    logReg()
    adaBoost()
    NB()
    voting()
    knn()
    end = time.time()
    print("\nRuntime:", end-start, "seconds")

if __name__ == "__main__":
    main()
