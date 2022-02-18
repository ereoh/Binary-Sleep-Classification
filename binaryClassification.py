import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs

from createDataset import getDataFromFile, binaryDataset
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

def logReg():
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = binaryDataset(1, 130)
    print(yTrain.shape)
    print("Successfully loaded dataset.")

    print("Training Models...")
    lr = LogisticRegression(solver="liblinear", max_iter=100)
    lr.fit(xTrain, yTrain)

    print("Testing Models...")
    lrAcc = testModel(lr, xTest, yTest)

    print("logistic regression: ", lrAcc)

def trainSVMModel(kernel, xTrain, yTrain):
    model = svm.SVC(kernel=kernel, C=1000)
    model.fit(xTrain, yTrain)

    return model

def svms():
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = binaryDataset(4, 100)
    # print("Train")
    # print(xTrain.shape)
    # print(yTrain.shape)
    #
    # print("Test")
    # print(xTest.shape)
    # print(yTest.shape)
    print("Successfully loaded dataset.")

    print("Training Models...")
    rbfModel = trainSVMModel("rbf", xTrain, yTrain)
    linearModel = trainSVMModel("linear", xTrain, yTrain)
    polyModel = trainSVMModel("poly", xTrain, yTrain)
    sigmoidModel = trainSVMModel("sigmoid", xTrain, yTrain)

    print("Testing Models...")
    rbfAcc = testModel(rbfModel, xTest, yTest)
    linearAcc = testModel(linearModel, xTest, yTest)
    polyAcc = testModel(polyModel, xTest, yTest)
    sigmoidAcc = testModel(sigmoidModel, xTest, yTest)

    print("rbf model: ", rbfAcc)
    print("linear model: ", linearAcc)
    print("poly model:", polyAcc)
    print("sigmoid model:", sigmoidAcc)

def ldas():
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = binaryDataset(4, 100)

    print("Successfully loaded dataset.")

    print("Training Models...")
    lda = LinearDiscriminantAnalysis(solver='svd', n_components=1)
    lda.fit(xTrain, yTrain)

    print("Testing Models...")
    ldaAcc = testModel(lda, xTest, yTest)

    print("lda model: ", ldaAcc)

def main():
    # svms()
    # ldas()
    logReg()

if __name__ == "__main__":
    main()
