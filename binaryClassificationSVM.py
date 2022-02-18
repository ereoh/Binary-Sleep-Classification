import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_blobs

from createDataset import getDataFromFile

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

# each sample is 3000 data points long
width = 3000

# zeroClass = class that gets the zero values, other classes get 1
def makeBinary(l, zeroClass):
    for i in range(len(l)):
        if l[i] == zeroClass:
            l[i] = 0
        else:
            l[i] = 1

    return l

# returns x train, y train, x test, y test, height train, height test
def binaryDataset(trainNum, testNum):
    xTrain, yTrain = getDataFromFile(trainNum)
    xTest, yTest = getDataFromFile(testNum)

    heightTrain = xTrain.shape[0]
    heightTest = xTest.shape[0]

    yTrain = makeBinary(yTrain, W)
    yTest = makeBinary(yTest, W)
    # print(yTrain)
    # print(yTest)

    xTrain = np.reshape(xTrain, (heightTrain, width))
    yTrain = np.reshape(yTrain, (heightTrain))
    xTest = np.reshape(xTest, (heightTest, width))
    yTest = np.reshape(yTest, (heightTest))

    return (xTrain, yTrain, xTest, yTest, heightTrain, heightTest)

def trainModel(kernel, xTrain, yTrain):
    model = svm.SVC(kernel=kernel, C=1000)
    model.fit(xTrain, yTrain)

    return model

def testModel(model, xTest, yTest):
    pred = model.predict(xTest)

    totalPreds = len(pred)
    totalCorrect = 0.0

    for i in range(totalPreds):
        if(pred[i] == yTest[i]):
            totalCorrect += 1

    accuracy = totalCorrect/totalPreds

    return accuracy

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
    rbfModel = trainModel("rbf", xTrain, yTrain)
    linearModel = trainModel("linear", xTrain, yTrain)
    polyModel = trainModel("poly", xTrain, yTrain)
    sigmoidModel = trainModel("sigmoid", xTrain, yTrain)

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
    ldas()

if __name__ == "__main__":
    main()
