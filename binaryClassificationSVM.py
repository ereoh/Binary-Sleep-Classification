import numpy as np
from sklearn import svm
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
def binaryDataset():
    xTrain, yTrain = getDataFromFile(0)
    xTest, yTest = getDataFromFile(1)

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

def main():

    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = binaryDataset()
    # print("Train")
    # print(xTrain.shape)
    # print(yTrain.shape)
    #
    # print("Test")
    # print(xTest.shape)
    # print(yTest.shape)
    print("Successfully loaded dataset.")

    print("Training Model...")
    clf1 = svm.SVC(kernel="rbf", C=1000)
    clf1.fit(xTrain, yTrain)

    print("Testing Model...")
    pred = clf1.predict(xTest)

    totalPreds = len(pred)
    totalCorrect = 0.0

    for i in range(totalPreds):
        if(pred[i] == yTest[i]):
            totalCorrect += 1

    accuracy = totalCorrect/totalPreds

    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
