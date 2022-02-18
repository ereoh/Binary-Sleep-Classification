import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

def multiDataset(trainNum, testNum):
    xTrain, yTrain = getDataFromFile(trainNum)
    xTest, yTest = getDataFromFile(testNum)

    heightTrain = xTrain.shape[0]
    heightTest = xTest.shape[0]

    xTrain = np.reshape(xTrain, (heightTrain, width))
    yTrain = np.reshape(yTrain, (heightTrain))
    xTest = np.reshape(xTest, (heightTest, width))
    yTest = np.reshape(yTest, (heightTest))

    return (xTrain, yTrain, xTest, yTest, heightTrain, heightTest)

def randomForests():
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = multiDataset(1, 2)
    print("Successfully loaded dataset.")

    print("Training Models...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5)
    rf.fit(xTrain, yTrain)

    print("Testing Models...")
    rfAcc = testModel(rf, xTest, yTest)

    print("random forests: ", rfAcc)

def main():
    randomForests()

if __name__ == "__main__":
    main()
