import os
import numpy as np

from utility import validSubject
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

def multiDataset(trainNum, testNum):
    xTrain, yTrain = getDataFromFile(trainNum, 1)
    xTest, yTest = getDataFromFile(testNum, 2)

    heightTrain = xTrain.shape[0]
    heightTest = xTest.shape[0]

    xTrain = np.reshape(xTrain, (heightTrain, width))
    yTrain = np.reshape(yTrain, (heightTrain))
    xTest = np.reshape(xTest, (heightTest, width))
    yTest = np.reshape(yTest, (heightTest))

    return (xTrain, yTrain, xTest, yTest, heightTrain, heightTest)

def multiDatasetPersonal(subjectNum):
    xTrain, yTrain = getDataFromFile(subjectNum, 1)
    xTest, yTest = getDataFromFile(subjectNum, 2)

    heightTrain = xTrain.shape[0]
    heightTest = xTest.shape[0]

    xTrain = np.reshape(xTrain, (heightTrain, width))
    yTrain = np.reshape(yTrain, (heightTrain))
    xTest = np.reshape(xTest, (heightTest, width))
    yTest = np.reshape(yTest, (heightTest))

    return (xTrain, yTrain, xTest, yTest, heightTrain, heightTest)

# zeroClass = class that gets the zero values, other classes get 1
def makeBinary(l, zeroClass):
    for i in range(len(l)):
        if l[i] == zeroClass:
            l[i] = 0
        else:
            l[i] = 1

    return l

# returns x train, y train, x test, y test, height train, height test
def binaryDataset(posClass, trainNum, testNum):
    xTrain, yTrain = getDataFromFile(trainNum)
    xTest, yTest = getDataFromFile(testNum)

    heightTrain = xTrain.shape[0]
    heightTest = xTest.shape[0]

    yTrain = makeBinary(yTrain, posClass)
    yTest = makeBinary(yTest, posClass)
    # print(yTrain)
    # print(yTest)

    xTrain = np.reshape(xTrain, (heightTrain, width))
    yTrain = np.reshape(yTrain, (heightTrain))
    xTest = np.reshape(xTest, (heightTest, width))
    yTest = np.reshape(yTest, (heightTest))

    return (xTrain, yTrain, xTest, yTest, heightTrain, heightTest)

def binaryDatasetPersonal(posClass, subjectNum):
    xTrain, yTrain = getDataFromFile(subjectNum, 1)
    xTest, yTest = getDataFromFile(subjectNum , 2)

    heightTrain = xTrain.shape[0]
    heightTest = xTest.shape[0]

    yTrain = makeBinary(yTrain, posClass)
    yTest = makeBinary(yTest, posClass)
    # print(yTrain)
    # print(yTest)

    xTrain = np.reshape(xTrain, (heightTrain, width))
    yTrain = np.reshape(yTrain, (heightTrain))
    xTest = np.reshape(xTest, (heightTest, width))
    yTest = np.reshape(yTest, (heightTest))

    return (xTrain, yTrain, xTest, yTest, heightTrain, heightTest)

def makeAllBinaryDatasets(subjectNum):
    classes = [W, N1, N2, N3, REM]

    datasets = []

    for c in classes:
        datasetInfo = binaryDatasetPersonal(c, subjectNum)
        datasets.append(datasetInfo)

    datasets = np.array(datasets)

    print(datasets[0][4], datasets[0][5])

    return datasets

def loadAllFiles(num=None, night=None):
    # read in npz files
    currentDir = os.getcwd()
    #print("Current Directory =", currentDir)
    datasetDir = currentDir + "/prepare_datasets/edf_20_npz/"
    #print("Dataset Directory =", datasetDir)

    allFiles = os.listdir(datasetDir)

    #print("Files Found:", len(allFiles))

    # allXs = []
    # allYs = []

    for i in range(len(allFiles)):
        # print("processing file", i, "---")
        filename = allFiles[i]
        file = np.load(datasetDir + "/" + filename, allow_pickle=True)
        # print("Loaded in", allFiles[i])
        # print(file.files)
        # print(filename[3:5], filename[5:6])
        # exit()
        # print(file['x'].shape[0])
        # exit()
        #print(file['y'].shape)
        # print("total rows:", file['x'].shape[0])
        # for j in range(file['x'].shape[0]):
            # if j%50 == 0:
            #     print("\trow", j)
            # allXs.append(file['x'][j])
            # allYs.append(file['y'][j])
        # exclude patients: 36, 52, and 13
        subjectNum = int(filename[3:5])
        nightNum = int(filename[5:6])
        if num != None and subjectNum == num and nightNum == night:
            return (file['x'], file['y'])

    # x - np.array(allXs)
    # y = np.array(allYs)
    #
    # print(x.shape)
    # print(y.shape)
    #
    # numpy.save("x.npy", x)
    # numpy.save("y.npy", y)
    print("Did not find subject", num)
    exit()

def getDataFromFile(subjectNum, night):
    x, y = loadAllFiles(num=subjectNum, night=night)
    if x.all() == None or y.all() == None:
        print("Did not find:", subjectNum)
        exit()
    return (x,y)

def saveAllDatasets():
    dir = os.getcwd() + "/prepare_datasets/processedDatasets/"
    # binary saveAllDatasets
    for i in range(83):
        print("on subject", i)
        if validSubject(i):
            for t in range(5):
                xTrain, yTrain, xTest, yTest, _, _ = binaryDatasetPersonal(t, i)
                filename = dir + "binary/" + class_dict[t] + "/" + "Subject" + str(i)

                np.savez(filename, xTrain=xTrain, yTrain=yTrain, xTest=xTest, yTest=yTest)

    # multiclass datasets
    for i in range(83):
        print("on subject", i)
        if validSubject(i):
            xTrain, yTrain, xTest, yTest, _, _ = multiDatasetPersonal(i)
            filename = dir + "multiclass/" + "Subject" + str(i)

            np.savez(filename, xTrain=xTrain, yTrain=yTrain, xTest=xTest, yTest=yTest)

def getBinaryDataset(posClass, subjectNum):
    filename = os.getcwd() + "/prepare_datasets/processedDatasets/binary/" + class_dict[posClass] + "/" + "Subject" + str(subjectNum) + ".npz"

    data = np.load(filename)

    xTrain = data["xTrain"]
    yTrain = data["yTrain"]
    xTest = data["xTest"]
    yTest = data["yTest"]

    heightTrain = xTrain.shape[0]
    heightTest = xTest.shape[0]

    return (xTrain, yTrain, xTest, yTest, heightTrain, heightTest)

def getBinaryDatasetAll(subjectNum):

    datasetW = getBinaryDataset(W, subjectNum)
    dataset1 = getBinaryDataset(N1, subjectNum)
    dataset2 = getBinaryDataset(N2, subjectNum)
    dataset3 = getBinaryDataset(N3, subjectNum)
    datasetREM = getBinaryDataset(REM, subjectNum)

    return (datasetW, dataset1, dataset2, dataset3, datasetREM)


def getMulticlassDataset(subjectNum):
    filename = os.getcwd() + "/prepare_datasets/processedDatasets/multiclass/Subject" + str(subjectNum) + ".npz"
    data = np.load(filename)

    xTrain = data["xTrain"]
    yTrain = data["yTrain"]
    xTest = data["xTest"]
    yTest = data["yTest"]

    heightTrain = xTrain.shape[0]
    heightTest = xTest.shape[0]

    return (xTrain, yTrain, xTest, yTest, heightTrain, heightTest)

def main():
    # loadAllFiles()
    # makeAllBinaryDatasets(0)
    saveAllDatasets()


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()

    print(end - start)
