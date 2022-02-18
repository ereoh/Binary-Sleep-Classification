import os
import numpy as np

label2ann = {
    0: "Sleep stage W",
    1: "Sleep stage NREM 1",
    2: "Sleep stage NREM 2",
    3: "Sleep stage NREM 3",
    4: "Sleep stage REM",
    5: "Unknown/Movement"
}

def loadAllFiles(num=None):
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
        #print("processing file", i, "---")
        filename = allFiles[i]
        file = np.load(datasetDir + "/" + filename, allow_pickle=True)
        #print("Loaded in", allFiles[i])
        #print(file.files)
        # print(file['x'].shape[0])
        # exit()
        #print(file['y'].shape)
        # print("total rows:", file['x'].shape[0])
        # for j in range(file['x'].shape[0]):
            # if j%50 == 0:
            #     print("\trow", j)
            # allXs.append(file['x'][j])
            # allYs.append(file['y'][j])
        if num != None and i == num:
            return (file['x'], file['y'])

    # x - np.array(allXs)
    # y = np.array(allYs)
    #
    # print(x.shape)
    # print(y.shape)
    #
    # numpy.save("x.npy", x)
    # numpy.save("y.npy", y)

def getDataFromFile(num):
    x, y = loadAllFiles(num=num)
    return (x,y)

def main():
    loadAllFiles()


if __name__ == "__main__":
    main()
