'''
https://github.com/akaraspt/deepsleepnet
Copyright 2017 Akara Supratak and Hao Dong.  All rights reserved.
'''

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

def printClassMetrics():
    # read in npz files
    currentDir = os.getcwd()
    print("Current Directory =", currentDir)
    datasetDir = currentDir + "/prepare_datasets/edf_20_npz/"
    print("Dataset Directory =", datasetDir)

    allFiles = os.listdir(datasetDir)

    print("Files Found:", len(allFiles))

    # count how many of each classification

    # wake, N1, N2, N3, REM, Unknown/movement
    numClasses = [0, 0, 0, 0, 0, 0]
    classRatioPerSample = [0, 0, 0, 0, 0, 0]

    for i in range(len(allFiles)):
        sampleNumClasses = [0, 0, 0, 0, 0, 0]
        filename = allFiles[i]
        file = np.load(datasetDir + "/" + filename)
        print("Loaded in", allFiles[i])
        #print(file.files)
        #print(file['y'])

        numSamples = float(len(file['y']))
        for c in file['y']:
            numClasses[c] += 1
            sampleNumClasses[c] += 1

        for j in range(len(sampleNumClasses)):
            avg = sampleNumClasses[j]/numSamples
            classRatioPerSample[j] += avg

    #print(numClasses)
    for num in range(len(classRatioPerSample)):
        classRatioPerSample[num] /= float(len(allFiles))
        classRatioPerSample[num] = "{:.2f}".format(classRatioPerSample[num]*100)

    print(classRatioPerSample)

    # Calculate percentages

    total = 0.0
    percentages = [0, 0, 0, 0, 0, 0]

    for stageNum in numClasses:
        total += stageNum

    for i in range(len(numClasses)):
        percentages[i] = "{:.2f}".format(float((numClasses[i]/total)*100))

    print("Total 30s Samples: ", int(total))
    for n in range(len(numClasses)):
        print("\t" + label2ann[n] + ": " + percentages[n] + "%  (" +  str(numClasses[n]) +" samples)")


def main():
    printClassMetrics()


if __name__ == "__main__":
    main()
