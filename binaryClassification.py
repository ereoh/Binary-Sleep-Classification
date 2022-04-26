import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from joblib import dump, load
from tqdm import tqdm

from createDataset import getDataFromFile, binaryDatasetPersonal, makeAllBinaryDatasets,getBinaryDataset, getBinaryDatasetAll, getBinaryDatasetBalanced, getBinaryDatasetAllBalanced
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

def runBestModels(sleepStage, dataset, subjectNum):
    model = None
    acc = -1

    if sleepStage == W:
        acc, model = bestWakeModel(dataset)
    elif sleepStage == N1:
        acc, model = bestN1Model(dataset)
    elif sleepStage == N2:
        acc, model = bestN2Model(dataset)
    elif sleepStage == N3:
        acc, model = bestN3Model(dataset)
    elif sleepStage == REM:
        acc, model = bestREMModel(dataset)
    else:
        print("Unknown sleep stage entered:", sleepStage)
        exit()

    if acc < 0:
        print("Error: Accuracy score below zero:", acc)
        exit()

    return acc

def saveBestModels(sleepStage, dataset, subjectNum):
    model = None
    acc = -1

    if sleepStage == W:
        acc, model = bestWakeModel(dataset)
    elif sleepStage == N1:
        acc, model = bestN1Model(dataset)
    elif sleepStage == N2:
        acc, model = bestN2Model(dataset)
    elif sleepStage == N3:
        acc, model = bestN3Model(dataset)
    elif sleepStage == REM:
        acc, model = bestREMModel(dataset)
    else:
        print("Unknown sleep stage entered:", sleepStage)
        exit()

    if acc < 0:
        print("Error: Accuracy score below zero:", acc)
        exit()

    # save model

    # check if directory exists, if not create it
    saveModelDir = os.getcwd() + "/saved_models/best/" + str(subjectNum) + "/"
    isExist = os.path.exists(saveModelDir)

    if not isExist:
        os.makedirs(saveModelDir)
        print("Folder created:", saveModelDir)

    filename = "best_" + class_dict[sleepStage] +".joblib"
    s = dump(model,saveModelDir + filename)
    print("\t\tsaved")

    return acc

def bestREMModel(dataset):
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset
    rbfModel = trainSVMModel("rbf", xTrain, yTrain)
    rbfAcc = testModel(rbfModel, xTest, yTest)

    return rbfAcc, rbfModel

def bestN3Model(dataset):
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset

    knn = KNeighborsClassifier(3)
    knn.fit(xTrain, yTrain)

    knnAcc = testModel(knn, xTest, yTest)

    return knnAcc, knn

def bestN2Model(dataset):
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset

    rbfModel = trainSVMModel("rbf", xTrain, yTrain)
    polyModel = trainSVMModel("poly", xTrain, yTrain)
    ad = AdaBoostClassifier()
    knn = KNeighborsClassifier(3)
    estimators=[("rbf", rbfModel),("poly", polyModel),("ada boost", ad),("knn", knn)]
    weights=[1.5, 1, 1.5, 1]
    vc = VotingClassifier(estimators,voting="hard", weights=weights)

    rbfModel.fit(xTrain, yTrain)
    polyModel.fit(xTrain, yTrain)
    ad.fit(xTrain, yTrain)
    knn.fit(xTrain, yTrain)
    vc.fit(xTrain, yTrain)

    vcAcc = testModel(vc, xTest, yTest)

    return vcAcc, vc

def bestN1Model(dataset):
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset

    nb = GaussianNB()
    nb.fit(xTrain, yTrain)

    nbAcc = testModel(nb, xTest, yTest)

    return nbAcc, nb

def bestWakeModel(dataset):
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset

    rbfModel = trainSVMModel("rbf", xTrain, yTrain)
    nb = GaussianNB()
    ad = AdaBoostClassifier()
    polyModel = trainSVMModel("poly", xTrain, yTrain)
    estimators=[("rbf", rbfModel), ("naive bayes", nb), ("ada boost", ad), ("poly", polyModel)]
    weights=[1.5, 1.5, 1, 1]
    vc = VotingClassifier(estimators,voting="hard", weights=weights)

    rbfModel.fit(xTrain, yTrain)
    nb.fit(xTrain, yTrain)
    ad.fit(xTrain, yTrain)
    polyModel.fit(xTrain, yTrain)
    vc.fit(xTrain, yTrain)

    vcAcc = testModel(vc, xTest, yTest)

    return vcAcc, vc

def wakeVoting(dataset):
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset

    rbfModel = trainSVMModel("rbf", xTrain, yTrain)
    nb = GaussianNB()
    ad = AdaBoostClassifier()
    polyModel = trainSVMModel("poly", xTrain, yTrain)
    estimators=[("rbf", rbfModel), ("naive bayes", nb), ("ada boost", ad), ("poly", polyModel)]
    weights=[1.5, 1.5, 1, 1]
    vc = VotingClassifier(estimators,voting="hard", weights=weights)

    rbfModel.fit(xTrain, yTrain)
    nb.fit(xTrain, yTrain)
    ad.fit(xTrain, yTrain)
    polyModel.fit(xTrain, yTrain)
    vc.fit(xTrain, yTrain)

    vcAcc = testModel(vc, xTest, yTest)

    return vcAcc

def N1Voting(dataset):
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset

    rbfModel = trainSVMModel("rbf", xTrain, yTrain)
    nb = GaussianNB()
    ad = AdaBoostClassifier()
    estimators=[("naive bayes", nb), ("rbf", rbfModel), ("ada boost", ad)]
    weights=[1.5, 1, 1]
    vc = VotingClassifier(estimators,voting="hard", weights=weights)

    nb.fit(xTrain, yTrain)
    rbfModel.fit(xTrain, yTrain)
    ad.fit(xTrain, yTrain)
    vc.fit(xTrain, yTrain)

    vcAcc = testModel(vc, xTest, yTest)

    return vcAcc

def N2Voting(dataset):
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset

    rbfModel = trainSVMModel("rbf", xTrain, yTrain)
    polyModel = trainSVMModel("poly", xTrain, yTrain)
    ad = AdaBoostClassifier()
    knn = KNeighborsClassifier(3)
    estimators=[("rbf", rbfModel),("poly", polyModel),("ada boost", ad),("knn", knn)]
    weights=[1.5, 1, 1.5, 1]
    vc = VotingClassifier(estimators,voting="hard", weights=weights)

    rbfModel.fit(xTrain, yTrain)
    polyModel.fit(xTrain, yTrain)
    ad.fit(xTrain, yTrain)
    knn.fit(xTrain, yTrain)
    vc.fit(xTrain, yTrain)

    vcAcc = testModel(vc, xTest, yTest)

    return vcAcc

def N3Voting(dataset):
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset

    knn = KNeighborsClassifier(3)
    # rbfModel = trainSVMModel("rbf", xTrain, yTrain)
    # ad = AdaBoostClassifier()
    # estimators=[("knn", knn), ("rbf", rbfModel), ("ada boost", ad)]
    # weights=[2.5, 1, 1,1]
    # vc = VotingClassifier(estimators,voting="hard", weights=weights)

    knn.fit(xTrain, yTrain)
    # rbfModel.fit(xTrain, yTrain)
    # ad.fit(xTrain, yTrain)
    # vc.fit(xTrain, yTrain)

    vcAcc = testModel(knn, xTest, yTest)

    return vcAcc

def REMVoting(dataset):
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset

    rbfModel = trainSVMModel("rbf", xTrain, yTrain)
    nb = GaussianNB()
    ad = AdaBoostClassifier()
    estimators=[("rbf", rbfModel), ("naive bayes", nb), ("ada boost", ad)]
    weights=[2, 1, 1]
    vc = VotingClassifier(estimators,voting="hard", weights=weights)

    rbfModel.fit(xTrain, yTrain)
    nb.fit(xTrain, yTrain)
    ad.fit(xTrain, yTrain)
    vc.fit(xTrain, yTrain)

    vcAcc = testModel(vc, xTest, yTest)

    return vcAcc

def knn(dataset):
    # print("Using k Nearest Neighbors----------")
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset
    # print("Successfully loaded dataset.")

    # print("Training Models...")
    knn = KNeighborsClassifier(3)
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
    knn = KNeighborsClassifier(3)
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
    model = svm.SVC(kernel=kernel)
    model.fit(xTrain, yTrain)

    return model

def trainSVMModelWithWeights(kernel, xTrain, yTrain):
    model = svm.SVC(kernel=kernel, class_weight="balanced")
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

def svmRBFSave(dataset, subjectNum, binaryClass):
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset
    rbfModel = trainSVMModel("rbf", xTrain, yTrain)
    # save model
    # check if directory exists, if not create it
    saveModelDir = os.getcwd() + "/saved_models/" + str(subjectNum) + "/"
    isExist = os.path.exists(saveModelDir)

    if not isExist:
        os.makedirs(saveModelDir)
        print("Folder created:", saveModelDir)

    filename = "svm_" + class_dict[binaryClass] +".joblib"
    s = dump(rbfModel,saveModelDir + filename)
    print("\t\tsaved")
    rbfAcc = testModel(rbfModel, xTest, yTest)

    return rbfAcc

def bestModelSave(dataset, subjectNum, binaryClass):
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset
    # rbfModel = trainSVMModel("rbf", xTrain, yTrain)
    # save model
    # check if directory exists, if not create it
    saveModelDir = os.getcwd() + "/saved_models/" + str(subjectNum) + "/"
    isExist = os.path.exists(saveModelDir)

    if not isExist:
        os.makedirs(saveModelDir)
        print("Folder created:", saveModelDir)

    filename = "svm_" + class_dict[binaryClass] +".joblib"
    s = dump(rbfModel,saveModelDir + filename)
    print("\t\tsaved")
    rbfAcc = testModel(rbfModel, xTest, yTest)

    return rbfAcc

def svmRBF(dataset):
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset
    rbfModel = trainSVMModel("rbf", xTrain, yTrain)
    rbfAcc = testModel(rbfModel, xTest, yTest)

    return rbfAcc

def svmRBFWithWeights(dataset):
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset
    rbfModel = trainSVMModelWithWeights("rbf", xTrain, yTrain)
    rbfAcc = testModel(rbfModel, xTest, yTest)

def testSVMRBF(model, dataset):
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset
    rbfAcc = testModel(model, xTest, yTest)

    return rbfAcc

def testModelwithCM(model, dataset):
    xTrain, yTrain, xTest, yTest, heightTrain, heightTest = dataset

    preds = model.predict(xTest)

    cm = confusion_matrix(yTest, preds, labels=[0, 1], normalize=None)

    return cm

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
    modelNames = ["rbf", "lda", "naive bayes","ada boost", "voting", "knn"]
    scores = np.zeros((len(modelNames,)))

    N = 83 - len(skip)

    for i in range(83):
        print("on subject", i)
        if validSubject(i):
            dataset = getBinaryDataset(W, i)
            print("\tcreated dataset")

            r = svmRBF(dataset)
            scores[0] += r
            print("\tsvm done")

            scores[1] += lda(dataset)
            print("\tlogistic regression done")

            scores[2] += NB(dataset)
            print("\tnaive bayes done")

            scores[3] += adaBoost(dataset)
            print("\tada boost done")

            scores[4] += voting(dataset)
            print("\tvoting done")

            scores[5] += knn(dataset)
            print("\tknn done")
        else:
            print("\tskipped", i)

    for s in range(scores.shape[0]):
        scores[s] /= float(N)
        print(modelNames[s], " score:", scores[s])

def testAlgorithmsBalanced():
    # scores = [rbfAcc, linearAcc, polyAcc, logAcc, nbAcc, dAcc, vcAcc, knnAcc]
    modelNames = ["rbf", "lda", "naive bayes","ada boost", "voting", "knn", "best"]
    scores = np.zeros((len(modelNames,)))
    datasets = [W, N1, N2, N3, REM]

    N = 83 - len(skip)

    for c in datasets:
        print("on class", c)
        for i in range(83):
            print("on subject", i)
            if validSubject(i):
                dataset = getBinaryDataset(c, i)
                print("\tcreated dataset")

                r = svmRBF(dataset)
                scores[0] += r
                print("\tsvm done")

                scores[1] += lda(dataset)
                print("\tlogistic regression done")

                scores[2] += NB(dataset)
                print("\tnaive bayes done")

                scores[3] += adaBoost(dataset)
                print("\tada boost done")

                scores[4] += voting(dataset)
                print("\tvoting done")

                scores[5] += knn(dataset)
                print("\tknn done")

                scores[6] += runBestModels(c, dataset, i)
                print("\tbest model done")
            else:
                print("\tskipped", i)

    for s in range(scores.shape[0]):
        scores[s] /= float(N)
        print(modelNames[s], " score:", scores[s])

def testAllVotingBalanced():
    # scores = [rbfAcc, linearAcc, polyAcc, logAcc, nbAcc, dAcc, vcAcc, knnAcc]
    modelNames = ["wake voting", "n1 voting", "n2 voting", "n3 voting", "rem voting"]
    scores = np.zeros((len(modelNames,)))

    N = 83 - len(skip)

    for i in tqdm(range(83)):
        # print("on subject", i)
        if validSubject(i):
            datasetW, dataset1, dataset2, dataset3, datasetREM = getBinaryDatasetAllBalanced(i)

            scores[0] += wakeVoting(datasetW)

            scores[1] += N1Voting(dataset1)

            scores[2] += N2Voting(dataset2)

            scores[3] += N3Voting(dataset3)

            scores[4] += REMVoting(datasetREM)

    for s in range(scores.shape[0]):
        scores[s] /= float(N)
        print(modelNames[s], " score:", scores[s])

def createBinaryClassifiers():
    # Wake, 1, 2, 3, REM
    scores = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    N = 83 - len(skip)
    for i in range(83):
        print("on subject", i)
        if validSubject(i):
            datasetW, dataset1, dataset2, dataset3, datasetREM = getBinaryDatasetAll(i)

            # insert algorthm to use where
            scores[0] += svmRBFWithWeights(datasetW)
            print("\tW done")
            scores[1] += svmRBFWithWeights(dataset1)
            print("\tN1 done")
            scores[2] += svmRBFWithWeights(dataset2)
            print("\tN2 done")
            scores[3] += svmRBFWithWeights(dataset3)
            print("\tN3 done")
            scores[4] += svmRBFWithWeights(datasetREM)
            print("\tREM done")

    for s in range(scores.shape[0]):
        scores[s] /= N
        print(label2ann[s] + ": " + str(scores[s]))

    scoresNP = np.array(scores)
    filename = os.getcwd() + "/results/svm-rbf-weights"
    np.save(filename, scoresNP)

def trainSaveBinaryClassifiers():
    print("Training and Saving Binary Classifiers: SVM with RBF")
    # Wake, 1, 2, 3, REM
    scores = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    N = 83 - len(skip)

    for i in range(83):
        print("on subject", i)
        if validSubject(i):
            datasetW, dataset1, dataset2, dataset3, datasetREM = getBinaryDatasetAll(i)

            # insert algorthm to use where
            scores[0] += svmRBFSave(datasetW, i, W)
            print("\tW done")
            scores[1] += svmRBFSave(dataset1, i, N1)
            print("\tN1 done")
            scores[2] += svmRBFSave(dataset2, i, N2)
            print("\tN2 done")
            scores[3] += svmRBFSave(dataset3, i, N3)
            print("\tN3 done")
            scores[4] += svmRBFSave(datasetREM, i, REM)
            print("\tREM done")

    for s in range(scores.shape[0]):
        scores[s] /= N
        print(label2ann[s] + ": " + str(scores[s]))

    scoresNP = np.array(scores)
    filename = os.getcwd() + "/results/svm-rbf"
    np.save(filename, scoresNP)

def confusionMatrixBinaryClassifiers(modelTypes, balanced=True):
    print("Testing Binary Classifiers with Confusion Matrix Output")
    totalCM = np.zeros((5,2, 2))

    # Wake, 1, 2, 3, REM
    scores = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    N = 83 - len(skip)
    # 34
    # 70
    for i in tqdm(range(83)):
        # print("on subject", i)
        if validSubject(i):
            if balanced:
                datasetW, dataset1, dataset2, dataset3, datasetREM = getBinaryDatasetAllBalanced(i)
            else:
                atasetW, dataset1, dataset2, dataset3, datasetREM = getBinaryDatasetAll(i)

            datasets = []
            datasets.append(datasetW)
            datasets.append(dataset1)
            datasets.append(dataset2)
            datasets.append(dataset3)
            datasets.append(datasetREM)

            models = []
            for c,data in tqdm(enumerate(datasets)):
                # load saved Models
                saveModelDir = os.getcwd() + "/saved_models/" + modelTypes + "/" + str(i) + "/"
                filename = modelTypes + "_" + class_dict[c] +".joblib"
                isExist = os.path.exists(saveModelDir + filename)

                if not isExist:
                    # train model and save
                    print("model for subject " + str(i) + ", dataset " + class_dict[c] + " does not exist. Traning model...")
                    loadBinaryModel(modelType, i, c)

                m = load(saveModelDir + filename)
                models.append(m)

            for index,data in enumerate(datasets):
                cm = testModelwithCM(models[index], data)
                # print(cm)
                # exit()
                for r in range(2):
                    for c in range(2):
                        totalCM[index][r][c] += cm[r][c]
                # print(totalCM)
                # exit()
                # print("\t" + class_dict[index] +" done")

    np.set_printoptions(suppress=True)
    for z in range(5):
        print(class_dict[z])
        print(totalCM[z])
    np.set_printoptions(suppress=False)

    totalCM = np.array(totalCM)
    filename = os.getcwd() + "/results/svm-rbf-confusion-matrix"
    np.save(filename, totalCM)
    print("DONE!")

def loadBinaryModel(modelName, subjectNum, posClass):

    saveModelDir = os.getcwd() + "/saved_models/" + modelName + "/" + str(subjectNum) + "/"
    filename = modelName + "_" + class_dict[posClass] +".joblib"
    isExist = os.path.exists(saveModelDir + filename)

    # print(filename)

    if not isExist:
        # model file does not exist
        print("model for subject " + str(subjectNum) + ", dataset " + class_dict[posClass] + " does not exist. Traning model...")
        datasetW, dataset1, dataset2, dataset3, datasetREM = getBinaryDatasetAllBalanced(subjectNum)
        datasets = [datasetW, dataset1, dataset2, dataset3, datasetREM]
        if modelName == "svm":
            svmRBFSave(datasets[posClass], subjectNum, posClass)
        elif modelName == "best":
            saveBestModels(posClass, datasets[posClass], subjectNum)
        # exit()

    m = load(saveModelDir + filename)

    return m

def main():
    start = time.time()

    # testAlgorithms() # run all implemented binary classifier algorithms on the dataset
    testAlgorithmsBalanced() # run all implemented binary classifier algorithms on balanced dataset
    # testAllVotingBalanced() # run custom voting classifiers on balanced datasets
    # createBinaryClassifiers() # run SVM with RBF kernel on dataset
    # trainSaveBinaryClassifiers() # run SVM with RBF kernel on datasets and save model
    # confusionMatrixBinaryClassifiers("best") # test best models on datasets, get confusion matrix

    end = time.time()
    print("\nRuntime:", end-start, "seconds")

if __name__ == "__main__":
    main()
