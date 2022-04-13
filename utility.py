import sklearn
import os
from joblib import dump, load

skip = [36, 52, 13, 39, 68, 69, 78, 79, 32, 33, 34, 72, 73, 57, 74, 75, 64]

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}

def loadBinaryModel(subjectNum, posClass):
    saveModelDir = os.getcwd() + "/saved_models/" + str(subjectNum) + "/"
    filename = "svm_" + class_dict[posClass] +".joblib"
    isExist = os.path.exists(saveModelDir + filename)

    if not isExist:
        # model file does not exist
        print("model for subject " + str(subjectNum) + ", dataset " + class_dict[posClass] + " does not exist. Traning model...")
        exit()

    m = load(saveModelDir + filename)

    return m

def evaluateAcc(true, predictions):
    total = true.shape[0]
    totalCorrect = 0.0

    for i in range(total):
        if(predictions[i] == true[i]):
            totalCorrect += 1

    accuracy = totalCorrect/total

    return accuracy

def testModel(model, xTest, yTest):
    pred = model.predict(xTest)

    totalPreds = len(pred)
    totalCorrect = 0.0

    for i in range(totalPreds):
        if(pred[i] == yTest[i]):
            totalCorrect += 1

    accuracy = totalCorrect/totalPreds

    return accuracy

def validSubject(num):
    for s in skip:
        if num == s:
            return False
    return True
