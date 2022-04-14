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
