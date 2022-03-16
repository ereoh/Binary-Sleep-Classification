import sklearn

skip = [36, 52, 13, 39, 68, 69, 78, 79, 32, 33, 34, 72, 73, 57, 74, 75, 64]

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
