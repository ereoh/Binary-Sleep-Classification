import time

from multiClassification import confusionMatrixBalanced

def main():
    start = time.time()

    confusionMatrixBalanced() # run Random Forests on balanced dataset for confusion matrix

    end = time.time()
    print("Runtime:", end-start, "seconds")

if __name__ == "__main__":
    main()
