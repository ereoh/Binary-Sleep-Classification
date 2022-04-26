import time
from binaryClassification import confusionMatrixBinaryClassifiers

def main():
    start = time.time()

    confusionMatrixBinaryClassifiers("best") # test best models on datasets, get confusion matrix

    end = time.time()
    print("\nRuntime:", end-start, "seconds")

if __name__ == "__main__":
    main()
