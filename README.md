# Sleep Stage Classification using Binary Neural Networks
### Erebus Oh, Kenneth Barkdoll

We use code from AttnSleep: https://github.com/emadeldeen24/AttnSleepAll. Do not use link to reproduce our code. All code is not our own except for the following files:
- binaryClassification.py
- hierarchicalBinaryClassification.py
- createDataset.py
- metrics.py
- multiClassification.py
- utlity.py
- classes.py
- prepare_datasets/processedDatasets
- results/
- environment.yaml

See READMEAttnSleep.md for their README file. Note: We only use Sleep-EDF Database Expanded Cassette Data.

# How to run our experiments:
Note: Code was run and developed on Linux Ubuntu 20.04 LTS.

Clone or download the source code at ([Github](https://github.com/ereoh/Binary-Sleep-Classification)).

## Creating and Activating the Conda Environment
Ensure you have Anaconda ([Installation instructions here](https://www.anaconda.com/products/individual)).
```
conda env create --file environment.yml
conda activate asenv
```

If you are unable to create the Conda Environemnt, below is a list of requirements that you can install:
- jq 1.6 (sudo apt-get install jq)
- python 3.7
- pytorch 1.4.0
- numpy 1.19.2
- scikit-learn 1.0.2
- scipy 1.6.2
- pandas 1.3.4
- openpyxl 3.0.9
- mne 0.20.7

## Prepare Dataset
We use [Sleep-EDF Database Expanded](https://www.physionet.org/content/sleep-edfx/1.0.0/).
Note: Downloading and preparing the dataset might take some time.

First download and prepare the dataset with the following commands:
```
mkdir dataset
wget -r -N -c -np -nd -nH -P dataset/ https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/
```
Estimated time: 15 minutes
```
cd prepare_datasets
python prepare_physionet.py --data_dir ../dataset/ --output_dir edf_20_npz --select_ch "EEG Fpz-Cz"
```
Estimated time: 20 minutes
```
cd ..
python createDataset.py
```
Estimated time: 15 minutes

## Run Models

### Binary Classification
To recreate our binary classifiers using SVM with Radial Basis Function kernel, run:
```
python binaryClassification.py
```

### Multiclass classification
To recreate our confusion matrix from Random Forests, run:
```
python multiClassification.py
```

### Dataset metrics
To recreate our dataset metrics calculations, run:
```
python metrics.py
```
