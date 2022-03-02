# Sleep Stage Classification using Binary Neural Networks
### Erebus Oh, Kenneth Barkdoll

We use code from AttnSleep: https://github.com/emadeldeen24/AttnSleepAll. All code is not our own except for the following files:
- binaryClassification.py
- createDataset.py
- metrics.py
- multiClassification.py
- environment.yaml
- utlity.py
- prepare_datasets/processedDatasets
- results/

See READMEAttnSleep.md for their README file. Note: We only use Sleep-EDF Database Expanded Cassette Data.

# How to run our experiments:

## Creating and Activating the Conda Environemnt
```
conda env create --file environemnt.yml
conda activate asenv
```

## Prepare Dataset
We use [Sleep-EDF Database Expanded](https://www.physionet.org/content/sleep-edfx/1.0.0/).

First download the dataset on the website.
The path to the PSG files should be: [where you downloaded from link]/dataset/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette
```
cd prepare_datasets
python prepare_physionet.py --data_dir /path/to/PSG/files --output_dir edf_20_npz --select_ch "EEG Fpz-Cz"
cd ..
python createDataset.py
```
Note: Use forward slashes for the path to PSG files. Preparing the dataset might take some time.

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
