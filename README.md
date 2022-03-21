# Sleep Stage Classification using Binary Neural Networks
### [authors redacted]

We use code from AttnSleep: https://github.com/emadeldeen24/AttnSleepAll. All code is not our own except for the following files:
- binaryClassification.py
- createDataset.py
- metrics.py
- multiClassification.py
- utlity.py
- prepare_datasets/processedDatasets
- results/
- environment.yaml

See READMEAttnSleep.md for their README file. Note: We only use Sleep-EDF Database Expanded Cassette Data.

# How to run our experiments:
Note: Code was run and developed on Linux Ubuntu 20.04 LTS.

## Creating and Activating the Conda Environment
Ensure you have Anaconda ([Installation instructions here](https://www.anaconda.com/products/individual)).
```
conda env create --file environment.yml
conda activate asenv
```

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
