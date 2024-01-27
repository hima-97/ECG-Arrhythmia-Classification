#  Application Of Machine Learning For Single-Lead ECG-Based Arrhythmia Classification Via Smart Wearable Devices
This repository contains a comprehensive machine learning project aimed at classifying heartbeats from single-lead ECG data, pivotal for smart wearable devices. It demonstrates an advanced application of machine learning algorithms in biomedical signal processing, highlighting my proficiency in both fields. 


The project's core objective is to categorize three types of heartbeats: normal (N), supraventricular (S), and ventricular (V), using single-lead ECG signals. This classification is crucial for real-time cardiac monitoring in wearable technology. The project adopts a meticulous approach, encompassing data preprocessing, feature extraction, selection, and model evaluation, culminating in the implementation of a Random Forest classifier.


A detailed of the project, including methodology, experiments, and results, is accessible [here](https://github.com/hima-97/ECG-Arrhythmia-Classification/blob/master/Himanshu%20Kumar%20-%20Application%20of%20Machine%20Learning%20for%20Single-Lead%20ECG-Based%20Arrhythmia%20Classification%20via%20Smart%20Wearable%20Devices.pdf)

# Methodology
The approach adopted in this project is methodical and comprehensive, starting from the preprocessing of raw ECG data to the deployment of a Random Forest classifier. 

Key phases include data preprocessing, feature extraction, feature selection, model training, and evaluation. 

The project emphasizes on mutual information for feature selection, ensuring the model focuses on the most relevant aspects of the ECG data.

# Data Preprocessing
The preprocessing phase is fundamental in transforming raw ECG data into a format suitable for machine learning algorithms. 
This process involves several steps:

* Baseline Wander Removal: A high-pass Butterworth filter eliminates low-frequency noise, ensuring signal stability.
* Noise Reduction: A band-reject Butterworth filter diminishes 60 Hz AC interference, enhancing signal clarity.
* High-frequency Noise Removal: A low-pass Butterworth filter removes unwanted high-frequency components.
* Signal Normalization: Standardizes the signal magnitude within a uniform range for consistent analysis.

# Feature Extraction
Feature extraction is a crucial step in representing ECG signals in a way that highlights characteristics relevant to heartbeat classification. It includes:

* QRS Complex Analysis: Involves calculating the width, amplitude, and slope of the QRS complex, providing insights into heartbeat morphology.
* RR Interval Features: Focuses on the time intervals between consecutive R-peaks, which are indicative of heart rate variability.
* Advanced Descriptors: Incorporates techniques like Hermite Basis Functions (HBF), wavelet descriptors, and Higher Order Statistics (HOS) for a comprehensive signal analysis.

# Feature Selection
Feature selection is conducted using Mutual Information (MI), a statistical measure that assesses the dependency between variables. It identifies features that are most informative about the heartbeat classes, thereby enhancing model performance and reducing computational complexity.

# Model Training and Testing
The project utilizes a Random Forest classifier, an ensemble learning method known for its accuracy and robustness. The classifier is fine-tuned through hyperparameter optimization, ensuring optimal performance. The model undergoes extensive evaluation, including accuracy assessment and cross-validation, to validate its effectiveness in heartbeat classification.

# Requirements
The following Python libraries are required to execute the code: 

* numpy: For data manipulation and numerical computations
* scipy: For scientific and technical computations
* sklearn: Comprehensive toolkit for machine learning and data mining tasks, including model training and evaluation
* wfdb: For reading ECG data from the MIT-BIH Arrhythmia Database
* PyWavelets: For wavelet analysis in feature extraction
* pandas: Data manipulation and analysis library, ideal for handling structured data
* matplotlib (optional): A versatile plotting library for creating static, interactive, and animated visualizations in Python

# Project Structure

```
ECG-Arrhythmia-Classification/
│
├── data/
│   ├── mit-bih-arrhythmia-database-1.0.0/
│   ├── Preprocessed Data 360 Hz/
│   ├── Training/
│   ├── Testing/
│   ├── Heartbeats Data/
│   └── Heartbeats Classifier
│
├── src/
│   ├── __pycache__/
│   ├── feature_extraction_utils/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── advanced_features.py
│   │   ├── pqrs_features.py
│   │   ├── rr_features.py
│   │   └── signal_buffer.py
│   ├── ecg_components.txt
│   ├── ecg_types.py
│   ├── feature_extraction.py
│   ├── feature_selection.py
│   ├── plotting.py
│   ├── preprocessing.py
│   ├── resampling.py
│   ├── split_data.py
│   └── training_and_testing.py
│
├── logs/
├── results/
├── .gitattributes
├── project_structure.txt
├── main.py
└── README.md
└── requirements.txt
```

# How to run the code
Execute the Python 'main.py' file in the _/src_ folder. The following steps will be executed in order:

1. Preprocessing of the original data by reading the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).
2. Resampling of the ECG signals from 360 Hz to the new sampling rate of 256 Hz.
3. Split and save the dataset into training and testing sets according to the literature defined inter-patient paradigm.
4. Segmentation of the ECG signals into individual heartbeats and extraction of the heatbeat features using the extractors defined in the _/src/feature_extraction_utils_ folder
5. Ranking and selecting features based on mutual information and constructing feature datasets for training and testing.
6. Training and testing the Random Forest classifier.