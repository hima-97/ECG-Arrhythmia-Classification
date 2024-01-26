#  Application Of Machine Learning For Single-lead ECG-based Arrhythmia Classification Via Smart Wearable Devices
This repo contains the code used for 

# Requirements
The following Python libraires are required to execute the code: 

* numpy
* sklearn
* matplotlib (optional)
* wfdb
* scipy
* pandas
* PyWavelets

# Project Structure

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

# How to run the code
Execute the Python 'main.py' file in the _/src_ folder. The following steps will be executed in order:

1. Preprocessing of the original data by reading the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).
2. Resampling of the ECG signals from 360 Hz to the new sampling rate of 256 Hz.
3. Split and save the dataset into training and testing sets according to the literature defined inter-patient paradigm.
4. Segmentation of the ECG signals into individual heartbeats and extraction of the heatbeat features using the extractors defined in the _/src/feature_extraction_utils_ folder
5. Ranking and selecting features based on mutual information and constructing feature datasets for training and testing.
6. Training and testing the Random Forest classifier.


Symbols used in plots:
[An expanded and updated version of the table below can be found at http://www.physionet.org/physiobank/annotations.shtml.]


Beat annotations:

Symbol:	              Meaning:

· or N	              Normal beat
L	                  Left bundle branch block beat
R	                  Right bundle branch block beat
A	                  Atrial premature beat
a	                  Aberrated atrial premature beat
J	                  Nodal (junctional) premature beat
S	                  Supraventricular premature beat
V	                  Premature ventricular contraction
F	                  Fusion of ventricular and normal beat
[	                  Start of ventricular flutter/fibrillation
!	                  Ventricular flutter wave
]	                  End of ventricular flutter/fibrillation
e	                  Atrial escape beat
j	                  Nodal (junctional) escape beat
E	                  Ventricular escape beat
/	                  Paced beat
f	                  Fusion of paced and normal beat
x	                  Non-conducted P-wave (blocked APB)
Q	                  Unclassifiable beat
|	                  Isolated QRS-like artifact




Rhythm annotations appear below the level used for beat annotations:

(AB	                  Atrial bigeminy
(AFIB	              Atrial fibrillation
(AFL	              Atrial flutter
(B	                  Ventricular bigeminy
(BII	              2° heart block
(IVR	              Idioventricular rhythm
(N	                  Normal sinus rhythm
(NOD	              Nodal (A-V junctional) rhythm
(P	                  Paced rhythm
(PREX	              Pre-excitation (WPW)
(SBR	              Sinus bradycardia
(SVTA	              Supraventricular tachyarrhythmia
(T	                  Ventricular trigeminy
(VFL	              Ventricular flutter
(VT	                  Ventricular tachycardia




Signal quality and comment annotations appear above the level used for beat annotations:

qq	                  Signal quality change: the first character (`c' or `n') indicates the quality of the upper signal (clean or noisy), 
                      and the second character indicates the quality of the lower signal
U	                  Extreme noise or signal loss in both signals: ECG is unreadable
M (or MISSB)	      Missed beat
P (or PSE)	          Pause
T (or TS)	          Tape slippage

