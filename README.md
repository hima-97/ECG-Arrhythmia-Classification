#  Application Of Machine Learning For Single-Lead ECG-Based Arrhythmia Classification Via Smart Wearable Devices
This repository presents an intricate machine learning project focused on classifying heartbeats from single-lead ECG data, crucial in the domain of smart wearable devices. It showcases an advanced application of machine learning techniques in biomedical signal processing, underscoring my expertise in both areas.

The primary goal is to categorize three heartbeat types: normal (N), supraventricular (S), and ventricular (V) beats, using single-lead ECG signals. This classification is essential for real-time cardiac monitoring through wearable technology. The project embodies a detailed methodology, encompassing data preprocessing, feature extraction and selection, and the implementation and evaluation of a Random Forest classifier.

A comprehensive outline of the project, including methodologies, experiments, and results, is available [here](https://github.com/hima-97/ECG-Arrhythmia-Classification/blob/master/Himanshu%20Kumar%20-%20Application%20of%20Machine%20Learning%20for%20Single-Lead%20ECG-Based%20Arrhythmia%20Classification%20via%20Smart%20Wearable%20Devices.pdf)

## Methodology
The methodology employed in this project reflects a rigorous and holistic approach to classifying cardiac rhythms using single-lead ECG data, a pivotal feature in contemporary smart wearable devices like the Apple Watch and Samsung Watch. The project pivots on the Random Forest (RF) classifier, selected for its effectiveness in handling complex datasets.

Comprehensive Approach:

* Data Exploration and Analysis:  
Initial exploration to grasp the intrinsic characteristics of the ECG dataset.

* Signal Preprocessing:  
Noise removal and resampling of ECG signals to align with the required sampling rate.

* Heartbeat Segmentation:  
Heartbeat segmentation to isolate individual cardiac cycles, critical for accurate feature extraction.

* Feature Extraction:  
Deriving key attributes from heartbeats, transforming raw data into a machine-learning-ready format.  
Employing advanced techniques for a thorough representation of ECG signals.

* Feature Selection with Mutual Information (MI):  
Applying MI for prioritizing features based on relevance, ensuring model efficiency and focus.

* Model Training:  
Training the RF model using selected features, fine-tuning its ability to classify arrhythmias.

* Hyperparameter Tuning:  
Optimizing model parameters to enhance training efficiency and accuracy.

* Model Validation and Evaluation:  
Implementing a custom leave-one-out cross-validation strategy to validate the model’s generalizability and mitigate overfitting.
Thorough evaluation in a separate testing phase to assess the model's performance on unseen data.

* Performance Assessment:  
Utilizing Key Performance Indicators (KPIs) such as accuracy, precision, recall, and F1 score for final model evaluation.
These metrics provide a comprehensive view of the model's diagnostic capabilities in classifying arrhythmias from single-lead ECG data.

This methodical and detailed approach ensures a seamless transition from raw ECG data acquisition to developing a proficient model capable of classifying different arrhythmias, thereby demonstrating the project's applicability in smart wearable technology for remote healthcare monitoring.


### Data Exploration and Analysis
For this project, the MIT-BIH Arrhythmia Database, sourced from Physionet.org, was meticulously selected.  
This dataset is a gold standard in cardiac arrhythmia research, thanks to its extensive collection of accurately annotated ECG recordings.

The database comprises 48 half-hour excerpts from two-channel ambulatory ECG recordings, carefully curated from over 4000 long-term Holter recordings at the Beth Israel Hospital Arrhythmia Laboratory between 1975 and 1979. Each record encapsulates 30 minutes of ECG data, sampled at 360 Hz per lead, amounting to 648,000 data points per lead per recording.

Recording Quality:  
The ECG equipment was battery-powered, minimizing 60 Hz noise typically introduced during recording.  
However, noise at 30 Hz, resulting from double-speed digitization during playback, was noted.

Exclusion Criteria:  
Records 102, 104, 107, and 217, containing paced beats, were excluded in line with the AAMI recommended practice and similar studies. This exclusion led to a refined dataset of 44 records and 100,733 labeled heartbeats.
 
Each record includes:
* Header file (.hea): Details the signal's attributes like format, type, and sample count.  
* Binary file (.dat): Contains the digitized ECG signal.  
* Annotation file (.atr): Houses expert annotations aligned with the R-wave peaks, ensuring accuracy for beat characterization.

Annotation and Classification Focus:  
Two independent cardiologists annotated the heartbeats, providing high-reliability labels for each beat.  
The heartbeats are classified into five categories based on the AAMI EC57 standard:  
* N (Normal Beat)
* S (Supraventricular Ectopic Beat)
* V (Ventricular Ectopic Beat)
* F (Fusion Beat)
* Q (Unknown Beat)

For the purpose of this study, Fusion Beats (F) and Unknown Beats (Q) were excluded, aligning with previous studies.  
Consequently, the project focuses on classifying three beat types: Normal, Supraventricular, and Ventricular beats.

### Data Preprocessing
The preprocessing of ECG signals is a critical phase in this project, ensuring the raw data's accuracy and suitability for machine learning algorithms. The preprocessing pipeline has been meticulously designed to address various artifacts commonly present in raw ECG signals, such as noise induced by muscle contractions, power-line interference, and baseline wander.

* Baseline Wander Removal:  
Baseline wander, often introduced by patient movements or respiration, manifests as low-frequency noise in ECG signals. A high-pass Butterworth filter, with a default cutoff frequency of 1 Hz, is employed to counter this.  
The frequency can be adjusted to 0.5 Hz in cases of significant baseline wander around this range. Signal padding is incorporated to minimize edge effects, and the filter order is kept at 1 to avoid over-attenuation.

* Noise Reduction:  
To mitigate external electrical noise, especially the 60 Hz interference from power lines, a band-reject Butterworth filter with cutoff frequencies of 59 Hz and 61 Hz is utilized. This approach effectively eliminates AC power line interference, ensuring the signal's integrity. Notably, the majority of the 60 Hz noise in the database originates from the playback stage of the recording equipment, which was battery-powered. Signal padding is again used here to reduce edge effects during filtering.

* High-frequency Noise Removal:  
To suppress high-frequency noise components while preserving clinically relevant information, a low-pass Butterworth filter with a 25 Hz cut-off frequency is applied. This step is critical in maintaining the balance between noise reduction and data integrity. Signal padding is also employed here to mitigate edge effects.

* Normalization:  
Normalizing the ECG signals between 0 and 1 is vital for standardizing signal amplitude across different recordings. This step is crucial when dealing with diverse datasets, ensuring a consistent analytical approach. Normalization thus facilitates accurate and interpretable data analysis.

* ECG Signal Resampling:  
Considering the standard 256 Hz sampling rate of modern smart wearables like the Hexoskin Pro Kit, Apple Watch, and Samsung Watch, the ECG signals from the MIT-BIH Arrhythmia Database, originally at 360 Hz, are resampled to 256 Hz. This resampling process aligns the data sampling rates with those of the target devices, ensuring the model's applicability and accuracy upon deployment. The resampling is achieved through the formula  
num_samples_resampled = int((256/360) * num_samples_original)  
maintaining the proportionality between the original and the new sampling rates. The annotation points are also adjusted correspondingly to accurately locate R-peaks in the resampled ECG signals.

### Heartbeat segmentation
Heartbeat segmentation is a pivotal step in the analysis of ECG signals, particularly for arrhythmia classification.  
This process entails segmenting ECG signals into individual heartbeats, each representing a single cardiac cycle.  
Such segmentation is crucial for precise feature extraction and effective classification.

Dataset Division Using Inter-patient Paradigm:  
To ensure a realistic and clinically applicable approach, the dataset, sourced from the MIT-BIH Arrhythmia Database,  
is divided into training and testing sets based on an inter-patient paradigm. This division ensures that the model is trained and tested on data from different patients, bolstering its ability to generalize effectively to new, unseen data.

The 44 recordings are evenly split as follows:
* Training Set:  
101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230
* Testing Set:  
100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234

Segmentation Process:  
The segmentation process leverages the R spike annotations from the MIT-BIH Arrhythmia Database as markers to identify individual heartbeats. These annotations, typically located at the R-wave peak of the QRS complex, are used to segment the ECG signal on a beat-by-beat basis. The approach recognizes each cardiac cycle as an independent unit, allowing for a detailed analysis of inter-beat variability and morphological differences.

The segmentation process begins with the detection of the R peak using annotations provided in the record's .atr files. Each heartbeat is segmented by selecting a 640 ms window around the annotated R peak, comprising 373 ms before and 267 ms after the R peak. This window size is chosen to encompass the complete QRS complex and adjacent parts of the ECG waveform.

To normalize the signal and remove baseline wander, the mean value of each segment is subtracted from its individual samples. This normalization centers the signal around the R peak, ensuring an accurate and consistent analysis of the ECG waveform.

### Feature Extraction
Feature extraction is a crucial step in representing ECG signals in a way that highlights characteristics relevant to heartbeat classification. It includes:

* QRS Complex Analysis: Involves calculating the width, amplitude, and slope of the QRS complex, providing insights into heartbeat morphology.
* RR Interval Features: Focuses on the time intervals between consecutive R-peaks, which are indicative of heart rate variability.
* Advanced Descriptors: Incorporates techniques like Hermite Basis Functions (HBF), wavelet descriptors, and Higher Order Statistics (HOS) for a comprehensive signal analysis.

### Feature Selection
Feature selection is conducted using Mutual Information (MI), a statistical measure that assesses the dependency between variables. It identifies features that are most informative about the heartbeat classes, thereby enhancing model performance and reducing computational complexity.

Implements Mutual Information (MI) ranking for feature selection to reduce computational complexity and enhance model accuracy.
Features are ranked according to their relevance in predicting heartbeat classes.

This stage is critical in enhancing model accuracy and reducing computational complexity. Utilizing Mutual Information (MI), a statistical measure of dependency, this process identifies features that significantly contribute to classifying the heartbeat types. The top 10 MI-ranked features are selected for model training, ensuring a focus on the most relevant ECG signal characteristics.

### Model Training and Testing
The project harnesses a Random Forest classifier, renowned for its accuracy and resilience. The model undergoes rigorous hyperparameter tuning for optimal performance. Extensive evaluation methods, including accuracy assessment and cross-validation, are employed to affirm the classifier's efficacy in heartbeat categorization.

A robust Random Forest classifier with 84 decision trees is implemented. Extensive model evaluation includes a leave-one-out cross-validation strategy, considering the data's uniqueness. The model's performance is meticulously assessed through accuracy, precision, recall, F1 score, and specificity metrics. These evaluations demonstrate the classifier's proficiency in distinguishing between normal, supraventricular, and ventricular heartbeats, validating its potential for integration into smart wearable technologies.

### Results:
The model is assessed through three experimental setups, each refining the classifier's performance.
Hyperparameter tuning is performed to optimize the number of trees and features.
Performance metrics demonstrate the model's high accuracy and reliability in arrhythmia classification.

## Requirements
The following Python libraries are required to execute the code: 

* numpy: For data manipulation and numerical computations
* scipy: For scientific and technical computations
* sklearn: For machine learning and data mining tasks, including model training and evaluation
* wfdb: For reading ECG data from the MIT-BIH Arrhythmia Database
* PyWavelets: For wavelet analysis in feature extraction
* pandas: For structured data handling and analysis
* matplotlib (optional): For creating static, interactive, and animated visualizations in Python

## Project Structure

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

## How to run the code
Execute the Python 'main.py' file in the _/src_ folder. The following steps will be executed in order:

1. Preprocessing of the original data by reading the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).
2. Resampling of the ECG signals from 360 Hz to the new sampling rate of 256 Hz.
3. Split and save the dataset into training and testing sets according to the literature defined inter-patient paradigm.
4. Segmentation of the ECG signals into individual heartbeats and extraction of the heatbeat features using the extractors defined in the _/src/feature_extraction_utils_ folder
5. Ranking and selecting features based on mutual information and constructing feature datasets for training and testing.
6. Training and testing the Random Forest classifier.