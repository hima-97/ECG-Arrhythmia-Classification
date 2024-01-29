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
Resampling of ECG signals to align with the required sampling rate.
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
For this project, the MIT-BIH Arrhythmia Database, sourced from Physionet.org, was meticulously selected. This dataset is a gold standard in cardiac arrhythmia research, thanks to its extensive collection of accurately annotated ECG recordings.

The database comprises 48 half-hour excerpts from two-channel ambulatory ECG recordings, carefully curated from over 4000 long-term Holter recordings at the Beth Israel Hospital Arrhythmia Laboratory between 1975 and 1979. Each record encapsulates 30 minutes of ECG data, sampled at 360 Hz per lead, amounting to 648,000 data points per lead per recording.

Recording Quality:  
The ECG equipment was battery-powered, minimizing 60 Hz noise typically introduced during recording. However, noise at 30 Hz, resulting from double-speed digitization during playback, was noted.

Exclusion Criteria:  
Records 102, 104, 107, and 217, containing paced beats, were excluded in line with the AAMI recommended practice and similar studies.  
This exclusion led to a refined dataset of 44 records and 100,733 labeled heartbeats.

File Structure:  
Each record includes:  
* Header file (.hea): Details the signal's attributes like format, type, and sample count.  
* Binary file (.dat): Contains the digitized ECG signal.  
* Annotation file (.atr): Houses expert annotations aligned with the R-wave peaks, ensuring accuracy for beat characterization.

Annotation and Classification Focus:  
Expert Annotations: Two independent cardiologists annotated the heartbeats, providing high-reliability labels for each beat.
Beat Classification: The heartbeats are classified into five categories based on the AAMI EC57 standard.  
* N (Normal Beat)
* S (Supraventricular Ectopic Beat)
* V (Ventricular Ectopic Beat)
* F (Fusion Beat)
* Q (Unknown Beat)

For the purpose of this study, Fusion Beats (FB) and Unknown Beats (UB) were excluded, aligning with previous research paradigms. Consequently, the project zeroes in on classifying three critical beat types: Normal, Supraventricular, and Ventricular beats.

### Data Preprocessing
The preprocessing phase is fundamental in transforming raw ECG data into a format suitable for machine learning algorithms. 
This process involves several steps:

* Baseline Wander Removal: A high-pass Butterworth filter eliminates low-frequency noise for signal stability.
* Noise Reduction: A band-reject Butterworth filter diminishes 60 Hz AC interference, enhancing signal clarity.
* High-frequency Noise Removal: A low-pass Butterworth filter removes unwanted high-frequency components.
* Signal Normalization: Standardizes the signal magnitude within a uniform range for consistent analysis.

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