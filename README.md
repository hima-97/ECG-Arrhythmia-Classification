#  Application Of Machine Learning For Single-Lead ECG-Based Arrhythmia Classification Via Smart Wearable Devices
This repository presents an intricate machine learning project focused on classifying heartbeats from single-lead ECG data, crucial in the domain of contemporary smart wearable devices like the Apple Watch and Samsung Watch.  
This project showcases advanced application of machine learning in the field of biomedical signal processing.

The primary goal is to categorize three heartbeat types: normal (N), supraventricular (S), and ventricular (V) beats, using single-lead ECG signals. This classification is essential for real-time cardiac monitoring through wearable technology. The project embodies a detailed methodology, encompassing data preprocessing, feature extraction and selection, and the implementation and evaluation of a Random Forest classifier.

Harnessing the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/), this project transforms two-lead ECG data to emulate single-lead data acquisition, aligning with the capabilities of current smart wearable devices. The projects culminates in a Random Forest model that demonstrates remarkable accuracy, achieving an overall accuracy of 95.69% and a weighted average F1 score of 95.37%. This performance illustrates the model's proficiency in distinguishing between normal, supraventricular, and ventricular heartbeats, reinforcing its potential for integration into smart wearable technologies for effective cardiac health monitoring.

A comprehensive outline of the project, including methodologies, experiments, and results, is available [here](https://github.com/hima-97/ECG-Arrhythmia-Classification/blob/master/Himanshu%20Kumar%20-%20Application%20of%20Machine%20Learning%20for%20Single-Lead%20ECG-Based%20Arrhythmia%20Classification%20via%20Smart%20Wearable%20Devices.pdf).

## Methodology
Comprehensive Approach:

* Data Exploration and Analysis:  
Initial exploration to grasp the intrinsic characteristics of the ECG dataset.

* Signal Preprocessing:  
Noise removal and resampling of ECG signals to align with the required sampling rate.

* Heartbeat Segmentation:  
Heartbeat segmentation to isolate individual cardiac cycles, critical for accurate feature extraction.

* Feature Extraction:  
Deriving key attributes from heartbeats, transforming raw data into a machine-learning-ready format.  

* Feature Selection with Mutual Information (MI):  
Applying MI for prioritizing features based on relevance, ensuring model efficiency and focus.

* Model Training:  
Training the RF model using selected features, fine-tuning its ability to classify arrhythmias.

* Hyperparameter Tuning:  
Optimizing model parameters to enhance training efficiency and accuracy.

* Model Validation and Evaluation:  
Custom leave-one-out cross-validation strategy to validate the model’s generalizability and mitigate overfitting.
Thorough evaluation in a separate testing phase to assess the model's performance on unseen data.

* Performance Assessment:  
Utilizing Key Performance Indicators (KPIs) such as accuracy, precision, recall, and F1 score for final model evaluation.
These metrics provide a comprehensive view of the model's diagnostic capabilities in classifying arrhythmias from single-lead ECG data.

This methodical and detailed approach ensures a seamless transition from raw ECG data acquisition to developing a proficient model capable of classifying different arrhythmias, thereby demonstrating the project's applicability in smart wearable technology for remote healthcare monitoring.


### Data Exploration and Analysis
For this project, the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/), sourced from Physionet.org, was meticulously selected.  
This dataset is a gold standard in cardiac arrhythmia research, thanks to its extensive collection of accurately annotated ECG recordings.

The database comprises 48 half-hour excerpts from two-channel ambulatory ECG recordings, carefully curated from over 4000 long-term Holter recordings at the Beth Israel Hospital Arrhythmia Laboratory between 1975 and 1979. Each record encapsulates 30 minutes of ECG data, sampled at 360 Hz per lead, amounting to 648,000 data points per lead per recording.

Exclusion Criteria:  
Records 102, 104, 107, and 217, containing paced beats, were excluded in line with the Association for the Advancement of Medical Instrumentation (AAMI) recommended practice and similar studies. This exclusion led to a refined dataset of 44 records and 100,733 labeled heartbeats.
 
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
The preprocessing of ECG signals ensures the raw data's accuracy and suitability for machine learning algorithms.  
The preprocessing pipeline has been meticulously designed to address various artifacts commonly present in raw ECG signals, such as noise induced by muscle contractions, power-line interference, and baseline wander.

* Baseline Wander Removal:  
Baseline wander, often introduced by patient movements or respiration, manifests as low-frequency noise in ECG signals. A high-pass Butterworth filter, with a default cutoff frequency of 1 Hz, is employed to counter this.  
The frequency can be adjusted to 0.5 Hz in cases of significant baseline wander around this range. Signal padding is incorporated to minimize edge effects, and the filter order is kept at 1 to avoid over-attenuation.

* Noise Reduction:  
To mitigate external electrical noise, especially the 60 Hz interference from power lines, a band-reject Butterworth filter with cutoff frequencies of 59 Hz and 61 Hz is utilized. This approach effectively eliminates AC power line interference, ensuring the signal's integrity. Notably, the majority of the 60 Hz noise in the database originates from the playback stage of the recording equipment, which was battery-powered. Signal padding is again used here to reduce edge effects during filtering.

* High-frequency Noise Removal:  
To suppress high-frequency noise components while preserving clinically relevant information, a low-pass Butterworth filter with a 25 Hz cut-off frequency is applied. This step is critical in maintaining the balance between noise reduction and data integrity. Signal padding is also employed here to mitigate edge effects.

* Normalization:  
Normalizing the ECG signals between 0 and 1 is vital for standardizing signal amplitude across different recordings. This step is crucial when dealing with diverse datasets, ensuring a consistent analytical approach.

* ECG Signal Resampling:  
Considering the standard 256 Hz sampling rate of modern smart wearables like the Hexoskin Pro Kit, Apple Watch, and Samsung Watch, the ECG signals from the MIT-BIH Arrhythmia Database, originally at 360 Hz, are resampled to 256 Hz. This resampling process aligns the data sampling rates with those of the target devices, ensuring the model's applicability and accuracy upon deployment.

    The resampling is achieved through the formula

        num_samples_resampled = int((256/360) * num_samples_original)  
  
  maintaining the proportionality between the original and the new sampling rates. The annotation points are also adjusted correspondingly to accurately locate R-peaks in the resampled ECG signals.

### Heartbeat segmentation
Heartbeat segmentation entails segmenting ECG signals into individual heartbeats, each representing a single cardiac cycle. Such segmentation is crucial for precise feature extraction and effective classification.

* Dataset Division Using Inter-patient Paradigm:  
To ensure a realistic and clinically applicable approach, the dataset is divided into training and testing sets based on an inter-patient paradigm. This division ensures that the model is trained and tested on data from different patients, improving its ability to generalize effectively to new, unseen data.

  The 44 recordings are evenly split as follows:

        - Training Set: 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230
        - Testing Set: 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234

* Data Processing and Serialization:  
During the dataset division into training and testing sets, each heartbeat in the ECG data undergoes a processing step to create a structured representation. More specifically, each heartbeat is represented by a structured dictionary with details such as time, type, and rhythm class.  

  To ensure efficient data handling and preserve the integrity and structure of the data, the Python ```pickle``` module is utilized. This module is instrumental for serialization, converting the structured data into a byte stream format that can be easily stored or transmitted. When the model is set for training or evaluation, these pickle files can be deserialized back into Python objects. This process ensures that all the structured data is accurately reconstructed and can be directly utilized in our machine learning models without additional preprocessing.

* Segmentation Process:  
The segmentation process leverages the R spike annotations from the MIT-BIH Arrhythmia Database as markers to identify individual heartbeats. These annotations, typically located at the R-wave peak of the QRS complex, are used to segment the ECG signal on a beat-by-beat basis. The approach recognizes each cardiac cycle as an independent unit, allowing for a detailed analysis of inter-beat variability and morphological differences.

The segmentation process begins with the detection of the R peak using annotations from the record's .atr files.  
Each heartbeat is segmented by selecting a 640 ms window around the annotated R peak, comprising 373 ms before and 267 ms after the R peak. This window size is chosen to encompass the complete QRS complex and adjacent parts of the ECG waveform.

To normalize the signal and remove baseline wander, the mean value of each segment is subtracted from its individual samples. This normalization centers the signal around the R peak, ensuring an accurate and consistent analysis of the ECG waveform.

### Feature Extraction
Feature extraction is a pivotal phase in this study, where distinct and quantifiable attributes are derived from segmented ECG heartbeats. These features are instrumental in distinguishing between different types of heartbeats and are essential for the accurate classification of arrhythmias.

Feature extraction is a crucial step in representing ECG signals in a way that highlights characteristics relevant to heartbeat classification. It includes:

* QRS Complex Analysis: Involves calculating the width, amplitude, and slope of the QRS complex, providing insights into heartbeat morphology.
* RR Interval Features: Focuses on the time intervals between consecutive R-peaks, which are indicative of heart rate variability.
* Advanced Descriptors: Incorporates techniques like Hermite Basis Functions (HBF), wavelet descriptors, and Higher Order Statistics (HOS) for a comprehensive signal analysis.

A total of 141 features are extracted and categorized as follows:

* Heart Rate Related Features:  
These include the current, previous, and next RR intervals, which are the time intervals between successive R peaks. They provide critical insights into the heart rate variability.

* Normalized Heart Rate Related Features:  
The RR interval features are normalized by dividing them by their average value in the last 32 heartbeats, enhancing their comparability across different heartbeats.

* QRS Temporal Features:  
These encompass the total duration of the QRS complex, its width at half and a quarter of its peak value, and the distance between the Q wave peak and the S wave peak, shedding light on the morphology of each heartbeat.

* Normalized QRS Temporal Features:  
Similar to heart rate features, these QRS temporal characteristics are normalized against their average values in the last 32 beats.

* Hermite Basis Function (HBF) Coefficients:  
Utilizing Hermite polynomials of degrees 3, 4, and 5, we decompose each beat segment into a series of orthogonal polynomials. These coefficients effectively summarize the waveform's characteristics, capturing subtle variations indicative of different arrhythmias. Each beat segment is analyzed using the hermfit function in Python, generating coefficients that represent the ECG waveform's characteristics.

* Discrete Wavelet Transform (DWT) Coefficients:  
We apply DWT using the db1 wavelet at three levels of decomposition. This multi-resolution analysis dissects the ECG signal into various frequency bands, enabling detailed examination of specific signal characteristics. The ECG signal is decomposed into different frequency bands, facilitating the analysis of time-varying characteristics of the ECG waveform.

* Higher Order Statistics (HOS) Features:  
Including third and fourth-order cumulants like kurtosis and skewness, HOS features provide a deeper understanding of the signal's shape and distribution, especially in the context of heart rhythm irregularities. Calculating kurtosis and skewness within specific intervals around the R peak, we gain insights into the signal distribution's asymmetry and 'tailedness'.

* Euclidean Distances:  
We measure the Euclidean distances between the R peak and four strategically chosen points in the ECG waveform. These distances quantify the variations in the waveform relative to the R peak, offering insights into the heart's electrical activity. These are calculated between the R peak and four points in the ECG segment, highlighting amplitude variations relative to the R peak.

* Heartbeat Amplitude Features:  
These features include the amplitude differences between various wave peaks (P, Q, R, S), providing an understanding of the electrical forces generated by the heart. Fiducial points, including the peaks of the P, Q, R, S waves, are identified through inflection points in the ECG signal. The differences in amplitude between these points are critical in analyzing the signal morphology. In cases where the QRS complex exhibits complex morphology, the maximum signal value within a 100 ms window around the annotated R peak (𝑄𝑅𝑆𝑚𝑎𝑥) serves as a reference for accurately identifying the R peak. This adaptation is crucial for ensuring the reliability of feature extraction, especially in abnormal heartbeats.

    A detailed explanation of the algorithm used to extract the key fiducial points from each heartbeat heartbeat amplitude features is available [here](https://github.com/hima-97/ECG-Arrhythmia-Classification/blob/master/src/heartbeat_amplitude_features.txt).

    After pinpointing the fiducial points, calculating all the features becomes straightforward by assessing the differences in values or positions of these corresponding fiducial points.

    A figure showing the temporal properties and variations in amplitude derived from the cardiac cycle in a normal ECG, including the identification of key fiducial points used to extract these measurements, is available [here](https://github.com/hima-97/ECG-Arrhythmia-Classification/blob/master/src/Key%20Fiducial%20Points%20of%20Heartbeat.jpg).


### Feature Selection
The feature selection process in this project is plays a crucial role in balancing computational efficiency with classification accuracy. This project employs Mutual Information (MI) ranking, a robust method ideally suited for ECG classification due to its ability to quantify the shared information between features and class labels, thereby ranking them according to their relevance. It identifies features that are most informative about the heartbeat classes, thereby enhancing model performance and reducing computational complexity.

* Methodology:  
Using the mutual_info_classif function from Scikit-Learn, we effectively detect both linear and nonlinear relationships between features and the class labels. This approach is particularly effective for complex models like Random Forest, which must interpret intricate patterns present in ECG data.

* Process:  
We construct feature vectors, labels, and source information for each heartbeat. The MI for each feature relative to the class labels is calculated, focusing our model on the most informative aspects for heartbeat classification.

* Application:  
The MI ranking reduces computational demands by ranking and selecting the top features based on their MI scores. This strategic selection balances computational efficiency with the need for classification accuracy.

* Consistency Across Datasets:  
The same feature ranking applied to the training dataset is also used for the testing dataset. This consistency ensures the model is equally efficient and effective across different datasets.

* Efficiency:  
MI ranking effectively identifies the most relevant features for ECG classification, ensuring the model's efficiency. By concentrating on a subset of highly informative features, we manage computational complexity without compromising the model's ability to differentiate various heartbeat types.

* Interpretability:  
The emphasis on interpretability in feature selection is vital. By using MI ranking, we not only enhance model performance but also ensure that the selected features are truly indicative of the underlying cardiac rhythms. This interpretability is crucial for clinical applications where understanding the model's decision-making process is as important as its accuracy.

* Application in Random Forest Model:  
In developing our Random Forest model for arrhythmia classification, the MI ranking of features ensures that we are utilizing the most relevant ECG signal characteristics. This focus is integral to the model's success in accurately classifying arrhythmias, a key aspect of cardiac monitoring and diagnosis.

### Model Training and Testing
* Model Training:  
The project harnesses a Random Forest classifier, renowned for its accuracy and resilience. Utilizing an ensemble of decision trees, the RF model capitalizes on the diversity of its composite learners. This approach diminishes individual biases and reduces variance, ultimately enhancing generalizability.  

  Post-training, the model calculates importance scores for each feature. This process, inherent to the RF algorithm, provides valuable insights into the most influential factors in classification.
  
  A custom function is implemented for cross-validation. The function first identifies unique sources in the training data, representing different subsets. For each source, the training set is divided into two: a subset for training and another for testing. The function yields pairs of indices for training and testing splits, facilitating a thorough cross-validation process where each data part is used for both training and validation. This approach ensures every unique data source is utilized once as the test set, providing a comprehensive and unbiased evaluation of the model's training performance. 

* Model Testing:  
A separate test dataset, unexposed to the model during training, is used to rigorously assess the model's classification accuracy and generalization capabilities. The test dataset undergoes the same preprocessing and feature extraction procedures as the training set, ensuring uniformity in data representation.

  A confusion matrix is constructed to visualize the model's performance across different arrhythmia classes. This matrix is instrumental in pinpointing the model's strengths and potential areas of improvement.

  The model’s efficacy is gauged using key metrics like precision, recall, sensitivity, specificity, positive predictivity, false positive rate, accuracy, and F1 score. These metrics offer a detailed and nuanced understanding of the model's diagnostic abilities.

### Results:
The model is assessed through three experimental setups, each refining the classifier's performance.

* Experiment 1:
  The model utilized the entire set of 141 features, with 101 decision trees.

* Experiment 2:  
  The model was trained with the top 6 features and 40 decision trees, following the same approach of similar successfull studies.

* Experiment 3:
  Hyperparameter tuning is implemented to identify the optimal configuration, focusing on the number of trees and top ranked features.

    The hyperparameter tuning was performed as follows:

        - Created a dictionary specifying the range of hyperparameters to test:
            'n_estimators': Number of trees in the forest, ranging from 1 to 100.
            'max_features': Number of top-ranked features to consider, ranging from 1 to 20.
        
        - Initialized GridSearchCV with the Random Forest classifier, parameter grid,
          and set it to perform 5-fold cross-validation.
        
        - Utilized two evaluation metrics: 'accuracy' and 'f1_weighted',
          focusing on overall accuracy and a balance between precision and recall.

        - Fitted the Random Forest model over the defined hyperparameter grid.
        
        - Iterated over search results, assessing model performance (accuracy and F1 score)
          across different hyperparameter combinations (trees and features).

        - Identified best hyperparameters (search.best_params_) and highest score (search.best_score_).
        
        - Refitted the model on the entire dataset using these optimal settings to maximize performance.

        - Retrieved the best model estimator (search.best_estimator_) for detailed analysis.

    The 10 top-ranked features and 84 decision trees were identified as the most optimal parameters.

    The model performance improved by 4%, showing an overall accuracy of 95.69% and an overall F1 score of 95.37% (weighted average).

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
│   ├── heartbeat_amplitude_features.txt
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
5. Ranking and selecting features by mutual information and constructing feature datasets for training and testing.
6. Training and testing the Random Forest classifier.