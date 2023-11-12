import os
import numpy as np
import wfdb
import pandas as pd # Library to work with dataframes
from scipy.signal import find_peaks
import pickle # Library to work with pickle files
from .ecgtypes import BeatType
from .Feature_Extraction_Utils.pqrs_features import ExtractQRS
from .Feature_Extraction_Utils.rr_features import RRFeatures
# from .Feature_Extraction_Utils.rr_features import compute_wavelet_descriptor as wt_features
# from .Feature_Extraction_Utils.rr_features import compute_hos_descriptor as hos_features
# from .Feature_Extraction_Utils.rr_features import compute_my_own_descriptor as mg_features
# from .Feature_Extraction_Utils.rr_features import compute_HBF as hbf_features
# from .Feature_Extraction_Utils.rr_features import compute_Uniform_LBP as lbp_features
import matplotlib.pyplot as plt
import time



# In this file we first segment each ECG signal in individual heartbeats,
# and then we extract features from each heartbeat.

# Heartbeat segmentation refers to the process of identifying and isolating individual heartbeats (or cardiac cycles) from a continuous ECG recording.
# The goal of this process is to extract individual heartbeats from the ECG signal for further processing and analysis.

# File Structure after this file is run:
# Each pickle file contains a list of dictionaries, with each dictionary representing a single heartbeat and its corresponding fiducial points.
# The keys in each dictionary correspond to the beat label and the fiducial points, like 'q_peak', 'r_peak', 's_peak', 'p_peak', 'qrs_start', and 'qrs_end'.
# The values are the indices within the signal where these fiducial points occur.


# Constants for dataset paths:
ORIGINAL_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
RESAMPLED_PATH = './data/Preprocessed Data 256 Hz'
SEGMENTED_PATH = './data/Segmented Data/'
TRAINING_PATH = './data/Training/'
TESTING_PATH = './data/Testing/'
LOGS_PATH = './logs/'

# Constant for sampling rate:
SAMPLE_RATE = 256 # Sample rate in Hz


# Debugging options:
DEBUG = False
DEBUG_RECORD = '207'
DEBUG_BEAT = 0

# Timer for measuring the execution time of feature extractors
timer = time.process_time()


# Ensure the logs directory exists
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)


# Function to log segmentation errors
def log_segmentation_error(record, error_message):
    with open(os.path.join(LOGS_PATH, 'segmentation_errors.log'), 'a') as log_file:
        log_file.write(f"Record {record}: {error_message}\n")


# Function to reset timer:
def tic():
    """
    Reset timer
    """
    global timer
    timer = None


# Function to get elapsed time:
def toc(reset=False):
    """
    Get elapsed time
    """
    global timer
    if timer is None:
        timer = time.perf_counter()
        return 0.0
    t = time.perf_counter() - timer
    if reset:
        tic()
    return t


# Function to extract a segment from ECG signal around the R peak:
# The length of this segment is determined by the window parameter.
def get_qrs_waveform(beatTime, signal, window=((180) * (SAMPLE_RATE / 150))):  # Adjusted window size for 256 Hz
    beatSample = int(beatTime * SAMPLE_RATE)
    qrsWaveform = np.zeros(window)
    k = int(window / 2)
    for n in range(beatSample, -1, -1):
        if k >= 0:
            qrsWaveform[k] = signal[n]
        else:
            break
        k -= 1
    k = int(window / 2 + 1)
    for n in range(beatSample + 1, len(signal)):
        if k < window:
            qrsWaveform[k] = signal[n]
        else:
            break
        k += 1
    return qrsWaveform


# Function to extract features from all labeled heartbeats in the given dataset:
def extract_heartbeat_features(signals, labels, records):
    
    beats = []  # Create an empty list to store extracted features for each heartbeat
    morph_features = ExtractQRS()  # Initialize an object for extracting QRS morphology features
    rr_features = RRFeatures()  # Initialize an object for extracting RR interval features
    
    # Iterate through the records in the dataset
    for recordIndex, recordName in enumerate(records):
        # If in debug mode and the current record is not the one we want to debug, skip it
        if DEBUG and recordName != DEBUG_RECORD:
            continue
        print(f'Processing record {recordName} ({recordIndex + 1} of {len(records)})')
        
        # Iterate through the labels (heartbeats) in the current record
        for labelIndex, label in enumerate(labels[recordIndex]):
            labeledBeatTime = label['time']  # Get the timestamp of the labeled heartbeat
            labeledBeat = label['beat']  # Get the type of the labeled heartbeat (e.g., 'N', 'V', etc.)
            
            # Ignore noise and label artifacts (heartbeat type 'OTHER')
            if labeledBeat == BeatType.OTHER:
                continue
            
            # Start a timer to measure the execution time of feature extraction
            tic()
            
            # Extract RR interval features from the labeled heartbeat
            rr = rr_features(labels[recordIndex], labelIndex)
            rr_time = toc(True)  # Stop the timer and record the execution time
            
            # Extract QRS morphology features from the signal around the labeled R peak
            morph = morph_features(labeledBeatTime, signals[recordIndex])
            morph_time = toc(True)
            
            # Get the QRS waveform around the labeled R peak with the adjusted window size for 256 Hz
            qrsWaveform = get_qrs_waveform(labeledBeatTime, signals[recordIndex], window=int((76) *(256 / 150)))  # Adjust window size based on sampling rate of 256 Hz
            
            # If in debug mode and the current heartbeat is beyond the debug index, plot the signal
            if DEBUG and labelIndex >= DEBUG_BEAT:
                plt.plot(signals[recordIndex])
                plt.title(labeledBeat.symbol())
                plt.show()
            
            # Extract additional features from the QRS waveform using various methods
            # wt = wt_features(qrsWaveform)
            # wt_time = toc(True)
            # hos = hos_features(qrsWaveform)
            # hos_time = toc(True)
            # mg = mg_features(qrsWaveform)
            # mg_time = toc(True)
            # hbf = hbf_features(qrsWaveform)
            # hbf_time = toc(True)
            # lbp = lbp_features(qrsWaveform)
            # lbp_time = toc(True)
            
            # Create a dictionary to store all the extracted features for the current heartbeat
            beat = {
                'beatType': labeledBeat,  # Heartbeat type (e.g., 'N', 'V', etc.)
                'source': recordName,
                'rr': rr,  # RR interval features
                'morph': morph,  # QRS morphology features
                #'wt': wt,  # Wavelet features
                #'hos': hos,  # Higher-order statistics features
                'rr_time': rr_time,  # Execution time for RR interval feature extraction
                'morph_time': morph_time,  # Execution time for QRS morphology feature extraction
                #'wt_time': wt_time,  # Execution time for wavelet feature extraction
                #'hos_time': hos_time,  # Execution time for higher-order statistics feature extraction
                'mg': mg,  # Custom feature
                #'mg_time': mg_time,  # Execution time for custom feature extraction
                #'hbf': hbf,  # Custom feature (replace with an appropriate description)
                #'hbf_time': hbf_time,  # Execution time for custom feature extraction
                #'lbp': lbp,  # Local binary pattern feature
                #'lbp_time': lbp_time  # Execution time for LBP feature extraction
            }
            
            # Append the dictionary of features to the list of beats
            beats.append(beat)
    
    # Return the list of extracted features for all heartbeats in the dataset
    return beats


# Function to save extracted beat features to a pickle file:
def save_beat_features(beats, output_path):
    
    with open(output_path, 'wb') as file:
        # Open a binary file for writing (specified by 'wb')
        # Create a dictionary with a key 'beats' and the list of beat features as its value
        pickle.dump({'beats': beats}, file)


def segment_and_extract_features():
    
    print('Extracting training dataset heartbeats features...')
    # Load the resampled training dataset from a pickle file
    pickle_in = open(TRAINING_PATH + 'training_dataset_signals.pickle', "rb")
    data = pickle.load(pickle_in)
    pickle_in.close()
    
    # Extract beat features from the training dataset
    beats = extract_heartbeat_features(data['signals'], data['labels'], data['records'])
    
    # Print a message indicating that the training set beat features are being saved
    print('Saving train_set_beats.pickle...')
    
    # Save the extracted training dataset beat features to a pickle file
    save_beat_features(beats, SEGMENTED_PATH + 'training_dataset_heartbeats.pickle')







    print('Extracting testing dataset heartbeats features...')
    # Load the resampled testing dataset from a pickle file
    pickle_in = open(TESTING_PATH + 'testing_dataset_signals.pickle', "rb")
    data = pickle.load(pickle_in)
    pickle_in.close()
    
    # Extract beat features from the testing dataset
    beats = extract_heartbeat_features(data['signals'], data['labels'], data['records'])
    
    # Print a message indicating that the test set beat features are being saved
    print('Saving test_set_beats.pickle...')
    
    # Save the extracted testind dataset beat features to a pickle file
    save_beat_features(beats, SEGMENTED_PATH + 'testing_dataset_heartbeats.pickle')