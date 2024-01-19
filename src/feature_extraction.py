import os
import numpy as np
import pandas as pd # Library to work with dataframes
from scipy.signal import find_peaks
import pickle # Library to work with pickle files
import time
import matplotlib.pyplot as plt
from .ecgtypes import BeatType
from .Feature_Extraction_Utils.pqrs_features import ExtractPQRS
from .Feature_Extraction_Utils.rr_features import RRFeatures
from .Feature_Extraction_Utils.advanced_features import compute_wavelet_descriptor as wt_features
from .Feature_Extraction_Utils.advanced_features import compute_hos_descriptor as hos_features
from .Feature_Extraction_Utils.advanced_features import compute_HBF as hbf_features
from .Feature_Extraction_Utils.advanced_features import compute_Uniform_LBP as lbp_features
from .Feature_Extraction_Utils.advanced_features import compute_morphological_features as m_features



# In this file we first segment each ECG signal in individual heartbeats and then we extract features from each heartbeat.

# Heartbeat segmentation refers to the process of identifying and isolating individual heartbeats (or cardiac cycles) from a continuous ECG recording.
# The goal of this process is to extract individual heartbeats from the ECG signal for further processing and analysis.

# File Structure after this file is run:
# Each pickle file contains a list of dictionaries, with each dictionary representing a single heartbeat and its corresponding fiducial points.


# Debugging options:
DEBUG = False
DEBUG_RECORD = '207'
DEBUG_BEAT = 0

# Timer for measuring the execution time of feature extractors:
timer = time.process_time()

# Constant for logs files path:
LOGS_PATH = './logs/'




# Function to log segmentation errors:
def log_feature_extraction_error(record, error_message):
    with open(os.path.join(LOGS_PATH, 'feature_extraction_errors.log'), 'a') as log_file:
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
def get_qrs_waveform(beatTime, signal, window):
    
    # Calculate the sample index of the R peak:                                                                   
    beatSample = int(beatTime * 256)
    
    # Initialize an empty array to store the QRS waveform:
    # The size of the array is determined by the window parameter.
    qrsWaveform = np.zeros(window)
    
    # `k` is index of the middle of the array to store the QRS waveform:
    k = int(window / 2)
    
    # Iterate through the signal in both directions from the R peak:
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
def extract_heartbeat_features(signals, labels, records, debug=False):
    
    beats = []  # Create an empty list to store extracted features for each heartbeat
    qrs_morph_features = ExtractPQRS(debug)  # Initialize an object for extracting QRS morphology features
    rr_features = RRFeatures()  # Initialize an object for extracting RR interval features
    
    # Iterate through the records in the dataset:
    for recordIndex, recordName in enumerate(records):
        
        # If in debug mode and the current record is not the one we want to debug, skip it:
        if DEBUG and recordName != DEBUG_RECORD:
            continue
        print(f'Processing record {recordName} ({recordIndex + 1} of {len(records)})')
        
        # Iterate through the labels (heartbeats) in the current record:
        for labelIndex, label in enumerate(labels[recordIndex]):
            
            # Get the timestamp of the labeled heartbeat:
            # This parameter represents the time (in seconds) at which the R peak of a heartbeat occurs. It's used to locate the R peak within the ECG signal.
            labeledBeatTime = label['time']
            
            # Get the type of the labeled heartbeat (e.g., 'N', 'V', etc.):
            labeledBeat = label['beat']
            
            # Ignore noise and label artifacts (heartbeat type 'OTHER'):
            if labeledBeat == BeatType.OTHER:
                continue
            
            # Start a timer to measure the execution time of feature extraction:
            tic()
            
            # Extract RR interval features from the labeled heartbeat:
            rr = rr_features(labels[recordIndex], labelIndex)
            rr_time = toc(True)  # Stop the timer and record the execution time
            
            # Extract QRS morphology features from the signal around the labeled R peak:
            qrs_morph = qrs_morph_features(labeledBeatTime, signals[recordIndex])
            qrs_morph_time = toc(True)
            
            # Get the QRS waveform around the labeled R peak with the adjusted window size for 256 Hz:
            # At each beat location, a segment of 640 ms of signal (164 samples for 256 Hz) is considered,
            qrsWaveform = get_qrs_waveform(labeledBeatTime, signals[recordIndex], window=164)
            
            # If in debug mode and the current heartbeat is beyond the debug index, plot the signal:
            if DEBUG and labelIndex >= DEBUG_BEAT:
                plt.plot(signals[recordIndex])
                plt.title(labeledBeat.symbol())
                plt.show()
                pass
            
            # Extracting advanced features:
            wt = wt_features(qrsWaveform)
            wt_time = toc(True)
            hos = hos_features(qrsWaveform)
            hos_time = toc(True)
            #mg = m_features(qrsWaveform)
            #mg_time = toc(True)
            hbf = hbf_features(qrsWaveform)
            hbf_time = toc(True)
            lbp = lbp_features(qrsWaveform)
            lbp_time = toc(True)
            
            # Create a dictionary to store all the extracted features for the current heartbeat:
            beat = {
                'beatType': labeledBeat, # Heartbeat type (e.g., 'N', 'V', etc.)
                'source': recordName,
                'rr': rr, # RR interval features
                'morph': qrs_morph, # QRS morphology features
                'wt': wt, # Wavelet features
                'hos': hos, # HOS features
                'rr_time': rr_time, # Execution time for RR interval feature extraction
                'morph_time': qrs_morph_time, # Execution time for QRS morphology feature extraction
                'wt_time': wt_time,
                'hos_time': hos_time,
                #'mg': mg,
                #'mg_time': mg_time,
                'hbf': hbf,
                'hbf_time': hbf_time,
                'lbp': lbp,
                'lbp_time': lbp_time
            }
            
            # Append the dictionary of features to the list of beats:
            beats.append(beat)
    
    # Return the list of extracted features for all heartbeats in the dataset:
    return beats




# Function to save extracted beat features to a pickle file:
def save_beat_features(beats, output_path):
    
    # Extract the directory from the output path
    directory = os.path.dirname(output_path)
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Open a binary file for writing (specified by 'wb')
    # Create a dictionary with a key 'beats' and the list of beat features as its value
    with open(output_path, 'wb') as file:
        pickle.dump({'beats': beats}, file)




# Function to orchestrate the segmentation and feature extraction process:
def segment_and_extract_features(training_dataset_path, testing_dataset_path, heartbeats_path, debug=False):
    
    print('\nExtracting training dataset heartbeats features...\n')
    # Load the resampled training dataset from a pickle file
    pickle_in = open(training_dataset_path + 'training_dataset_signals.pickle', "rb")
    data = pickle.load(pickle_in)
    pickle_in.close()
    
    # Extract beat features from the training dataset
    beats = extract_heartbeat_features(data['signals'], data['labels'], data['records'], debug)
    
    # Print a message indicating that the training set beat features are being saved
    print('\nSaving training_dataset_heartbeats.pickle...')
    
    # Save the extracted training dataset beat features to a pickle file
    save_beat_features(beats, heartbeats_path + 'training_dataset_heartbeats.pickle')




    print('\nExtracting testing dataset heartbeats features...\n')
    # Load the resampled testing dataset from a pickle file
    pickle_in = open(testing_dataset_path + 'testing_dataset_signals.pickle', "rb")
    data = pickle.load(pickle_in)
    pickle_in.close()
    
    # Extract beat features from the testing dataset
    beats = extract_heartbeat_features(data['signals'], data['labels'], data['records'], debug)
    
    # Print a message indicating that the test set beat features are being saved
    print('\nSaving testing_dataset_heartbeats.pickle...')
    
    # Save the extracted testind dataset beat features to a pickle file
    save_beat_features(beats, heartbeats_path + 'testing_dataset_heartbeats.pickle')




# Function to verify segmented heartbeats and the extracted features:
# This function will load the pickle file, iterate through the first few heartbeats (up to the specified num_beats_to_inspect), 
# and print various details about each heartbeat, including its type, source, RR interval features, and morphological features.
def verify_heartbeats_and_features(heartbeats_path, record_name, num_beats_to_inspect=5):
    
    try:
        # Load the data from the pickle file
        with open(os.path.join(heartbeats_path, 'training_dataset_heartbeats.pickle'), 'rb') as file:
            data = pickle.load(file)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return
    
    heartbeats = data['beats']
    
    print(f"\nTotal heartbeats in the file: {len(heartbeats)}")
    
    filtered_heartbeats = [beat for beat in heartbeats if beat['source'] == record_name]

    if not filtered_heartbeats:
        print(f"Error: Record '{record_name}' not found in the dataset.")
        return

    # Print the number of heartbeats extracted for the specified record:
    print(f"\nTotal heartbeats extracted for record '{record_name}': {len(filtered_heartbeats)}")
    
    # Inspect first few heartbeats for of the record, until the specified `num_beats_to_inspect`` is reached:
    for i, beat in enumerate(filtered_heartbeats[:num_beats_to_inspect]):
        print(f"\nHeartbeat {i}:")
        print(f"Keys: {beat.keys()}")
        
        if 'beatType' in beat:
            print(f"Type: {beat['beatType']}")
        
        if 'rr' in beat:
            print("RR Features:", beat['rr'])
        else:
            print("RR Features key is missing")

        if 'morph' in beat:
            print("Morphological Features:", beat['morph'])
            if 'morph' in beat:
                print(f"Morph keys: {beat['morph'].keys()}")
        else:
            print("Morphological Features key is missing")