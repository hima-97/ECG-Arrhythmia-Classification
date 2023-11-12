import os
import numpy as np
import wfdb
import pandas as pd # Library to work with dataframes
from scipy.signal import find_peaks
import pickle # Library to work with pickle files
from .ecgtypes import BeatType
from .Feature_Extraction_Utils.pqrs_features import ExtractQRS
import matplotlib.pyplot as plt




# Heartbeat segmentation refers to the process of identifying and isolating individual heartbeats (or cardiac cycles) from a continuous ECG recording.
# The goal of this process is to extract individual heartbeats from the ECG signal for further processing and analysis.

# File Structure after this file is run:
# Each pickle file contains a list of dictionaries, with each dictionary representing a single heartbeat and its corresponding fiducial points.
# The keys in each dictionary correspond to the beat label and the fiducial points, like 'q_peak', 'r_peak', 's_peak', 'p_peak', 'qrs_start', and 'qrs_end'.
# The values are the indices within the signal where these fiducial points occur.


# Constants for dataset paths:
ORIGINAL_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
RESAMPLED_PATH = './data/Preprocessed Data 256 Hz/'
SEGMENTED_PATH = './data/Segmented Data/'
LOGS_PATH = './logs/'
SAMPLE_RATE = 256 # Sample rate in Hz

# Debugging options:
DEBUG = False
DEBUG_RECORD = '207'
DEBUG_BEAT = 0


# Ensure the logs directory exists
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)

# Function to log segmentation errors
def log_segmentation_error(record, error_message):
    with open(os.path.join(LOGS_PATH, 'segmentation_errors.log'), 'a') as log_file:
        log_file.write(f"Record {record}: {error_message}\n")
        


# Function to extract a segment from ECG signal around the R peak:
# The length of this segment is determined by the window parameter.
def get_qrs_waveform(beatTime, signal, window=int((180/150)*(256))): # Adjust default window size based on sampling rate of 256 Hz
    
    beatSample = int(beatTime * 256)  # Adjust for 256 Hz sampling rate
    
    # Check if beatSample index is within the bounds of the signal array
    if beatSample >= len(signal) or beatSample < 0:
        print(f"Error: beatSample index {beatSample} out of bounds for signal length {len(signal)}")
        return np.zeros(window)  # Return a default waveform
    
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




# Function to extract fiducial points (P, Q, S, T waves) from a QRS waveform (i.e. individual heartbeat):
def extract_fiducial_points(qrs_waveform, qrs_extractor, beatTime):
    
    qrs_features = qrs_extractor(beatTime, qrs_waveform)

    # Extract the Q, R, S peaks' indices or positions
    q_index = qrs_features.get('Qpeak', None)  # Adjust key names based on actual keys in the dictionary
    r_index = qrs_features.get('Rpeak', None)
    s_index = qrs_features.get('Speak', None)

    # Assert that the returned values are of the expected type (int or float)
    assert isinstance(q_index, (int, float)), "Qpeak should be a single numeric value"
    assert isinstance(r_index, (int, float)), "Rpeak should be a single numeric value"
    assert isinstance(s_index, (int, float)), "Speak should be a single numeric value"
    
    # Return a dictionary with fiducial point indices
    return {
        'q_peak': q_index,
        'r_peak': r_index,
        's_peak': s_index,
        #'p_peak': p_index,  # Index of P peak (if extracted)
        #'t_peak': t_index,  # Index of T peak (if extracted)
    }





# Function to segment a single heartbeat of a resampled ECG recording through the segmentation process:
def segment_data():
    # Create 'Segmented Data' directory if it does not exist
    if not os.path.exists(SEGMENTED_PATH):
        os.makedirs(SEGMENTED_PATH)

    # Get the list of segmented files to check if they have already been processed
    segmented_records = {f.split('_')[0] for f in os.listdir(SEGMENTED_PATH) if f.endswith('_segmented.pkl')}

    # List all annotation files in the original dataset directory
    annotation_files = [f for f in os.listdir(ORIGINAL_PATH) if f.endswith('.atr')]

    # Extract unique record numbers from annotation files
    # records = set(f.split('.')[0] for f in annotation_files)
    records = {'100'}

    # Create an instance of ExtractQRS class
    qrs_extractor = ExtractQRS()

    for record in records:
        if record in segmented_records:
            print(f"Record {record} has already been segmented. Skipping.")
            continue

        if DEBUG and record != DEBUG_RECORD:
            continue

        try:
            # Load the resampled signal
            signal = np.loadtxt(os.path.join(RESAMPLED_PATH, record + '_preprocessed_256hz.dat'), delimiter=',')

            # Check the length of the loaded signal
            print(f"Length of signal for record {record}: {len(signal)}")

            # Check if the signal has multiple leads and extract the first lead
            if signal.ndim > 1:
                signal = signal[:, 0]

            # Load annotations from the original directory
            annotations = wfdb.rdann(os.path.join(ORIGINAL_PATH, record), 'atr')

            # Adjust annotations for the resampled data using linear transformation
            resampling_ratio = SAMPLE_RATE / 360
            adjusted_annotations = [(int(ann_sample * resampling_ratio), ann_symbol) for ann_sample, ann_symbol in
                                    zip(annotations.sample, annotations.symbol)]

            segmented_heartbeats = []

            for ann_index, (beatTime, ann_symbol) in enumerate(adjusted_annotations):
                print(f"beatTime: {beatTime}, ann_symbol: {ann_symbol}")
                beatSample = int(beatTime * 256)  # Add this line
                print(f"beatSample: {beatSample}")  # Add this line
                if ann_symbol == BeatType.OTHER:
                    continue
                try:
                    # Calculate window size
                    pre_samples = int(128 * (256 / 150))  # Adjusted for 256 Hz
                    post_samples = int(40 * (256 / 150))  # Adjusted for 256 Hz

                    # Print values of pre_samples and post_samples
                    print(f"pre_samples: {pre_samples}, post_samples: {post_samples}")

                    # Check if beatTime - pre_samples is non-negative
                    if beatTime - pre_samples < 0:
                        print(f"Warning: Negative index for beatTime - pre_samples in record {record}")

                    # Check if beatTime + post_samples is within the signal length
                    if beatTime + post_samples >= len(signal):
                        print(f"Warning: Out-of-bounds index for beatTime + post_samples in record {record}")

                    heartbeat_segment = get_qrs_waveform(beatTime, signal,
                                                        window=int(76 * (256 / 150)))  # Adjust window size based on sampling rate of 256 Hz

                    fiducial_points = extract_fiducial_points(heartbeat_segment, ExtractQRS(), beatTime)

                    segmented_heartbeat = {
                        'heartbeat_segment': heartbeat_segment,
                        'fiducial_points': fiducial_points
                    }
                    segmented_heartbeats.append(segmented_heartbeat)

                    if DEBUG and ann_index >= DEBUG_BEAT:
                        plt.plot(heartbeat_segment)
                        plt.title(ann_symbol)
                        plt.show()

                except Exception as inner_e:
                    log_segmentation_error(record, f"Error segmenting heartbeat at index {ann_index}: {inner_e}")

            with open(os.path.join(SEGMENTED_PATH, record + '_segmented.pkl'), 'wb') as f:
                pickle.dump(segmented_heartbeats, f)

            print(f"Record {record} segmented and saved in '{SEGMENTED_PATH}'.")

        except Exception as e:
            log_segmentation_error(record, f"General error: {e}")
