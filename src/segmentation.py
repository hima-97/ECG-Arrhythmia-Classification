import os
import numpy as np
import wfdb
import pandas as pd # Library to work with dataframes
from scipy.signal import find_peaks
import pickle
from ecgtypes import BeatType

# Heartbeat segmentation refers to the process of identifying and isolating individual heartbeats (or cardiac cycles) from a continuous ECG recording.
# The goal of this process is to extract individual heartbeats from the ECG signal for further processing and analysis.

# Steps for heartbeat segmentation:
# 1) Storing Peak Locations: Store the identified P, Q, R, S, and T peaks in a structured format (e.g., a dictionary or dataframe). 
# This will allow to easily access and manipulate these points for further analysis.
# 2) Segmenting Individual Heartbeats: One common approach is to segment the ECG signal into individual heartbeats using the R-peaks as reference points. 
# The segment might start from the midpoint between one R-peak and the previous R-peak and end at the midpoint between the same R-peak and the next R-peak.
# 3) Storing Segmented Heartbeats: Store the segmented heartbeats in a structured format (e.g., a list or dataframe) for further processing or analysis.


# File Structure after this file is run:
# Each pickle file contains a list of dictionaries, with each dictionary representing a single heartbeat and its corresponding fiducial points.
# The keys in each dictionary correspond to the fiducial points, like 'q_peak', 'r_peak', 's_peak', 'p_peak', 'qrs_start', and 'qrs_end'.
# The values are the indices within the signal where these fiducial points occur.



# Constants for dataset paths:
ORIGINAL_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
RESAMPLED_PATH = './data/Preprocessed Data 256 Hz/'
SEGMENTED_PATH = './data/Segmented Data/'
LOGS_PATH = './logs/'
SAMPLE_RATE = 256 # Sample rate in Hz
MS_IN_SECOND = 1000 # Number of milliseconds in one second




# Ensure the logs directory exists
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)


# Function to log segmentation errors
def log_segmentation_error(record, error_message):
    with open(os.path.join(LOGS_PATH, 'segmentation_errors.log'), 'a') as log_file:
        log_file.write(f"Record {record}: {error_message}\n")

# Function to find inflection points using numerical differentiation
def find_inflection_points(signal):
    derivative = np.diff(signal)  # Compute the first derivative
    inflection_points = np.where(np.diff(np.sign(derivative)))[0]  # Find where the sign of the derivative changes
    return inflection_points + 1  # Shift by one due to the nature of np.diff reducing array length


# Function to analyze the segment backward from QRSmax to identify Qpeak, QRSstart, and Rpeak
def analyze_backward(segment, qrs_max_index, qrs_max, segment_start, sample_rate):
    # Initialize variables
    q_peak = r_peak = s_peak = p_peak = 0
    qrs_start = None

    # If QRSmax is positive, then make Rpeak equal to QRSmax
    if qrs_max > 0:
        r_peak = qrs_max_index + segment_start

    # Get the backward segment and inflection points
    backward_segment = segment[:qrs_max_index]
    inflection_points_backward = find_inflection_points(backward_segment)
    derivative_backward = np.diff(backward_segment)  # Compute the derivative of the backward segment

    # Look for QRSmax/2a and QRSmax/4a
    qrs_half_index = np.where(backward_segment < qrs_max / 2)[0]
    qrs_quarter_index = np.where(backward_segment < qrs_max / 4)[0]
    if len(qrs_half_index) > 0:
        qrs_max_half_a = qrs_half_index[0] + segment_start
    if len(qrs_quarter_index) > 0:
        qrs_max_quarter_a = qrs_quarter_index[0] + segment_start

    # Analyze the signal backward from QRSmax
    for idx in reversed(range(qrs_max_index)):
        # Check for the first inflection point
        if idx in inflection_points_backward:
            inflection_value = derivative_backward[idx - 1]  # Use idx-1 because derivative array is one element shorter
            if inflection_value < 0 and r_peak != 0 and q_peak == 0:
                q_peak = idx + segment_start
            elif inflection_value >= 0 and r_peak != 0:
                if qrs_start is None:
                    qrs_start = idx + segment_start
                q_peak = 0
                break  # Since this is the first inflection point, we can break the loop after processing
            elif inflection_value > 0 and r_peak == 0:
                r_peak = idx + segment_start
                s_peak = qrs_max_index + segment_start
                qrs_start = idx + segment_start
                break  # Since this is the first inflection point, we can break the loop after processing

    # If Qpeak is not zero and the signal crosses zero, mark the first non-negative point as QRSstart
    if q_peak != 0:
        zero_crossing_idx = np.where(backward_segment[:q_peak - segment_start] >= 0)[0]
        if len(zero_crossing_idx) > 0:
            qrs_start = zero_crossing_idx[0] + segment_start

    # If the second inflection point is positive or zero and QRSstart has not been found yet, mark it as QRSstart
    if qrs_start is None and len(inflection_points_backward) > 1:
        second_inflection_value = derivative_backward[inflection_points_backward[-2] - 1]
        if second_inflection_value >= 0:
            qrs_start = inflection_points_backward[-2] + segment_start

    return q_peak, qrs_start, r_peak


# Function to analyze the segment forward from QRSmax to identify Speak and QRSend
def analyze_forward(segment, qrs_max_index, qrs_max, segment_start, sample_rate):
    # Initialize Speak and QRSend
    s_peak = None
    qrs_end = None

    # Forward segment and its inflection points
    forward_segment = segment[qrs_max_index:]
    inflection_points_forward = find_inflection_points(forward_segment)
    derivative_forward = np.diff(forward_segment)  # Compute the derivative of the forward segment

    # Look for QRSmax/2b and QRSmax/4b
    qrs_half_index_forward = np.where(forward_segment < qrs_max / 2)[0]
    qrs_quarter_index_forward = np.where(forward_segment < qrs_max / 4)[0]
    if len(qrs_half_index_forward) > 0:
        qrs_max_half_b = qrs_half_index_forward[0] + qrs_max_index + segment_start
    if len(qrs_quarter_index_forward) > 0:
        qrs_max_quarter_b = qrs_quarter_index_forward[0] + qrs_max_index + segment_start

    # Analyze the signal forward from QRSmax
    for idx in range(len(forward_segment)):
        # Check for the first inflection point
        if idx in inflection_points_forward:
            inflection_value = derivative_forward[idx - 1]  # Use idx-1 because derivative array is one element shorter
            if inflection_value < 0 and s_peak is None:
                s_peak = idx + qrs_max_index + segment_start
            elif inflection_value >= 0:
                if qrs_end is None:
                    qrs_end = idx + qrs_max_index + segment_start
                break  # Since this is the first positive inflection point, we can break the loop after processing

    # If Speak is not zero and the signal crosses zero, mark the first non-negative point as QRSend
    if s_peak is not None:
        zero_crossing_idx_forward = np.where(forward_segment[s_peak - qrs_max_index:] >= 0)[0]
        if len(zero_crossing_idx_forward) > 0:
            qrs_end = zero_crossing_idx_forward[0] + s_peak

    # If the second inflection point is positive or zero and QRSend has not been found yet, mark it as QRSend
    if qrs_end is None and len(inflection_points_forward) > 1:
        second_inflection_value_forward = derivative_forward[inflection_points_forward[1] - 1]
        if second_inflection_value_forward >= 0:
            qrs_end = inflection_points_forward[1] + qrs_max_index + segment_start

    return s_peak, qrs_end




# Function to find the P peak:
def find_p_peak(signal, qrs_start_index, sample_rate):
    # Constants for the windows
    window_start_offset = int(0.233 * sample_rate)  # 233 ms before QRS start
    window_end_offset = int(0.067 * sample_rate)    # 67 ms before QRS start

    # Calculate the start and end of the window to search for the P peak
    window_start = max(0, qrs_start_index - window_start_offset)
    window_end = max(0, qrs_start_index - window_end_offset)

    # Segment to analyze for P peak
    p_peak_segment = signal[window_start:window_end]

    # Standard deviation of the signal during the 67 ms preceding the window
    std_preceding_segment = np.std(signal[max(0, window_start - window_end_offset):window_start])

    # Find the maximum value within the P peak segment
    p_peak_value = np.max(p_peak_segment)
    p_peak_index = np.argmax(p_peak_segment) + window_start

    # Check if the maximum value is an inflection point and greater than three times the standard deviation
    inflection_points = find_inflection_points(signal[window_start:qrs_start_index])
    if p_peak_value > 3 * std_preceding_segment:
        # Check if the p_peak_index corresponds to an inflection point
        # Inflection points need to be offset since they are relative to window_start
        inflection_indices = [ip + window_start for ip in inflection_points]
        if p_peak_index in inflection_indices:
            return p_peak_index  # Return the index of the P peak if the conditions are met

    return None  # Return None if no P peak is found based on the criteria





# Function to segment a single heartbeat and identify PQRS points from an ECG signal:
def segment_with_pqrst(signal, r_peak_index, sample_rate, annotation_symbol):
    # Constants for time windows around R peak
    samples_before_r_peak = int(0.373 * sample_rate)  # Samples before R peak based on 373 ms window
    samples_after_r_peak = int(0.267 * sample_rate)   # Samples after R peak based on 267 ms window

    # Extract the signal segment centered around the R peak
    segment_start = max(0, r_peak_index - samples_before_r_peak)
    segment_end = min(len(signal), r_peak_index + samples_after_r_peak)
    segment = signal[segment_start:segment_end]

    # Baseline removal: Subtract the mean of the segment from each sample to remove baseline wander
    segment -= np.mean(segment)

    # Find QRSmax within a 100 ms window around the R peak
    window_100ms = int(0.100 * sample_rate)
    window_start = max(0, r_peak_index - window_100ms - segment_start)
    window_end = min(len(segment), r_peak_index + window_100ms - segment_start)
    qrs_max_index = np.argmax(np.abs(segment[window_start:window_end])) + window_start
    qrs_max = segment[qrs_max_index]

    # Initialize fiducial points
    q_peak, qrs_start, r_peak = analyze_backward(segment, qrs_max_index, qrs_max, segment_start, sample_rate)
    s_peak, qrs_end = analyze_forward(segment, qrs_max_index, qrs_max, segment_start, sample_rate)

    # Identify the P peak
    p_peak = find_p_peak(signal, qrs_start, sample_rate)
    
    # Find the closest annotation to the R peak
    #closest_annotation_index, closest_annotation_symbol = min(adjusted_annotations, key=lambda ann: abs(ann[0] - r_peak_index))

    # Use ecgtypes.py to interpret the symbol into a label
    beat_label = BeatType.new_from_symbol(annotation_symbol)

    # Once all fiducial points are identified, they are stored in a dictionary
    fiducial_points = {
        'beat_label': beat_label,
        'q_peak': q_peak,
        'r_peak': r_peak,
        's_peak': s_peak,
        'p_peak': p_peak,
        'qrs_start': qrs_start,
        'qrs_end': qrs_end,
    }

    return fiducial_points



# Function to segment a single heartbeat of a resampled ECG recording through the segmentation process:
def segment_data():
    # Create 'Segmented Data' directory if it does not exist
    if not os.path.exists(SEGMENTED_PATH):
        os.makedirs(SEGMENTED_PATH)

    # Get the list of files in the resampled data directory
    resampled_files = os.listdir(RESAMPLED_PATH)
    segmented_records = {f.split('_')[0] for f in resampled_files if 'segmented' in f}

    # List all annotation files in the original dataset directory
    annotation_files = [f for f in os.listdir(ORIGINAL_PATH) if f.endswith('.atr')]

    # Extract unique record numbers from annotation files
    records = set(f.split('.')[0] for f in annotation_files)

    for record in records:
        if record in segmented_records:
            print(f"Record {record} has already been segmented. Skipping.")
            continue

        try:
            # Load the resampled signal
            signal = np.loadtxt(os.path.join(RESAMPLED_PATH, record + '_resampled.dat'), delimiter=',')

            # Load annotations from the original directory
            annotations = wfdb.rdann(os.path.join(ORIGINAL_PATH, record), 'atr')

            # Adjust annotations for the resampled data using linear transformation
            resampling_ratio = SAMPLE_RATE / 360
            adjusted_annotations = [(int(ann_sample * resampling_ratio), ann_symbol) for ann_sample, ann_symbol in zip(annotations.sample, annotations.symbol)]

            # Prepare a list to hold segmented heartbeat data
            segmented_heartbeats = []

            # Segment the signal around each R peak and find fiducial points
            for ann_tuple in adjusted_annotations:
                r_peak_index, annotation_symbol = ann_tuple  # Unpack the tuple
                try:
                    heartbeat_segment = segment_with_pqrst(signal, r_peak_index, SAMPLE_RATE, annotation_symbol)
                    segmented_heartbeats.append(heartbeat_segment)
                except Exception as inner_e:
                    log_segmentation_error(record, f"Error segmenting heartbeat at index {r_peak_index}: {inner_e}")

            # Save the segmented data to the new 'Segmented Data' directory using pickle
            with open(os.path.join(SEGMENTED_PATH, record + '_segmented.pkl'), 'wb') as f:
                pickle.dump(segmented_heartbeats, f)

            print(f"Record {record} segmented and saved in '{SEGMENTED_PATH}'.")

        except Exception as e:
            log_segmentation_error(record, f"General error: {e}")



# Function to check if all resampled files have been segmented:
def check_all_files_segmented():
    # List all resampled files in the resampled data directory
    resampled_files = [f for f in os.listdir(RESAMPLED_PATH) if f.endswith('_preprocessed_256hz.dat')]
    
    # Convert the resampled filenames to the expected segmented filenames
    expected_segmented_files = {f.replace('_preprocessed_256hz.dat', '_segmented.pkl') for f in resampled_files}
    
    # List all segmented files in the segmented data directory
    segmented_files = set(os.listdir(SEGMENTED_PATH))
    
    # Find any expected files that are not present in the segmented directory
    missing_files = expected_segmented_files - segmented_files
    
    # Display the results
    if not missing_files:
        print("\nAll resampled files have been successfully segmented and saved in the 'Segmented Data' directory.\n")
    else:
        print("\nThe following files have not been segmented or are missing:")
        for missing in missing_files:
            print(missing)
