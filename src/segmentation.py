import os
import numpy as np
import wfdb
import pandas as pd # Library to work with dataframes
from scipy.signal import find_peaks

# Heartbeat segmentation refers to the process of identifying and isolating individual heartbeats (or cardiac cycles) from a continuous ECG recording.
# The goal of this process is to extract individual heartbeats from the ECG signal for further processing and analysis.

# Steps for heartbeat segmentation:
# 1) Storing Peak Locations: Store the identified P, Q, R, S, and T peaks in a structured format (e.g., a dictionary or dataframe). 
# This will allow to easily access and manipulate these points for further analysis.
# 2) Segmenting Individual Heartbeats: One common approach is to segment the ECG signal into individual heartbeats using the R-peaks as reference points. 
# The segment might start from the midpoint between one R-peak and the previous R-peak and end at the midpoint between the same R-peak and the next R-peak.
# 3) Storing Segmented Heartbeats: Store the segmented heartbeats in a structured format (e.g., a list or dataframe) for further processing or analysis.


# Constants for dataset paths and beat labels:
ORIGINAL_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
RESAMPLED_PATH = './data/Preprocessed Data 256 Hz/'
SAMPLE_RATE = 256 # Sample rate in Hz
MS_IN_SECOND = 1000 # Number of milliseconds in one second



# Function to find inflection points using numerical differentiation
def find_inflection_points(signal):
    derivative = np.diff(signal)  # Compute the first derivative
    inflection_points = np.where(np.diff(np.sign(derivative)))[0]  # Find where the sign of the derivative changes
    return inflection_points



# Function to analyze the segment backward from QRSmax to find Q peak and QRS start
def analyze_backward(segment, qrs_max_index, qrs_max, segment_start):
    backward_segment = segment[:qrs_max_index]
    inflection_points_backward = find_inflection_points(backward_segment)

    q_peak = None
    qrs_start = None

    # Look backward from QRSmax to evaluate the signal
    for idx in inflection_points_backward[::-1]:  # Reverse the array to look backward
        # Check if we're at half the value of QRSmax
        if segment[idx] <= qrs_max / 2 and qrs_start is None:
            qrs_start = idx + segment_start  # Absolute index in the signal

        # Check if we're at a quarter the value of QRSmax
        if segment[idx] <= qrs_max / 4 and q_peak is None:
            q_peak = idx + segment_start  # Absolute index in the signal

        # Determine Q peak and QRS start
        if segment[idx] < 0 and q_peak is None:
            q_peak = idx + segment_start  # Absolute index in the signal

        if segment[idx] >= 0 and qrs_start is None:
            qrs_start = idx + segment_start  # Absolute index in the signal
            if q_peak is None:
                q_peak = 0  # If there's no Q peak, we assume its value is 0

    return q_peak, qrs_start



# Function to analyze the segment forward from QRSmax to identify S peak and QRS end
def analyze_forward(segment, qrs_max_index, qrs_max, segment_start):
    # Initialize the S peak and QRS end points
    s_peak = None
    qrs_end = None

    # Convert the segment to the forward portion starting from QRSmax
    forward_segment = segment[qrs_max_index:]

    # Find inflection points in the forward segment
    inflection_points_forward = find_inflection_points(forward_segment)

    # If there are inflection points, proceed with the analysis
    if inflection_points_forward.size > 0:
        # Analyze the first inflection point forward from QRSmax
        first_inflection = inflection_points_forward[0]

        # Check if the inflection point is negative and if the R peak is not zero (meaning it exists)
        if forward_segment[first_inflection] < 0 and qrs_max > 0:
            s_peak = first_inflection + qrs_max_index  # The S peak is at this inflection point

        # Check for QRS end
        for inflection in inflection_points_forward:
            if forward_segment[inflection] >= 0:
                qrs_end = inflection + qrs_max_index  # The QRS end is at the first non-negative inflection point
                break

    # Return the identified S peak and QRS end
    return s_peak, qrs_end



# Function to find the P peak as described in the paper
def find_p_peak(signal, qrs_start_index, sample_rate):
    # Constants for the windows specified in the paper
    window_start_offset = int((233 / MS_IN_SECOND) * sample_rate)  # 233 ms before QRS start
    window_end_offset = int((67 / MS_IN_SECOND) * sample_rate)    # 67 ms before QRS start

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
    if p_peak_value > 3 * std_preceding_segment and p_peak_index in find_inflection_points(signal):
        return p_peak_index  # Return the index of the P peak if the conditions are met

    return None  # Return None if no P peak is found based on the criteria



# Function to segment a single heartbeat and identify PQRS points from an ECG signal:
def segment_with_pqrst(signal, r_peak_index, sample_rate):
    
    # Constants for time windows around R peak
    samples_before_r_peak = int((373 / MS_IN_SECOND) * sample_rate)  # Samples before R peak based on 373 ms window
    samples_after_r_peak = int((267 / MS_IN_SECOND) * sample_rate)   # Samples after R peak based on 267 ms window

    # Extract the signal segment centered around the R peak
    segment_start = max(0, r_peak_index - samples_before_r_peak)
    segment_end = min(len(signal), r_peak_index + samples_after_r_peak)
    segment = signal[segment_start:segment_end]

    # Baseline removal: Subtract the mean of the segment from each sample to remove baseline wander
    segment -= np.mean(segment)

    # Find QRSmax within a 100 ms window around the R peak
    window_100ms = int((100 / MS_IN_SECOND) * sample_rate)
    window_start = max(0, r_peak_index - window_100ms - segment_start)
    window_end = min(len(segment), r_peak_index + window_100ms - segment_start)
    qrs_max_index = np.argmax(np.abs(segment[window_start:window_end]))  # Index of QRSmax within the window
    qrs_max = segment[qrs_max_index + window_start]  # Value of QRSmax
    qrs_max_index += window_start  # Absolute index of QRSmax within the segment

    # Initialize fiducial points
    fiducial_points = {
        'q_peak': None,
        'r_peak': None,
        's_peak': None,
        'p_peak': None,
        'qrs_start': None,
        'qrs_end': None,
    }

    # If QRSmax is positive, mark it as R peak
    if qrs_max > 0:
        fiducial_points['r_peak'] = qrs_max_index + segment_start  # Absolute index of R peak within the signal

    # The above TODOs should implement the detailed steps provided in the paper's pseudocode.
    # This includes checking the sign of inflection points and whether they cross certain amplitude thresholds.
    # For example, you would look for the point where the signal goes below half of QRSmax (QRSmax/2a and QRSmax/2b).
    # You would also determine the start and end of the QRS complex by locating inflection points where the signal crosses zero.

    # Analyze backward from QRSmax to identify Q peak and QRS start
    fiducial_points['q_peak'], fiducial_points['qrs_start'] = analyze_backward(segment[:qrs_max_index], qrs_max_index, qrs_max, segment_start, sample_rate)

    # Analyze forward from QRSmax to identify S peak and QRS end
    fiducial_points['s_peak'], fiducial_points['qrs_end'] = analyze_forward(segment[qrs_max_index:], qrs_max_index, qrs_max, segment_start + qrs_max_index, sample_rate)


    # Identify the P peak
    if fiducial_points['qrs_start'] is not None:
        fiducial_points['p_peak'] = find_p_peak(signal, fiducial_points['qrs_start'], sample_rate)

    # Once all fiducial points are identified, they are stored in the fiducial_points dictionary
    return fiducial_points