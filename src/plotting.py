import os
import matplotlib.pyplot as plt
import wfdb # Library to read the WFDB formatted data
import numpy as np # Library to work with arrays
import pandas as pd # Library to work with dataframes
import pickle # Library to save and load python objects
from src import feature_extraction
from src import ecgtypes



# Constants for dataset paths:
ORIGINAL_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
PREPROCESSED_PATH = './data/Preprocessed Data 360 Hz'
RESAMPLED_PATH = './data/Preprocessed Data 256 Hz'
HEARTBEATS_PATH = './data/Heartbeats Data/'
TRAINING_PATH = './data/Training/'
TESTING_PATH = './data/Testing/'
LOGS_PATH = './logs/'

# Constant for sampling rate:
SAMPLE_RATE = 256 # Sample rate in Hz


# Function to plot original ECG signal:
def plot_original_ecg(record_name, start, end, original_path=ORIGINAL_PATH):
    
    # Load the specified record
    record = wfdb.rdrecord(f'{original_path}/{record_name}')
    
    # Extract the ECG signal data
    ecg_signal = record.p_signal[start:end, 0]  # Assuming we are interested in the first lead

    # Plotting
    plt.figure(figsize=(15, 6))
    plt.grid(True)
    plt.plot(range(start, end), ecg_signal)
    plt.title(f'Original ECG Signal for Record {record_name} - Lead 1')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()  # This ensures that everything fits well, especially when moving the legend outside.
    plt.show()




# Function to plot preprocessed ECG signal with R-peak annotations:
def plot_preprocessed_ecg_with_rpeaks(record_name, start, end, preprocessed_path=PREPROCESSED_PATH, original_path=ORIGINAL_PATH):
    
    # Load the preprocessed ECG signal data
    preprocessed_signal = np.loadtxt(f'{preprocessed_path}/{record_name}_preprocessed_360hz.dat', delimiter=',')

    # Check if the signal is one-dimensional or two-dimensional
    if preprocessed_signal.ndim > 1:
        preprocessed_signal = preprocessed_signal[:, 0]  # Assuming we want the first lead

    # Load annotations from the original directory
    annotations = wfdb.rdann(f'{original_path}/{record_name}', 'atr')

    # Extract R-peak locations and symbols
    r_peak_indices = [idx for idx, symbol in zip(annotations.sample, annotations.symbol) if symbol in ecgtypes.BeatType]

    # Plotting
    plt.figure(figsize=(15, 6))
    plt.grid(True)
    plt.plot(range(start, end), preprocessed_signal[start:end], label='Lead 1', linestyle='-')

    # Plot R-peaks using scatter to avoid connecting lines
    for r_index in r_peak_indices:
        if start <= r_index < end:
            plt.scatter(r_index - start, preprocessed_signal[r_index - start], color='red', marker='o', s=20, zorder=3, label='R-peak' if 'R-peak' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title(f'Preprocessed ECG Signal for Record {record_name} - Lead 1 with R-peaks')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()  # This ensures that everything fits well, especially when moving the legend outside.
    plt.show()
    



# Plot a resampled ECG segment with adjusted PQRST annotations:
# Since I am resampling the signal, the sample points of the annotations will also change.
# You can calculate the new sample points using a simple ratio based on the resampling rate:
# new_sample_point = original_sample_point Ã— (new_sampling_rate / original_sampling_rate).
def plot_resampled_ecg_segment_with_pqrst4(record, start, end, resampled_path=RESAMPLED_PATH):
    
    # Load resampled preprocessed signal from the directory
    resampled_signal = np.loadtxt(f'{resampled_path}/{record}_preprocessed_256hz.dat', delimiter=',')
    
    # Check if the provided segment (start to end) is valid and within the bounds of the data
    if start < 0 or end > len(resampled_signal) or start >= end:
        print(f"Invalid segment provided: start={start}, end={end}. Ensure 0 <= start < end <= {len(resampled_signal)}")
        return
    
    # Load annotations from the original directory
    annotations = wfdb.rdann(f'{ORIGINAL_PATH}/{record}', 'atr')
    
    # Adjust annotations for the resampled data using linear transformation
    resampling_ratio = 256 / 360
    adjusted_sample_annotations = [int(x * resampling_ratio) for x in annotations.sample]
    
    # Extract R-peak locations for the beats of interest
    r_peak_indices = [idx for idx, symbol in zip(adjusted_sample_annotations, annotations.symbol) if symbol in BEAT_LABELS]
    
    # Re-identify R-peaks in the resampled signal
    window_size = int(0.1 * 256)  # ~100 ms window around the estimated R-peak
    r_peak_indices_resampled = [np.argmax(resampled_signal[max(0, r - window_size):min(len(resampled_signal), r + window_size), 0]) + max(0, r - window_size) for r in r_peak_indices]

    # Detect P, Q, S, T peaks around the R peaks
    p_peaks = []
    q_peaks = []
    s_peaks = []
    t_peaks = []
    
    for r in r_peak_indices_resampled:
        
        # P peak detection
        p_window_start = max(0, r - 26) # P wave is searched in a window from 26 to 15 samples (101 - 59 ms) before the R-peak. 
        p_window_end = r - 15
        if p_window_start < p_window_end:
            p_peak = p_window_start + np.argmax(resampled_signal[p_window_start:p_window_end, 0])
            p_peaks.append(p_peak)
        
        # Q peak detection
        q_window_start = max(0, r - 15) # Q peak is searched 15 samples (59 ms) before the R-peak. 
        q_peak = q_window_start + np.argmin(resampled_signal[q_window_start:r, 0])
        q_peaks.append(q_peak)
        
        # S peak detection
        s_window_end = min(len(resampled_signal), r + 15) # S peak is searched 15 samples (59 ms) after the R-peak. 
        s_peak = r + np.argmin(resampled_signal[r:s_window_end, 0])
        s_peaks.append(s_peak)
        
        # T peak detection
        t_window_start = r + 41 # T wave is searched in a window 41 to 82 samples (160 - 320 ms) after the R-peak. 
        t_window_end = min(len(resampled_signal), r + 82)
        if t_window_start < t_window_end:
            t_peak = t_window_start + np.argmax(resampled_signal[t_window_start:t_window_end, 0])
            t_peaks.append(t_peak)

    # Plotting
    plt.figure(figsize=(15, 6))
    plt.grid(True)
    
    x_values = np.arange(start, end)
    plt.plot(x_values, resampled_signal[start:end, 0], label='Lead 1', linewidth=0.8)
    
    for p, q, r, s, t in zip(p_peaks, q_peaks, r_peak_indices_resampled, s_peaks, t_peaks):
        # if start <= p <= end:
        #     plt.plot(p, resampled_signal[p, 0], 'cv', markersize=4, label='P-peak' if 'P-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        # if start <= q <= end:
        #     plt.plot(q, resampled_signal[q, 0], 'b^', markersize=4, label='Q-peak' if 'Q-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        if start <= r <= end:
            plt.plot(r, resampled_signal[r, 0], 'ro', markersize=4, label='R-peak' if 'R-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        # if start <= s <= end:
        #     plt.plot(s, resampled_signal[s, 0], 'g^', markersize=4, label='S-peak' if 'S-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        # if start <= t <= end:
        #     plt.plot(t, resampled_signal[t, 0], 'mv', markersize=4, label='T-peak' if 'T-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f"Preprocessed ECG signal for record {record} - Lead 1 with R-peaks annotations (Resampled at 256 Hz)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.tight_layout()  # This ensures that everything fits well, especially when moving the legend outside.
    plt.show()





# Function to plot a heartbeat from training dataset with extracted features:
# function to accept both the record number and the number of heartbeats to plot as parameters. 
# The function will then find the specified number of heartbeats for the given record 
# and plot them with their PQRS features overlaid on the corresponding segments of the resampled ECG signal.
def plot_training_heartbeat_with_features(record_name, heartbeats_number):
    # Load the heartbeat features data
    with open(os.path.join(HEARTBEATS_PATH, 'testing_dataset_heartbeats.pickle'), 'rb') as file:
        feature_data = pickle.load(file)['beats']

    # Load resampled signal
    resampled_signal = np.loadtxt(f'{RESAMPLED_PATH}/{record_name}_preprocessed_256hz.dat', delimiter=',')

    # Filter out the beats for the specified record
    record_beats = [beat for beat in feature_data if beat['source'] == record_name]

    # Check if the specified number of heartbeats is available
    if heartbeats_number > len(record_beats):
        print(f"Only {len(record_beats)} heartbeats available for record {record_name}.")
        heartbeats_number = len(record_beats)

    # Plot each heartbeat
    for i in range(heartbeats_number):
        beat = record_beats[i]
        # Convert labeledBeatTime to sample index if necessary
        labeledBeatTime = beat['rr']['RR0'] * SAMPLE_RATE  # If labeledBeatTime is in seconds
        # Extract the ECG segment for the heartbeat
        signal_segment, startIndex = feature_extraction.get_qrs_waveform(labeledBeatTime, resampled_signal)

        plt.figure(figsize=(10, 4))

        # Plotting the ECG signal segment
        plt.plot(signal_segment, label='ECG Signal')

        # Plotting fiducial points
        for key in ['Ppeak', 'QRSstart', 'Qpeak', 'Rpeak', 'Speak', 'QRSend']:
            adjusted_index = beat['morph'][key] - startIndex  # Adjust index for segment
            plt.plot(adjusted_index, signal_segment[adjusted_index], 'o', label=key)

        # Adding labels and title
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.title(f'Heartbeat {i+1} of Record {record_name}')
        plt.legend()
        plt.show()
