import matplotlib.pyplot as plt
import wfdb # Library to read the WFDB formatted data
import numpy as np # Library to work with arrays
import pandas as pd # Library to work with dataframes
import pickle # Library to save and load python objects


# Constants for dataset paths:
PREPROCESSED_PATH = './data/Preprocessed Data 360 Hz'
ORIGINAL_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
RESAMPLED_PATH = './data/Preprocessed Data 256 Hz/'
SEGMENTED_PATH = './data/Segmented Data/'


# Function to plot original ECG signal:
def plot_original_ecg(record_name, start, end, original_path=ORIGINAL_PATH):
    """
    Plot the ECG signal for a specified record from start to end samples.

    Parameters:
    - record_name (str): The name of the record.
    - start (int): The starting sample number to plot.
    - end (int): The ending sample number to plot.
    - data_path (str): The directory path where the data files are located.
    """
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
    """
    Plot the preprocessed ECG signal for a specified record from start to end samples,
    with R-peak annotations marked.

    Parameters:
    - record_name (str): The name of the record.
    - start (int): The starting sample number to plot.
    - end (int): The ending sample number to plot.
    - preprocessed_path (str): The directory path where the preprocessed data files are located.
    - original_path (str): The directory path where the original data files are located.
    """
    # Load the preprocessed ECG signal data
    preprocessed_signal = np.loadtxt(f'{preprocessed_path}/{record_name}_preprocessed_360hz.dat', delimiter=',')

    # Check if the signal is one-dimensional or two-dimensional
    if preprocessed_signal.ndim > 1:
        preprocessed_signal = preprocessed_signal[:, 0]  # Assuming we want the first lead

    # Load annotations from the original directory
    annotations = wfdb.rdann(f'{original_path}/{record_name}', 'atr')

    # Extract R-peak locations and symbols
    r_peak_indices = [idx for idx, symbol in zip(annotations.sample, annotations.symbol) if symbol in BEAT_LABELS]

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


# Define the path to the segmented data
SEGMENTED_PATH = './data/Segmented Data/'

def plot_heartbeat(record_name, beat_number):
    """
    Plot a heartbeat for a specific record name with the content of the dictionary
    (i.e., beat label, fiducial points) by reading from the pickle files directory.

    Parameters:
    - record_name (str): The name of the record.
    - beat_number (int): The index number of the heartbeat to plot.
    """
    # Construct the filename for the pickle file
    pickle_filename = f'{SEGMENTED_PATH}/{record_name}_segmented.pkl'

    # Load the list of heartbeats from the pickle file
    try:
        with open(pickle_filename, 'rb') as file:
            heartbeats = pickle.load(file)
    except FileNotFoundError:
        print(f"The file for record {record_name} was not found.")
        return
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return

    # Check if the beat number is valid
    if beat_number >= len(heartbeats) or beat_number < 0:
        print(f"Beat number {beat_number} is out of range for record {record_name}.")
        return

    # Extract the specific heartbeat data
    heartbeat_data = heartbeats[beat_number]
    signal_segment = heartbeat_data['signal']
    fiducial_points = {k: v for k, v in heartbeat_data.items() if k not in ['signal', 'beat_label']}

    # Plot the ECG segment
    plt.figure(figsize=(10, 4))
    plt.plot(signal_segment, label='ECG Segment', color='blue')

    # Plot fiducial points
    for point_label, point_index in fiducial_points.items():
        if point_index is not None and point_index >= 0:  # Check if the fiducial point was identified
            plt.scatter(point_index, signal_segment[point_index], label=point_label, zorder=3)

    plt.title(f'Heartbeat {beat_number} for Record {record_name} - {heartbeat_data["beat_label"]}')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()






# Function to plot an ECG segment with PQRST annotations:
def plot_ecg_segment_with_pqrst(record, start, end, preprocessed_path=PREPROCESSED_PATH, original_path=ORIGINAL_PATH):
    
    # Load preprocessed signal from the "Preprocessed Data 360 Hz" directory:
    preprocessed_signal = np.loadtxt(f'{preprocessed_path}/{record}_preprocessed_360hz.dat', delimiter=',')
    
    # Load annotations from the "mit-bih-arrhythmia-database-1.0.0" directory:
    annotations = wfdb.rdann(f'{original_path}/{record}', 'atr')
    
    # Extract R-peak locations for the beats of interest
    r_peak_indices = [idx for idx, symbol in zip(annotations.sample, annotations.symbol) if symbol in BEAT_LABELS]
    
    # Calculate the average distance between consecutive R-peaks
    average_r_distance = np.mean(np.diff(r_peak_indices))
    window_size = int(average_r_distance * 0.2)
    
    q_peak_indices = []
    s_peak_indices = []
    p_peak_indices = []
    t_peak_indices = []
    
    for r in r_peak_indices:
        # Q and S peak detection:
        q_window_start = max(0, r - window_size)
        s_window_end = min(len(preprocessed_signal), r + window_size)
        
        q_peak = q_window_start + np.argmin(preprocessed_signal[q_window_start:r, 0])
        q_peak_indices.append(q_peak)
        
        s_peak = r + np.argmin(preprocessed_signal[r:s_window_end, 0])
        s_peak_indices.append(s_peak)
        
        # P peak detection
        p_window_start = max(0, q_peak - 2*window_size)
        p_peak = p_window_start + np.argmax(preprocessed_signal[p_window_start:q_peak, 0])
        p_peak_indices.append(p_peak)
        
        # T peak detection
        t_window_end = min(len(preprocessed_signal), s_peak + 2*window_size)
        t_peak = s_peak + np.argmax(preprocessed_signal[s_peak:t_window_end, 0])
        t_peak_indices.append(t_peak)
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.grid(True)
    
    # Correctly plot the ECG segment against the actual sample indices
    x_values = np.arange(start, end)
    plt.plot(x_values, preprocessed_signal[start:end, 0], label='Lead 1', linewidth=0.8)
    
    for p, q, r, s, t in zip(p_peak_indices, q_peak_indices, r_peak_indices, s_peak_indices, t_peak_indices):
        if start <= p <= end:
            plt.plot(p, preprocessed_signal[p, 0], 'cv', markersize=4, label='P-peak' if 'P-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        if start <= q <= end:
            plt.plot(q, preprocessed_signal[q, 0], 'bs', markersize=4, label='Q-peak' if 'Q-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        if start <= r <= end:
            plt.plot(r, preprocessed_signal[r, 0], 'ro', markersize=4, label='R-peak' if 'R-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        if start <= s <= end:
            plt.plot(s, preprocessed_signal[s, 0], 'g^', markersize=4, label='S-peak' if 'S-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        if start <= t <= end:
            plt.plot(t, preprocessed_signal[t, 0], 'mv', markersize=4, label='T-peak' if 'T-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f"Preprocessed ECG signal for record {record} - Lead 1 with PQRST annotations")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.tight_layout()  # This ensures that everything fits well, especially when moving the legend outside.
    plt.show()




# Plot a resampled ECG segment with adjusted PQRST annotations:
# Since I am resampling the signal, the sample points of the annotations will also change.
# You can calculate the new sample points using a simple ratio based on the resampling rate:
# new_sample_point = original_sample_point × (new_sampling_rate / original_sampling_rate).
def plot_resampled_ecg_segment_with_pqrst(record, start, end, resampled_path=RESAMPLED_PATH):
    
    # Load resampled preprocessed signal from the directory
    resampled_signal = np.loadtxt(f'{resampled_path}/{record}_preprocessed_256hz.dat', delimiter=',')
    
    # Load annotations from the original directory
    annotations = wfdb.rdann(f'./data/mit-bih-arrhythmia-database-1.0.0/{record}', 'atr')
    
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
        # Q peak detection
        q_window_start = max(0, r - window_size)
        q_peak = q_window_start + np.argmin(resampled_signal[q_window_start:r, 0])
        q_peaks.append(q_peak)
        
        # S peak detection
        s_window_end = min(len(resampled_signal), r + window_size)
        s_peak = r + np.argmin(resampled_signal[r:s_window_end, 0])
        s_peaks.append(s_peak)
        
        # P peak detection
        p_window_start = max(0, q_peak - 2*window_size)
        p_peak = p_window_start + np.argmax(resampled_signal[p_window_start:q_peak, 0])
        p_peaks.append(p_peak)
        
        # T peak detection
        t_window_end = min(len(resampled_signal), s_peak + 2*window_size)
        t_peak = s_peak + np.argmax(resampled_signal[s_peak:t_window_end, 0])
        t_peaks.append(t_peak)
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.grid(True)
    
    x_values = np.arange(start, end)
    plt.plot(x_values, resampled_signal[start:end, 0], label='Lead 1', linewidth=0.8)
    
    for p, q, r, s, t in zip(p_peaks, q_peaks, r_peak_indices_resampled, s_peaks, t_peaks):
        if start <= p <= end:
            plt.plot(p, resampled_signal[p, 0], 'cv', markersize=4, label='P-peak' if 'P-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        if start <= q <= end:
            plt.plot(q, resampled_signal[q, 0], 'b^', markersize=4, label='Q-peak' if 'Q-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        if start <= r <= end:
            plt.plot(r, resampled_signal[r, 0], 'ro', markersize=4, label='R-peak' if 'R-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        if start <= s <= end:
            plt.plot(s, resampled_signal[s, 0], 'g^', markersize=4, label='S-peak' if 'S-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        if start <= t <= end:
            plt.plot(t, resampled_signal[t, 0], 'mv', markersize=4, label='T-peak' if 'T-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f"Preprocessed ECG signal for record {record} - Lead 1 with PQRST annotations (Resampled at 256 Hz)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.tight_layout()  # This ensures that everything fits well, especially when moving the legend outside.
    plt.show()




# Plot a resampled ECG segment with adjusted PQRST annotations:
# Since I am resampling the signal, the sample points of the annotations will also change.
# You can calculate the new sample points using a simple ratio based on the resampling rate:
# new_sample_point = original_sample_point × (new_sampling_rate / original_sampling_rate).
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






def plot_resampled_ecg_segment_with_pqrst3(record, start, end, resampled_path=RESAMPLED_PATH):
    
    # Load resampled preprocessed signal from the directory
    resampled_signal = np.loadtxt(f'{resampled_path}/{record}_preprocessed_256hz.dat', delimiter=',')
    
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

    # Adjust window sizes for P and T peak detection
    p_window_size = int(0.12 * 256)  # ~120 ms window
    t_window_size = int(0.15 * 256)  # ~150 ms window

    for idx, r in enumerate(r_peak_indices_resampled):
        # Q peak detection
        q_window_start = max(0, r - window_size)
        q_peak = q_window_start + np.argmin(resampled_signal[q_window_start:r, 0])
        q_peaks.append(q_peak)
        
        # S peak detection
        s_window_end = min(len(resampled_signal), r + window_size)
        s_peak = r + np.argmin(resampled_signal[r:s_window_end, 0])
        s_peaks.append(s_peak)
        
        # P peak detection: Incorporate preceding R peak for better estimation
        if idx > 0:
            r_prev = r_peak_indices_resampled[idx - 1]
            expected_p_position = int((r + r_prev) / 2)
            p_window_start = max(0, expected_p_position - p_window_size)
            p_window_end = expected_p_position + p_window_size
            p_peak = p_window_start + np.argmax(resampled_signal[p_window_start:p_window_end, 0])
        else:
            p_window_start = max(0, r - 3 * window_size)
            p_window_end = q_peak
            derivative = np.diff(resampled_signal[p_window_start:p_window_end, 0])
            zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
            if zero_crossings.size > 0:
                p_peak = p_window_start + zero_crossings[-1]  # Taking the last zero-crossing before Q peak
            else:
                p_peak = p_window_start + np.argmax(resampled_signal[p_window_start:q_peak, 0])
        p_peaks.append(p_peak)
        
        # T peak detection: Incorporate succeeding R peak for better estimation
        if idx < len(r_peak_indices_resampled) - 1:
            r_next = r_peak_indices_resampled[idx + 1]
            expected_t_position = int((r + r_next) / 2)
            t_window_start = max(0, expected_t_position - t_window_size)
            t_window_end = expected_t_position + t_window_size
            t_peak = t_window_start + np.argmax(resampled_signal[t_window_start:t_window_end, 0])
        else:
            t_window_start = s_peak
            t_window_end = min(len(resampled_signal), r + 3 * window_size)
            derivative = np.diff(resampled_signal[t_window_start:t_window_end, 0])
            zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
            if zero_crossings.size > 0:
                t_peak = t_window_start + zero_crossings[0]  # Taking the first zero-crossing after S peak
            else:
                t_peak = s_peak + np.argmax(resampled_signal[s_peak:t_window_end, 0])
        t_peaks.append(t_peak)

    # Plotting
    plt.figure(figsize=(15, 6))
    plt.grid(True)
    
    x_values = np.arange(start, end)
    plt.plot(x_values, resampled_signal[start:end, 0], label='Lead 1', linewidth=0.8)
    
    for p, q, r, s, t in zip(p_peaks, q_peaks, r_peak_indices_resampled, s_peaks, t_peaks):
        if start <= p <= end:
            plt.plot(p, resampled_signal[p, 0], 'cv', markersize=4, label='P-peak' if 'P-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        if start <= q <= end:
            plt.plot(q, resampled_signal[q, 0], 'b^', markersize=4, label='Q-peak' if 'Q-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        if start <= r <= end:
            plt.plot(r, resampled_signal[r, 0], 'ro', markersize=4, label='R-peak' if 'R-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        if start <= s <= end:
            plt.plot(s, resampled_signal[s, 0], 'g^', markersize=4, label='S-peak' if 'S-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        if start <= t <= end:
            plt.plot(t, resampled_signal[t, 0], 'mv', markersize=4, label='T-peak' if 'T-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f"Preprocessed ECG signal for record {record} - Lead 1 with PQRST annotations (Resampled at 256 Hz)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.tight_layout()  # This ensures that everything fits well, especially when moving the legend outside.
    plt.show()

#def plot_resampled_ecg_segment_with_pqrst2(record, start, end, resampled_path=RESAMPLED_PATH):
    
    # Load resampled preprocessed signal from the directory
    resampled_signal = np.loadtxt(f'{resampled_path}/{record}_preprocessed_256hz.dat', delimiter=',')
    
    # Ensure the provided start and end indices are valid
    if start < 0 or end > len(resampled_signal) or start >= end:
        raise ValueError("Invalid start or end indices provided.")
    
    # Load annotations from the original directory
    annotations = wfdb.rdann(f'./data/mit-bih-arrhythmia-database-1.0.0/{record}', 'atr')
    
    # Adjust annotations for the resampled data using linear transformation:
    resampling_ratio = 256 / 360
    adjusted_sample_annotations = [int(x * resampling_ratio) for x in annotations.sample]
    
    # Extract R-peak locations for the beats of interest
    r_peak_indices = [idx for idx, symbol in zip(adjusted_sample_annotations, annotations.symbol) if symbol in BEAT_LABELS]
    
    # Re-identify R-peaks in the resampled signal
    window_size = int(0.12 * 256)  # ~120 ms window around the estimated R-peak
    r_peak_indices_resampled = [np.argmax(resampled_signal[max(0, r - window_size):min(len(resampled_signal), r + window_size), 0]) + max(0, r - window_size) for r in r_peak_indices]
    
    # Detect P, Q, S, T peaks around the R peaks
    p_peaks = []
    q_peaks = []
    s_peaks = []
    t_peaks = []
    
    # Window sizes based on research provided
    q_window_size = int(0.06 * 256)  # Half of QRS duration (max 120 ms)
    p_window_size = int(0.1 * 256)   # P-wave duration (max 100 ms)
    t_window_size = int(0.14 * 256)  # Half of T wave duration (max 140 ms)
    
    for r in r_peak_indices_resampled:
        # Ensure we don't venture beyond the signal boundaries
        q_window_start = max(0, r - q_window_size)
        s_window_end = min(len(resampled_signal), r + q_window_size)
        p_window_start = max(0, r - p_window_size)
        t_window_end = min(len(resampled_signal), r + t_window_size)
        
        # Q peak detection
        q_peak = q_window_start + np.argmin(resampled_signal[q_window_start:r, 0])
        q_peaks.append(q_peak)
        
        # S peak detection
        s_peak = r + np.argmin(resampled_signal[r:s_window_end, 0])
        s_peaks.append(s_peak)
        
        # P peak detection
        p_peak = p_window_start + np.argmax(resampled_signal[p_window_start:q_peak, 0])
        p_peaks.append(p_peak)
        
        # T peak detection
        t_peak = s_peak + np.argmax(resampled_signal[s_peak:t_window_end, 0])
        t_peaks.append(t_peak)
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.grid(True)
    
    x_values = np.arange(start, end)
    plt.plot(x_values, resampled_signal[start:end, 0], label='Lead 1', linewidth=0.8)
    
    for p, q, r, s, t in zip(p_peaks, q_peaks, r_peak_indices_resampled, s_peaks, t_peaks):
        # Plot only if the peak is within the segment to be plotted
        if start <= p <= end:
            plt.plot(p, resampled_signal[p, 0], 'cv', markersize=4, label='P-peak' if 'P-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        if start <= q <= end:
            plt.plot(q, resampled_signal[q, 0], 'b^', markersize=4, label='Q-peak' if 'Q-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        if start <= r <= end:
            plt.plot(r, resampled_signal[r, 0], 'ro', markersize=4, label='R-peak' if 'R-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        if start <= s <= end:
            plt.plot(s, resampled_signal[s, 0], 'g^', markersize=4, label='S-peak' if 'S-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        if start <= t <= end:
            plt.plot(t, resampled_signal[t, 0], 'mv', markersize=4, label='T-peak' if 'T-peak' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f"Preprocessed ECG signal for record {record} - Lead 1 with PQRST annotations (Resampled at 256 Hz)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()