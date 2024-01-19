import matplotlib.pyplot as plt
import wfdb # Library to read the WFDB formatted data
import numpy as np # Library to work with arrays
from src import ecg_types



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
