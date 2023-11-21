import os
import logging # To log errors
import traceback # To print detailed error information
import wfdb # Library to read the WFDB formatted data
import numpy as np # Library to work with arrays
from scipy.signal import BadCoefficients, butter, filtfilt # Library to apply filters to the signal


# Set up logging:
# if you're processing many records in batch mode, you can log these errors to a file for easier post-processing. 
# Any errors during preprocessing will be logged to the file `preprocessing_errors.log`, inside the `logs` directory.
def log_preprocessing_error(error_message):
    with open('./logs/preprocessing_errors.log', 'a') as log_file:
        log_file.write(f"{error_message}\n")




# 1) Baseline Wander Removal:
# Baseline wander is a low-frequency noise introduced in the ECG signal. 
# Common sources include respiration or movement artifacts.
# You can use a high-pass Butterworth filter to remove baseline wander and DC components.
# Note: if the ECG data has significant baseline wander with frequency content around 1 Hz, 
# a lowcut of 0.5 Hz might be preferable as it will be less aggressive in attenuating those frequencies.
# On the other hand, if you want a sharper attenuation of the low-frequency components, a lowcut of 1 Hz might be better.

# Padding is a technique used to avoid edge effects when filtering a signal. 
# Edge effects occur when the filter is applied to the signal near the beginning or end of the signal, 
# where there are not enough samples to apply the filter properly.
# The length of the input vector must be greater than `padlen` to avoid a `ValueError` exception. 
# Padding involves adding extra samples to the beginning and end of the signal to ensure that the filter is applied correctly.

# This function applies the high-pass filter to the signal after padding it at the beginning and end:
# By extending the signal at both the beginning and the end, the filter can be applied more uniformly, and the edge effects are pushed into the padded regions. 
# After filtering, the padded regions are then removed, leaving a signal that has been more evenly filtered throughout its length.
def highpass_filter_with_padding(signal, lowcut=1, fs=360, order=1, pad_length=100):
    
    try:
        # Create an empty array to store the filtered signals for each lead
        filtered_signals = np.zeros_like(signal)
    
        # Loop through each lead (column)
        for i in range(signal.shape[1]):
            lead_signal = signal[:, i]
            
            # Pad the signal for the current lead
            padded_signal = np.pad(lead_signal, (pad_length, pad_length), 'edge')
            
            # High-pass filter the padded signal for the current lead
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            b, a = butter(order, low, btype='high')
            filtered_padded_signal = filtfilt(b, a, padded_signal)
            
            # Remove the padding and store the filtered signal for the current lead
            filtered_signals[:, i] = filtered_padded_signal[pad_length:-pad_length]
        
        return filtered_signals
    
    except ValueError as ve:
        log_preprocessing_error(f"ValueError encountered in highpass_filter_with_padding: {ve}")
        raise
    except np.core._exceptions._UFuncOutputCastingError as np_err:
        log_preprocessing_error(f"NumPy UFunc Output Casting Error in highpass_filter_with_padding: {np_err}")
        raise
    

# Note:
# In summary, the issue seems to originate during the high-pass filtering stage for Lead 2. 
# The filter might be overly attenuating the signal, rendering it almost flat. 
# This could be due to the characteristics of Lead 2 in this specific dataset or the high-pass filter parameters.

# To address this, you can try the following:
# Investigate the frequency content of Lead 2 before filtering to ensure that the high-pass filter isn't overly aggressive.
# Experiment with different filter parameters or designs to retain more of Lead 2's original content.
# Consider the possibility that Lead 2 in this specific dataset might inherently have a low frequency content.




# 2) Noise Reduction:
# Apply a band-reject Butterworth filter to the signal to reduce the 60 Hz AC interference.
# The band-reject filter is also known as a notch filter.

# This function applies the band-reject filter to the signal after padding it at the beginning and end:
# By extending the signal at both the beginning and the end, the filter can be applied more uniformly, and the edge effects are pushed into the padded regions.
# After filtering, the padded regions are then removed, leaving a signal that has been more evenly filtered throughout its length.
def bandreject_filter_with_padding(signal, lowcut=59, highcut=61, fs=360, order=1, pad_length=100):
    
    try:
        # Create an empty array to store the filtered signals for each lead
        filtered_signals = np.zeros_like(signal)
        
        # Loop through each lead (column)
        for i in range(signal.shape[1]):
            lead_signal = signal[:, i]
            
            # Pad the signal for the current lead
            padded_signal = np.pad(lead_signal, (pad_length, pad_length), 'edge')
            
            # Apply the band-reject filter to the padded signal for the current lead
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='bandstop')
            filtered_padded_signal = filtfilt(b, a, padded_signal)
            
            # Remove the padding and store the filtered signal for the current lead
            filtered_signals[:, i] = filtered_padded_signal[pad_length:-pad_length]
        
        return filtered_signals
    
    except ValueError as ve:
        log_preprocessing_error(f"ValueError encountered in bandreject_filter_with_padding: {ve}")
        raise
    except np.core._exceptions._UFuncOutputCastingError as np_err:
        log_preprocessing_error(f"NumPy UFunc Output Casting Error in bandreject_filter_with_padding: {np_err}")
        raise
    



# 3) High-frequency noise removal.
# Apply a low-pass Butterworth filter to high-frequency noise with cut-off frequency 25 Hz.

# Low-pass Butterworth filter (with cut-off frequency 25 Hz) after padding it at the beginning and end:
def lowpass_filter_with_padding(signal, highcut=25, fs=360, order=1, pad_length=100):
    
    try:
        # Create an empty array to store the filtered signals for each lead
        filtered_signals = np.zeros_like(signal)
        
        # Loop through each lead (column)
        for i in range(signal.shape[1]):
            lead_signal = signal[:, i]
            
            # Pad the signal for the current lead
            padded_signal = np.pad(lead_signal, (pad_length, pad_length), 'edge')
            
            # Apply the low-pass filter to the padded signal for the current lead
            nyquist = 0.5 * fs
            high = highcut / nyquist
            b, a = butter(order, high, btype='low')
            filtered_padded_signal = filtfilt(b, a, padded_signal)
            
            # Remove the padding and store the filtered signal for the current lead
            filtered_signals[:, i] = filtered_padded_signal[pad_length:-pad_length]
        
        return filtered_signals

    except ValueError as ve:
        log_preprocessing_error(f"ValueError encountered in lowpass_filter_with_padding: {ve}")
        raise
    except np.core._exceptions._UFuncOutputCastingError as np_err:
        log_preprocessing_error(f"NumPy UFunc Output Casting Error in lowpass_filter_with_padding: {np_err}")
        raise
    



# 4) Normalization:
# Normalization is a technique used to scale the ECG signal to a fixed range of values (typically between 0 and 1 or -1 and 1).
# This can be useful when comparing ECG signals from different patients or when training a machine learning model.

# The normalize function takes a signal as input and returns the normalized signal to range [-1, 1] for each lead individually:
#   return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
def normalize(signal):
    
    # Compute the minimum and maximum values for each lead
    min_values = np.min(signal, axis=0)
    max_values = np.max(signal, axis=0)
    
    # Compute the range (difference between max and min) for each lead
    range_values = max_values - min_values
    
    # If the range is zero (i.e., constant signal), return the signal unchanged
    # This prevents division by zero and retains the constant lead value
    constant_lead_indices = np.where(range_values == 0)[0]
    if len(constant_lead_indices) > 0:
        log_preprocessing_error(f"WARNING: Leads {constant_lead_indices} have constant values. They won't be normalized.")
        range_values[range_values == 0] = 1.0  # Set to 1 to avoid division by zero, but the signal remains unchanged for these leads
    
    # Normalize the signal to [-1, 1]
    normalized_signal = 2 * ((signal - min_values) / range_values) - 1
    # Normalize the signal to [0, 1]
    #normalized_signal = (signal - min_values) / range_values
    
    return normalized_signal




# Function to load the original ECG signal and preprocess it:
def load_and_preprocess_signal(record_name, source_dataset_path): 

    try:
        record = wfdb.rdrecord(os.path.join(source_dataset_path, record_name))
        original_signal = record.p_signal

        # High-pass filtering:
        cleaned_signal = highpass_filter_with_padding(original_signal)
        
        # Band-reject filtering:
        denoised_signal = bandreject_filter_with_padding(cleaned_signal)
        
        # Low-pass filtering:
        final_cleaned_signal = lowpass_filter_with_padding(denoised_signal)
        
        # Normalization:
        normalized_signal = normalize(final_cleaned_signal)

        return normalized_signal

    except ValueError as ve:
        log_preprocessing_error(f"ValueError encountered in load_and_preprocess_signal: {ve}")
        raise
    



# Function to start preprocessing phase for all records in the source dataset directory:
def preprocess_data(source_dataset_path, save_dataset_path):
    
    # Check the existence of the source directory:
    if not os.path.exists(source_dataset_path):
        print(f"Error: Source directory {source_dataset_path} does not exist.")
        return

    # Create target directory if it doesn't exist:
    if not os.path.exists(save_dataset_path):
        os.makedirs(save_dataset_path)
        
    # List all .dat files (i.e. ECG signal files) in the original dataset directory:
    all_dat_files = [f for f in os.listdir(source_dataset_path) if f.endswith('.dat')]

    # Extract unique record numbers:
    records = set(f.split('.')[0] for f in all_dat_files)
    
    # Get the set of already preprocessed records:
    preprocessed_files = os.listdir(save_dataset_path)
    preprocessed_records = {f.split('_')[0] for f in preprocessed_files if f.endswith('_preprocessed_360hz.dat')}
    
    # Print a new line before preprocessing of the records starts:
    print("\n")
    
    # Preprocess each record from the original dataset that hasn't been preprocessed yet:
    for record_name in records:
        if record_name in preprocessed_records:
            print(f"Record {record_name} has already been preprocessed. Skipping.")
            continue # Skip to the next record
        
        # Start of preprocessing phase for the current record's ECG signal:
        try:
            
            logging.info(f"Starting preprocessing for record: {record_name}")
            
            # Calling function to load and preprocess signal of current record:
            preprocessed_signal = load_and_preprocess_signal(record_name, source_dataset_path)
            
            # Construct the filename for the preprocessed signal to be saved:
            preprocessed_filename = os.path.join(save_dataset_path, f"{record_name}_preprocessed_360hz.dat")
            
            # Saving the preprocessed signal:
            np.savetxt(preprocessed_filename, preprocessed_signal, delimiter=',')
            print(f"Successfully preprocessed and saved {record_name} to {preprocessed_filename}.")
            
        
        except FileNotFoundError as fnf_error:
            print(fnf_error)
            log_preprocessing_error(fnf_error)
        except ValueError as ve:
            print(f"ValueError encountered: {ve}")
            log_preprocessing_error(ve)
            traceback.print_exc()
        # To handle cases where filter coefficients might lead to instability:
        except BadCoefficients as bc:
            print(f"Bad coefficients encountered during filtering: {bc}")
            log_preprocessing_error(bc)
            traceback.print_exc()
        # To handle potential linear algebra-related errors:
        except np.linalg.LinAlgError as lae:
            print(f"Linear algebra error encountered: {lae}")
            log_preprocessing_error(lae)
            traceback.print_exc()
        except Exception as e:
            print(f"General error processing record {record_name}. Error: {e}")
            log_preprocessing_error(e)
            traceback.print_exc()




# Function to check if all files have been preprocessed:
def check_all_files_preprocessed(source_dataset_path, save_dataset_path):
    # List of all .dat files in the MIT-BIH directory
    original_files = [f for f in os.listdir(source_dataset_path) if f.endswith('.dat')]
    
    # List of all preprocessed files in the 'Preprocessed Data 360 Hz' directory
    preprocessed_files = [f for f in os.listdir(save_dataset_path) if f.endswith('_preprocessed_360hz.dat')]
    
    # Check if each original .dat file has a corresponding preprocessed file
    missing_files = []
    for orig_file in original_files:
        expected_preprocessed_file = orig_file.split('.')[0] + '_preprocessed_360hz.dat'
        if expected_preprocessed_file not in preprocessed_files:
            missing_files.append(orig_file)
    
    # Display the results
    if not missing_files:
        print(f"\nAll .dat files have been successfully preprocessed and saved in the {save_dataset_path} directory.\n")
    else:
        print("\nThe following files have not been preprocessed:")
        for missing in missing_files:
            print(missing)
