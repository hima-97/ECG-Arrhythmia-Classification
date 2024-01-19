import os
import wfdb
import pickle
import numpy as np
from src.ecg_types import HeartRhythm, BeatType


# Defining training and testing datasets:
TRAINING_SET = {"101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"}
TESTING_SET = {"100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"}




# Function to read the dataset information:
def read_dataset_info(original_dataset_path, resampled_dataset_path, records):

    # Prepare containers
    signals, labels = [], []
    
    # Iterate over the records:
    for record_name in records:
        
        # Reading annotations from original database
        annotations = wfdb.rdann(f'{original_dataset_path}/{record_name}', 'atr')
    
        # Reading original ECG record from original database:
        original_record = wfdb.rdrecord(os.path.join(original_dataset_path, record_name))
    
        header = {
            "label": original_record.sig_name[0],
            "dimension": original_record.units[0],
            "sample_rate": original_record.fs,
            "digital_max": (2 ** original_record.adc_res[0]) - 1,
            "digital_min": 0,
            "transducer": "transducer type not recorded",
            "prefilter": "prefiltering not recorded",
        }
        header["physical_max"] = (header["digital_max"] - original_record.baseline[0]) / original_record.adc_gain[0]
        header["physical_min"] = (header["digital_min"] - original_record.baseline[0]) / original_record.adc_gain[0]
        
        
        # Reading annotations from original database:
        # Initializes a variable rhythmClass with a default value indicating a normal heart rhythm. 
        # This will be updated if different rhythm information is found in the annotations.
        rhythmClass = HeartRhythm.NORMAL
        label = []
        
        # For each annotation, calculate the time in seconds at which it occurs: 
        # This is done by dividing the original sample index of the annotation (annotations.sample[s]) by the original sampling rate (record.fs).
        for s in range(len(annotations.sample)):
            t = annotations.sample[s] / original_record.fs # `original_record.fs` is the original sampling rate (360 Hz)
            ann = annotations.symbol[s]

            # Checking for auxiliary notes in the annotations:
            if len(annotations.aux_note[s]) > 0:
                if annotations.aux_note[s][0] == "(":
                    rhythmClass = annotations.aux_note[s].strip("\x00")[1:]

            # If the annotation is not empty, append it to the label list:
            if len(ann) == 0:
                continue
            elif ann:
                label.append(
                    {
                        "time": t,
                        "beat": BeatType.new_from_symbol(ann),
                        "rhythm": HeartRhythm.new_from_symbol(rhythmClass),
                    }
                )
                
        
        # Reading resampled ECG signals from local directory
        record_path = os.path.join(resampled_dataset_path, f"{record_name}_preprocessed_256hz.dat")
        try:
            # Load resampled preprocessed signal from the directory
            resampled_signal = np.loadtxt(record_path, delimiter=',')
        except IOError as e:
            print(f"Error reading file {record_path}: {e}")
            # Handle the error or continue with the next record
            continue

        # Append the resampled signal and the label to the containers:
        signals.append(resampled_signal) # `resampled_signal` is the resampled ECG signal (256 Hz)
        labels.append(label)
        
    return signals, labels




# Function to create and save training and testing datasets:
def split_and_save_dataset(original_dataset_path, resampled_dataset_path, training_dataset_path, testing_dataset_path):
    
    # Check and create the training directory if it doesn't exist:
    if not os.path.exists(training_dataset_path):
        os.makedirs(training_dataset_path)
        print(f"Created training directory at {training_dataset_path}")

    # Check and create the testing directory if it doesn't exist:
    if not os.path.exists(testing_dataset_path):
        os.makedirs(testing_dataset_path)
        print(f"Created testing directory at {testing_dataset_path}")
    
    print("\nCreating training dataset...")
    signals, labels = read_dataset_info(original_dataset_path, resampled_dataset_path, TRAINING_SET)
    
    print("\nSaving training_dataset file...")
    pickle_out = open(training_dataset_path + "training_dataset_signals.pickle", "wb")
    pickle.dump({"signals": signals, "labels": labels, "records": TRAINING_SET}, pickle_out)
    pickle_out.close()

    print("\nCreating testing dataset...")
    signals, labels = read_dataset_info(original_dataset_path, resampled_dataset_path, TESTING_SET)
    
    print("\nSaving testing_dataset file...")
    pickle_out = open(testing_dataset_path + "testing_dataset_signals.pickle", "wb")
    pickle.dump({"signals": signals, "labels": labels, "records": TESTING_SET}, pickle_out)
    pickle_out.close()
    
    


# Function to view and print first 10 samples and first 5 labels of first record in training dataset:
def view_training_pickle_file():
    
    # Path to the training pickle file
    file_path = './data/Training/training_dataset_signals.pickle'

    # Load the data from the pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Print a summary or specific details of the data
    print("\nNumber of records in training dataset:\n", len(data['signals']))
    print("\nFirst 10 samples of first record from training dataset:\n", data['signals'][0][:10])  # prints first 10 samples of the first record
    print("\nFirst 5 labels of first record from training dataset:\n", data['labels'][0][:5])  # prints first 5 labels of the first record
    
    
    
    
# Function to view and print first 10 samples and first 5 labels of first record in testing dataset:
def view_testing_pickle_file():
    
    # Path to the testing pickle file
    file_path = './data/Testing/testing_dataset_signals.pickle'

    # Load the data from the pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Print a summary or specific details of the data
    print("\nNumber of records in testing dataset:\n", len(data['signals']))
    print("\nFirst 10 samples of first record from testing dataset:\n", data['signals'][0][:10])  # prints first 10 samples of the first record
    print("\nFirst 10 samples of first record from testing dataset:\n", data['labels'][0][:5])  # prints first 5 labels of the first record
