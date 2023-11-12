import os
import wfdb
import pickle
import numpy as np
from src.ecgtypes import HeartRhythm, BeatType



# Constants for dataset paths
ORIGINAL_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
RESAMPLED_PATH = './data/Preprocessed Data 256 Hz'
TRAINING_PATH = './data/Training/'
TESTING_PATH = './data/Testing/'

# Defining training and testing datasets
training_dataset = {"101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"}
testing_dataset = {"100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"}



# Function to read the dataset information:
def read_dataset_info(records):

    # Prepare containers
    signals, labels = [], []
    
    # Iterate over the records:
    for record_name in records:
        
        # Reading annotations from original database
        annotations = wfdb.rdann(f'{ORIGINAL_PATH}/{record_name}', 'atr')
    
        # Reading original ECG record from original database:
        original_record = wfdb.rdrecord(os.path.join(ORIGINAL_PATH, record_name))
    
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
        record_path = os.path.join(RESAMPLED_PATH, f"{record_name}_preprocessed_256hz.dat")
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
def split_and_save_dataset():
    
    print("Creating training dataset...")
    signals, labels = read_dataset_info(training_dataset)
    
    print("Saving training_dataset file...")
    pickle_out = open(TRAINING_PATH + "training_dataset_signals.pickle", "wb")
    pickle.dump({"signals": signals, "labels": labels, "records": training_dataset}, pickle_out)
    pickle_out.close()

    print("Creating testing dataset...")
    signals, labels = read_dataset_info(testing_dataset)
    
    print("Saving testing_dataset file...")
    pickle_out = open(TESTING_PATH + "testing_dataset_signals.pickle", "wb")
    pickle.dump({"signals": signals, "labels": labels, "records": testing_dataset}, pickle_out)
    pickle_out.close()
    
    

# Function to view the testing dataset:
def view_training_pickle_file():
    
    # Path to the training pickle file
    file_path = './data/Training/training_dataset_signals.pickle'

    # Load the data from the pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Print a summary or specific details of the data
    print("Number of records in training dataset:", len(data['signals']))
    print("Sample record signal from training dataset:", data['signals'][0][:10])  # prints first 10 samples of the first record
    print("Sample record labels from training dataset:", data['labels'][0][:5])  # prints first 5 labels of the first record
    
    
# Function to view the testing dataset:
def view_testing_pickle_file():
    # Path to the testing pickle file
    file_path = './data/Testing/testing_dataset_signals.pickle'

    # Load the data from the pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Print a summary or specific details of the data
    print("Number of records in testing dataset:", len(data['signals']))
    print("Sample record signal from testing dataset:", data['signals'][0][:10])  # prints first 10 samples of the first record
    print("Sample record labels from testing dataset:", data['labels'][0][:5])  # prints first 5 labels of the first record
