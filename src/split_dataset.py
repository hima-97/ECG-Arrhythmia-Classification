import os
import wfdb
import pickle
import numpy as np

# Constants for dataset paths
ORIGINAL_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
RESAMPLED_PATH = './data/Preprocessed Data 256 Hz'
TRAINING_PATH = './data/Training/'
TESTING_PATH = './data/Testing/'

# Defining training and testing datasets
training_dataset = {"101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"}
testing_dataset = {"100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"}

def read_resampled_data(record_name, directory):
    file_path = os.path.join(directory, f"{record_name}_preprocessed_256hz.dat")
    try:
        # Load resampled preprocessed signal from the directory
        data = np.loadtxt(file_path, delimiter=',')
        return {'signal': data}
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def read_labels(record_name, directory):
    record_path = os.path.join(directory, record_name)
    annotation = wfdb.rdann(record_path, 'atr')
    labels = []
    for ann_sample, ann_symbol in zip(annotation.sample, annotation.symbol):
        labels.append({'time': ann_sample / 360, 'symbol': ann_symbol})
    return labels

def read_header_info(record_name, directory):
    record_path = os.path.join(directory, record_name)
    record = wfdb.rdrecord(record_path)
    return {
        "label": record.sig_name[0],
        "dimension": record.units[0],
        "sample_rate": record.fs,
        "digital_max": (2 ** record.adc_res[0]) - 1,
        "digital_min": 0,
        "physical_max": (record.adc_zero[0] - record.baseline[0]) / record.adc_gain[0],
        "physical_min": (-record.baseline[0]) / record.adc_gain[0],
        "transducer": "transducer type not recorded",
        "prefilter": "prefiltering not recorded"
    }

def split_dataset(dataset, resampled_directory, original_directory):
    signals, labels, headers = [], [], []
    for record in dataset:
        data = read_resampled_data(record, resampled_directory)
        label = read_labels(record, original_directory)
        header = read_header_info(record, original_directory)
        signals.append(data['signal'])
        labels.append(label)
        headers.append(header)
    return signals, labels, headers

def save_dataset(data, labels, headers, directory, filename):
    with open(os.path.join(directory, filename), 'wb') as file:
        pickle.dump({'signals': data, 'labels': labels, 'headers': headers}, file)



def view_training_pickle_file():
    # Path to one of the saved pickle files
    file_path = './data/Training/training_dataset_signals.pickle'

    # Load the data
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Inspect the contents
    print("Number of records:", len(data['signals']))
    print("Sample record signal:", data['signals'][0])
    print("Sample record labels:", data['labels'][0])
    print("Sample record header:", data['headers'][0])
    
    
def view_testing_pickle_file():
    # Path to one of the saved pickle files
    file_path = './data/Testing/testing_dataset_signals.pickle'

    # Load the data
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Inspect the contents
    print("Number of records:", len(data['signals']))
    print("Sample record signal:", data['signals'][0])
    print("Sample record labels:", data['labels'][0])
    print("Sample record header:", data['headers'][0])




def split_and_save_datasets():
    print("Splitting and saving training dataset...")
    train_signals, train_labels, train_headers = split_dataset(training_dataset, RESAMPLED_PATH, ORIGINAL_PATH)
    save_dataset(train_signals, train_labels, train_headers, TRAINING_PATH, 'training_dataset_signals.pickle')

    print("Splitting and saving testing dataset...")
    test_signals, test_labels, test_headers = split_dataset(testing_dataset, RESAMPLED_PATH, ORIGINAL_PATH)
    save_dataset(test_signals, test_labels, test_headers, TESTING_PATH, 'testing_dataset_signals.pickle')
