import os
import wfdb
import pickle

# Constants for dataset paths
ORIGINAL_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
RESAMPLED_DIRECTORY = './data/Preprocessed Data 256 Hz'  # Directory containing resampled files
TRAINING_DIRECTORY = './data/Training/'
TESTING_DIRECTORY = './data/Testing/'

# Defining training and testing datasets
training_dataset = {"101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"}
testing_dataset = {"100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"}

# Function to read the resampled data
def read_resampled_data(record_name, directory):
    file_path = os.path.join(directory, f"{record_name}_preprocessed_256hz.dat")
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# Function to read labels from the original .atr files
def read_labels(record_name, directory):
    record_path = os.path.join(directory, record_name)
    annotation = wfdb.rdann(record_path, 'atr')
    labels = []
    for ann_sample, ann_symbol in zip(annotation.sample, annotation.symbol):
        labels.append({'time': ann_sample / 360, 'symbol': ann_symbol})  # Convert time to seconds
    return labels

# Function to split the dataset
def split_dataset(dataset, resampled_directory, original_directory):
    signals, labels = [], []
    for record in dataset:
        data = read_resampled_data(record, resampled_directory)
        label = read_labels(record, original_directory)
        signals.append(data['signal'])  # Assuming data has 'signal'
        labels.append(label)
    return signals, labels

# Function to save the split datasets
def save_dataset(data, labels, directory, filename):
    with open(os.path.join(directory, filename), 'wb') as file:
        pickle.dump({'signals': data, 'labels': labels}, file)

# Main function to split datasets and save them
def split_and_save_datasets():
    print("Splitting and saving training dataset...")
    train_signals, train_labels = split_dataset(training_dataset, RESAMPLED_DIRECTORY, ORIGINAL_PATH)
    save_dataset(train_signals, train_labels, TRAINING_DIRECTORY, 'training_dataset_signals.pickle')

    print("Splitting and saving testing dataset...")
    test_signals, test_labels = split_dataset(testing_dataset, RESAMPLED_DIRECTORY, ORIGINAL_PATH)
    save_dataset(test_signals, test_labels, TESTING_DIRECTORY, 'testing_dataset_signals.pickle')
