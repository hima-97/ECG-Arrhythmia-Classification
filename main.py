import pandas as pd # Library to work with dataframes
from src import preprocessing, plotting, resampling, split_data, feature_extraction, feature_selection, training_and_testing, training_and_testing2




# Constants for dataset paths:
ORIGINAL_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
PREPROCESSED_PATH = './data/Preprocessed Data 360 Hz/'
RESAMPLED_PATH = './data/Preprocessed Data 256 Hz/'  # Directory containing resampled files
HEARTBEATS_PATH = './data/Heartbeats Data/' # Directory containing segmented heartbeats with extracted features
TRAINING_PATH = './data/Training/'
TESTING_PATH = './data/Testing/'
CLASSIFIER_PATH = './data/Heartbeats Classifier/'
LOGS_PATH = './logs/'

# Constant for sampling rate:
SAMPLE_RATE = 256 # Sample rate in Hz




# Main Function:
def main():

    # Preprocess original data:
    preprocessing.preprocess_data(ORIGINAL_PATH, PREPROCESSED_PATH)
    
    # Check if all files have been preprocessed:
    preprocessing.check_all_files_preprocessed(ORIGINAL_PATH, PREPROCESSED_PATH)
    
    # Resample preprocessed data from 360 Hz to 256 Hz:
    resampling.resample_preprocessed_data(PREPROCESSED_PATH, RESAMPLED_PATH, SAMPLE_RATE)
    
    # Check if all preprocessed files have been resampled:
    resampling.check_all_files_resampled(PREPROCESSED_PATH, RESAMPLED_PATH)
    
    # Split and save the dataset into training and testing sets:
    split_data.split_and_save_dataset(ORIGINAL_PATH, RESAMPLED_PATH, TRAINING_PATH, TESTING_PATH)
    
    # Function to view and print first 10 samples and first 5 labels of first record in training dataset:
    split_data.view_training_pickle_file()
    
    # Function to view and print first 10 samples and first 5 labels of first record in testing dataset:
    split_data.view_testing_pickle_file()
    
    # Segment ECG signals into heartbeats and extract features:
    feature_extraction.segment_and_extract_features(TRAINING_PATH, TESTING_PATH, HEARTBEATS_PATH, debug=False)
    
    # Function to verify segmented heartbeats and the extracted features for a specific record:
    # This function will load the pickle file, iterate through the first few heartbeats (up to the specified num_beats_to_inspect), 
    # and print various details about each heartbeat, including its type, source, RR interval features, and morphological features.
    #feature_extraction.verify_heartbeats_and_features(HEARTBEATS_PATH, '101', num_beats_to_inspect=1)
    
    # Function for feature selection process and constructing the training and testing features datasets:
    feature_selection.rank_features_and_construct_features_datasets(HEARTBEATS_PATH, TRAINING_PATH, TESTING_PATH)
    
    # Function to train and test the model:
    training_and_testing.train_and_test_model(TRAINING_PATH, TESTING_PATH, CLASSIFIER_PATH)



# Call to `main` function:
if __name__ == '__main__':
    main()