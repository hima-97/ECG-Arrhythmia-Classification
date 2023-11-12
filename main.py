import os
import pandas as pd # Library to work with dataframes
from src import preprocessing, plotting, resampling, split_dataset, feature_extraction, segmentation, segmentation_2
import matplotlib.pyplot as plt
import wfdb
import numpy as np
import pickle


# Constants for dataset paths:
ORIGINAL_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
PREPROCESSED_PATH = './data/Preprocessed Data 360 Hz/'
RESAMPLED_DIRECTORY = './data/Preprocessed Data 256 Hz'  # Directory containing resampled files

# Main Function:
def main():

    # Get the set of already preprocessed records
    #preprocessed_files = os.listdir(PREPROCESSED_PATH)
    #preprocessed_records = {f.split('_')[0] for f in preprocessed_files if f.endswith('_preprocessed_360hz.dat')}

    # List all .dat files in the dataset directory
    #all_dat_files = [f for f in os.listdir(ORIGINAL_PATH) if f.endswith('.dat')]

    # Extract unique record numbers
    #records = set(f.split('.')[0] for f in all_dat_files)

    # Preprocess each record that hasn't been preprocessed yet
    #for record in records:
    #    if record in preprocessed_records:
    #        print(f"Record {record} has already been preprocessed. Skipping.")
    #        continue
    #    preprocessing.preprocess_record(record, ORIGINAL_PATH)
    
    # Check if all files have been preprocessed:
    #preprocessing.check_all_files_preprocessed()
    
    # Resample preprocessed data from 360 Hz to 256 Hz:
    #resampling.resample_preprocessed_data()
    
    # Now each resampled ECG signal has the following properties:
    # Resampled Length: 462222
    # Actual Sampling Rate: 256.0 Hz
    
    # Check if all preprocessed files have been resampled:
    #resampling.check_all_files_resampled()
    
    
    
    
    
    
    # Split the dataset into training and testing datasets:
    #split_dataset.split_and_save_datasets()
    
    # split_dataset.view_training_pickle_file()
    # split_dataset.view_testing_pickle_file()
    
    feature_extraction.segment_and_extract_features()
    

    

    
    
    
    # Segment the resampled ECG signal into individual beats:
    #segmentation.segment_data()
    #segmentation_2.segment_data()

    #count = segmentation.count_heartbeats('100')
    #print(f"Number of heartbeats for record 100: {count}")
    
    #plotting.plot_heartbeat('100', 0)  # Record '100', 5th heartbeat (index 4)

    
    # Chek if all resampled files have been segmented:
    #segmentation.check_all_files_segmented()
    
    
    
    
    
    
    # Plot original ECG segment:
    #plotting.plot_original_ecg('100', 0, 1000)
    
    # Plot preprocessed ECG segment with R-peaks:
    #plotting.plot_preprocessed_ecg_with_rpeaks('100', 0, 1000)
    
    # Plot a resampled ECG segment with adjusted PQRST annotations:
    #plotting.plot_resampled_ecg_segment_with_pqrst4('202', 324864, 328472) # Example of Atrial Fibrillation, PVC
    #plotting.plot_resampled_ecg_segment_with_pqrst4('100', 0, 2000)




if __name__ == '__main__':
    main()