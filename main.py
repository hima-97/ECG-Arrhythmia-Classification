import os
import pandas as pd # Library to work with dataframes
from src import preprocessing, plotting, resampling
import matplotlib.pyplot as plt


# Main Function:
def main():
    # Path to the dataset
    dataset_path = './data/mit-bih-arrhythmia-database-1.0.0'  # MIT-BIH Arhhythmia Database directory.
    
    # List all files in the dataset directory
    all_files = os.listdir(dataset_path)
    
    # Extract unique record numbers by checking the first part of the file names before any extensions
    #records = set([f.split('.')[0] for f in all_files if f.endswith('.dat')])
    records = ['100'] 

    # Preprocess each record in the dataset:
    # for record in records:
    #     preprocessing.preprocess_record(record, dataset_path)

    # Check if all files have been preprocessed:
    #preprocessing.check_all_files_preprocessed()
    
    # Resample preprocessed data from 360 Hz to 256 Hz:
    #resampling.resample_preprocessed_data()
    
    # Check if all preprocessed files have been resampled:
    #resampling.check_all_files_resampled()
    
    # Plot original ECG segment:
    #plotting.plot_original_ecg('100', 0, 1000)
    
    # Plot preprocessed ECG segment with R-peaks:
    #plotting.plot_preprocessed_ecg_with_rpeaks('100', 0, 1000)
    
    # Plot an ECG segment with PQRST annotations:
    #plotting.plot_ecg_segment_with_pqrst('100', 0, 500)
    
    # Plot a resampled ECG segment with adjusted PQRST annotations:
    plt.ioff()  # Turn off interactive mode to avoid multiple plots on the same figure
    # plotting.plot_resampled_ecg_segment_with_pqrst4('202', 324864, 328472) # Example of Atrial Fibrillation, PVC
    plotting.plot_resampled_ecg_segment_with_pqrst4('100', 0, 1000)
    # plotting.plot_resampled_ecg_segment_with_pqrst4('100', 200, 400)
    # plotting.plot_resampled_ecg_segment_with_pqrst4('100', 400, 600)






if __name__ == '__main__':
    main()