import os
import numpy as np # Library to work with arrays
from scipy.signal import resample


# Also, consider whether your analysis would benefit from an anti-aliasing filter applied before resampling, 
# which can be critical when downsampling to avoid aliasing of higher-frequency components into your band of interest. 
# The resample function applies an anti-aliasing filter by default, but depending on your specific requirements, 
# you might want to apply a custom filter that better suits your data.


# Constants:
SOURCE_DIRECTORY = './data/Preprocessed Data 360 Hz'
TARGET_DIRECTORY = './data/Preprocessed Data 256 Hz'
    
# Function to resample pre-processed data from 360 Hz to 256 Hz:
def resample_preprocessed_data():
    
    # Check the existence of the source directory
    if not os.path.exists(SOURCE_DIRECTORY):
        print(f"Error: Source directory {SOURCE_DIRECTORY} does not exist.")
        return

    # Create target directory if it doesn't exist
    if not os.path.exists(TARGET_DIRECTORY):
        os.makedirs(TARGET_DIRECTORY)
    
    # List all preprocessed files in the source directory
    preprocessed_files = [f for f in os.listdir(SOURCE_DIRECTORY) if f.endswith('_preprocessed_360hz.dat')]
    
    # Counter to keep track of successfully resampled files:
    resampled_count = 0
    
    for file in preprocessed_files:
        # Define the output filename
        output_filename = file.replace('_preprocessed_360hz', '_preprocessed_256hz')
        
        # Check if the output file already exists in the target directory
        if os.path.exists(os.path.join(TARGET_DIRECTORY, output_filename)):
            print(f"{output_filename} already exists. Skipping resampling for {file}.")
            continue
        
        # Load preprocessed data, assuming it's a multi-column array with the first column as the first lead
        data = np.loadtxt(os.path.join(SOURCE_DIRECTORY, file), delimiter=',')
        first_lead_data = data[:, 0] if data.ndim > 1 else data

    
        original_duration = first_lead_data.shape[0] / 360
        new_length = int(np.ceil(original_duration) * 360)

        # Extend the original signal
        extended_data = np.zeros(new_length)
        extended_data[0:first_lead_data.shape[0]] = first_lead_data

        # Resample the extended data to 256 Hz
        resampled_duration = int(np.ceil(original_duration) * 256)
        resampled_data = resample(extended_data, resampled_duration)

        # Save the resampled data
        np.savetxt(os.path.join(TARGET_DIRECTORY, output_filename), resampled_data, delimiter=',')
        
        print(f"Resampled and saved {file} to {output_filename}.")
        resampled_count += 1

    print(f"\nResampled {resampled_count} out of {len(preprocessed_files)} preprocessed files successfully.")
        

# Function to check if all preprocessed files have been resampled:
def check_all_files_resampled():
    
    # List of all preprocessed files in the source directory:
    preprocessed_files = [f for f in os.listdir(SOURCE_DIRECTORY) if f.endswith('_preprocessed_360hz.dat')]
    
    # List of all resampled files in the target directory:
    resampled_files = [f for f in os.listdir(TARGET_DIRECTORY) if f.endswith('_preprocessed_256hz.dat')]
    
    # Check if each preprocessed file in the source directory has a corresponding resampled file in the target directory:
    missing_files = []
    for preprocessed_file in preprocessed_files:
        expected_resampled_file = preprocessed_file.replace('_preprocessed_360hz', '_preprocessed_256hz')
        if expected_resampled_file not in resampled_files:
            missing_files.append(preprocessed_file)
    
    # Display the results
    if not missing_files:
        print("\nAll preprocessed files have been successfully resampled and saved in the 'Preprocessed Data ' directory.\n")
    else:
        print("\nThe following files have not been resampled:")
        for missing in missing_files:
            print(missing)
