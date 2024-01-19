import os
import numpy as np # Library to work with arrays
from scipy.signal import resample


# Also, consider whether your analysis would benefit from an anti-aliasing filter applied before resampling, 
# which can be critical when downsampling to avoid aliasing of higher-frequency components into your band of interest. 
# The resample function applies an anti-aliasing filter by default, but depending on your specific requirements, 
# you might want to apply a custom filter that better suits your data.



    
# Function to resample pre-processed data from 360 Hz to 256 Hz:
def resample_preprocessed_data(source_dataset_path, save_dataset_path, sample_rate):
    
    # Check the existence of the source directory:
    if not os.path.exists(source_dataset_path):
        print(f"Error: Source directory {source_dataset_path} does not exist.")
        return

    # Create target directory if it doesn't exist:
    if not os.path.exists(save_dataset_path):
        os.makedirs(save_dataset_path)
    
    # List all preprocessed files in the source directory:
    preprocessed_files = [f for f in os.listdir(source_dataset_path) if f.endswith('_preprocessed_360hz.dat')]
    
    # Counter to keep track of successfully resampled files:
    resampled_count = 0
    
    for file in preprocessed_files:
        # Define the output filename
        output_filename = file.replace('_preprocessed_360hz', '_preprocessed_256hz')
        
        # Check if the output file already exists in the target directory:
        if os.path.exists(os.path.join(save_dataset_path, output_filename)):
            print(f"{output_filename} already exists. Skipping resampling for {file}.")
            continue
        
        # Load preprocessed data, assuming it's a multi-column array with the first column as the first lead:
        data = np.loadtxt(os.path.join(source_dataset_path, file), delimiter=',')
        first_lead_data = data[:, 0] if data.ndim > 1 else data

        # Calculate the new length of the signal:
        original_duration = first_lead_data.shape[0] / 360
        new_length = int(np.ceil(original_duration) * 360)

        # Extend the original signal:
        extended_data = np.zeros(new_length)
        extended_data[0:first_lead_data.shape[0]] = first_lead_data

        # Resample the extended data to 256 Hz:
        resampled_duration = int(np.ceil(original_duration) * sample_rate)
        resampled_data = resample(extended_data, resampled_duration)

        # Save the resampled data:
        np.savetxt(os.path.join(save_dataset_path, output_filename), resampled_data, delimiter=',')
        
        print(f"Resampled and saved {file} to {output_filename}.")
        resampled_count += 1

    print(f"\nResampled {resampled_count} out of {len(preprocessed_files)} preprocessed files successfully.")
        
        
        

# Function to check if all preprocessed files have been resampled:
def check_all_files_resampled(source_dataset_path, save_dataset_path):
    
    # List of all preprocessed files in the source directory:
    preprocessed_files = [f for f in os.listdir(source_dataset_path) if f.endswith('_preprocessed_360hz.dat')]
    
    # List of all resampled files in the target directory:
    resampled_files = [f for f in os.listdir(save_dataset_path) if f.endswith('_preprocessed_256hz.dat')]
    
    # Check if each preprocessed file in the source directory has a corresponding resampled file in the target directory:
    missing_files = []
    for preprocessed_file in preprocessed_files:
        expected_resampled_file = preprocessed_file.replace('_preprocessed_360hz', '_preprocessed_256hz')
        if expected_resampled_file not in resampled_files:
            missing_files.append(preprocessed_file)
    
    # Display the results
    if not missing_files:
        print(f"\nAll preprocessed files have been successfully resampled and saved in the {source_dataset_path} directory.\n")
    else:
        print("\nThe following files have not been resampled:")
        for missing in missing_files:
            print(missing)
