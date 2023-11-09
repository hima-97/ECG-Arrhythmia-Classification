import os
import numpy as np
import wfdb
import pandas as pd # Library to work with dataframes

# Heartbeat segmentation refers to the process of identifying and isolating individual heartbeats (or cardiac cycles) from a continuous ECG recording.
# The goal of this process is to extract individual heartbeats from the ECG signal for further processing and analysis.

# Steps for heartbeat segmentation:
# 1) Storing Peak Locations: Store the identified P, Q, R, S, and T peaks in a structured format (e.g., a dictionary or dataframe). 
# This will allow to easily access and manipulate these points for further analysis.
# 2) Segmenting Individual Heartbeats: One common approach is to segment the ECG signal into individual heartbeats using the R-peaks as reference points. 
# The segment might start from the midpoint between one R-peak and the previous R-peak and end at the midpoint between the same R-peak and the next R-peak.
# 3) Storing Segmented Heartbeats: Store the segmented heartbeats in a structured format (e.g., a list or dataframe) for further processing or analysis.


# Constants for dataset paths and beat labels:
ORIGINAL_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
RESAMPLED_PATH = './data/Pre-processed Data (256 Hz)/'
BEAT_LABELS = ['·', 'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q']
SEGMENTATION_PATH = './data/segmented_heartbeats.parquet'
SEGMENTATION_PATH_CSV = './data/segmented_heartbeats.csv'