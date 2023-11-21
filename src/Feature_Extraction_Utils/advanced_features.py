
# The code below is an adaptation of the code from Mondejar-Guerra et. al (http://www.sciencedirect.com/science/article/pii/S1746809418301976)
# The original code is located at https://github.com/mondejar/ecg-classification/blob/master/python/features_ECG.py


import numpy as np
import scipy.stats
import pywt
import operator
from numpy.polynomial.hermite import hermfit

# The code in this file contains an adaptation of the code from Mondejar-Guerra et. al (http://www.sciencedirect.com/science/article/pii/S1746809418301976)
# which can be found at https://github.com/mondejar/ecg-classification/blob/master/python/features_ECG.py


# This file is dedicated to extrating the following types of features:
# Hermite Basis Function (HBF) Coefficients
# Higher Order Statistics (HOS) features
# DWT coefficients
# Uniform Local Binary Patterns (LBP)
# Custom Descriptor Based on Amplitude Intervals




# Function that uses Discrete Wavelet Transform (DWT) to compute coefficients for a heartbeat:
# Parameters include:
# The wavelet family to use, defaulting to 'db1' (Daubechies wavelet),
# and the level of wavelet decomposition, defaulting to 3.
def compute_wavelet_descriptor(beat, family='db1', level=3):
    wave_family = pywt.Wavelet(family)
    coeffs = pywt.wavedec(beat, wave_family, level=level)
    return coeffs[0]




# Compute the HOS descriptor for a heartbeat:
# Skewness (3 cumulant) and kurtosis (4 cumulant)
# The third-, and fourth-order cumulant functions (kurtosis and skewness, respectively) for each beat segment are computed. 
# The lag parameters range from a -250 ms to 250 ms centered on the R spike and five equally spaced sample points.
def compute_hos_descriptor(beat, n_intervals=6):
    
    # The length of the beat signal is divided by the number of intervals to determine the size of each interval:
    lag = len(beat) / n_intervals
    hos_b = np.zeros(((n_intervals-1) * 2))
    
    for i in range(0, n_intervals-1):
        pose = (lag * (i+1))
        interval = beat[int(pose - (lag/2)):int(pose + (lag/2))]

        # Skewness
        hos_b[i] = scipy.stats.skew(interval, 0, True)

        if np.isnan(hos_b[i]):
            hos_b[i] = 0.0

        # Kurtosis
        hos_b[(n_intervals-1) + i] = scipy.stats.kurtosis(interval, 0, False, True)
        if np.isnan(hos_b[(n_intervals-1) + i]):
            hos_b[(n_intervals-1) + i] = 0.0
    
    # This contains the calculated skewness and kurtosis values for each segment of the beat:        
    return hos_b




# This function calculates the uniform Local Binary Pattern (LBP) histogram for a 1D signal: 
# Local Binary Patterns are a texture descriptor, often used in image processing but also applicable to 1D signals like ECGs. 
# They are useful for capturing local patterns in data.

# This list contains all possible uniform patterns for 8 neighbors, which are used to classify each local pattern as either uniform or non-uniform:
uniform_pattern_list = np.array([0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128,
                                 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255])
# Compute the uniform LBP 1D from signal with neigh equal to number of neighbours:
# and return the 59 histogram:
# 0-57: uniform patterns
# 58: the non uniform pattern
# NOTE: this method only works with neigh = 8
def compute_Uniform_LBP(signal, neigh=8):
    
    # Initializes a histogram with 59 bins (58 for each uniform pattern and 1 for all non-uniform patterns):
    hist_u_lbp = np.zeros(59, dtype=float)

    avg_win_size = 2
    # NOTE: Reduce sampling by half
    #signal_avg = scipy.signal.resample(signal, len(signal) / avg_win_size)

    for i in range(int(neigh/2), len(signal) - int(neigh/2)):
        pattern = np.zeros(neigh)
        ind = 0
        for n in [x for x in range(-1*int(neigh/2), 0)] + [x for x in range(1, int(neigh/2)+1)]:
            if signal[i] > signal[i+n]:
                pattern[ind] = 1
            ind += 1
            
        # Convert pattern to id-int 0-255 (for neigh == 8)
        pattern_id = int("".join(str(c) for c in pattern.astype(int)), 2)

        # Convert id to uniform LBP id 0-57 (uniform LBP)  58: (non uniform LBP)
        if pattern_id in uniform_pattern_list:
            pattern_uniform_id = int(np.argwhere(
                uniform_pattern_list == pattern_id))
        else:
            pattern_uniform_id = 58  # Non uniforms patternsuse

        hist_u_lbp[pattern_uniform_id] += 1.0

    return hist_u_lbp




# This function is designed to calculate a custom set of features from an ECG heartbeat, based on amplitudes of several intervals:
# This function focuses on analyzing specific intervals within the beat and calculating distances related to the R peak.
# the Euclidean distance (sample, amplitude) between the R spike peak and four points of the beat segment are obtained as described in [25]. 
# These points correspond to the maximum amplitude value between 250 ms and 139 ms before the R spike, 
# the minimum value between 42 ms and 14 ms before the R spike, the minimum value between 14 ms and 42 ms after the R spike, 
# and maximum value between 139 ms and 250 ms after the R spike. 
def compute_morphological_features(beat):
    
    # Length of the beat:
    L = len(beat)
    
    # Assuming R peak is located in the middle of the beat array:
    # This is a common assumption when the beat is pre-segmented around the R peak.
    R_pos = int(len(beat) / 2) 

    R_value = beat[R_pos]
    my_morph = np.zeros((4))
    y_values = np.zeros(4)
    x_values = np.zeros(4)
    
    # Identifying maximum and minimum values (amplitudes) in specified intervals around the R peak:
    # These intervals are scaled based on the length of the beat (L).
    [x_values[0], y_values[0]] = max(enumerate(beat[0:int(40*L/180)]), key=operator.itemgetter(1))
    [x_values[1], y_values[1]] = min(enumerate(beat[int(75*L/180):int(85*L/180)]), key=operator.itemgetter(1))
    [x_values[2], y_values[2]] = min(enumerate(beat[int(95*L/180):int(105*L/180)]), key=operator.itemgetter(1))
    [x_values[3], y_values[3]] = max(enumerate(beat[int(150*L/180):L]), key=operator.itemgetter(1))

    x_values[1] = x_values[1] + int(75*L/180)
    x_values[2] = x_values[2] + int(95*L/180)
    x_values[3] = x_values[3] + int(150*L/180)

    # Normalizing data before computing Euclidean distances:
    # The normalization of data before computing the distances ensures that the distances are relative to the range of the signal, 
    # accounting for variability in signal amplitude and length.
    x_max = max(x_values)
    y_max = max(np.append(y_values, R_value))
    x_min = min(x_values)
    y_min = min(np.append(y_values, R_value))

    R_pos = (R_pos - x_min) / (x_max - x_min)
    R_value = (R_value - y_min) / (y_max - y_min)

    # Calculate the Euclidean distance, but not in the conventional sense of distance between two points in space. 
    # Instead, it calculates the normalized Euclidean distance between the R peak and four other points in the beat segment. 
    # These points are determined based on maximum and minimum amplitudes within specified intervals.
    # The calculated 'distance' is a form of morphological feature, reflecting the relationship of the R peak to other key points in the ECG waveform.
    for n in range(0, 4):
        x_values[n] = (x_values[n] - x_min) / (x_max - x_min)
        y_values[n] = (y_values[n] - y_min) / (y_max - y_min)
        x_diff = (R_pos - x_values[n])
        y_diff = R_value - y_values[n]
        my_morph[n] = np.linalg.norm([x_diff, y_diff])
        # TODO test with np.sqrt(np.dot(x_diff, y_diff))

    if np.isnan(my_morph[n]):
        my_morph[n] = 0.0

    return my_morph




# https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.polynomials.hermite.html
# Support Vector Machine-Based Expert System for Reliable Heartbeat Recognition
# 15 hermite coefficients!

# Function to compute Hermite Basis Function (HBF) coefficients for a heartbeat:
# HBF coefficients are a way to represent a signal using a series of orthogonal polynomials, 
# providing a compact and efficient description of the signal's shape.
def compute_HBF(beat):

    coeffs_hbf = np.zeros(15, dtype=float)
    coeffs_HBF_3 = hermfit(range(0, len(beat)), beat, 3)
    coeffs_HBF_4 = hermfit(range(0, len(beat)), beat, 4)
    coeffs_HBF_5 = hermfit(range(0, len(beat)), beat, 5)

    coeffs_hbf = np.concatenate((coeffs_HBF_3, coeffs_HBF_4, coeffs_HBF_5))

    return coeffs_hbf
