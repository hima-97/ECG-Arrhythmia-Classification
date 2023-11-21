import os
import pandas as pd # Library to work with dataframes
from src import preprocessing, plotting, resampling, split_data, feature_extraction, feature_selection, training_and_testing, training_and_testing2
import matplotlib.pyplot as plt
import wfdb
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import re


# Constants for dataset paths:
ORIGINAL_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
PREPROCESSED_PATH = './data/Preprocessed Data 360 Hz/'
RESAMPLED_PATH = './data/Preprocessed Data 256 Hz/'  # Directory containing resampled files
HEARTBEATS_PATH = './data/Heartbeats Data/' # Directory containing segmented heartbeats with extracted features
TRAINING_PATH = './data/Training/'
TESTING_PATH = './data/Testing/'




# Main Function:
def main():

    # Preprocess original data:
    preprocessing.preprocess_data(ORIGINAL_PATH, PREPROCESSED_PATH)
    
    # Check if all files have been preprocessed:
    preprocessing.check_all_files_preprocessed(ORIGINAL_PATH, PREPROCESSED_PATH)
    
    # Resample preprocessed data from 360 Hz to 256 Hz:
    resampling.resample_preprocessed_data(PREPROCESSED_PATH, RESAMPLED_PATH)
    
    # Check if all preprocessed files have been resampled:
    resampling.check_all_files_resampled(PREPROCESSED_PATH, RESAMPLED_PATH)
    
    # Split and save the dataset into training and testing sets:
    split_data.split_and_save_dataset(ORIGINAL_PATH, RESAMPLED_PATH, TRAINING_PATH, TESTING_PATH)
    
    # Function to view and print first 10 samples and first 5 labels of first record in training dataset:
    split_data.view_training_pickle_file()
    
    # Function to view and print first 10 samples and first 5 labels of first record in testing dataset:
    split_data.view_testing_pickle_file()
    
    
    #feature_extraction.segment_and_extract_features(debug=False)
    #feature_extraction.verify_heartbeats_and_features('101', num_beats_to_inspect=1) # Verify that the heartbeats and features were extracted correctly
    #plotting.plot_training_heartbeat_with_features('100', heartbeats_number=5)
    
    
    # Function for feature selection process and constructing the training and testing features datasets:
    #feature_selection.rank_features_and_construct_features_datasets()
    
    
    # Function to train and test the model:
    #tuning_results = training_and_testing2.train_and_test_model()
    #training_and_testing.train_and_test_model()
    
    # # Splitting the results
    # trees, features, accuracies, f1_scores = zip(*tuning_results)
    # # Plotting Accuracy vs Number of Trees
    # plt.figure(figsize=(10, 6))
    # plt.plot(trees, accuracies, marker='o')
    # plt.title('Accuracy vs Number of Trees')
    # plt.xlabel('Number of Trees')
    # plt.ylabel('Accuracy')
    # plt.grid(True)
    # plt.show()
    # # Plotting Accuracy vs Number of Features
    # plt.figure(figsize=(10, 6))
    # plt.plot(features, accuracies, marker='o', color='red')
    # plt.title('Accuracy vs Number of Features')
    # plt.xlabel('Number of Features')
    # plt.ylabel('Accuracy')
    # plt.grid(True)
    # plt.show()
    
    
    
    # Optionally, find optimal number of features
    # Uncomment the following lines if feature optimization analysis is needed
    # best_num_features, best_accuracy, results = find_optimal_features()
    # print(f"Optimal number of features: {best_num_features} with accuracy: {best_accuracy}")
    # for num_features, accuracy in results.items():
    #     print(f"{num_features} features: Accuracy = {accuracy}")
    
    
    
    
    
    # # Load training and testing data
    # train_features, train_labels = load_dataset("./data/Training/training_dataset_features.pickle")
    # test_features, test_labels = load_dataset("./data/Testing/testing_dataset_features.pickle")

    # # Find the optimal number of trees
    # optimal_trees_results = find_optimal_trees(train_features, train_labels, test_features, test_labels)

    # # Find the best tree count based on desired criteria (e.g., highest accuracy or F1 score)
    # best_tree_count = max(optimal_trees_results, key=lambda x: optimal_trees_results[x]['accuracy'])
    # best_f1_score = optimal_trees_results[best_tree_count]['f1_score']
    # print(f"Optimal tree count: {best_tree_count} with accuracy: {optimal_trees_results[best_tree_count]['accuracy']} and F1-score: {best_f1_score}")





if __name__ == '__main__':
    main()