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
ORIGINAL_PATH = './data/mit-bih-arrhythmia-database-1.0.0'
PREPROCESSED_PATH = './data/Preprocessed Data 360 Hz'
RESAMPLED_PATH = './data/Preprocessed Data 256 Hz'  # Directory containing resampled files
HEARTBEATS_PATH = './data/Heartbeats Data' # Directory containing segmented heartbeats with extracted features
TRAINING_PATH = './data/Training'
TESTING_PATH = './data/Testing'



def load_dataset(filepath):
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    return data['features'], data['labels']

def find_optimal_features(start=6, end=20):
    results = {}
    best_accuracy = 0
    best_num_features = 0

    # Load training and testing data
    train_features, train_labels = load_dataset("./data/Training/training_dataset_features.pickle")
    test_features, test_labels = load_dataset("./data/Testing/testing_dataset_features.pickle")
    
    for n_features in range(start, end + 1):
        print(f"\nTraining model with top {n_features} features...")

        # Select the top n features
        selected_train_features = train_features[:, :n_features]
        selected_test_features = test_features[:, :n_features]

        # Train and evaluate the model
        rf_classifier = RandomForestClassifier(n_estimators=40, random_state=42)
        rf_classifier.fit(selected_train_features, train_labels)
        accuracy = accuracy_score(test_labels, rf_classifier.predict(selected_test_features))
        print(f"Accuracy with {n_features} features: {accuracy}")

        results[n_features] = accuracy

        # Update best accuracy and number of features if current accuracy is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_num_features = n_features

    return best_num_features, best_accuracy, results





def find_optimal_trees(train_features, train_labels, test_features, test_labels, start=40, end=100, step=1):
    results = {}
    
    for n_trees in range(start, end + 1, step):
        print(f"\nTraining model with {n_trees} trees...")
        
        # Train the Random Forest classifier
        rf_classifier = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        rf_classifier.fit(train_features, train_labels)
        
        # Validate the model
        test_predictions = rf_classifier.predict(test_features)
        accuracy = accuracy_score(test_labels, test_predictions)
        f1 = f1_score(test_labels, test_predictions, average='weighted')

        print(f"Accuracy with {n_trees} trees: {accuracy}")
        print(f"F1-score with {n_trees} trees: {f1}")
        
        results[n_trees] = {'accuracy': accuracy, 'f1_score': f1}
    
    return results


def plot_optimal_trees():
    # Initialize lists to store data
    num_trees = []
    accuracy_scores = []

    # Read data from the text file
    with open('output_optimal_trees.txt', 'r') as file:
        lines = file.readlines()

    # Parse the data using regular expressions
    for line in lines:
        match = re.search(r'Accuracy with (\d+) trees: ([0-9.]+)', line)
        if match:
            num_trees.append(int(match.group(1)))
            accuracy_scores.append(float(match.group(2)) * 100)  # Multiply by 100 to convert to percentage

    # Find the maximum accuracy score and corresponding number of trees
    max_accuracy = max(accuracy_scores)
    best_num_trees = num_trees[accuracy_scores.index(max_accuracy)]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(num_trees, accuracy_scores, marker='o', linestyle='-', markersize=5, linewidth=1)
    plt.title('Number of Trees vs. Accuracy Score')
    plt.xlabel('Number of Trees', fontsize=12)  # Increase x-axis label font size
    plt.ylabel('Overall Accuracy Score (%)', fontsize=12)  # Increase y-axis label font size
    plt.grid(True)

    # Highlight the point with the highest accuracy score by making it red and filled
    plt.scatter(best_num_trees, max_accuracy, color='red', label=f'Max Accuracy: {max_accuracy:.2f}%', s=50,
                marker='o', edgecolors='red')
    
    # Annotate the point with the highest accuracy score
    plt.annotate(f'Max Accuracy: {max_accuracy:.2f}%\nTrees: {best_num_trees}',
                xy=(best_num_trees, max_accuracy),
                xytext=(best_num_trees - 10, max_accuracy - 0.5),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=11)
    
    # Format y-axis labels with two decimal places
    plt.gca().set_yticklabels(['{:.2f}%'.format(x) for x in plt.gca().get_yticks()])
    
    # Set the x-axis limits
    plt.xlim(35, 105)
    plt.ylim(94.40, 95.00)  # Adjust the y-axis limits for percentages
    # Show the plot
    plt.show()

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