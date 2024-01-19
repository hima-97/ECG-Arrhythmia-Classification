import os
import numpy as np
import pandas as pd
import pickle
from .ecg_types import BeatType
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_recall_fscore_support


# This file is dedicated to training the Random Forest classifier and evaluating its performance.


# Number of top ranked features to consider:
NUMBER_OF_FEATURES = 10
# Number of trees to use for the Random Forest classifier:
NUMBER_OF_TREES = 84
# Initializing a standard scaler object for normalizing the feature data:
SCALER = StandardScaler()

# List of output labels, representing the different types of heartbeats (i.e. classes): 
# These are the classes that the classifier will predict.
CLASS_LABELS = [
    BeatType.NORMAL.symbol(), 
    BeatType.AURICULAR_PREMATURE_CONTRACTION.symbol(),
    BeatType.PREMATURE_VENTRICULAR_CONTRACTION.symbol(),
    BeatType.FUSION.symbol(),
    BeatType.UNKNOWN.symbol()
    ]




# Function to load dataset:
def load_dataset(path):
    
    # Opening and loading pickle file with features:
    with open(path, "rb") as file:
        data = pickle.load(file)
    
    # Extracting the labels from the loaded dataset:
    labels = data['labels']
    
    # Extracting the top `NUMBER_OF_FEATURES` features from the training ranked features dataset:
    features = data['ranked_features'][:, :NUMBER_OF_FEATURES]
    
    # Converting the list of sources (identifying the origin of each data point) into a NumPy array:
    sources = np.array(data['sources'])
    
    # Returning the features, labels, and sources of the features dataset:
    return features, labels, sources




# Function to verify the order of features
def verify_feature_order(train_set_features, train_set_ranked_feature_names):
    # Assuming train_set_features is a NumPy array, and the features are ordered in columns
    # We can't directly compare the column names (since it's an array), but we can ensure the number of features matches
    if train_set_features.shape[1] != len(train_set_ranked_feature_names):
        print("\nWarning: The number of features in the training set does not match the number of ranked feature names.")
    else:
        print("\nFeature number is consistent with ranked feature names.")





# Function to train the Random Forest classifier:
def train_model(train_set_features, train_set_labels, feature_names):
    
    # The commented line below should scale the training features using the standard scaler:
    # However, the results might be better without scaling the data.
    # Therefore, we must test both approaches and see what's best.
    #print('\nScaling training features...')
    #train_set_features = SCALER.fit_transform(train_set_features)
    
    print('\nTraining model...')
    # Initializing a Random Forest classifier object:
    # The `random_state` is set for reproducibility.
    # The `random_state` parameter is used in functions that have a random component to them, 
    # ensuring that the same sequence of random numbers is generated each time the code is run. 
    # This is crucial for reproducibility, particularly in scenarios where the results need to be demonstrable 
    # and consistent across different runs or for different users.
    rf_classifier = RandomForestClassifier(random_state=42, n_estimators=NUMBER_OF_TREES)
    
    # Training the Random Forest classifier using the training features and labels:
    rf_classifier.fit(train_set_features, train_set_labels)
    
    # Retrieve feature importance scores:
    # Feature importance gives you a score for each feature of your data.
    # The higher the score more important or relevant is the feature towards your output variable.
    # Feature importance is generally evaluated after the model has been trained. 
    # Itâ€™s an intrinsic model property for tree-based models like Random Forests and Gradient Boosting Machines.
    feature_importances = rf_classifier.feature_importances_

    # Ensure that the number of feature names matches the number of feature importances
    assert len(feature_names) == len(feature_importances), "Mismatch in feature names and importances"
    
    # Pair the feature names with their importance scores:
    feature_importance_pairs = zip(feature_names, feature_importances)

    # Print the names of features and their importance scores:
    print('\nFeature Importances:')
    for name, importance in feature_importance_pairs:
        print(f'{name}: {importance}')
    

    # Returning the trained Random Forest classifier:
    return rf_classifier




# Function to calculate various quality parameters or performance metrics for a classifier, based on its confusion matrix:
def evaluate_classifier(conf_matrix, output_class_labels):
    
    print('\nCalculating quality parameters...')

    # List of quality measures:
    # The list includes the following measures: Sensitivity, Specificity, Positive Predictivity value, False Positive Rate, Accuracy, and F1 Score.
    qualityMeasures = ['Se', 'Sp', 'Pp', 'FPR', 'Ac', 'F1']
    
    # Initializing an empty NumPy array to store the calculated quality measures. 
    # The array's shape is determined by the number of quality measures and the number of output classes.
    Q = np.empty((len(qualityMeasures), len(output_class_labels)))
    
    # Initializing a small value to avoid division by zero:
    epsilon = 1e-7

    
    # Loop to calculate the quality measures for each output class:
    for k, label in enumerate(output_class_labels):
        
        # Calculates the number of true positives (tp) for class `k`: 
        # It's the diagonal element of the confusion matrix for that class.
        tp = conf_matrix[k,k]
        
        # Calculates the number of false negatives (fn) for class `k`:
        fn = np.sum(conf_matrix[k,:]) - tp
        
        # Calculates the number of true negatives (tn) for class `k`:
        tn = np.sum(conf_matrix) - np.sum(conf_matrix[k,:])
        
        # Calculates the number of false positives (fp) for class `k`:
        fp = np.sum(conf_matrix[:,k]) - tp
        
        # Calculating the different quality measures (Sensitivity, Specificity, etc.) based on these values:
        Q[0, k] = tp / (tp + fn) # Calculates Sensitivity for class `k`
        Q[1, k] = tn / (tn + fp)
        Q[2, k] = tp / (tp + fp + epsilon)  # Adjusted to avoid division by zero
        Q[3, k] = fp / (tp + fn)
        Q[4, k] = (tp + tn) / (tp + tn + fp + fn)
        Q[5, k] = 2 * (Q[2, k] * Q[0, k]) / (Q[2, k] + Q[0, k] + epsilon) # Calculates F1 Score for class `k`.
                                                                          # Adjusted to avoid division by zero
    
    # Returning a Pandas DataFrame containing the calculated quality measures:
    # The rows are labeled with the names of the quality measures, and the columns are labeled with the output class labels.
    return pd.DataFrame(Q, columns=output_class_labels, index=qualityMeasures)




# Function to evaluate the trained Random Forest classifier:
# The trained Random Forest classifier is used to predict the labels of the test set features.
def evaluate_model(rf_classifier, test_features, test_set_labels):
    
    # The commented line below should scale the training features using the standard scaler:
    # However, the results might be better without scaling the data.
    # Therefore, we must test both approaches and see what's best.
    #print('\nScaling testing features...')
    #test_features = SCALER.fit_transform(test_features)
    
    print('\nValidating model...')
    test_predictions = rf_classifier.predict(test_features)
    
    # Calculating the confusion matrix for the testing set predictions:
    conf_matrix = confusion_matrix(test_set_labels, test_predictions)
    print('\nConfusion_matrix:\n', conf_matrix)
    
    # Calculating the accuracy of the model on the testing data:
    accuracy = accuracy_score(test_set_labels, test_predictions)
    print('\nAccuracy:', accuracy)
    
    # Generating and printing classification report:
    # `digits=4` specifies the number of digits after the decimal point.
    print('\nClassification report:\n', classification_report(test_set_labels, test_predictions, target_names=CLASS_LABELS, digits=4, zero_division=1))
    
    
    # Calling function to calculate the quality measures for the testing set predictions, based on the confusion matrix:
    evaluation_test = evaluate_classifier(conf_matrix, CLASS_LABELS)
    print(f'\nEvaluation details with {NUMBER_OF_FEATURES} features, and {NUMBER_OF_TREES} trees:\n')
    print(evaluation_test)
    
    # Returning the confusion matrix, the accuracy, and the evaluation test of the model on the testing data:
    return conf_matrix, accuracy, evaluation_test




# Defining a custom function for leave-one-out cross-validation:
# This function will be used to generate train-test splits for cross-validation.
def leave_one_out_cv(train_set_sources):
    
    print('\nCross-validating model...')
    
    # List of unique sources from the training set:
    # This is used to ensure that each cross-validation fold leaves out data from one source at a time.
    set_sources = list(set(train_set_sources))
    
    # For each source `s` in `set_sources`, the indices of the training set are split into `test_indexes` and `train_indexes`: 
    # If the source of a data point matches `s`, it's included in `test_indexes`,
    # otherwise, it's added to `train_indexes`.
    for s in set_sources:
        test_indexes = []
        train_indexes = []
        for i, ts in enumerate(train_set_sources):
            if ts == s:
                test_indexes.append(i)
            else:
                train_indexes.append(i)
        
        # This returns a generator that produces pairs of indices for training and testing: 
        # This is used in the cross-validation.
        yield train_indexes, test_indexes
    
    


# Function to save the trained model to a pickle file:
def save_trained_rf_model(rf_classifier, evaluation_test, output_directory):
    
    # Defining the path and the name of the output pickle file:
    output_path = os.path.join(output_directory, 'heartbeats_rf_classifier.pickle')
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    # Saving the trained model to a pickle file:
    print('\nSaving model...')
    with open(output_path, "wb") as file:
        pickle.dump({'preprocessor': None, 'model': rf_classifier, 'Evaluation_test': evaluation_test}, file)
        print('\nModel saved successfully.')

        
        

# Function to orchestrate the training and testing process:
def train_and_test_model(training_path, testing_path, classifier_path):
    
    # Load datasets:
    train_features, train_labels, train_sources = load_dataset(training_path + "training_dataset_features.pickle")
    test_features, test_labels, test_sources = load_dataset(testing_path + "testing_dataset_features.pickle")

    # Extracting the names of the top-ranked features:
    with open(training_path + "training_dataset_mi_ranked_features.pickle", "rb") as file:
        mi_data = pickle.load(file)
    top_feature_names = mi_data["ranked_features_names"][:NUMBER_OF_FEATURES]

    # Verify the order of features before training the model
    verify_feature_order(train_features, top_feature_names)

    # Train model with top feature names:
    rf_classifier = train_model(train_features, train_labels, top_feature_names)

    # Using the Random Forest classifier to perform predictions on the training set using the custom cross-validation strategy:
    # This function will generate predictions for each part of the data when it is in the test fold.
    train_predictions = cross_val_predict(rf_classifier, train_features, train_labels, cv=leave_one_out_cv(train_sources))

    # Calculating the accuracy of the model on the training data based on these cross-validated predictions:
    train_accuracy = accuracy_score(train_labels, train_predictions)
    print(f'\nTraining accuracy with {NUMBER_OF_FEATURES} features, and {NUMBER_OF_TREES} trees:')
    print(train_accuracy)

    # Evaluating the model on the testing set:
    conf_matrix, test_accuracy, evaluation_test = evaluate_model(rf_classifier, test_features, test_labels)
    
    # Saving the trained Random Forest classifier:
    save_trained_rf_model(rf_classifier, evaluation_test, classifier_path)