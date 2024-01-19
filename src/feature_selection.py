import numpy as np
import pickle
from .ecg_types import BeatType
from sklearn.feature_selection import mutual_info_classif

# This file is dedicated to the feature selection/ranking process and constructing the training and testing features datasets.
# Feature ranking/selection is a process used to select a subset of relevant features for use in model construction. 
# It involves ranking all available features by some criterion before training the model.

# Mutual information (MI) between each feature and the class labels is used here.
# Mutual Information measures the amount of information gained about one random variable through observing another random variable. 
# In this case, it quantifies the amount of information each feature provides about the heartbeat type classes. 
# The higher the MI score, the more relevant the feature is in predicting the class.
# MI measures how much information the presence/absence of a feature contributes to making the correct prediction on the class.
# The aim is to reduce the number of input variables to those that are believed to be most useful to predict the target variable. 
# This can improve model accuracy, reduce overfitting, and decrease computational cost.
# Feature ranking/selection is typically done before training the model.


# Function to construct the features vector, feature names, labels, and source information for each beat from the beats dictionary:
def construct_vectors(beats):
    
    # List of all features names:
    # This list is compiled from the keys of various feature dictionaries present in the first beat of the `beats` array.
    features_names = list(beats[0]["rr"].keys())
    features_names += list(beats[0]["morph"].keys())
    features_names += ["wt_coef_" + str(i) for i, x in enumerate(beats[0]["wt"])]
    features_names += ["hos_" + str(i) for i, x in enumerate(beats[0]["hos"])]
    #features_names += ["mg_" + str(i) for i, x in enumerate(beats[0]["mg"])]
    features_names += ["hbf_" + str(i) for i, x in enumerate(beats[0]["hbf"])]
    features_names += ["lbp_" + str(i) for i, x in enumerate(beats[0]["lbp"])]

    features_names = np.array(features_names)

    # Vector to store the feature vectors:
    features_vector = np.empty((len(beats), len(features_names)))

    # Array to store the labels (i.e. tyep of beat):
    labels = np.empty((len(beats)), dtype=int)
    
    # List of sources (i.e. the file from which the beat was extracted):
    sources = list(range(len(beats)))

    # Iterate over each beat in the `beats` array and construct the feature vector:
    # This process results in a feature matrix where each row is a feature vector corresponding to a beat, and each column represents a specific feature.
    for beatIndex, beat in enumerate(beats):
        labels[beatIndex] = beat["beatType"].value # The label (i.e. beat type) is also stored in its respective array
        sources[beatIndex] = beat["source"] # The source is also stored in its respective array
        beat_features = list(beat["rr"].values())
        beat_features += list(beat["morph"].values())
        beat_features += list(beat["wt"])
        beat_features += list(beat["hos"])
        #beat_features += list(beat["mg"])
        beat_features += list(beat["hbf"])
        beat_features += list(beat["lbp"])

        features_vector[beatIndex] = np.array(beat_features)
        
    # Return the matrix containing the feature vectors for all beats,
    # the names of the features corresponding to the columns of the feature matrix,
    # the array with the labels for each beat, and the list of sources for each beat.
    return features_vector, features_names, labels, sources




# Function to rank features based on their mutual information scores and construct the training features dataset:
def construct_training_features_dataset(heartbeats_path, training_path):
    print("\nConstructing train set features vector...")
    pickle_in = open(heartbeats_path + "training_dataset_heartbeats.pickle", "rb")
    data = pickle.load(pickle_in)
    pickle_in.close()

    # Calling function to construct feature vectors, feature names, labels, and source information for each beat in the training dataset:
    train_features, train_features_names, train_labels, train_sources = construct_vectors(data["beats"])

    print("\nEstimating the most informative features for S and V from training dataset...")

    # Converting the `train_labels` NumPy array to a Python list for easier processing:
    labels_L = train_labels.tolist()

    # Array containing indices of Normal (N) beats in the training dataset:
    N_ind = np.array([k for k, l in enumerate(labels_L) if l == BeatType.NORMAL.value], dtype=int)

    # Array containing indices of Supraventricular ectopic beat (S) beats in the training dataset:
    S_ind = np.array(
        [
            k
            for k, l in enumerate(labels_L)
            if l == BeatType.AURICULAR_PREMATURE_CONTRACTION.value
        ],
        dtype=int,
    )

    # Array containing indices of Ventricular ectopic beat (V) beats in the training dataset:
    V_ind = np.array(
        [
            k
            for k, l in enumerate(labels_L)
            if l == BeatType.PREMATURE_VENTRICULAR_CONTRACTION.value
        ],
        dtype=int,
    )

    # Combining the indices for the specified beat types (i.e. N, S, V) into a single array:
    indixes = np.block([N_ind, S_ind, V_ind])

    # Filtering the labels for the specified beat types:
    labels_SV = train_labels[indixes]

    # Filtering the features for the specified beat types:
    features_SV = train_features[indixes]

    # Computes the mutual information of features with respect to the selected labels:
    # The `random_state` is set for reproducibility.
    # The `random_state` parameter is used in functions that have a random component to them, 
    # ensuring that the same sequence of random numbers is generated each time the code is run. 
    # This is crucial for reproducibility, particularly in scenarios where the results need to be demonstrable 
    # and consistent across different runs or for different users.
    mi_features = mutual_info_classif(features_SV, labels_SV, random_state=42)

    # Sorting the features based on their mutual information scores in descending order:
    mi_rank = np.argsort(mi_features)[-1:0:-1]

    # Retrieving the names of the higher ranked features:
    ranked_features_names = train_features_names[mi_rank]

    print("\nMI ranked features: " + str(ranked_features_names))
    print("\nMI of ranked features: " + str(mi_features[mi_rank]))

    # Saving the ranked features and their mutual information scores to a pickle file:
    print("\nSaving features rank file...")
    pickle_out = open(training_path + "training_dataset_mi_ranked_features.pickle", "wb")
    pickle.dump(
        {
            "ranked_features_names": ranked_features_names,
            "mi_ranked_features": mi_features[mi_rank],
            "mi_rank": mi_rank,
        },
        pickle_out,
    )
    pickle_out.close()

    # Saving the entire set of training features, feature names, labels, sources, ranked features, and ranked feature names to a pickle file:
    print("\nSaving train set file...")
    pickle_out = open(training_path + "training_dataset_features.pickle", "wb")
    pickle.dump(
        {
            "features": train_features,
            "features_names": train_features_names,
            "labels": train_labels,
            "sources": train_sources,
            "ranked_features": train_features[:, mi_rank],
            "ranked_features_names": ranked_features_names,
        },
        pickle_out,
    )
    pickle_out.close()
    
    # Returning the top ranked features and their names:
    # The same feature ranking obtained here will also be used for the testing dataset in order to ensure consistency.
    return mi_rank, ranked_features_names
    



# Function to rank features based on their mutual information scores and construct the training features dataset:
def construct_testing_features_dataset(heartbeats_path, testing_path, mi_rank, ranked_features_names):
    
    print("\nConstructing test set features vector...")
    pickle_in = open(heartbeats_path + "testing_dataset_heartbeats.pickle", "rb")
    data = pickle.load(pickle_in)
    pickle_in.close()
    
    # Calling function to construct feature vectors, feature names, labels, and source information for each beat in the testing dataset:
    test_features, test_features_names, test_labels, test_sources = construct_vectors(data["beats"])

    # Saving the entire set of testing features, feature names, labels, sources, ranked features, and ranked feature names to a pickle file:
    print("\nSaving test set file...")
    pickle_out = open(testing_path + "testing_dataset_features.pickle", "wb")
    pickle.dump(
        {
            "features": test_features,
            "features_names": test_features_names,
            "labels": test_labels,
            "sources": test_sources,
            "ranked_features": test_features[:, mi_rank],
            "ranked_features_names": ranked_features_names,
        },
        pickle_out,
    )
    pickle_out.close()




# Main function to orchestrate the feature selection process and construct the training and testing features datasets:
def rank_features_and_construct_features_datasets(heartbeats_path, training_path, testing_path):
    
    # Calling function to rank features based on their mutual information scores and construct the training features dataset:
    mi_rank, ranked_features_names = construct_training_features_dataset(heartbeats_path, training_path)
    
    # Calling function to construct the testing features dataset:
    # The top ranked features and their names are passed as arguments to this function.
    # The same feature ranking obtained from the training dataset is also used here in order to ensure consistency.
    construct_testing_features_dataset(heartbeats_path, testing_path, mi_rank, ranked_features_names)