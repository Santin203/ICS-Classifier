# Introduction to Artificial Intelligence
# Credit Default Dataset
# k-Nearest Neighbors Classifier
# By Juan Carlos Rojas
# Copyright 2024, Texas Tech University - Costa Rica
# modified by: William He Yu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.neighbors
import sklearn.metrics
import imblearn

# Load and prepare the data
df = pd.read_csv("data/train.csv", header=0)
labels = df["Response"]
df = df.drop(columns="Response")
df = df.drop(columns="id")

test_df = pd.read_csv("data/test.csv")
test_ids = test_df['id']
test_df.drop(columns=["id"], inplace=True)

# Define the categorical mappings
categorical_mappings = [
    {'Female': 0, 'Male': 1},
    {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2, 'Unknown': -1},
    {'No': 0, 'Yes': 1, 'Unknown': -1}
]

# Apply the mappings to the categorical columns
df['Gender'] = df['Gender'].replace(categorical_mappings[0])
df['Vehicle_Age'] = df['Vehicle_Age'].replace(categorical_mappings[1])
df['Vehicle_Damage'] = df['Vehicle_Damage'].replace(categorical_mappings[2])

# Apply the mappings to the categorical columns
test_df["Gender"] = test_df['Gender'].replace(categorical_mappings[0])
test_df["Vehicle_Age"] = test_df['Vehicle_Age'].replace(categorical_mappings[1])
test_df["Vehicle_Damage"] = test_df['Vehicle_Damage'].replace(categorical_mappings[2])

train_data, test_data, train_labels, test_labels = \
            sklearn.model_selection.train_test_split(df, labels,
            test_size=0.2, shuffle=True, random_state=2024)


# Balance classes 
num_ones = np.count_nonzero(train_labels)
num_zeros = len(train_labels) - num_ones

# resampler = imblearn.under_sampling.RandomUnderSampler(
#         sampling_strategy={0:num_ones, 1:num_ones}, random_state=2024)
# resampler = imblearn.under_sampling.RandomUnderSampler(
#         sampling_strategy={0:(num_ones), 1:num_ones}, random_state=2024)
# resampler = imblearn.over_sampling.RandomOverSampler(
#        sampling_strategy={0:num_zeros, 1:num_zeros}, random_state=2024)

# train_data, train_labels = resampler.fit_resample(train_data, train_labels)


#
# Create and train classifier
#

print("Training kNN Classifier")

n=100
model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
model.fit(train_data, train_labels)

print("Predicting Probabilities against Test Data");
pred_proba = model.predict_proba(test_data)[:,1]

#
# Performance Metrics
#


# Compute the area under the ROC curve (ROC AUC)
auc_score = sklearn.metrics.roc_auc_score(test_labels, pred_proba)
print("With n={}, Test AUC score: {:.4f}".format(n, auc_score))

# # Get the prediction probabilities for the test data
# predictions_test = model.predict_proba(test_df)[:, 1]

# result = pd.DataFrame({'id': test_ids, 'Response': predictions_test.flatten()},
#                       columns=['id', 'Response'])

# result.to_csv("data/submission.csv", index=False)
