# Introduction to Artificial Intelligence
# Credit Default Dataset
# Gradient Boosting Classifier
# By Juan Carlos Rojas
# Copyright 2024, Texas Tech University - Costa Rica
# Modified by: Santiago Jimenez and Jose Campos

import numpy as np
import pandas as pd
import imblearn
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.ensemble
import sklearn.metrics

# Load and prepare the data
train_df = pd.read_csv("data/train.csv", header=0)
labels = train_df["Response"]
train_df = train_df.drop(columns="Response")
train_df = train_df.drop(columns="id")

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
train_df['Gender'] = train_df['Gender'].replace(categorical_mappings[0])
train_df['Vehicle_Age'] = train_df['Vehicle_Age'].replace(categorical_mappings[1])
train_df['Vehicle_Damage'] = train_df['Vehicle_Damage'].replace(categorical_mappings[2])

# Apply the mappings to the categorical columns
test_df["Gender"] = test_df['Gender'].replace(categorical_mappings[0])
test_df["Vehicle_Age"] = test_df['Vehicle_Age'].replace(categorical_mappings[1])
test_df["Vehicle_Damage"] = test_df['Vehicle_Damage'].replace(categorical_mappings[2])

"""
# Balance classes 
num_ones = np.count_nonzero(train_labels)
num_zeros = len(train_labels) - num_ones

resampler = imblearn.under_sampling.RandomUnderSampler(
        sampling_strategy={0:num_ones, 1:num_ones}, random_state=2024)
# resampler = imblearn.under_sampling.RandomUnderSampler(
#         sampling_strategy={0:(num_ones/3), 1:num_ones}, random_state=2024)
# resampler = imblearn.over_sampling.RandomOverSampler(
#        sampling_strategy={0:num_zeros, 1:num_zeros}, random_state=2024)

train_data, train_labels = resampler.fit_resample(train_data, train_labels)
"""

msl = 10
learning_rate = 0.1
max_iter = 1100
max_depth = 5
l2_regularization = 1
early_stopping = False
class_weight = None

model = sklearn.ensemble.HistGradientBoostingClassifier(
    min_samples_leaf=msl,
    learning_rate=learning_rate,
    max_iter=max_iter,
    max_depth=max_depth,
    l2_regularization=l2_regularization, early_stopping=False, class_weight=class_weight,
    verbose=2)

print("With min_samples_leaf={}, learning_rate={}, max_iter={}, max_depth={}, l2_regularization={}".format(
    msl, learning_rate, max_iter, max_depth, l2_regularization))

model.fit(train_df, labels)
pred_proba = model.predict_proba(test_df)[:, 1]

# # Get the prediction probabilities for the test data
print("Creating submission file")
predictions_test = model.predict_proba(test_df)[:,1]

result = pd.DataFrame({'id' : test_ids, 'Response' : predictions_test.flatten()}, 
                      columns=['id', 'Response'])

result.to_csv("data/submission.csv",index=False)