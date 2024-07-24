# Introduction to Artificial Intelligence
# Credit Default Dataset
# Ensemble classifier
# By Juan Carlos Rojas
# Copyright 2024, Texas Tech University - Costa Rica
# Modified by: William He Yu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model
import sklearn.ensemble
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


#
# Create and train classifier
#

# HistGradientBoosting classifier
msl = 10
learning_rate = 0.3
max_iter = 600
max_depth = 5
l2_regularization = 1.0
early_stopping = False

histgboost = sklearn.ensemble.HistGradientBoostingClassifier(
    min_samples_leaf=msl,
    learning_rate=learning_rate,
    max_iter=max_iter,
    max_depth=max_depth,
    l2_regularization=l2_regularization,
    verbose=2)


n_est = 100
msl = 20
max_features = 'log2'
class_weight = 'balanced'

rf = sklearn.ensemble.RandomForestClassifier(\
    n_estimators=n_est,
    min_samples_leaf=msl,
    max_features=max_features,
    class_weight= class_weight,
    n_jobs=-1)

# Create a voting ensemble of classifiers
model = sklearn.ensemble.AdaBoostClassifier(
    estimator=rf,
    n_estimators=2)

# Train it with the training data and labels
model.fit(train_data, train_labels)

# Get prediction probabilities
pred_proba = model.predict_proba(test_data)[:,1]

# Compute the area under the ROC curve (ROC AUC)
auc_score = sklearn.metrics.roc_auc_score(test_labels, pred_proba)
print("Test AUC score: {:.4f}".format(auc_score))

# Get the prediction probabilities for the test data
print("Creating submission file")
predictions_test = model.predict_proba(test_df)[:,1]

result = pd.DataFrame({'id' : test_ids, 'Response' : predictions_test.flatten()}, 
                      columns=['id', 'Response'])

result.to_csv("data/submission.csv",index=False)