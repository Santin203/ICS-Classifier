# Introduction to Artificial Intelligence
# Credit Default Dataset
# Logistic regression
# By Juan Carlos Rojas
# Copyright 2024, Texas Tech University - Costa Rica
# Modified by: Santiago Jimenez and William He Yu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics

# Load and prepare the data
df = pd.read_csv("data/train.csv", header=0)
labels = df["Response"]
df = df.drop(columns="Response")
df = df.drop(columns="id")

test_df = pd.read_csv("data/test.csv")
test_ids = test_df['id']
test_df.drop(columns = ["id"], inplace = True)

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

# Feature scale standardization
for col in train_data.columns:
    mean = train_data[col].mean()
    stddev = train_data[col].std()
    train_data[col] = train_data[col] - mean
    train_data[col] = train_data[col]/stddev
    test_data[col] = test_data[col] - mean
    test_data[col] = test_data[col]/stddev
    test_df[col] = test_df[col] - mean
    test_df[col] = test_df[col]/stddev
    
# Select columns of interest (all columns)
cols = train_data.columns

solver = 'newton-cg'
tol = 1e-4
class_weight = 'balanced'

print(f"Solver: {solver}, Tolerance: {tol}, Class weight: {class_weight}")
# Create and train a new logistic regression classifier
model = sklearn.linear_model.LogisticRegression(\
        solver=solver, 
        tol=tol,
        class_weight=class_weight,
        n_jobs=-1)

# Train it with the training data and labels
model.fit(train_data[cols], train_labels)

# Get the prediction probabilities
pred_proba = model.predict_proba(test_data[cols])[:,1]

# Print a few predictions
# print(pred_proba[:5])

# Compute the area under the ROC curve (ROC AUC)
auc_score = sklearn.metrics.roc_auc_score(test_labels, pred_proba)
print("Test AUC score: {:.4f}".format(auc_score))

# Compute ROC AUC against training data
pred_proba_training = model.predict_proba(train_data[cols])[:,1]

auc_score_training = sklearn.metrics.roc_auc_score(\
    train_labels, pred_proba_training)
print("Train AUC score: {:.4f}".format(auc_score_training))


# # Get the prediction probabilities for the test data
# predictions_test = model.predict_proba(test_df)[:,1]


# result = pd.DataFrame({'id' : test_ids, 'Response' : predictions_test.flatten()}, 
#                       columns=['id', 'Response'])

# result.to_csv("data/submission.csv",index=False)