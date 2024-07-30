# Introduction to Artificial Intelligence
# Project 2: Kaggle competetion
# Ensemble AdaBoost classifier
# By William He Yu, Jose Campos and Santiago Jimenez

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors

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



# Define the hyperparameters for the model
msl = 10
learning_rate = 0.8
max_iter = 200
max_depth = 5

histgboost = sklearn.ensemble.HistGradientBoostingClassifier(
    min_samples_leaf=msl,
    learning_rate=learning_rate,
    max_iter=max_iter,
    max_depth=max_depth,
    verbose=2)


# Create a AdaBoost Classifier of histgboost
model = sklearn.ensemble.AdaBoostClassifier(
    estimator=histgboost,
    n_estimators=500,
    learning_rate=0.8)

# Train it with the training data and labels
model.fit(train_data, train_labels)

# Get prediction probabilities
pred_proba = model.predict_proba(test_data)[:,1]

# Compute the area under the ROC curve (ROC AUC)
auc_score = sklearn.metrics.roc_auc_score(test_labels, pred_proba)
print("Test AUC score: {:.4f}".format(auc_score))

#"""
# Compute a precision & recall graph
precisions, recalls, thresholds = \
    sklearn.metrics.precision_recall_curve(test_labels, pred_proba)
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.legend(loc="center left")
plt.xlabel("Threshold")
plt.axis([0.0, 1.0, 0.0, 1.0])
plt.show()

# Plot a ROC curve (Receiver Operating Characteristic)
fpr, tpr, _ = sklearn.metrics.roc_curve(test_labels, pred_proba)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.show()
#"""


# Get the prediction probabilities for the test data
# print("Creating submission file")
# predictions_test = model.predict_proba(test_df)[:,1]

# result = pd.DataFrame({'id' : test_ids, 'Response' : predictions_test.flatten()}, 
#                       columns=['id', 'Response'])

# result.to_csv("data/submission.csv",index=False)