# Introduction to Artificial Intelligence
# Credit Default Dataset
# Ensemble classifier
# By Juan Carlos Rojas
# Copyright 2024, Texas Tech University - Costa Rica

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

# kNN classifier
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=300)

# Logistic regression classifier
logistic = sklearn.linear_model.LogisticRegression(\
        solver='newton-cg', tol=1e-6)

# Create a Random Forest classifier
randforest = sklearn.ensemble.RandomForestClassifier(\
    n_estimators=50,
    min_samples_leaf=0.001)

# Gradient Boosting classifier
gboost = sklearn.ensemble.GradientBoostingClassifier(\
    loss='log_loss', subsample=0.2,
    n_estimators=100, min_samples_leaf=0.001)


# Create a voting ensemble of classifiers
model = sklearn.ensemble.VotingClassifier(
    estimators=[('knn', knn),
                ('logistic', logistic),
                ('randforest', randforest),
                ('gboost', gboost),
                ],voting='soft')

# Train it with the training data and labels
model.fit(train_data, train_labels)

# Get prediction probabilities
pred_proba = model.predict_proba(test_data)[:,1]

#
# Performance Metrics
#

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

# Compute the area under the ROC curve (ROC AUC)
auc_score = sklearn.metrics.roc_auc_score(test_labels, pred_proba)
print("Test AUC score: {:.4f}".format(auc_score))

# Compute ROC AUC against training data
pred_proba_training = model.predict_proba(train_data)[:,1]
auc_score_training = sklearn.metrics.roc_auc_score(\
    train_labels, pred_proba_training)
print("Train AUC score: {:.4f}".format(auc_score_training))
