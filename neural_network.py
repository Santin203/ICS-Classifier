# Introduction to Artificial Intelligence
# Based on code made by Juan Carlos Rojas
# Adapted by Jose Campos
# Copyright 2024, Texas Tech University - Costa Rica


import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.metrics
import imblearn
import math
import matplotlib.pyplot as plt

#Load and prepare train data
train_df = pd.read_csv("data/train.csv", header=0)
labels = train_df["Response"]
train_df = train_df.drop(columns="Response")
train_df = train_df.drop(columns="id")

#Load and prepare test data
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
train_df['Gender'] = train_df['Gender'].replace(categorical_mappings[0])
train_df['Vehicle_Age'] = train_df['Vehicle_Age'].replace(categorical_mappings[1])
train_df['Vehicle_Damage'] = train_df['Vehicle_Damage'].replace(categorical_mappings[2])

# Apply the mappings to the categorical columns
test_df["Gender"] = test_df['Gender'].replace(categorical_mappings[0])
test_df["Vehicle_Age"] = test_df['Vehicle_Age'].replace(categorical_mappings[1])
test_df["Vehicle_Damage"] = test_df['Vehicle_Damage'].replace(categorical_mappings[2])

#Use part of the training data for supervised training and testing
#Test data does not have labels
train_data, test_data, train_labels, test_labels = \
            sklearn.model_selection.train_test_split(train_df, labels,
            test_size=0.2, shuffle=True, random_state=2024)

# Get some lengths
n_inputs = train_data.shape[1]
nsamples = train_data.shape[0]

# Training constants
n_nodes_l1 = 500
batch_size = math.ceil(nsamples / 10)
n_epochs = 750
eval_step = 5
learning_rate = 1e-4

n_batches = math.ceil(nsamples / batch_size)

#
# Keras definitions
#

# Create a neural network model
model = tf.keras.models.Sequential()

# Add an Input layer
model.add(tf.keras.layers.Input(shape=(n_inputs,)))

# Hidden layer 1:
#   Inputs: n_inputs
#   Outputs: n_nodes_l1
#   Activation: ELU
model.add(tf.keras.layers.Dense( n_nodes_l1,
        activation='elu',
        kernel_initializer='he_normal', bias_initializer='zeros'))


# Output layer:
#   Inputs: n_nodes_l1
#   Outputs: 1
#   Activation: sigmoid
model.add(tf.keras.layers.Dense(1,
        activation='sigmoid',
        kernel_initializer='glorot_normal', bias_initializer='zeros'))

# Define the optimizer

# ADAM optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Define model
model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['AUC']    
        )

# Train the neural network
history = model.fit(
        train_data,
        train_labels,
        epochs=n_epochs,
        batch_size=batch_size,
        validation_data=(test_data, test_labels),
        validation_freq=eval_step,
        verbose=2,
        )


cost_test, auc_test = model.evaluate(test_data, test_labels, batch_size=None, verbose=0)
cost_train, auc_train = model.evaluate(train_data, train_labels, batch_size=None, verbose=0)

# Print the configuration
print("Batch size: {} Num batches: {} Num epochs: {}".format(batch_size, n_batches, n_epochs))

print("Layer 1: Num nodes={}  Activation=ELU".format(n_nodes_l1))

print("Layer 2: Num nodes={}  Activation=Sigmoid".format(1))

print("Optimizer: ADAM.  Learning rate = {}".format(learning_rate))

print("Final Test AUC:          {:.4f}".format(auc_test))
print("Final Training Cost:     {:.4f}".format(cost_train))

# Compute the best test result from the history
epoch_hist = [i for i in range(0, n_epochs, eval_step)]
test_auc_hist = history.history['val_AUC']
test_best_val = max(test_auc_hist)
test_best_idx = test_auc_hist.index(test_best_val)
print("Best Test AUC:           {:.4f} at epoch: {}".format(test_best_val, epoch_hist[test_best_idx]))

#Predict responses in test data set
model.predict(test_df, verbose = "auto", steps = None , callbacks = None)

# Plot the history of the loss
plt.plot(history.history['loss'])
plt.title('Training Cost')
plt.ylabel('Cost')
plt.xlabel('Epoch')

# Plot the history of the test accuracy
plt.figure()
plt.plot(epoch_hist, history.history['val_AUC'], "r")
plt.title('Test AUC')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.show()
