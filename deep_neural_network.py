# Introduction to Artificial Intelligence
# MNIST dataset
# Deep Neural Network, version 2
# By Juan Carlos Rojas
# Modified by Jose Campos
# Copyright 2024, Texas Tech University - Costa Rica

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import sklearn.preprocessing
import math
import time
import matplotlib.pyplot as plt

#
# Load and prepare data
#

##Load and prepare train data
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
n_nodes = 500
n_hidden_layers = 5
batch_size = math.ceil(nsamples / 10)
n_epochs = 750
eval_step = 10
learning_rate = 1e-4

# Print configuration summary
n_nodes_per_layer = n_nodes // n_hidden_layers
print("Num nodes: {} Num layers: {} Nodes per layer: {}".format(n_nodes, n_hidden_layers, n_nodes_per_layer))
n_batches = math.ceil(nsamples / batch_size)
print("Batch size: {} Num batches: {} Num epochs: {}".format(batch_size, n_batches, n_epochs))

#
# Keras definitions
#

# Create a neural network model
model = tf.keras.models.Sequential()

# Add an Input layer
model.add(tf.keras.layers.Input(shape=(n_inputs,)))

# Hidden layers
for n in range(n_hidden_layers):
    model.add(tf.keras.layers.Dense(
            n_nodes_per_layer,
            activation='elu',
            kernel_initializer='he_normal', bias_initializer='zeros'))
    print("Layer {}: Num nodes={}  Activation=ELU".format(n+1, n_nodes_per_layer))

# Output layer
model.add(tf.keras.layers.Dense(
        1,
        activation='softmax',
        kernel_initializer='glorot_normal', bias_initializer='zeros'))
print("Output Layer: Num nodes={}  Activation=Softmax".format(1))

# Display the network topology
model.summary()

# Define the optimizer

# ADAM optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
print("Optimizer: ADAM.  Learning rate = {}".format(learning_rate))

# Define model
model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )

# Train the neural network
start_time = time.time()
history = model.fit(
        train_data,
        train_labels,
        epochs=n_epochs,
        batch_size=batch_size,
        validation_data=(test_data, test_labels),
        validation_freq=eval_step,
        verbose=2,
        )
elapsed_time = time.time() - start_time
print("Execution time: {:.1f}".format(elapsed_time))

cost_test, acc_test = model.evaluate(test_data, test_labels, batch_size=None, verbose=0)
cost_train, acc_train = model.evaluate(train_data, train_labels, batch_size=None, verbose=0)

print("Final Test Accuracy:     {:.4f}".format(acc_test))
print("Final Training Cost:     {:.8f}".format(cost_train))

# Compute the best test result from the history
epoch_hist = [i for i in range(0, n_epochs, eval_step)]
test_acc_hist = history.history['val_accuracy']
test_best_val = max(test_acc_hist)
test_best_idx = test_acc_hist.index(test_best_val)
print("Best Test Accuracy:      {:.4f} at epoch: {}".format(test_best_val, epoch_hist[test_best_idx]))

#Predict responses in test data set
prediction_test = model.predict(test_df, verbose = "auto", steps = None , callbacks = None)

result = pd.DataFrame({'id' : test_ids, 'Response' : prediction_test.flatten()}, 
                      columns=['id', 'Response'])

result.to_csv("data/submission.csv",index=False)

# Plot the history of the loss
plt.plot(history.history['loss'])
plt.title('Training Cost')
plt.ylabel('Cost')
plt.xlabel('Epoch')

# Plot the history of the test accuracy
plt.figure()
plt.plot(epoch_hist, history.history['val_accuracy'], "r")
plt.title('Test Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
