# Introduction to Artificial Intelligence
# Credit Default Dataset
# Shallow Neural Network using Keras, version 1
# By Juan Carlos Rojas
# Copyright 2024, Texas Tech University - Costa Rica
# Modified by: Santiago Jimenez


import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.metrics
import imblearn
import time
import math
import matplotlib.pyplot as plt

#
# Load and prepare data
#

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
    
# Balance classes
# num_ones = np.count_nonzero(train_labels)
# num_zeros = len(train_labels) - np.count_nonzero(train_labels)
# sm = imblearn.over_sampling.RandomOverSampler(sampling_strategy={0:num_zeros, 1:num_zeros})
# train_data, train_labels = sm.fit_resample(train_data, train_labels)
# print("Balancing classes with Upsampling")

# Undersampling
# num_ones = np.count_nonzero(train_labels)
# num_zeros = len(train_labels) - np.count_nonzero(train_labels)
# max_samples = min(num_zeros, num_ones)
# sm = imblearn.under_sampling.RandomUnderSampler(sampling_strategy={0:max_samples, 1:max_samples})
# train_data, train_labels = sm.fit_resample(train_data, train_labels)
# print("Balancing classes with Undersampling")

# Get some lengths
n_inputs = train_data.shape[1]
nsamples = train_data.shape[0]

# Training constants
n_nodes = 1000
#batch_size = math.ceil(nsamples / 10)
batch_size = 1024
n_epochs = 100
eval_step = 5
learning_rate = 1e-3
n_hidden_layers = 7
#dropout_rate = 0.1

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
            activation='relu',
            kernel_initializer='he_normal', bias_initializer='zeros'))
    #model.add(tf.keras.layers.Dropout(dropout_rate))
    print("Layer {}: Num nodes={}  Activation=RELU".format(n+1, n_nodes_per_layer))

# Output layer
model.add(tf.keras.layers.Dense(
        1,
        activation='sigmoid',
        kernel_initializer='glorot_normal', bias_initializer='zeros'))
print("Output Layer: Num nodes={}  Activation=Sigmoid".format(1))

# Display the network topology
model.summary()

# Define the optimizer

# ADAM optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
print("Optimizer: ADAM.  Learning rate = {}".format(learning_rate))

# Define model
model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['AUC']    
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

cost_test, auc_test = model.evaluate(test_data, test_labels, batch_size=None, verbose=0)
cost_train, auc_train = model.evaluate(train_data, train_labels, batch_size=None, verbose=0)

print("Final Test AUC:          {:.4f}".format(auc_test))
print("Final Training Cost:     {:.4f}".format(cost_train))

try:
    # Compute the best test result from the history
    epoch_hist = [i for i in range(0, n_epochs, eval_step)]
    test_auc_hist = history.history['val_AUC']
    test_best_val = max(test_auc_hist)
    test_best_idx = test_auc_hist.index(test_best_val)
    test_best_idx = int(test_best_idx / 2)
    print("Best Test AUC:           {:.4f} at epoch: {}".format(test_best_val, epoch_hist[test_best_idx]))
except Exception as e:
    print(e)
    pass

#Predict responses in test data set
prediction_test = model.predict(test_df, verbose = "auto", steps = None , callbacks = None)

result = pd.DataFrame({'id' : test_ids, 'Response' : prediction_test.flatten()}, 
                      columns=['id', 'Response'])

result.to_csv("data/submission.csv",index=False)

