import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import sys
import os
sys.path.append(os.path.abspath('.'))
import data_gen

# Load your data here
# X_train, y_train, X_val, y_val

def generate_data(N, data_gen):
    X = []
    y = []

    for _ in range(N):
        data = np.hstack(data_gen.generate_data(1000))
        X.append(data)
        y.append(np.mean(data_gen.f(data[:1000], data[1000:])))

    return np.array(X), np.array(y)

# Create training and validation data
N_train = 10000  # Number of training datapoints
N_val = 20  # Number of validation datapoints

X_train, y_train = generate_data(N_train, data_gen)
X_val, y_val = generate_data(N_val, data_gen)

# Set the dimensions of input and output
input_dim = X_train.shape[1]
output_dim = y_train.shape[0]

# Define the MLP model
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(32, activation='relu'),
    Dense(output_dim, activation='linear')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-5), loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_val, y_val))

model.save("models/mlp.h5")
