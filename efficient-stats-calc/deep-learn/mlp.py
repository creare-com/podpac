import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import sys
import os
sys.path.append(os.path.abspath('.'))
from data_gen import load_cdf_data


# Example usage:
X_train, y_train = load_cdf_data("data/X_train.npy", "data/y_train.npy")
X_val, y_val = load_cdf_data("data/X_val.npy", "data/y_val.npy")

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)


# Set the dimensions of input and output
input_dim = X_train.shape[1]
output_dim = y_train.shape[2]

print("Input Dim", input_dim)
print("Output Dim:", output_dim )

# Define the MLP model
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(32, activation='relu'),
    Dense(output_dim, activation='linear')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-5), loss=tf.keras.losses.Huber(delta=1.0), metrics=['mean_absolute_error'])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_val, y_val))

print(model.predict(np.expand_dims(X_train[0], axis=0)))
print(y_train[0])

model.save("models/mlp.h5")
