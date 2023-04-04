import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import concurrent.futures



import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./deep-learn'))
from data_gen import load_cdf_data


X_train, y_train = load_cdf_data("data/X_train.npy", "data/y_train.npy")
X_val, y_val = load_cdf_data("data/X_val.npy", "data/y_val.npy")

input_dim = X_train.shape[1]
output_dim=y_train.shape[2]

print(input_dim)
print(output_dim)
# Update input_shape for the 1D CNN model
input_shape = (input_dim, 1)

# Reshape the input data
X_train_cnn = X_train.reshape(-1, input_dim, 1)
X_val_cnn = X_val.reshape(-1, input_dim, 1)

# Define the 1D CNN model
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
    Conv1D(64, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(output_dim, activation='linear')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
history = model.fit(X_train_cnn, y_train, batch_size=32, epochs=10, validation_data=(X_val_cnn, y_val))

model.save("models/cnn_1d.h5")