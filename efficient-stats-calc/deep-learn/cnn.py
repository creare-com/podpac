import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


import sys
import os
sys.path.append(os.path.abspath('.'))
import data_gen

# Load your data here
# X_train, y_train, X_val, y_val

# Ensure your input data is in the correct format: (samples, height, width, channels)
# For grayscale grid data, channels = 1
# For RGB grid data, channels = 3

def generate_data_cnn(N, data_gen, width, height):
    X = []
    y = []

    for _ in range(N):
        x_data, y_data = data_gen.generate_data(width * height)
        x_data = x_data.reshape(height, width)
        y_data = y_data.reshape(height, width)
        data = np.stack([x_data, y_data], axis=-1)  # shape: (height, width, 2)
        X.append(data)
        y.append(np.mean(data_gen.f(x_data, y_data)))

    return np.array(X), np.array(y)

# Create training and validation data
N_train = 100  # Number of training datapoints
N_val = 20  # Number of validation datapoints
width, height = 32, 32  # Dimensions of the square images

X_train, y_train = generate_data_cnn(N_train, data_gen, width, height)
X_val, y_val = generate_data_cnn(N_val, data_gen, width, height)

# Set the dimensions of input and output
input_shape = X_train.shape[1:]
output_dim = y_train.shape[0]

# Define the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(output_dim, activation='linear')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_val, y_val))

model.save("models/cnn.h5")