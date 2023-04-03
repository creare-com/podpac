import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import sys
import os
sys.path.append(os.path.abspath('.'))
import data_gen

from scipy.stats import gaussian_kde


# Load your data here
# X_train, y_train, X_val, y_val

# def generate_data(N, data_gen):
#     X = []
#     y = []

#     for _ in range(N):
#         data = np.hstack(data_gen.generate_data(1024))
#         X.append(data)
#         y.append(np.mean(data_gen.f(data[:1024], data[1024:])))

#     return np.array(X), np.array(y)


# # Create training and validation data
# N_train = 1000  # Number of training datapoints
# N_val = 20  # Number of validation datapoints

# X_train, y_train = generate_data(N_train, data_gen)
# X_val, y_val = generate_data(N_val, data_gen)


def generate_pdf_data(N, data_gen, num_points=1024):
    X = []
    y = []

    for _ in range(N):
        data = np.hstack(data_gen.generate_data(num_points))
        x_data = data[:num_points]
        y_data = data[num_points:]

        # Estimate the PDFs of x and y using KDE
        kde_x = gaussian_kde(x_data)
        kde_y = gaussian_kde(y_data)

        # Compute f(x, y) and estimate its PDF using KDE
        f_data = data_gen.f(x_data, y_data)
        kde_f = gaussian_kde(f_data)

        # Stack the PDFs of x, y, and f(x, y) as input features and target
        x_pdf = kde_x(x_data)
        y_pdf = kde_y(y_data)
        f_pdf = kde_f(f_data)

        X.append(np.hstack([x_pdf, y_pdf]))
        y.append([f_pdf])


    return np.array(X), np.array(y)

# Example usage:
N = 5000  # Number of training datapoints
val = 100
X_train, y_train = generate_pdf_data(N, data_gen)
X_val, y_val = generate_pdf_data(val, data_gen)


# Set the dimensions of input and output
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

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
