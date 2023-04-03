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
from scipy.integrate import cumtrapz



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

def generate_cdf_data(N, data_gen, num_points=1024):
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

        # Compute the CDFs of x, y, and f(x, y) using the PDFs
        x_cdf = cumtrapz(kde_x(x_data), x_data, initial=0)
        y_cdf = cumtrapz(kde_y(y_data), y_data, initial=0)
        f_cdf = cumtrapz(kde_f(f_data), f_data, initial=0)

        # Normalize the CDFs
        x_cdf /= x_cdf[-1]
        y_cdf /= y_cdf[-1]
        f_cdf /= f_cdf[-1]

        X.append(np.hstack([x_cdf, y_cdf]))
        y.append([f_cdf])

    return np.array(X), np.array(y)

# Example usage:
N = 1000  # Number of training datapoints
val = 20
X_train, y_train = generate_cdf_data(N, data_gen)
X_val, y_val = generate_cdf_data(val, data_gen)


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
history = model.fit(X_train, y_train, batch_size=8, epochs=100, validation_data=(X_val, y_val))

model.save("models/mlp.h5")
