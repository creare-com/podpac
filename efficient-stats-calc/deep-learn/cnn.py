import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Load your data here
# X_train, y_train, X_val, y_val

# Ensure your input data is in the correct format: (samples, height, width, channels)
# For grayscale grid data, channels = 1
# For RGB grid data, channels = 3

# Set the dimensions of input and output
input_shape = X_train.shape[1:]
output_dim = y_train.shape[1]

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

# Make predictions with the trained model
# X_test: your test data
predictions = model.predict(X_test)
