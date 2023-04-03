import time
import numpy as np
from data_gen import *
from tensorflow.keras.models import load_model

MLP_MODEL = "models/mlp.h5"

# Constants
CNN_MODEL = 'models/cnn.h5'
WIDTH = 32
HEIGHT = 32

# Brute Force -- computing s(f(x,y))
def brute_force(x, y):
    return np.mean(f(x,y))

# Multi-layer Perceptron
def mlp(x, y):
    model = load_model(MLP_MODEL)
    return model.predict(np.array([np.hstack([x,y])]))[0][0]
    
    
# Convolutional Neural Network
def cnn(x, y):
    # Load the trained CNN model
    model = load_model(CNN_MODEL)

    # Reshape x and y into the format expected by the CNN
    x_data = x.reshape(HEIGHT, WIDTH)
    y_data = y.reshape(HEIGHT, WIDTH)
    data = np.stack([x_data, y_data], axis=-1)  # shape: (height, width, 2)
    input_data = np.expand_dims(data, axis=0)  # shape: (1, height, width, 2)

    # Perform prediction using the model
    prediction = model.predict(input_data)[0][0]

    return prediction


# Benchmarking tool
def benchmark(methods, x, y, f, n_runs=10):
    results = {}
    
    # Calculate the ground truth s(c)
    c_true = f(x, y)
    s_c_true = np.mean(c_true)
    
    for name, method in methods.items():
        # Initialize variables to store execution time and errors
        exec_times = []
        errors = []
        
        for _ in range(n_runs):
            # Measure execution time
            start_time = time.time()
            s_c_approx = method(x, y)
            exec_time = time.time() - start_time
            exec_times.append(exec_time)
            
            # Calculate the error
            error = np.abs(s_c_approx - s_c_true)
            errors.append(error)
        
        # Calculate average execution time and error
        avg_exec_time = np.mean(exec_times)
        avg_error = np.mean(errors)
        
        results[name] = {
            'average_execution_time': avg_exec_time,
            'average_error': avg_error
        }
        
    return results

# Define methods to benchmark
methods = {
    'brute_force': brute_force,
    'multilayer_perceptron': mlp,
    'convolutional': cnn
}

# Generate toy data
n_samples = WIDTH*HEIGHT
x, y = generate_data(n_samples)


# Run the benchmark
results = benchmark(methods, x, y, f)
print("Benchmark results:")
for method, result in results.items():
    print(f"{method}:")
    print(f"  Average execution time: {result['average_execution_time']:.6f} seconds")
    print(f"  Average error: {result['average_error']:.6f}")
