import time
import numpy as np

import sys
import os
sys.path.append(os.path.abspath('.'))
from data_gen import *
import data_gen
from tensorflow.keras.models import load_model
from scipy.stats import gaussian_kde
from scipy.integrate import cumtrapz

import time

MLP_MODEL = "models/mlp.h5"

# Constants
CNN_MODEL = 'models/cnn_1d.h5'
N_SAMPLES = 1024

# Brute Force -- computing s(f(x,y))
def brute_force(x_cdf, y_cdf):
    return NotImplementedError
    

# Multi-layer Perceptron
def mlp(x_cdf, y_cdf):
    model = load_model(MLP_MODEL)
    print(x_cdf)
    print(model.predict(np.array([np.hstack([x_cdf,y_cdf])]))[0])
    return model.predict(np.array([np.hstack([x_cdf,y_cdf])]))[0]
    
    
# Convolutional Neural Network
def cnn(x_cdf, y_cdf):
    # Load the trained CNN model
    model = load_model(CNN_MODEL)

    cdf = np.array([np.hstack([x_cdf,y_cdf])])
    cdf = cdf.reshape(-1, cdf.shape[1], 1)
    
    
    # Perform prediction using the model
    prediction = model.predict(cdf)[0]
    
    print(prediction)

    return prediction

def benchmark(methods, data_gen, f, n_runs=1, n_sets=10):
    results = {}
    
    def calculate_cdf(data):
        sorted_data = np.sort(data)
        probabilities = np.linspace(0, 1, len(sorted_data))
        return sorted_data, probabilities

    for name, method in methods.items():
        # Initialize variables to store execution time and errors
        exec_times = []
        errors = []
        
        for _ in range(n_runs):
            set_errors = []
            for i in range(n_sets):
                x, y = data_gen.generate_data(N_SAMPLES)
                c_true = f(x, y)

                # Calculate the CDFs of x, y, and f(x, y)
                x_sorted, x_cdf = calculate_cdf(x)
                y_sorted, y_cdf = calculate_cdf(y)
                c_sorted, c_cdf_true = calculate_cdf(c_true)

                # Measure execution time
                start_time = time.time()
                c_cdf_approx = method(x_cdf, y_cdf)
                exec_time = time.time() - start_time
                
                # Calculate the error
                error = np.mean(np.abs(c_cdf_approx - c_cdf_true))
                set_errors.append(error)

            exec_times.append(exec_time)
            errors.append(np.mean(set_errors))
        
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
    'mlp': mlp,
    'cnn': cnn
}   

# Run the benchmark
results = benchmark(methods, data_gen, f, n_runs=1, n_sets=1)
print("Benchmark results:")
for method, result in results.items():
    print(f"{method}:")
    print(f"  Average execution time: {result['average_execution_time']:.6f} seconds")
    print(f"  Average error: {result['average_error']:.6f}")
