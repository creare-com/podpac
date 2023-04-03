import numpy as np

# Non-linear n-dimensional function f(x, y)
def f(x, y):
    return np.sin(x) * np.cos(y)

# Generate random x and y data
def generate_data(n_samples):
    x = np.random.uniform(-np.pi, np.pi, n_samples)
    y = np.random.uniform(-np.pi, np.pi, n_samples)
    return x, y

# Calculate statistics s(x), s(y) and s(c)
def calculate_statistics(x, y, func):
    s_x = np.mean(x)
    s_y = np.mean(y)
    c = func(x, y)
    s_c = np.mean(c)
    return s_x, s_y, s_c

# Generate toy data and calculate statistics
def create_toy_data(n_samples, func):
    x, y = generate_data(n_samples)
    s_x, s_y, s_c = calculate_statistics(x, y, func)
    return x, y, s_x, s_y, s_c

"""
Example Usage
"""
n_samples = 1000
x, y, s_x, s_y, s_c = create_toy_data(n_samples, f)
print("s(x):", s_x)
print("s(y):", s_y)
print("s(c):", s_c)
