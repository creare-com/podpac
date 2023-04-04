import numpy as np
import concurrent.futures
from scipy.stats import gaussian_kde
from scipy.integrate import cumtrapz

# Non-linear n-dimensional function f(x, y)
def f(x, y):
    return np.sin(x) * np.cos(y)

def generate_data(n_samples, seed=None):
        
    if seed is not None:
        np.random.seed(seed)
    
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
# n_samples = 1000
# x, y, s_x, s_y, s_c = create_toy_data(n_samples, f)
# print("s(x):", s_x)
# print("s(y):", s_y)
# print("s(c):", s_c)




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

def generate_cdf_data(N, data_gen, num_points=1024, grid_size=100):
    X = []
    y = []

    for _ in range(N):
        data = np.hstack(data_gen.generate_data(num_points))
        x_data = data[:num_points]
        y_data = data[num_points:]

        # Create a lower resolution grid
        x_grid = np.linspace(x_data.min(), x_data.max(), grid_size)
        y_grid = np.linspace(y_data.min(), y_data.max(), grid_size)

        # Estimate the PDFs of x and y using KDE
        # Example: Use Silverman's rule of thumb
        kde_x = gaussian_kde(x_data, bw_method='silverman')
        kde_y = gaussian_kde(y_data, bw_method='silverman')

        # Compute f(x, y) and estimate its PDF using KDE
        f_data = data_gen.f(x_data, y_data)
        kde_f = gaussian_kde(f_data, bw_method='silverman')

        # Compute the CDFs of x, y, and f(x, y) using the PDFs on the grid
        x_cdf = cumtrapz(kde_x(x_grid), x_grid, initial=0)
        y_cdf = cumtrapz(kde_y(y_grid), y_grid, initial=0)
        f_cdf = cumtrapz(kde_f(f_data), f_data, initial=0)

        # Normalize the CDFs
        x_cdf /= x_cdf[-1]
        y_cdf /= y_cdf[-1]
        f_cdf /= f_cdf[-1]

        X.append(np.hstack([x_cdf, y_cdf]))
        y.append([f_cdf])

    return np.array(X), np.array(y)

def generate_cdf_data_parallel(N, data_gen, num_workers=8, num_points=1024):
    # Define a helper function to process each data point
    def process_datapoint(data_gen):
        data = np.hstack(data_gen.generate_data(num_points))
        x_data = data[:num_points]
        y_data = data[num_points:]

        # Calculate the CDFs of x, y, and f(x, y) directly from the data points
        x_sorted, x_cdf = calculate_cdf(x_data)
        y_sorted, y_cdf = calculate_cdf(y_data)
        f_data = data_gen.f(x_data, y_data)
        f_sorted, f_cdf = calculate_cdf(f_data)

        return np.hstack([x_cdf, y_cdf]), f_cdf

    def calculate_cdf(data):
        sorted_data = np.sort(data)
        probabilities = np.linspace(0, 1, len(sorted_data))
        return sorted_data, probabilities

    X = []
    y = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Parallelize the generation of data points
        futures = [executor.submit(process_datapoint, data_gen) for _ in range(N)]

        for future in futures:
            x_cdf_y_cdf, f_cdf = future.result()
            X.append(x_cdf_y_cdf)
            y.append([f_cdf])

    return np.array(X), np.array(y)

def save_cdf_data(X, y, X_filename, y_filename):
    np.save(X_filename, X)
    np.save(y_filename, y)

def load_cdf_data(X_filename, y_filename):
    X = np.load(X_filename)
    y = np.load(y_filename)
    return X, y

N = 10000
import data_gen

X_train, y_train = generate_cdf_data_parallel(N, data_gen, num_workers=8)
save_cdf_data(X_train, y_train, "data/X_train.npy", "data/y_train.npy")


X_val, y_val = generate_cdf_data_parallel(200, data_gen, num_workers=8)
save_cdf_data(X_train, y_train, "data/X_val.npy", "data/y_val.npy")