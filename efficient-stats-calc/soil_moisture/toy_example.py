import podpac
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
from scipy.stats import gaussian_kde
import xarray as xr
'''
Toy Data Creation
'''

def generate_sm_data(N, random_seed=None, scale=1, sigma=3):
    if random_seed is not None:
        np.random.seed(random_seed)

    elevation_data = np.random.rand(N, N) * scale
    elevation_data = gaussian_filter(elevation_data, sigma=sigma)

    return elevation_data

def year_long_sm(mean_sm_data, monthly_coefficients, coefficient_std_dev=0.05):
    year_long_sm_data = []

    for month_coefficient in monthly_coefficients:
        month_coefficient_array = np.random.normal(loc=month_coefficient, scale=coefficient_std_dev, size=mean_sm_data.shape)
        month_sm_data = mean_sm_data * month_coefficient_array
        year_long_sm_data.append(month_sm_data)

    return year_long_sm_data

def decade_long_sm(mean_sm_data, monthly_coefficients, years=10, coefficient_std_dev=0.05):
    decade_long_sm_data = []

    for _ in range(years):
        year_long_data = year_long_sm(mean_sm_data, monthly_coefficients, coefficient_std_dev)
        decade_long_sm_data.extend(year_long_data)

    return decade_long_sm_data


def create_coordinates(N, YEARS):
    # create coordinates for data

    fine_lat = np.linspace(-10, 10, N)    # Fine Data
    fine_lon = np.linspace(-10, 10, N)    # Coarse Data



    time = np.linspace(0, (12*YEARS)-1, 12*YEARS)
    coords = podpac.Coordinates([time, fine_lat, fine_lon], ['time','lat', 'lon']) # Fine coords
    
    return coords


"""
Data Modification/Utilities
"""

def g(data):
    return 1 / (1 + np.exp(-data))

def create_pdf(all_data, N_bins):
    n_rows, n_cols = all_data.shape[1:3]

    # Initialize the PDFs, edges, and centers arrays
    pdfs_shape = (N_bins, n_rows, n_cols)
    edges_shape = (N_bins + 1, n_rows, n_cols)
    all_pdfs = np.zeros(pdfs_shape)
    all_edges = np.zeros(edges_shape)

    # Calculate the common edges using the flattened data
    flat_data = all_data.ravel()
    flat_data = flat_data[np.isfinite(flat_data)]
    common_edges = np.histogram_bin_edges(flat_data, bins=N_bins)

    # Calculate the histograms and centers for each element of the 2D grid
    for i in range(n_rows):
        for j in range(n_cols):
            tmp = all_data[:, i, j].ravel()
            tmp = tmp[np.isfinite(tmp)]
            all_pdfs[:, i, j], _ = np.histogram(tmp, density=True, bins=common_edges)
            all_edges[:, i, j] = common_edges

    # Calculate centers
    all_centers = (all_edges[:-1] + all_edges[1:]) / 2

    return all_pdfs, all_edges, all_centers

def create_stats_f_data(stats_coords_f, g_x):
    stats_f_data = []
    shape = stats_coords_f.shape
    shape = (1, shape[1], shape[2])  # set maximum time shape to 1
    for coords in stats_coords_f.iterchunks(shape):
        stats_f_data.append(g_x.eval(coords))  # should be fast -- all local cached data (Except vegetation for some reason?)
    stats_f_data = xr.concat(stats_f_data, 'time')
    
    return stats_f_data
    


"""
Data Visualization
"""
def plot_year_long_sm(year_long_data):
    n_months = len(year_long_data)
    n_cols = 3
    n_rows = (n_months + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    for month, sm_data in enumerate(year_long_data, start=1):
        ax = axes[month - 1]
        img = ax.imshow(sm_data, cmap='viridis', origin='lower')
        ax.set_title(f'Month {month}')
        ax.axis('off')

    # Remove unused subplots
    for ax in axes[n_months:]:
        ax.remove()

    # Add a colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(img, cax=cbar_ax, label='Soil Moisture')

    plt.show()

def create_histogram(m_sm_matrices, N_bins):
    # Calculate the average soil moisture value for each matrix
    avg_soil_moisture = [np.mean(matrix) for matrix in m_sm_matrices]

    # Create the histogram
    plt.hist(avg_soil_moisture, bins=N_bins)
    plt.xlabel('Average Soil Moisture')
    plt.ylabel('Frequency')
    plt.title('Histogram of Soil Moisture Matrices')
    plt.show()


def plot_pdf_for_single_grid_square(all_pdfs, all_edges, row, col):
    # Select the PDF and edges for the specific grid square
    pdf = all_pdfs[:, row, col]
    edges = all_edges[:, row, col]

    # Calculate the centers of the bins
    centers = (edges[:-1] + edges[1:]) / 2

    # Plot the PDF
    plt.plot(centers, pdf)
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.title(f"PDF for Grid Square ({row}, {col})")
    plt.show()

"""
Main
"""

def approx_sm(N, YEARS, N_BINS, DO_PLOT, monthly_coefficients):


    """
    Data Gen
    """

    # Data Gen
    average_sm = generate_sm_data(N, random_seed=0)
    decade_long_data = decade_long_sm(average_sm, monthly_coefficients, YEARS)

    
    if DO_PLOT:
        year_long_data = year_long_sm(average_sm, monthly_coefficients)
        plot_year_long_sm(year_long_data)
        
    print("Data Generated")

    """
    Coordinate Gen
    """
    
    coords = create_coordinates(N, YEARS)
    print(coords)
    
    
    """
    Create PDF's for each grid square
    """
    
    data = np.array(decade_long_data)
    all_pdfs, all_edges, all_centers = create_pdf(data, N_BINS)
    
    """
    Calculate Means
    """
    true_mean = g(data).mean(0)
    cheap_mean = (g(all_centers) * all_pdfs * (np.diff(all_edges, axis=0))).sum(axis=0)
    print("Mean Absolute Error", np.abs(true_mean - cheap_mean).mean())
    
    
    if DO_PLOT or True:
        plt.figure()
        ax1 = plt.subplot(131)
        plt.imshow(true_mean)
        plt.title("Truth")
        plt.colorbar()
        plt.subplot(132, sharex=ax1, sharey=ax1)
        plt.imshow(cheap_mean,)
        plt.title(f"Cheap,Approx, nbins={N_BINS}")
        plt.colorbar()
        plt.subplot(133, sharex=ax1, sharey=ax1)
        plt.title("Cheap - Truth")
        plt.imshow(cheap_mean - true_mean, cmap='BrBG')#, vmin=-0.02, vmax=0.02)
        plt.colorbar()
        plt.show()
    

if __name__ == '__main__':
    # Constants
    N = 50  # size of the fine coords
    YEARS = 10 # Total number of years to create years for
    N_BINS = 32 # Number of bins for the histogram
    DO_PLOT = False # plots or not
    monthly_coefficients = [1.2, 1.1, 1.0, 0.8, 0.6, 0.5, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2]
    approx_sm(N, YEARS, N_BINS, DO_PLOT, monthly_coefficients)