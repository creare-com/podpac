import podpac
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
from scipy.stats import gaussian_kde
import xarray as xr
from tqdm import tqdm
from matplotlib.colors import LogNorm


import soilmap.datalib.geowatch as geowatch


def get_coordinates():
    # Make a set of fine and coarse coordinates
    # Center coordinates about Creare
    center_lat, center_lon = 43.682102, -72.233455
    # get deltas using 110 km / deg and a 30 x 30 km box
    f_box_lat_lon = 30 / 2 / 110
    n_fine = int(30 / (30 / 1000))  # Number of 30 m boxes in a 30km square (hint, it's 1000)
    n_coarse = 3
    f_coords =  podpac.Coordinates([
        podpac.clinspace(center_lat + f_box_lat_lon, center_lat - f_box_lat_lon, n_fine),
        podpac.clinspace(center_lon - f_box_lat_lon, center_lon + f_box_lat_lon, n_fine),
            ["2022-06-01T12"]], ["lat", "lon", "time"]
    )
    c_coords = f_coords[::333, 100::400]
    
    return c_coords, f_coords

def fill_time_dimensions(c_coords, f_coords):
      # make the coordinates for the brute-force technique
    e1 = podpac.algorithm.ExpandCoordinates(time=["0,D", "30,D", '1,D'])
    day_time_coords = podpac.Coordinates([e1.get_modified_coordinates1d(f_coords, 'time')])
    e2 = podpac.algorithm.ExpandCoordinates(source=e1, time=["-4,Y", "0,Y", '1,Y'])
    all_time_coords = podpac.Coordinates([e2.get_modified_coordinates1d(day_time_coords, 'time')])

    all_f_coords = podpac.coordinates.merge_dims([f_coords.drop('time'), all_time_coords])
    all_c_coords = podpac.coordinates.merge_dims([c_coords.drop('time'), all_time_coords])
    

    return all_c_coords, all_f_coords

def get_veg_node(all_f_coords):
      # Also, use the average of the vegetation, for initial testing
    sm = geowatch.SoilMoisture()
    o_veg = []
    vegetation = sm.vegetation
    one_month_coords = all_f_coords[:, :, :32*5:5]
    shape = one_month_coords[:, :, :32*5:5].shape
    shape = (shape[0], shape[1], 1)  # set maximum time shape to 1
    for coords in one_month_coords.iterchunks(shape):
        o_veg.append(vegetation.eval(coords))
    o_veg = xr.concat(o_veg, 'time')
    print(o_veg.mean('time').data)
    veg_node = podpac.data.Array(source=o_veg.mean('time').data, coordinates=all_f_coords.drop('time'))
    
    return veg_node

def get_constant_veg_node(all_f_coords, constant_veg_value):
    # Get the spatial coordinates from all_f_coords
    spatial_coords = all_f_coords.drop('time')
    
    # Create an array with the same spatial shape as all_f_coords but filled with the constant vegetation value
    constant_veg_data = np.full(spatial_coords.shape, constant_veg_value)
    
    # Create a PODPAC Array node with the constant vegetation data and the spatial coordinates
    constant_veg_node = podpac.data.Array(source=constant_veg_data, coordinates=spatial_coords)
    
    return constant_veg_node

def get_constant_land_use(all_f_coords, constant_land_value):
    # Get the spatial coordinates from all_f_coords
    spatial_coords = all_f_coords.drop('time')
    
    # Create an array with the same spatial shape as all_f_coords but filled with the constant vegetation value
    constant_veg_data = np.full(spatial_coords.shape, constant_land_value)
    
    # Create a PODPAC Array node with the constant vegetation data and the spatial coordinates
    constant_land_node = podpac.data.Array(source=constant_veg_data, coordinates=spatial_coords)
    
    return constant_land_node

def get_calibrated_sm(all_c_coords, veg_node, land_use_node):
        # Quick test to make sure everything makes sense
    sm = geowatch.SoilMoisture(vegetation=veg_node, vegetation_mean=veg_node, land_use=land_use_node)
    sm_ca = sm.solmst_0_10
    sm_cr = sm.relsolmst_0_10  # you can compute this from the abs soil moisture above... but I'm lazy so I'll just get this data for now
    
    all_ca_data = [] #coarse absolute
    all_cr_data = [] #coarse relative
    shape = all_c_coords.shape
    shape = (shape[0], shape[1], 1)
    for coords in all_c_coords.iterchunks(shape):
        all_ca_data.append(sm_ca.eval(coords))
        all_cr_data.append(sm_cr.eval(coords))
    
    # Merge the data in all_ca_data along the 'time' dimension
    all_ca_data = xr.concat(all_ca_data, 'time')

    # Merge the data in all_cr_data along the 'time' dimension
    all_cr_data = xr.concat(all_cr_data, 'time')
   
    # All Coarse Absolute Data
    all_ca_arr = podpac.data.Array(
        source=all_ca_data,
        coordinates=all_c_coords,
        interpolation="bilinear"
    )
    # All Coarse Relative Data
    all_cr_arr = podpac.data.Array(
        source=all_cr_data,
        coordinates=all_c_coords,
        interpolation="bilinear"
    )

    # Create a new SoilMoisture node that adjusts the soil moisture data based on vegetation data
    return all_ca_data, all_cr_data, geowatch.SoilMoisture(solmst_0_10=all_ca_arr, relsolmst_0_10=all_cr_arr, vegetation=veg_node, vegetation_mean=veg_node, land_use=land_use_node)

def get_stats_data(all_c_coords, all_f_coords, all_ca_edges, all_cr_edges, constant_veg_value, constant_land_value, n_bins):
    # Get the centers of the bins
    all_ca_centers = (all_ca_edges[..., 1:] + all_ca_edges[..., :-1]) * 0.5
    all_cr_centers = (all_cr_edges[..., 1:] + all_cr_edges[..., :-1]) * 0.5

    # Get stats coords
    stats_coords_f = all_f_coords[:, :, :n_bins * 2:2]
    stats_coords = all_c_coords[:, :, :n_bins * 2:2]
    
    # Create abs soil moisture coords at the centers of the bins
    abs_ws_stats = podpac.data.Array(
        source=all_ca_centers, # Here's the weatherscale data at the centers of the bins
        coordinates=stats_coords,  # Mock the coordinates
        interpolation="bilinear"
    )
    # Create relative soil moisture coords at the centers of the bins
    rel_ws_stats = podpac.data.Array(
        source=all_cr_centers,
        coordinates=stats_coords,  # Mock the coordinates
        interpolation="bilinear"
    )
    # Get the spatial coordinates from stats_coords
    spatial_coords = stats_coords.drop('time')
    
    constant_veg_data = np.full(spatial_coords.shape, constant_veg_value)
    veg_node = podpac.data.Array(source=constant_veg_data, coordinates=spatial_coords)
    
    constant_land_data = np.full(spatial_coords.shape, constant_land_value)
    land_use_node = podpac.data.Array(source=constant_land_data, coordinates=spatial_coords)

    # Make the g(x) function
    sm_stats = geowatch.SoilMoisture(solmst_0_10=abs_ws_stats, relsolmst_0_10=rel_ws_stats, vegetation=veg_node, vegetation_mean=veg_node, land_use=land_use_node)

    # Evaluate g(x) 
    stats_f_data = []
    shape = stats_coords_f.shape
    shape = (shape[0], shape[1], 1)  # set maximum time shape to 1
    for coords in stats_coords_f.iterchunks(shape):
        stats_f_data.append(sm_stats.eval(coords))  # should be fast -- all local cached data (Except vegetation for some reason? )
    stats_f_data = xr.concat(stats_f_data, 'time')
        
    return stats_f_data

def get_fine_pdfs_edges(all_ca_pdfs, all_ca_edges, n_bins, all_c_coords, all_f_coords):
    ca_edges = podpac.data.Array(
        source=all_ca_edges,
        coordinates=all_c_coords[:, :, :n_bins + 1],  # Again, the time coordinate here is fake -- I just want to interpolate space
        interpolation='bilinear',    )
    ca_edges_f = ca_edges.eval(all_f_coords[:, :, :n_bins + 1])
    # pdfs
    ca_pdfs = podpac.data.Array(
        source=all_ca_pdfs,
        coordinates=all_c_coords[:, :, :n_bins],  # Again, the time coordinate here is fake -- I just want to interpolate space
        interpolation='bilinear',
    )
    ca_pdfs_f = ca_pdfs.eval(all_f_coords[:, :, :n_bins])
    
    return ca_edges_f, ca_pdfs_f

def get_fine_data(all_f_coords, sm_centered):
    # get the data for the cheap approach
    all_f_data = []
    shape = all_f_coords.shape
    shape = (shape[0], shape[1], 1)
    for coords in all_f_coords.iterchunks(shape):
        all_f_data.append(sm_centered.eval(coords))
    all_f_data = xr.concat(all_f_data, 'time')

    return all_f_data


def create_pdfs(n_bins, all_ca_data, all_cr_data):
    # Create new arrays with a third dimension of size n_bins.
    new_shape = all_ca_data.shape[:2] + (n_bins,)
    new_shape_edges = all_ca_data.shape[:2] + (n_bins + 1,)
    all_ca_pdfs = np.zeros((new_shape))
    all_ca_edges = np.zeros((new_shape_edges))
    all_cr_pdfs = np.zeros((new_shape))
    all_cr_edges = np.zeros((new_shape_edges))

    # Loop over each element in the 2D array, computing the histograms for the corresponding 1D data.
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            # Compute the histogram for the i,j-th element of all_ca_data.
            tmp = all_ca_data.data[i, j].ravel()
            tmp = tmp[np.isfinite(tmp)]
            all_ca_pdfs[i, j, :], all_ca_edges[i, j, :] = np.histogram(tmp, density=True, bins=n_bins)

            # Compute the histogram for the i,j-th element of all_cr_data.
            tmp = all_cr_data.data[i, j].ravel()
            tmp = tmp[np.isfinite(tmp)]
            all_cr_pdfs[i, j, :], all_cr_edges[i, j, :] = np.histogram(tmp, density=True, bins=n_bins)

    return all_ca_pdfs, all_ca_edges, all_cr_pdfs, all_cr_edges

def validate_pdfs(all_ca_pdfs, all_ca_edges):
         # Check on one of these
    assert np.abs(1 - (all_ca_pdfs[1, 1] * (all_ca_edges[1, 1][1:] - all_ca_edges[1, 1][:-1])).sum()) < 1e-14
    plt.stairs(all_ca_pdfs[1, 1], all_ca_edges[1, 1], fill=True)
    plt.stairs(all_ca_pdfs[1, 1], all_ca_edges[1, 1], color='k', fill=False)
    plt.show()

def compute_mse(cheap_mean, truth_mean):
    squared_diffs = np.nan_to_num((cheap_mean - truth_mean)**2)
    valid_mask = np.isfinite(cheap_mean) & np.isfinite(truth_mean)  # mask where neither array is NaN

    # Compute the mean of the squared differences, ignoring any NaN values
    mse = np.nanmean(squared_diffs[valid_mask])
    return mse
    
def plot_difference(arr1, arr2, n_bins, title=""):
    # Plot results
    plt.figure()
    kwargs = dict()
    ax1 = plt.subplot(131)
    plt.imshow(arr1, **kwargs)
    plt.title("Truth "+title)
    plt.colorbar()
    plt.subplot(132, sharex=ax1, sharey=ax1)
    plt.imshow(arr2, **kwargs)
    plt.title(f"Cheap,Approx, nbins={n_bins}, " + title)
    plt.colorbar()
    plt.subplot(133, sharex=ax1, sharey=ax1)
    plt.title("Cheap - Truth "+ title)
    plt.imshow(arr2 - arr1, cmap='BrBG')
    plt.colorbar()
    plt.show()
        
def eff_stats_calc(n_bins, veg_value, land_use_value, DO_LOG=False, DO_PLOT=False):
        # Get lat and lon coordinates for coarse data with empty time
        c_coords, f_coords = get_coordinates()
        
        # Fill in time dimension 
        all_c_coords, all_f_coords = fill_time_dimensions(c_coords, f_coords)
        
        if DO_LOG:
            print("Got Coords...")
        
        # Get a vegetation node for constant vegetation:
        # veg_node = get_veg_node(all_f_coords)
        veg_node = get_constant_veg_node(all_f_coords, veg_value)
        land_use_node = get_constant_land_use(all_f_coords, land_use_value)
        
        # Get a soil moisture node with centered on coarse coords
        all_ca_data, all_cr_data, sm_centered = get_calibrated_sm(all_c_coords, veg_node, land_use_node)
        
        if DO_LOG:
            print("Calibrated SM...")
        
        # Get Fine SM Data:
        all_f_data = get_fine_data(all_f_coords, sm_centered)
        
        if DO_LOG:
            print("Got Fine Data...")
        
        # Create a PDF from the data
        all_ca_pdfs, all_ca_edges, all_cr_pdfs, all_cr_edges = create_pdfs(n_bins, all_ca_data, all_cr_data)
        #TODO: Figure out how we can use coarse relative pdfs?
        # validate_pdfs(all_ca_pdfs, all_ca_edges)
        
        if DO_LOG:
            print("Created PDFs...")
        
        # Create a SM node with the centers
        stats_f_data = get_stats_data(all_c_coords, all_f_coords, all_ca_edges, all_cr_edges, veg_value, land_use_value ,n_bins)
        
        if DO_LOG:
            print("Got Stats Data...")
        
        # Get the fine pdfs by interpolating coarse-scale data edges
        # TODO: figure out ohow to use coarse relative pdfs?
        ca_edges_f, ca_pdfs_f = get_fine_pdfs_edges(all_ca_pdfs, all_ca_edges, n_bins, all_c_coords, all_f_coords)
        
        if DO_LOG:
            print("Got Fine PDFs...")
        
        # Calculate cheap mean and variance
        cheap_mean = (stats_f_data.data * ca_pdfs_f.data * (ca_edges_f[..., 1:].data - ca_edges_f[..., :-1].data)).sum(axis=-1)
        deviations = stats_f_data.data - cheap_mean[..., np.newaxis]
        cheap_var = (deviations**2 * ca_pdfs_f.data * (ca_edges_f[..., 1:].data - ca_edges_f[..., :-1].data)).sum(axis=-1)
        
        
        if DO_LOG:
            print("Calculated Cheap Mean...")
        
        # Calculate correct mean
        truth_mean = all_f_data.mean('time').data
        truth_var = all_f_data.var('time').data
        
        if DO_LOG:
            print("Calculated Truth Mean...")
        
        if DO_PLOT:
            plot_difference(truth_mean, cheap_mean, n_bins, title="Mean")
            plot_difference((truth_var), (cheap_var), n_bins, title="Variance")
            
        
        cheap_std_dev = np.sqrt(cheap_var)
        truth_std_dev = np.sqrt(truth_var)
        
        if DO_LOG:
            print("Calculated StdDev...")
        
        # Compute the squared differences, ignoring any NaN values
        mse = compute_mse(cheap_mean, truth_mean)
        
        return cheap_mean, truth_mean, mse
    

if __name__ == '__main__':
    """
    1. Generate PDFs out of coarse data
    2. Calculate centers for that data
    3. Create SoilMoisutre node using those centers
    4. Evaluate SM node at those centers
    5. Multiply eval by pdf and bin widths
    """
    with podpac.settings:
        # Constants
        VEG_VALUE = 50
        LAND_USE_VALUE = 0
        DO_LOG = True
        DO_PLOT = False

        # Cache
        podpac.settings["DEFAULT_CACHE"] = ["ram", "disk"]
        podpac.settings["MULTITHREADING"] = False
        # podpac.settings["CACHE_OUTPUT_DEFAULT"] = False
        # podpac.utils.clear_cache(mode='ram')
        # podpac.utils.clear_cache(mode='disk')

        # Initialize lists to store the bin counts and the corresponding MSE values
        bin_counts = []
        mse_values = []
        
        # Initialize the progress bar
        pbar = tqdm(total=7, desc='Processing bins', ncols=155)

        # Run eff_stats_calc for 1, 2, 4, 8, 16, 32, 64 bins
        # [1, 2, 4, 8, 16, 32, 64]:
        for n_bins in [1, 2, 4, 8, 16, 32, 64]:
            if n_bins == 64:
                _, _, mse = eff_stats_calc(n_bins, VEG_VALUE, LAND_USE_VALUE, DO_LOG, True)
            else:
                _, _, mse = eff_stats_calc(n_bins, VEG_VALUE, LAND_USE_VALUE, DO_LOG, DO_PLOT)
            bin_counts.append(n_bins)
            mse_values.append(mse)
            
            #Update the progress bar
            pbar.update(1)

        pbar.close()

        # Plot the results
        plt.figure(figsize=(10,6))
        plt.loglog(bin_counts, mse_values, marker='o')
        plt.title('Mean Squared Error vs Number of Bins')
        plt.xlabel('Number of Bins')
        plt.ylabel('Mean Squared Error')
        plt.grid(True)
        plt.show()

        

