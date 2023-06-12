# Create some toy data for testing zarr caching
import numpy as np
import podpac
from podpac.data import Zarr
from podpac.core.interpolation.selector import Selector
import zarr

"""
Part 1: Creating the zarr archive
"""

# Create some toy data
data = np.random.rand(100, 100, 100)

# Create Native Coords
lat = np.linspace(-90, 90, 100)
lon = np.linspace(-180, 180, 100)
time = np.arange("2023-06-02", 100, dtype='datetime64[D]')
native_coords = podpac.Coordinates([lat, lon, time], dims=["lat", "lon", "time"])

# Create a podpac array node with data and native coords
arr_node = podpac.data.Array(source=data, coordinates=native_coords)
arr_node.eval(native_coords)


# Create an *EMPTY* Zarr archive of the size of the native coords?
# Get data from array node and put it in the zarr archive using *SUBSET* of native coords
group = zarr.open('/home/cfoye/Projects/SoilMap/podpac/test_cache/mydata.zarr', mode='w')

# Get the shape of the data array:
data_shape = arr_node.shape
empty_data = np.empty(data_shape)


# Get the lat, lon, and time dimensions
lat_shape = arr_node.coordinates["lat"].shape
lon_shape = arr_node.coordinates["lon"].shape
time_shape = arr_node.coordinates["time"].shape

# Create coordinates for the zarr archive
lat_bounds = arr_node.coordinates["lat"].bounds
lat = np.linspace(lat_bounds[0], lat_bounds[1], lat_shape[0])

lon_bounds = arr_node.coordinates["lon"].bounds
lon = np.linspace(lon_bounds[0], lon_bounds[1], lon_shape[0])

time_bounds = arr_node.coordinates["time"].bounds
time = np.array([time_bounds[0]] * 100, dtype='datetime64[ns]')


group.array('data', empty_data, chunks=empty_data.shape, dtype='f8')



group2 = zarr.open('/home/cfoye/Projects/SoilMap/podpac/test_cache/mybool.zarr', mode='w')
false_bool = np.zeros(data_shape, dtype=bool)
group2.array('contains', false_bool, chunks=empty_data.shape, dtype='bool')


"""
Part 2, filling the zarr archive node
"""
# TODO: Figure out a way to access one of the two outputs when evalling a node??
# See for selecting: https://github.com/frocreare-com/podpac/blob/develop/podpac/core/interpolation/selector.py


z_node = Zarr(source='/home/cfoye/Projects/SoilMap/podpac/test_cache/mydata.zarr', coordinates=arr_node.coordinates, file_mode="r+")
z_bool = Zarr(source='/home/cfoye/Projects/SoilMap/podpac/test_cache/mybool.zarr', coordinates=arr_node.coordinates, file_mode="r+")


# change values in zarr archive
lat = np.linspace(-80, 80, 50)
lon = np.linspace(-150, 150, 50)
time = np.arange("2023-06-02", 10, dtype='datetime64[D]')
request_coords = podpac.Coordinates([lat, lon, time], dims=["lat", "lon", "time"])



# Get slices using a selector
s = Selector(method="nearest")

c3, index_arrays = s.select(native_coords, request_coords)

slices = {}
for dim, indices in zip(c3.keys(), index_arrays):
    indices = indices.flatten()  # convert to 1D array
    slices[dim] = slice(indices[0], indices[-1]+1)

# To use the slices:
for dim, slc in slices.items():
    print(f"{dim} slice:", native_coords[dim][slc])


# Eval Array node:
arr_node.eval(request_coords)

# Assign values in zarr archive using slices
z_node.dataset['data'][slices['lat'], slices['lon'], slices['time']] = arr_node.eval(request_coords).data
z_bool.dataset['contains'][slices['lat'], slices['lon'], slices['time']] = True





"""
Part 3:
Sub-selecting request coords according to z_bool
"""

