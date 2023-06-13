import numpy as np
import podpac
from podpac.data import Zarr
from podpac.core.interpolation.selector import Selector
import zarr

class ZarrCaching:
    def __init__(self, source_node, zarr_path_data, zarr_path_bool):
        self.source_node = source_node
        self.zarr_path_data = zarr_path_data
        self.zarr_path_bool = zarr_path_bool
        self.group_data = zarr.open(self.zarr_path_data, mode='w')
        self.group_bool = zarr.open(self.zarr_path_bool, mode='w')
        self.s = Selector(method="nearest")  # add Selector as a class variable


    def create_empty_zarr(self):
        data_shape = self.source_node.shape
        empty_data = np.empty(data_shape)
        false_bool = np.zeros(data_shape, dtype=bool)
        self.group_data.array('data', empty_data, chunks=empty_data.shape, dtype='f8')
        self.group_bool.array('contains', false_bool, chunks=empty_data.shape, dtype='bool')

    def get_source_data(self, request_coords):
        data = self.source_node.eval(request_coords).data
        return data

    def fill_zarr(self, data, request_coords):
        c3, index_arrays = self.s.select(self.source_node.coordinates, request_coords)
        slices = {}
        for dim, indices in zip(c3.keys(), index_arrays):
            indices = indices.flatten()  # convert to 1D array
            slices[dim] = slice(indices[0], indices[-1]+1)

        z_node = Zarr(source=self.zarr_path_data, coordinates=self.source_node.coordinates, file_mode="r+")
        z_bool = Zarr(source=self.zarr_path_bool, coordinates=self.source_node.coordinates, file_mode="r+")

        z_node.dataset['data'][slices['lat'], slices['lon'], slices['time']] = data
        z_bool.dataset['contains'][slices['lat'], slices['lon'], slices['time']] = True

    def subselect_has(self, request_coords):
        z_bool = Zarr(source=self.zarr_path_bool, coordinates=self.source_node.coordinates, file_mode="r")
        
        c3, index_arrays = self.s.select(self.source_node.coordinates, request_coords)
        slices = {}
        for dim, indices in zip(c3.keys(), index_arrays):
            indices = indices.flatten()  # convert to 1D array
            slices[dim] = slice(indices[0], indices[-1]+1)
        
        bool_data = z_bool.dataset['contains'][slices['lat'], slices['lon'], slices['time']]
        false_indices = np.where(bool_data == False)
        
        false_coords = {}
        for dim, indices in zip(['lat', 'lon', 'time'], false_indices):
            false_coords[dim] = self.source_node.coordinates[dim][indices]
        
        return podpac.Coordinates([false_coords['lat'], false_coords['lon'], false_coords['time']], dims=["lat", "lon", "time"])

    def eval(self, request_coords):
        # Get the coordinates for which the zarr cache doesn't have data yet
        subselect_coords = self.subselect_has(request_coords)

        # Get the missing data from the source node using the sub-selected coordinates
        missing_data = self.get_source_data(subselect_coords)

        # Fill the zarr cache with the missing data
        self.fill_zarr(missing_data, subselect_coords)
        
        # Read the requested data from the Zarr cache
        z_node = Zarr(source=self.zarr_path_data, coordinates=self.source_node.coordinates, file_mode="r")
        c3, index_arrays = self.s.select(self.source_node.coordinates, request_coords)
        slices = {}
        for dim, indices in zip(c3.keys(), index_arrays):
            indices = indices.flatten()  # convert to 1D array
            slices[dim] = slice(indices[0], indices[-1]+1)

        data = z_node.dataset['data'][slices['lat'], slices['lon'], slices['time']]

        return data
