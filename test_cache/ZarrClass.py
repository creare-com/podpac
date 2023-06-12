import numpy as np
import podpac
from podpac.data import Zarr
from podpac.core.interpolation.selector import Selector
import zarr

class ZarrCaching:
    def __init__(self, data, native_coords, zarr_path_data, zarr_path_bool):
        self.data = data
        self.native_coords = native_coords
        self.zarr_path_data = zarr_path_data
        self.zarr_path_bool = zarr_path_bool
        self.arr_node = podpac.data.Array(source=self.data, coordinates=self.native_coords)
        self.group_data = zarr.open(self.zarr_path_data, mode='w')
        self.group_bool = zarr.open(self.zarr_path_bool, mode='w')

    def create_empty_zarr(self):
        data_shape = self.arr_node.shape
        empty_data = np.empty(data_shape)
        false_bool = np.zeros(data_shape, dtype=bool)
        self.group_data.array('data', empty_data, chunks=empty_data.shape, dtype='f8')
        self.group_bool.array('contains', false_bool, chunks=empty_data.shape, dtype='bool')

    def fill_zarr(self, request_coords):
        s = Selector(method="nearest")
        c3, index_arrays = s.select(self.native_coords, request_coords)
        slices = {}
        for dim, indices in zip(c3.keys(), index_arrays):
            indices = indices.flatten()  # convert to 1D array
            slices[dim] = slice(indices[0], indices[-1]+1)

        z_node = Zarr(source=self.zarr_path_data, coordinates=self.arr_node.coordinates, file_mode="r+")
        z_bool = Zarr(source=self.zarr_path_bool, coordinates=self.arr_node.coordinates, file_mode="r+")

        z_node.dataset['data'][slices['lat'], slices['lon'], slices['time']] = self.arr_node.eval(request_coords).data
        z_bool.dataset['contains'][slices['lat'], slices['lon'], slices['time']] = True
