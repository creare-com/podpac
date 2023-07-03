# Create a PODPAC node out of a zarr archive
import podpac
from podpac.data import Zarr
from podpac.core.interpolation.selector import Selector
from podpac.core.node import Node
from podpac.core.utils import NodeTrait
from podpac.core.cache.cache_interface import CacheNode


import numpy as np
import traitlets as tl
import zarr

class ZarrCache(CacheNode):
    """
    A PODPAC CachingNode which uses Zarr archives to cache data from a source node.
    
    Attributes
    ----------

    zarr_path_data : str
        The path to the Zarr archive for storing data.
    zarr_path_bool : str
        The path to the Zarr archive for storing boolean indicators of data availability.
    group_data : zarr.hierarchy.Group
        The Zarr group for storing data.
    group_bool : zarr.hierarchy.Group
        The Zarr group for storing boolean indicators of data availability.
    selector : Selector
        Selector for selecting coordinates.
    chunks: list
        Chunk size for the Zarr array. If None, the default chunk size is used.
        
    _z_node : Zarr
        Zarr node for data.
    _z_bool : Zarr
        Zarr node for boolean indicators.
    _from_cache : bool
        Flag indicating whether the data was retrieved from the cache.
    """

    # Public Traits    
    zarr_path = tl.Unicode().tag(attr=True, required=True)

    group_data = tl.Instance(zarr.hierarchy.Group).tag(attr=True)
    group_bool = tl.Instance(zarr.hierarchy.Group).tag(attr=True)
    selector = tl.Instance(Selector, allow_none=True).tag(attr=True)
    chunks = tl.List(allow_none=True).tag(attr=True)

    # Private Traits
    _z_node = tl.Instance(Zarr).tag(attr=True)
    _z_bool = tl.Instance(Zarr).tag(attr=True)
    _from_cache = tl.Bool(allow_none=True, default_value=False)
    _zarr_path_data = tl.Unicode().tag(attr=True, required=True)
    _zarr_path_bool = tl.Unicode().tag(attr=True, required=True)
    
    
    @tl.default('selector')
    def _default_selector(self):
        return Selector(method='nearest')
    
    #Because of the way Zarr Nodes work, If I have one zarr node for both groups (making them into arrays for the same group), every time I eval the zarr node, it evals both datasets/arrays. I don't necessarily want this, given that first I need to eval the Boolean array without evalling the data array, get the data from the server, then only eval the data array.
    @tl.default('zarr_path')
    def _default_zarr_path(self):
        return f"{podpac.settings.cache_path}/zarr_cache_{self.uid}"
    
    @tl.default('_zarr_path_data')
    def _default_zarr_path_data(self):
        return f"{self.zarr_path}/data.zarr"
    
    @tl.default('_zarr_path_bool')
    def _default_zarr_path_bool(self):
        return f"{self.zarr_path}/bool.zarr"
    
    @tl.default('group_data')
    def _default_group_data(self):
        try:
            group = zarr.open(self._zarr_path_data, mode='a')
            if 'data' not in group:
                shape = self.source.coordinates.shape
                group.create_dataset('data', shape=shape, chunks = self.chunks if self.chunks is not None else True, dtype='float64', fill_value=np.nan)  # adjust dtype as necessary
                self._create_coordinate_zarr_dataset(group)
            return group
        except Exception as e:
            raise ValueError(f"Failed to open zarr data group. Original error: {e}")

    @tl.default('group_bool')
    def _default_group_bool(self):
        try:
            group = zarr.open(self._zarr_path_bool, mode='a') # no need to close, see https://zarr.readthedocs.io/en/stable/tutorial.html#persistent-arrays
            if 'contains' not in group:
                shape = self.source.coordinates.shape
                group.create_dataset('contains', shape=shape, dtype='bool', fill_value=False)
                self._create_coordinate_zarr_dataset(group)
            return group
        except Exception as e:
            raise ValueError(f"Failed to open zarr boolean group. Original error: {e}")
    

    @tl.default('_z_node')
    def _default_z_node(self):
        try:
            self.group_data  # ensure group exists
            return Zarr(source=self._zarr_path_data, coordinates=self.source.coordinates, file_mode="r+")
        except Exception as e:
            raise ValueError(f"Failed to create Zarr node. Original error: {e}")
        
    @tl.default ('_z_bool')
    def _default_z_bool(self):
        try:
            self.group_bool # ensure group exists
            return Zarr(source=self._zarr_path_bool, coordinates=self.source.coordinates, file_mode="r+")
        except Exception as e:
            raise ValueError(f"Failed to create Zarr boolean node. Original error: {e}")

    
    def _create_coordinate_zarr_dataset(self, group):
        """
        Create a Zarr dataset for storing coordinates.

        Returns
        -------
        zarr.Dataset
            The Zarr dataset for storing coordinates.
        """
        for dim in self.source.coordinates.dims:
            if dim not in group:
                if dim == 'time':
                    group.create_dataset(dim, shape=self.source.coordinates[dim].shape,dtype='datetime64[D]') # Need to specify specific datetime. Need to write a function to detect the datetime type
                else:
                    group.create_dataset(dim, shape=self.source.coordinates[dim].shape,dtype='float64')
                group[dim][:] = self.source.coordinates.xcoords[dim][1]
    
    def _create_slices(self, c3, index_arrays):
        """
        Create slices for the given coordinates and index arrays.

        Parameters
        ----------
        c3 : podpac.Coordinates
            The coordinates.
        index_arrays : list of np.ndarray
            The index arrays.

        Returns
        -------
        slices : dict
            The slices for the given coordinates and index arrays.
        """
        slices = {}
        for dim, indices in zip(c3.keys(), index_arrays):
            indices = indices.flatten()  # convert to 1D array
            slices[dim] = slice(indices[0], indices[-1]+1)
        return slices

    def rem_cache(self):
        self.group_data['data'][:] = np.nan
        self.group_bool['contains'][:] = False


    def get_source_data(self, request_coords):
        """
        Retrieve data from the source at the specified coordinates.

        Parameters
        ----------
        request_coords : podpac.Coordinates
            The coordinates at which data is requested from the source.

        Returns
        -------
        data : np.ndarray
            The data retrieved from the source at the specified coordinates.
        """
        data = self.source.eval(request_coords).data
        return data

    def fill_zarr(self, data, request_coords):
        """
        Fill the Zarr cache with data at the specified coordinates.

        Parameters
        ----------
        data : np.ndarray
            The data to be stored in the Zarr cache.
        request_coords : podpac.Coordinates
            The coordinates at which the data should be stored in the Zarr cache.
        """
        c3, index_arrays = self.selector.select(self.source.coordinates, request_coords)
        slices = self._create_slices(c3, index_arrays)

        self._z_node.dataset['data'][tuple(slices.get(dim) for dim in self.source.coordinates.dims)] = data
        self._z_bool.dataset['contains'][tuple(slices.get(dim) for dim in self.source.coordinates.dims)] = True

    def subselect_has(self, request_coords):
        """
        Fetch the coordinates for which the Zarr cache does not have data yet.

        Parameters
        ----------
        request_coords : podpac.Coordinates
            The coordinates at which data is requested.

        Returns
        -------
        false_coords : podpac.Coordinates or None
            The subset of the requested coordinates for which the Zarr cache does not have data yet.
            If the Zarr cache has data for all requested coordinates, returns None.
        """
        
        c3, index_arrays = self.selector.select(self.source.coordinates, request_coords)
        slices = self._create_slices(c3, index_arrays)
        
        bool_data = self._z_bool.dataset['contains'][tuple(slices.get(dim) for dim in self.source.coordinates.dims)]
        
        # check if all values are True
        if np.all(bool_data):
            return None  # or any other indicator that all data is present
        
        false_indices = np.where(bool_data == False)
        
        false_indices_unique = tuple(np.unique(indices) for indices in false_indices)

                    
        false_coords = {}
        for dim, indices in zip(self.source.coordinates.dims, false_indices_unique):
            false_coords[dim] = self.source.coordinates[dim][indices]
        
        return podpac.Coordinates([false_coords.get(dim) for dim in self.source.coordinates.dims], dims=self.source.coordinates.dims)

    def eval(self, request_coords):
        """
        Evaluate the data at the requested coordinates, fetching missing data from the source node and filling the Zarr cache as necessary.
        If requested coordinates are out of the source node's bounds, return an array filled with NaNs.

        Parameters
        ----------
        request_coords : podpac.Coordinates
            The coordinates at which data is requested.

        Returns
        -------
        data : np.ndarray
            The data at the requested coordinates. If coordinates were outside of the source's bounds, those positions will be filled with np.nan.
        """
        self._from_cache = False
        
        # Create a NaN-filled array with the shape of the request coordinates
        data_shape = [request_coords[d].size for d in request_coords.dims]
        data = np.full(data_shape, np.nan, dtype='float64')

        # Find valid request coordinates that are within the source's bounds
        valid_request_coords = request_coords.intersect(self.source.coordinates)

        if valid_request_coords.size > 0:
            subselect_coords = self.subselect_has(valid_request_coords)

            if subselect_coords is None:
                self._from_cache = True
            else:
                missing_data = self.get_source_data(subselect_coords)
                self.fill_zarr(missing_data, subselect_coords)

            c3, index_arrays = self.selector.select(self.source.coordinates, valid_request_coords)
            slices = self._create_slices(c3, index_arrays)

            # Use the slices to place data from Zarr cache into the correct location in the result array
            data[tuple(slices.get(dim) for dim in valid_request_coords.dims)] = self._z_node.dataset['data'][tuple(slices.get(dim) for dim in self.source.coordinates.dims)]

        return data