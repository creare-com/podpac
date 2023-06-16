# Create a PODPAC node out of a zarr archive
import numpy as np
import podpac
from podpac.data import Zarr
from podpac.core.interpolation.selector import Selector
import zarr
from podpac.core.node import Node, NodeDefinitionError, NodeException
from podpac.core.utils import NodeTrait
from podpac.core.utils import trait_is_defined
import traitlets as tl

class ZarrCache(Node):
    """
    ZarrCache class which extends the Node class from PODPAC.
    This class provides functionality for creating a PODPAC node from a zarr archive.
    It manages the caching of the data in a zarr format and implements the logic to fill the cache with missing data.
    
    Attributes
    ----------
    source_node : NodeTrait
        Source PODPAC node.
    zarr_path_data : str
        Path to the zarr file that will store the data.
    zarr_path_bool : str
        Path to the zarr file that will store the boolean mask indicating the presence of data.
    group_data : zarr.hierarchy.Group
        Zarr group for storing the data.
    group_bool : zarr.hierarchy.Group
        Zarr group for storing the boolean mask.
    selector : Selector
        Selector instance for data interpolation.

    Methods
    -------
    create_empty_zarr() -> None
        Create an empty zarr archive with the same shape as the source_node and a boolean mask.
    get_source_data(request_coords) -> np.ndarray
        Fetch data from the source_node for the specified coordinates.
    fill_zarr(data, request_coords) -> None
        Fill the zarr cache with data at the specified coordinates.
    subselect_has(request_coords) -> podpac.Coordinates
        Fetch the coordinates in the requested coordinates for which the zarr cache does not have data yet.
    eval(request_coords) -> np.ndarray
        Evaluate the data at the requested coordinates, fetching missing data from the source node and filling the zarr cache as necessary.
    """
    
    source_node = NodeTrait(allow_none=True).tag(attr=True, required=True)
    zarr_path_data = tl.Unicode().tag(attr=True, required=True)
    zarr_path_bool = tl.Unicode().tag(attr=True, required=True)
    group_data = tl.Instance(zarr.hierarchy.Group).tag(attr=True)
    group_bool = tl.Instance(zarr.hierarchy.Group).tag(attr=True)
    selector = tl.Instance(Selector, allow_none=True).tag(attr=True)
    
    
    @tl.default('selector')
    def _default_selector(self):
        return Selector(method='nearest')
    

    def create_empty_zarr(self):
        """
        Create an empty zarr archive and a boolean mask with the same shape as the source_node.

        This method initializes a zarr archive with an empty data array and a boolean mask indicating the presence of data.
        Both arrays have the same shape as the source_node. The data array is filled with NaNs, while the boolean mask is filled with False.
        """
        data_shape = self.source_node.shape
        empty_data = np.empty(data_shape)
        false_bool = np.zeros(data_shape, dtype=bool)
        self.group_data.array('data', empty_data, chunks=empty_data.shape, dtype='f8')
        self.group_bool.array('contains', false_bool, chunks=empty_data.shape, dtype='bool')

    def get_source_data(self, request_coords):
        """
        Retrieve data from the source_node at the specified coordinates.

        Parameters
        ----------
        request_coords : podpac.Coordinates
            The coordinates at which data is requested from the source_node.

        Returns
        -------
        data : np.ndarray
            The data retrieved from the source_node at the specified coordinates.
        """
        data = self.source_node.eval(request_coords).data
        return data

    def fill_zarr(self, data, request_coords):
        """
        Fill the zarr cache with data at the specified coordinates.

        Parameters
        ----------
        data : np.ndarray
            The data to be stored in the zarr cache.
        request_coords : podpac.Coordinates
            The coordinates at which the data should be stored in the zarr cache.
        """
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
        """
        Fetch the coordinates for which the zarr cache does not have data yet.

        Parameters
        ----------
        request_coords : podpac.Coordinates
            The coordinates at which data is requested.

        Returns
        -------
        false_coords : podpac.Coordinates
            The subset of the requested coordinates for which the zarr cache does not have data yet.
        """
        z_bool = Zarr(source=self.zarr_path_bool, coordinates=self.source_node.coordinates, file_mode="r")
        
        c3, index_arrays = self.selector.select(self.source_node.coordinates, request_coords)
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
        """
        Evaluate the data at the requested coordinates, fetching missing data from the source node and filling the zarr cache as necessary.

        Parameters
        ----------
        request_coords : podpac.Coordinates
            The coordinates at which data is requested.

        Returns
        -------
        data : np.ndarray
            The data at the requested coordinates.
        """
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
