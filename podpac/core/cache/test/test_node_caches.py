import pytest
import numpy as np
from podpac import Coordinates
from podpac.data import Array
from podpac.caches import ZarrCache

class TestZarrCache:
    
    @pytest.fixture
    def source(self):
        lat = np.linspace(0, 10, 11)
        lon = np.linspace(0, 10, 11)
        time = ['2018-01-01', '2018-01-02']
        coords = Coordinates([lat, lon, time], ['lat', 'lon', 'time'])
        return Array(source=np.random.rand(coords.shape[0], coords.shape[1], coords.shape[2]), coordinates=coords)

    def test_ZarrCache_fill_and_retrieve(self, source):

        # Initialize ZarrCache node
        node = ZarrCache(source=source)
        coords = source.coordinates

        # Eval the node, this will also fill the Zarr cache with source data
        data_filled = node.eval(coords)

        # Create a new node instance with same configuration
        node_retrieved = ZarrCache(source=source, 
                                zarr_path_data=node.zarr_path_data, 
                                zarr_path_bool=node.zarr_path_bool)

        # Retrieve data from the new node, which should come from the Zarr cache
        data_retrieved = node_retrieved.eval(coords)

        # Check the data retrieved from the Zarr cache is identical to the filled data
        np.testing.assert_allclose(data_filled, data_retrieved)
        assert node_retrieved._from_cache

    def test_ZarrCache_missing_data(self, source):

        # Initialize ZarrCache node
        node = ZarrCache(source=source)

        # Simulate some request coordinates that go beyond the existing source coordinates
        lat = np.linspace(0, 15, 16)  # goes up to 15 instead of 10
        request_coords = Coordinates([lat, source.coordinates['lon'], source.coordinates['time']], ['lat', 'lon', 'time'])

        # Eval the node, this should raise a ValueError because some requested data is not available
        with pytest.raises(ValueError):
            node.eval(request_coords)

    def test_ZarrCache_partial_caching(self, source):
        
        # Initialize ZarrCache node
        node = ZarrCache(source=source)

        # Create a subselection of the original coordinates
        lat_sub = np.linspace(0, 5, 6)  # only go up to 5 instead of 10
        request_coords_sub = Coordinates([lat_sub, source.coordinates['lon'], source.coordinates['time']], ['lat', 'lon', 'time'])

        # Eval the node with the subselection of coordinates, this will also fill the cache
        node.eval(request_coords_sub)

        # Now create a request that includes the previous coordinates plus new ones
        lat_extended = np.linspace(0, 7, 8)  # extends to 7, includes previously cached coords
        request_coords_extended = Coordinates([lat_extended, source.coordinates['lon'], source.coordinates['time']], ['lat', 'lon', 'time'])

        # Now check if subselect_has correctly identifies the non-cached coordinates
        missing_coords = node.subselect_has(request_coords_extended)
        
        # The missing coordinates should be those that were not included in the initial eval
        expected_missing_coords = Coordinates([np.linspace(6, 7, 2), source.coordinates['lon'], source.coordinates['time']], ['lat', 'lon', 'time'])

        assert missing_coords == expected_missing_coords

    def test_ZarrCache_clear_cache(self, source):

        # Initialize ZarrCache node
        node = ZarrCache(source=source)
        coords = source.coordinates

        # Eval the node, this will also fill the Zarr cache with source data
        node.eval(coords)

        # Clear the cache
        node.clear_cache()

        # Create a new node instance with same configuration
        node_retrieved = ZarrCache(source=source, 
                                zarr_path_data=node.zarr_path_data, 
                                zarr_path_bool=node.zarr_path_bool)
        
        # Since we have cleared the cache, it should be empty
        # All elements in the data array should be NaN, and all elements in the boolean array should be False
        np.testing.assert_array_equal(np.isnan(node_retrieved._z_node.dataset['data'][:]), True)
        np.testing.assert_array_equal(node_retrieved._z_bool.dataset['contains'][:], False)

        # An eval attempt should not be able to retrieve from the cache 
        node_retrieved.eval(coords)
        assert not node_retrieved._from_cache
      