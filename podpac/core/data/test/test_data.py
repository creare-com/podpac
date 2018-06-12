"""
Test podpac.core.data.data module
"""

import pytest

import numpy as np
from traitlets import TraitError

from podpac.core.units import UnitsDataArray
from podpac.core.data.data import DataSource, COMMON_DATA_DOC, COMMON_DOC
from podpac.core.node import COMMON_NODE_DOC
from podpac.core.coordinate import Coordinate

class TestDataSource(object):
    """Test podpac.core.data.data module"""

    def test_allow_missing_modules(self):
        """TODO: Allow user to be missing rasterio and scipy"""
        pass

    def test_common_doc(self):
        """Test that all COMMON_DATA_DOC keys make it into the COMMON_DOC and overwrite COMMON_NODE_DOC keys"""

        for key in COMMON_DATA_DOC:
            assert key in COMMON_DOC and COMMON_DOC[key] == COMMON_DATA_DOC[key]

        for key in COMMON_NODE_DOC:
            if key in COMMON_DATA_DOC:
                assert key in COMMON_DOC and COMMON_DOC[key] != COMMON_NODE_DOC[key]
            else:
                assert key in COMMON_DOC and COMMON_DOC[key] == COMMON_NODE_DOC[key]

    @pytest.mark.skip(reason="traitlets does not currently honor the `allow_none` field")
    def test_traitlets_allow_none(self):
        """TODO: it seems like allow_none = False doesn't work
        """
        with pytest.raises(TraitError):
            DataSource(source=None)

        with pytest.raises(TraitError):
            DataSource(no_data_vals=None)

    def test_traitlets_errors(self):
        """ make sure traitlet errors are reased with improper inputs """

        with pytest.raises(TraitError):
            DataSource(interpolation=None)

        with pytest.raises(TraitError):
            DataSource(interpolation='myowninterp')

    def test_methods_must_be_implemented(self):
        """These class methods must be implemented"""

        node = DataSource()

        with pytest.raises(NotImplementedError):
            node.get_native_coordinates()

        with pytest.raises(NotImplementedError):
            node.get_data(None, None)

    def test_init(self):
        """Test constructor of DataSource (inherited from Node)"""

        node = MockDataSource()
        assert node

    def test_definition(self):
        """Test definition property method"""

        node = DataSource(source='test')
        d = node.definition

        assert d
        assert 'node' in d
        assert d['source'] == node.source
        assert d['params']['interpolation'] == node.interpolation


    class TestNativeCoordinates(object):
        """Test Get Data Subset """

        def test_native_coordinates_trait(self):
            """must be a coordinate or None """

            with pytest.raises(TraitError):
                node = DataSource(source='test', native_coordinates='not a coordinate')

            # define with native_coordinates keyword, but get_native_coordinates will still raise error
            node = DataSource(source='test',
                              native_coordinates=Coordinate(lat=(-10, 0, 5), lon=(-10, 0, 5), order=['lat', 'lon']))
            assert node.native_coordinates

            with pytest.raises(NotImplementedError):
                node.get_native_coordinates()

            # define with native_coordinates as None in keyword
            node = DataSource(source='test', native_coordinates=None)
            assert node.native_coordinates is None


        def test_get_native_coordinates(self):
            """by default `native_coordinates` property should map to get_native_coordinates via _native_coordinates_default"""

            node = MockDataSource(source='test')
            get_native_coordinates = node.get_native_coordinates()
            native_coordinates_default = node._native_coordinates_default()
            native_coordinates = node.native_coordinates

            assert get_native_coordinates
            assert native_coordinates
            assert get_native_coordinates == native_coordinates and native_coordinates_default == native_coordinates


        def test_native_coordinates_overwrite(self):
            """user can overwrite the native_coordinates property and still get_native_coordinates() appropriately"""

            node = MockDataSource(source='test')

            # TODO: this does not throw an error - should traitlets stop you after the fact?
            # with pytest.raises(TraitError):
            #     node.native_coordinates = 'not a coordinate'

            new_native_coordinates = Coordinate(lat=(-10, 0, 5), lon=(-10, 0, 5), order=['lat', 'lon'])
            node.native_coordinates = new_native_coordinates
            get_native_coordinates = node.get_native_coordinates()

            assert get_native_coordinates == new_native_coordinates


    class TestGetDataSubset(object):
        """Test Get Data Subset """
        
        def test_no_intersect(self):
            """Test where the requested coordinates have no intersection with the native coordinates """
            node = MockDataSource()
            coords = Coordinate(lat=(-30, -27, 5), lon=(-30, -27, 5), order=['lat', 'lon'])
            data = node.get_data_subset(coords)
            
            assert isinstance(data, UnitsDataArray)     # should return a UnitsDataArray
            assert np.all(np.isnan(data.values))        # all values should be nan


        def test_subset(self):
            """Test the standard operation of get_subset """

            node = MockDataSource()
            coords = Coordinate(lat=(-25, 0, 50), lon=(-25, 0, 50), order=['lat', 'lon'])
            data, coords_subset = node.get_data_subset(coords)

            assert isinstance(data, UnitsDataArray)             # should return a UnitsDataArray
            assert isinstance(coords_subset, Coordinate)        # should return the coordinates subset

            assert not np.all(np.isnan(data.values))            # all values should not be nan
            assert data.shape == (52, 52)
            assert np.min(data.lat.values) == -25
            assert np.max(data.lat.values) == .5
            assert np.min(data.lon.values) == -25
            assert np.max(data.lon.values) == 0.5

        def test_interpolate_nearest_preview(self):
            """test nearest_preview interpolation method. this runs before get_data_subset"""

            # test with same dims as native coords
            node = MockDataSource(interpolation='nearest_preview')
            coords = Coordinate(lat=(-25, 0, 20), lon=(-25, 0, 20), order=['lat', 'lon'])
            data, coords_subset = node.get_data_subset(coords)

            assert data.shape == (18, 18)
            assert coords_subset.shape == (18, 18)

            # test with different dims and uniform coordinates
            node = MockDataSource(interpolation='nearest_preview')
            coords = Coordinate(lat=(-25, 0, 20), order=['lat'])
            data, coords_subset = node.get_data_subset(coords)

            assert data.shape == (18, 101)
            assert coords_subset.shape == (18, 101)

            # test with different dims and non uniform coordinates
            node = MockNonuniformDataSource(interpolation='nearest_preview')
            coords = Coordinate(lat=[-25, -10, -2], order=['lat'])
            data, coords_subset = node.get_data_subset(coords)

            assert data.shape == (3, 3)
            assert coords_subset.shape == (3, 3)

class MockDataSource(DataSource):
    """ Mock Data Source for testing """

    # mock 100 x 100 grid of random values
    source = np.random.rand(101, 101)
    native_coordinates = Coordinate(lat=(-25, 25, 101), lon=(-25, 25, 101), order=['lat', 'lon'])

    def get_native_coordinates(self):
        """ """
        return self.native_coordinates

    def get_data(self, coordinates, coordinates_index):
        """ """
        s = coordinates_index
        d = self.initialize_coord_array(coordinates, 'data', fillval=self.source[s])
        return d

class MockNonuniformDataSource(DataSource):
    """ Mock Data Source for testing """

    # mock 100 x 100 grid of random values
    source = np.random.rand(3, 3)
    native_coordinates = Coordinate(lat=[-10, -2, -1], lon=[4, 32, 1], order=['lat', 'lon'])

    def get_native_coordinates(self):
        """ """
        return self.native_coordinates

    def get_data(self, coordinates, coordinates_index):
        """ """
        s = coordinates_index
        d = self.initialize_coord_array(coordinates, 'data', fillval=self.source[s])
        return d
