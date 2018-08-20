"""
Test podpac.core.data.datasource module
"""

import pytest

import numpy as np
from traitlets import TraitError
from xarray.core.coordinates import DataArrayCoordinates

from podpac.core.units import UnitsDataArray
from podpac.core.node import Style, COMMON_NODE_DOC
from podpac.core.coordinate.coordinate import Coordinate
from podpac.core.data.datasource import DataSource, COMMON_DATA_DOC, DATA_DOC
from podpac.core.data.types import rasterio
from podpac.core.data import datasource

####
# Mock test fixtures
####

DATA = np.random.rand(101, 101)
COORDINATES = Coordinate(lat=(-25, 25, 101), lon=(-25, 25, 101), order=['lat', 'lon'])

class MockDataSource(DataSource):
    """ Mock Data Source for testing """

    # mock 101 x 101 grid of random values, and some specified values
    source = DATA
    source[0, 0] = 10
    source[0, 1] = 1
    source[1, 0] = 5
    source[1, 1] = None
    native_coordinates = COORDINATES

    def get_native_coordinates(self):
        """ see DataSource """

        return self.native_coordinates

    def get_data(self, coordinates, coordinates_index):
        """ see DataSource """

        s = coordinates_index
        d = self.initialize_coord_array(coordinates, 'data', fillval=self.source[s])
        return d

class MockNonuniformDataSource(DataSource):
    """ Mock Data Source for testing that is non-uniform """

    # mock 3 x 3 grid of random values
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

class MockEmptyDataSource(DataSource):
    """ Mock Empty Data Source for testing 
        requires passing in source, native_coordinates to work correctly
    """

    def get_native_coordinates(self):
        """ see DataSource """

        return self.native_coordinates

    def get_data(self, coordinates, coordinates_index):
        """ see DataSource """

        s = coordinates_index
        d = self.initialize_coord_array(coordinates, 'data', fillval=self.source[s])
        return d

####
# Tests
####
class TestDataSource(object):
    """Test datasource.py module"""

    def test_allow_missing_modules(self):
        """TODO: Allow user to be missing rasterio and scipy"""
        pass

    def test_common_doc(self):
        """Test that all DATA_DOC keys make it into the COMMON_DATA_DOC and overwrite COMMON_NODE_DOC keys"""

        for key in DATA_DOC:
            assert key in COMMON_DATA_DOC and COMMON_DATA_DOC[key] == DATA_DOC[key]

        for key in COMMON_NODE_DOC:
            if key in DATA_DOC:
                assert key in COMMON_DATA_DOC and COMMON_DATA_DOC[key] != COMMON_NODE_DOC[key]
            else:
                assert key in COMMON_DATA_DOC and COMMON_DATA_DOC[key] == COMMON_NODE_DOC[key]

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
        assert d['attrs']['interpolation'] == node.interpolation


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


    class TestExecute(object):
        """Test execute methods"""


        def test_requires_coordinates(self):
            """execute requires coordinates input"""
            
            node = MockDataSource()
            
            with pytest.raises(TypeError):
                node.execute()

        def test_execute_at_native_coordinates(self):
            """execute node at native coordinates"""

            node = MockDataSource()
            output = node.execute(node.native_coordinates)

            assert isinstance(output, UnitsDataArray)
            assert output.shape == (101, 101)
            assert output[0, 0] == 10
            assert output.lat.shape == (101,)
            assert output.lon.shape == (101,)

            # assert coordinates
            assert isinstance(output.coords, DataArrayCoordinates)
            assert output.coords.dims == ('lat', 'lon')

            # assert attributes
            assert isinstance(output.attrs['layer_style'], Style)
            # assert output.attrs['params']['interpolation'] == 'nearest' # TODO

            # should be evaluated
            assert node.evaluated

        def test_execute_with_output(self):
            """execute node at native coordinates passing in output to store in"""
            
            node = MockDataSource()
            output = UnitsDataArray(np.zeros(node.source.shape),
                                    coords=node.native_coordinates.coords,
                                    dims=node.native_coordinates.dims)
            node.execute(node.native_coordinates, output=output)

            assert isinstance(output, UnitsDataArray)
            assert output.shape == (101, 101)
            assert np.all(output[0, 0] == 10)


        def test_execute_with_output_no_overlap(self):
            """execute node at native coordinates passing output that does not overlap"""
            
            node = MockDataSource()
            coords = Coordinate(lat=(-55, -45, 101), lon=(-55, -45, 101), order=['lat', 'lon'])
            data = np.zeros(node.source.shape)
            output = UnitsDataArray(data, coords=coords.coords, dims=coords.dims)
            node.execute(coords, output=output)

            assert isinstance(output, UnitsDataArray)
            assert output.shape == (101, 101)
            assert np.all(np.isnan(output[0, 0]))

        def test_remove_dims(self):
            """execute node with coordinates that have more dims that data source"""

            node = MockDataSource()
            coords = Coordinate(lat=(-25, 0, 20), lon=(-25, 0, 20), time=(1, 10, 10), order=['lat', 'lon', 'time'])
            output = node.execute(coords)

            assert output.coords.dims == ('lat', 'lon')  # coordinates of the DataSource, no the evaluated coordinates

        def test_no_overlap(self):
            """execute node with coordinates that do not overlap"""

            node = MockDataSource()
            coords = Coordinate(lat=(-55, -45, 20), lon=(-55, -45, 20), order=['lat', 'lon'])
            output = node.execute(coords)

            assert np.all(np.isnan(output))
        
        def test_no_data_vals(self):
            """ execute note with no_data_vals """

            node = MockDataSource(no_data_vals=[10, None])
            output = node.execute(node.native_coordinates)

            assert output.values[np.isnan(output)].shape == (2,)


    class TestInterpolateData(object):
        """test interpolation functions"""

        def test_one_data_point(self):
            """ test when there is only one data point """
            # TODO: as this is currently written, this would never make it to the interpolater
            
            source = np.random.rand(1,1)
            coords_src = Coordinate(lat=[20], lon=[20], order=['lat', 'lon'])
            coords_dst = Coordinate(lat=[20], lon=[20], order=['lat', 'lon'])

            # TODO: this doesn't work, but I feel like it shold
            # coords_dst = Coordinate(lat=[21], lon=[21], order=['lat', 'lon'])

            node = MockEmptyDataSource(source=source, native_coordinates=coords_src)
            data, coords_subset = node.get_data_subset(coords_dst)
            output = node._interpolate_data(data, coords_subset, coords_dst)

            assert isinstance(output, UnitsDataArray)

        def test_nearest_preview(self):
            """ test interpolation == 'nearest_preview' """
            source = np.random.rand(5)
            coords_src = Coordinate(lat=(0, 10, 5), order=['lat'])
            coords_dst = Coordinate(lat=[1, 1.2, 1.5, 5, 9], order=['lat'])

            node = MockEmptyDataSource(source=source, native_coordinates=coords_src)
            node.interpolation = 'nearest_preview'
            output = node.execute(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords['lat'])


        def test_interpolate_time(self):
            """ for now time uses nearest neighbor """

            source = np.random.rand(5)
            coords_src = Coordinate(time=(0, 10, 5), order=['time'])
            coords_dst = Coordinate(time=(1, 11, 5), order=['time'])

            node = MockEmptyDataSource(source=source, native_coordinates=coords_src)
            output = node.execute(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.time.values == coords_dst.coords['time'])

        @pytest.mark.skip(reason="not implemented yet")
        def test_interpolate_lat_time(self):
            """interpolate with n dims and time"""
            pass

        def test_interpolate_alt(self):
            """ for now alt uses nearest neighbor """

            source = np.random.rand(5)
            coords_src = Coordinate(alt=(0, 10, 5), order=['alt'])
            coords_dst = Coordinate(alt=(1, 11, 5), order=['alt'])

            node = MockEmptyDataSource(source=source, native_coordinates=coords_src)
            output = node.execute(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.alt.values == coords_dst.coords['alt'])


        def test_interpolate_nearest(self):
            """ regular nearest interpolation """

            source = np.random.rand(5, 5)
            coords_src = Coordinate(lat=(0, 10, 5), lon=(0, 10, 5), order=['lat', 'lon'])
            coords_dst = Coordinate(lat=(2, 12, 5), lon=(2, 12, 5), order=['lat', 'lon'])

            node = MockEmptyDataSource(source=source, native_coordinates=coords_src)
            node.interpolation = 'nearest'
            output = node.execute(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords['lat'])
            assert output.values[0, 0] == source[1, 1]

    class TestInterpolateRasterio(object):
        """test interpolation functions"""

        def test_interpolate_rasterio(self):
            """ regular interpolation using rasterio"""

            assert rasterio is not None

            rasterio_interps = ['nearest', 'bilinear', 'cubic', 'cubic_spline',
                                'lanczos', 'average', 'mode', 'max', 'min',
                                'med', 'q1', 'q3']
            source = np.random.rand(5, 5)
            coords_src = Coordinate(lat=(0, 10, 5), lon=(0, 10, 5), order=['lat', 'lon'])
            coords_dst = Coordinate(lat=(2, 12, 5), lon=(2, 12, 5), order=['lat', 'lon'])

            node = MockEmptyDataSource(source=source, native_coordinates=coords_src)

            # make sure it raises trait error
            with pytest.raises(TraitError):
                node.interpolation = 'myowninterp'
                output = node.execute(coords_dst)

            # make sure rasterio_interpolation method requires lat and lon
            # with pytest.raises(ValueError):
            #     coords_not_lon = Coordinate(lat=(0, 10, 5), order=['lat'])
            #     node = MockEmptyDataSource(source=source, native_coordinates=coords_not_lon)
            #     node.rasterio_interpolation(node, coords_src, coords_dst)

            # try all other interp methods
            for interp in rasterio_interps:
                node.interpolation = interp
                print(interp)
                output = node.execute(coords_dst)

                assert isinstance(output, UnitsDataArray)
                assert np.all(output.lat.values == coords_dst.coords['lat'])


        def test_interpolate_rasterio_descending(self):
            """should handle descending"""

            source = np.random.rand(5, 5)
            coords_src = Coordinate(lat=(10, 0, 5), lon=(10, 0, 5), order=['lat', 'lon'])
            coords_dst = Coordinate(lat=(12, 2, 5), lon=(12, 2, 5), order=['lat', 'lon'])
            
            node = MockEmptyDataSource(source=source, native_coordinates=coords_src)
            output = node.execute(coords_dst)
            
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords['lat'])
            assert np.all(output.lon.values == coords_dst.coords['lon'])

    class TestInterpolateNoRasterio(object):
        """test interpolation functions"""


        def test_interpolate_irregular_arbitrary(self):
            """ irregular interpolation """

            # suppress module to force statement
            datasource.rasterio = None

            rasterio_interps = ['nearest', 'bilinear', 'cubic', 'cubic_spline',
                                'lanczos', 'average', 'mode', 'max', 'min',
                                'med', 'q1', 'q3']
            source = np.random.rand(5, 5)
            coords_src = Coordinate(lat=(0, 10, 5), lon=(0, 10, 5), order=['lat', 'lon'])
            coords_dst = Coordinate(lat=(2, 12, 5), lon=(2, 12, 5), order=['lat', 'lon'])

            node = MockEmptyDataSource(source=source, native_coordinates=coords_src)

            for interp in rasterio_interps:
                node.interpolation = interp
                print(interp)
                output = node.execute(coords_dst)

                assert isinstance(output, UnitsDataArray)
                assert np.all(output.lat.values == coords_dst.coords['lat'])

        def test_interpolate_irregular_arbitrary_2dims(self):
            """ irregular interpolation """

            datasource.rasterio = None

            # try >2 dims
            source = np.random.rand(5, 5, 3)
            coords_src = Coordinate(lat=(0, 10, 5), lon=(0, 10, 5), time=(2, 5, 3), order=['lat', 'lon', 'time'])
            coords_dst = Coordinate(lat=(2, 12, 5), lon=(2, 12, 5), time=(2, 5, 3), order=['lat', 'lon', 'time'])
            
            node = MockEmptyDataSource(source=source, native_coordinates=coords_src)
            node.interpolation = 'nearest'
            output = node.execute(coords_dst)
            
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords['lat'])
            assert np.all(output.lon.values == coords_dst.coords['lon'])
            assert np.all(output.time.values == coords_dst.coords['time'])

        def test_interpolate_irregular_arbitrary_descending(self):
            """should handle descending"""

            datasource.rasterio = None

            source = np.random.rand(5, 5)
            coords_src = Coordinate(lat=(10, 0, 5), lon=(10, 0, 5), order=['lat', 'lon'])
            coords_dst = Coordinate(lat=(2, 12, 5), lon=(2, 12, 5), order=['lat', 'lon'])
            
            node = MockEmptyDataSource(source=source, native_coordinates=coords_src)
            node.interpolation = 'nearest'
            output = node.execute(coords_dst)
            
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords['lat'])
            assert np.all(output.lon.values == coords_dst.coords['lon'])

        def test_interpolate_irregular_arbitrary_swap(self):
            """should handle descending"""

            datasource.rasterio = None

            source = np.random.rand(5, 5)
            coords_src = Coordinate(lon=(10, 0, 5), lat=(10, 0, 5), order=['lon', 'lat'])
            coords_dst = Coordinate(lat=(2, 12, 5), lon=(2, 12, 5), order=['lat', 'lon'])
            
            node = MockEmptyDataSource(source=source, native_coordinates=coords_src)
            node.interpolation = 'nearest'
            output = node.execute(coords_dst)
            
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords['lat'])
            assert np.all(output.lon.values == coords_dst.coords['lon'])

        def test_interpolate_irregular_lat_lon(self):
            """ irregular interpolation """

            datasource.rasterio = None

            source = np.random.rand(5, 5)
            coords_src = Coordinate(lat=(0, 10, 5), lon=(0, 10, 5), order=['lat', 'lon'])
            coords_dst = Coordinate(lat_lon=([0, 2, 4, 6, 8, 10], [0, 2, 4, 5, 6, 10]))

            node = MockEmptyDataSource(source=source, native_coordinates=coords_src)
            node.interpolation = 'nearest'
            output = node.execute(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat_lon.values == coords_dst.coords['lat_lon'])
            assert output.values[0] == source[0, 0]
            assert output.values[1] == source[1, 1]
            assert output.values[-1] == source[-1, -1]

        def test_interpolate_point(self):
            """ interpolate point data to nearest neighbor with various coords_dst"""

            datasource.rasterio = None

            source = np.random.rand(5)
            coords_src = Coordinate(lat_lon=([0, 2, 4, 6, 8, 10], [0, 2, 4, 5, 6, 10]))
            coords_dst = Coordinate(lat_lon=([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]))
            node = MockEmptyDataSource(source=source, native_coordinates=coords_src)

            output = node.execute(coords_dst)
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat_lon.values == coords_dst.coords['lat_lon'])
            assert output.values[0] == source[0]
            assert output.values[-1] == source[3]


            coords_dst = Coordinate(lat=[1, 2, 3, 4, 5], lon=[1, 2, 3, 4, 5], order=['lat', 'lon'])
            output = node.execute(coords_dst)
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords['lat'])
            assert output.values[0, 0] == source[0]
            assert output.values[-1, -1] == source[3]
