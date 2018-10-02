"""
Test podpac.core.data.datasource module
"""

import pytest

import numpy as np
import xarray as xr
from traitlets import TraitError
from xarray.core.coordinates import DataArrayCoordinates

from podpac.core.units import UnitsDataArray
from podpac.core.node import Style, COMMON_NODE_DOC
from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.data.datasource import DataSource, COMMON_DATA_DOC, DATA_DOC
from podpac.core.data.types import rasterio
from podpac.core.data import datasource

####
# Mock test fixtures
####


DATA = np.random.rand(101, 101)
COORDINATES = Coordinates([clinspace(-25, 25, 101), clinspace(-25, 25, 101)], dims=['lat', 'lon'])


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

class MockNonuniformDataSource(DataSource):
    """ Mock Data Source for testing that is non-uniform """

    # mock 3 x 3 grid of random values
    source = np.random.rand(3, 3)
    native_coordinates = Coordinates([[-10, -2, -1], [4, 32, 1]], dims=['lat', 'lon'])

    def get_native_coordinates(self):
        """ """
        return self.native_coordinates

    def get_data(self, coordinates, coordinates_index):
        """ """
        s = coordinates_index
        d = self.initialize_coord_array(coordinates, 'data', fillval=self.source[s])
        return d

class MockDataSourceReturnsArray(DataSource):
    """ Mock Data Source for testing that returns np.ndarray """

    # mock 101 x 101 grid of random values, and some specified values
    source = DATA
    native_coordinates = COORDINATES

    def get_native_coordinates(self):
        """ see DataSource """

        return self.native_coordinates

    def get_data(self, coordinates, coordinates_index):
        """ returns an np.ndarray from get data"""

        return self.source[coordinates_index]

class MockDataSourceReturnsDataArray(DataSource):
    """ Mock Data Source for testing returns xarray DataArray """

    # mock 101 x 101 grid of random values, and some specified values
    source = DATA
    native_coordinates = COORDINATES

    def get_native_coordinates(self):
        """ see DataSource """

        return self.native_coordinates

    def get_data(self, coordinates, coordinates_index):
        """ returns an xr.DataArray """

        return xr.DataArray(self.source[coordinates_index])

####
# Tests
####
class TestDataSource(object):
    """Test datasource.py module"""

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
            MockEmptyDataSource(source=None)

        with pytest.raises(TraitError):
            MockEmptyDataSource(nan_vals=None)

    def test_traitlets_errors(self):
        """ make sure traitlet errors are reased with improper inputs """

        with pytest.raises(TraitError):
            MockEmptyDataSource(nan_vals={})

        with pytest.raises(TraitError):
            MockEmptyDataSource(interpolation='myowninterp')

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

        # TODO: add interpolation definition testing


    class TestNativeCoordinates(object):
        """Test Get Data Subset """

        def test_missing_native_coordinates(self):
            """must be a coordinate or None """

            with pytest.raises(TraitError):
                node = DataSource(source='test', native_coordinates='not a coordinate')

            # if get_native_coordinates is not defined on data source class, try to return native_coordinates
            node = DataSource(source='test', native_coordinates=COORDINATES)
            assert node.native_coordinates

            # raise if native coordinates are not defined on input and get_native_coordinates is not defined on class
            node = DataSource(source='test')
            with pytest.raises(NotImplementedError):
                node.native_coordinates

            # raise if native_coordinates are none
            node = DataSource(source='test', native_coordinates=None)
            with pytest.raises(NotImplementedError):
                node.get_native_coordinates()

        def test_get_native_coordinates(self):
            """if native_coordinates is None, get_native_coordinates should set native_coordiantes property"""

            # if get_native_coordinates is not defined on data source class, try to return native_coordinates
            node = DataSource(source='test', native_coordinates=COORDINATES)
            native_coordinates = node.native_coordinates
            get_native_coordinates = node.get_native_coordinates()
            assert get_native_coordinates
            assert native_coordinates
            assert get_native_coordinates == native_coordinates

            # data source defines get_native_coordinates
            node = MockDataSource(source='test')
            get_native_coordinates = node.get_native_coordinates()
            native_coordinates = node.native_coordinates

            assert get_native_coordinates
            assert native_coordinates
            assert get_native_coordinates == native_coordinates


        def test_native_coordinates_overwrite(self):
            """user can overwrite the native_coordinates property and still get_native_coordinates() appropriately"""

            node = MockDataSource(source='test')

            # TODO: this does not throw an error - should traitlets stop you after the fact?
            # with pytest.raises(TraitError):
            #     node.native_coordinates = 'not a coordinate'

            new_native_coordinates = Coordinates([clinspace(-10, 0, 5), clinspace(-10, 0, 5)], dims=['lat', 'lon'])
            node.native_coordinates = new_native_coordinates
            get_native_coordinates = node.get_native_coordinates()

            assert get_native_coordinates == new_native_coordinates

        def test_no_get_native_coordinates(self):
            """implementing data source class can leave off get_native_coordinates if the user defines them on init"""

    @pytest.mark.xfail(reason="MockDataSource has no attribute get_data_subset")
    class TestGetDataSubset(object):
        """Test Get Data Subset """
        
        # def test_no_intersect(self):
        #     """Test where the requested coordinates have no intersection with the native coordinates """
        #     node = MockDataSource()
        #     coords = Coordinates([clinspace(-30, -27, 5), clinspace(-30, -27, 5)], dims=['lat', 'lon'])
        #     data = node.get_data_subset(coords)
            
        #     assert isinstance(data, UnitsDataArray)     # should return a UnitsDataArray
        #     assert np.all(np.isnan(data.values))        # all values should be nan


        def test_subset(self):
            """Test the standard operation of get_subset """

            node = MockDataSource()
            coords = Coordinates([clinspace(-25, 0, 50), clinspace(-25, 0, 50)], dims=['lat', 'lon'])
            data, coords_subset = node.get_data_subset(coords)

            assert isinstance(data, UnitsDataArray)             # should return a UnitsDataArray
            assert isinstance(coords_subset, Coordinates)       # should return the coordinates subset

            assert not np.all(np.isnan(data.values))            # all values should not be nan
            assert data.shape == (52, 52)
            assert np.min(data.lat.values) == -25
            assert np.max(data.lat.values) == .5
            assert np.min(data.lon.values) == -25
            assert np.max(data.lon.values) == 0.5

        def test_interpolate_nearest_preview(self):
            """test nearest_preview interpolation method. this runs before get_data_subset"""

            # test with same dims as native coords
            node = MockDataSource(interpolation='nearest')
            coords = Coordinates([clinspace(-25, 0, 20), clinspace(-25, 0, 20)], dims=['lat', 'lon'])
            data, coords_subset = node.get_data_subset(coords)

            assert data.shape == (18, 18)
            assert coords_subset.shape == (18, 18)

            # test with different dims and uniform coordinates
            node = MockDataSource(interpolation='nearest')
            coords = Coordinates([clinspace(-25, 0, 20)], dims=['lat'])
            data, coords_subset = node.get_data_subset(coords)

            assert data.shape == (18, 101)
            assert coords_subset.shape == (18, 101)

            # test with different dims and non uniform coordinates
            node = MockNonuniformDataSource(interpolation='nearest')
            coords = Coordinates([[-25, -10, -2]], dims=['lat'])
            data, coords_subset = node.get_data_subset(coords)

            # TODO: in the future this should have default to pass back native_coordinates
            # assert node.get_native_coordinates()


    class TestEvaluate(object):
        """Test evaluate methods"""


        def test_requires_coordinates(self):
            """evaluate requires coordinates input"""
            
            node = MockDataSource()
            
            with pytest.raises(TypeError):
                node.execute()

        def test_evaluate_at_native_coordinates(self):
            """evaluate node at native coordinates"""

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

            # should be evaluated
            assert node.evaluated

        def test_evaluate_with_output(self):
            """evaluate node at native coordinates passing in output to store in"""
            
            node = MockDataSource()
            output = UnitsDataArray(np.zeros(node.source.shape),
                                    coords=node.native_coordinates.coords,
                                    dims=node.native_coordinates.dims)
            node.execute(node.native_coordinates, output=output)

            assert isinstance(output, UnitsDataArray)
            assert output.shape == (101, 101)
            assert np.all(output[0, 0] == 10)


        def test_evaluate_with_output_no_overlap(self):
            """evaluate node at native coordinates passing output that does not overlap"""
            
            node = MockDataSource()
            coords = Coordinates([clinspace(-55, -45, 101), clinspace(-55, -45, 101)], dims=['lat', 'lon'])
            data = np.zeros(node.source.shape)
            output = UnitsDataArray(data, coords=coords.coords, dims=coords.dims)
            node.execute(coords, output=output)

            assert isinstance(output, UnitsDataArray)
            assert output.shape == (101, 101)
            assert np.all(np.isnan(output[0, 0]))

        def test_remove_dims(self):
            """evaluate node with coordinates that have more dims that data source"""

            node = MockDataSource()
            coords = Coordinates(
                [clinspace(-25, 0, 20), clinspace(-25, 0, 20), clinspace(1, 10, 10)], dims=['lat', 'lon', 'time'])
            output = node.execute(coords)

            assert output.coords.dims == ('lat', 'lon')  # coordinates of the DataSource, no the evaluated coordinates

        def test_no_overlap(self):
            """evaluate node with coordinates that do not overlap"""

            node = MockDataSource()
            coords = Coordinates([clinspace(-55, -45, 20), clinspace(-55, -45, 20)], dims=['lat', 'lon'])
            output = node.execute(coords)

            assert np.all(np.isnan(output))
        
        def test_nan_vals(self):
            """ evaluate note with nan_vals """

            node = MockDataSource(nan_vals=[10, None])
            output = node.execute(node.native_coordinates)

            assert output.values[np.isnan(output)].shape == (2,)

        def test_return_ndarray(self):
            
            node = MockDataSourceReturnsArray()
            output = node.execute(node.native_coordinates)

            assert isinstance(output, UnitsDataArray)
            assert node.native_coordinates['lat'].coordinates[4] == output.coords['lat'].values[4]

        def test_return_DataArray(self):
            node = MockDataSourceReturnsDataArray()
            output = node.execute(node.native_coordinates)

            assert isinstance(output, UnitsDataArray)
            assert node.native_coordinates['lat'].coordinates[4] == output.coords['lat'].values[4]


    class TestInterpolateData(object):
        """test interpolation functions"""

        def test_one_data_point(self):
            """ test when there is only one data point """
            # TODO: as this is currently written, this would never make it to the interpolater
            pass
            
        def test_nearest_preview(self):
            """ test interpolation == 'nearest_preview' """
            source = np.random.rand(5)
            coords_src = Coordinates([clinspace(0, 10, 5,)], dims=['lat'])
            coords_dst = Coordinates([[1, 1.2, 1.5, 5, 9]], dims=['lat'])

            node = MockEmptyDataSource(source=source, native_coordinates=coords_src, interpolation='nearest_preview')
            output = node.execute(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords['lat'])


        def test_interpolate_time(self):
            """ for now time uses nearest neighbor """

            source = np.random.rand(5)
            coords_src = Coordinates([clinspace(0, 10, 5,)], dims=['time'])
            coords_dst = Coordinates([clinspace(1, 11, 5,)], dims=['time'])

            node = MockEmptyDataSource(source=source, native_coordinates=coords_src)
            output = node.execute(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.time.values == coords_dst.coords['time'])

        def test_interpolate_lat_time(self):
            """interpolate with n dims and time"""
            pass

        def test_interpolate_alt(self):
            """ for now alt uses nearest neighbor """

            source = np.random.rand(5)
            coords_src = Coordinates([clinspace(0, 10, 5)], dims=['alt'])
            coords_dst = Coordinates([clinspace(1, 11, 5)], dims=['alt'])

            node = MockEmptyDataSource(source=source, native_coordinates=coords_src)
            output = node.execute(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.alt.values == coords_dst.coords['alt'])


        def test_interpolate_nearest(self):
            """ regular nearest interpolation """

            source = np.random.rand(5, 5)
            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=['lat', 'lon'])
            coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=['lat', 'lon'])

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
            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=['lat', 'lon'])
            coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=['lat', 'lon'])

            node = MockEmptyDataSource(source=source, native_coordinates=coords_src)

            # make sure it raises trait error
            with pytest.raises(TraitError):
                node.interpolation = 'myowninterp'
                output = node.execute(coords_dst)

            # make sure rasterio_interpolation method requires lat and lon
            # with pytest.raises(ValueError):
            #     coords_not_lon = Coordinates([clinspace(0, 10, 5)], dims=['lat'])
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
            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=['lat', 'lon'])
            coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=['lat', 'lon'])
            
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
            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=['lat', 'lon'])
            coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=['lat', 'lon'])

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
            coords_src = Coordinates(
                [clinspace(0, 10, 5), clinspace(0, 10, 5), [2, 3, 5]], dims=['lat', 'lon', 'time'])
            coords_dst = Coordinates(
                [clinspace(2, 12, 5), clinspace(2, 12, 5), [2, 3, 5]], dims=['lat', 'lon', 'time'])
            
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
            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=['lat', 'lon'])
            coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=['lat', 'lon'])
            
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
            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=['lat', 'lon'])
            coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=['lat', 'lon'])
            
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
            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=['lat', 'lon'])
            coords_dst = Coordinates([[[0, 2, 4, 6, 8, 10], [0, 2, 4, 5, 6, 10]]], dims=['lat_lon'])

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

            source = np.random.rand(6)
            coords_src = Coordinates([[[0, 2, 4, 6, 8, 10], [0, 2, 4, 5, 6, 10]]], dims=['lat_lon'])
            coords_dst = Coordinates([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]], dims=['lat_lon'])
            node = MockEmptyDataSource(source=source, native_coordinates=coords_src)

            output = node.execute(coords_dst)
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat_lon.values == coords_dst.coords['lat_lon'])
            assert output.values[0] == source[0]
            assert output.values[-1] == source[3]


            coords_dst = Coordinates([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dims=['lat', 'lon'])
            output = node.execute(coords_dst)
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords['lat'])
            assert output.values[0, 0] == source[0]
            assert output.values[-1, -1] == source[3]
