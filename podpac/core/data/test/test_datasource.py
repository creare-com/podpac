"""
Test podpac.core.data.datasource module
"""

from collections import OrderedDict

import pytest

import numpy as np
import traitlets as tl
import xarray as xr
from xarray.core.coordinates import DataArrayCoordinates

from podpac.core.units import UnitsDataArray
from podpac.core.node import COMMON_NODE_DOC, NodeException
from podpac.core.style import Style
from podpac.core.coordinates import Coordinates, clinspace, crange
from podpac.core.interpolation.interpolation import Interpolation, Interpolator
from podpac.core.interpolation.interpolator import Interpolator
from podpac.core.data.datasource import DataSource, COMMON_DATA_DOC, DATA_DOC


class MockDataSource(DataSource):
    data = np.ones((11, 11))
    data[0, 0] = 10
    data[0, 1] = 1
    data[1, 0] = 5
    data[1, 1] = None

    def get_coordinates(self):
        return Coordinates([clinspace(-25, 25, 11), clinspace(-25, 25, 11)], dims=["lat", "lon"])

    def get_data(self, coordinates, coordinates_index):
        return self.create_output_array(coordinates, data=self.data[coordinates_index])


class MockDataSourceStacked(DataSource):
    data = np.arange(11)

    def get_coordinates(self):
        return Coordinates([clinspace((-25, -25), (25, 25), 11)], dims=["lat_lon"])

    def get_data(self, coordinates, coordinates_index):
        return self.create_output_array(coordinates, data=self.data[coordinates_index])


class TestDataDocs(object):
    def test_common_data_doc(self):
        # all DATA_DOC keys should be in the COMMON_DATA_DOC
        for key in DATA_DOC:
            assert key in COMMON_DATA_DOC
            assert COMMON_DATA_DOC[key] == DATA_DOC[key]

        # DATA_DOC should overwrite COMMON_NODE_DOC keys
        for key in COMMON_NODE_DOC:
            assert key in COMMON_DATA_DOC

            if key in DATA_DOC:
                assert COMMON_DATA_DOC[key] != COMMON_NODE_DOC[key]
            else:
                assert COMMON_DATA_DOC[key] == COMMON_NODE_DOC[key]


class TestDataSource(object):
    def test_init(self):
        node = DataSource()

    def test_get_data_not_implemented(self):
        node = DataSource()

        with pytest.raises(NotImplementedError):
            node.get_data(None, None)

    def test_get_coordinates_not_implemented(self):
        node = DataSource()
        with pytest.raises(NotImplementedError):
            node.get_coordinates()

    def test_coordinates(self):
        # not implemented
        node = DataSource()
        with pytest.raises(NotImplementedError):
            node.coordinates

        # use get_coordinates (once)
        class MyDataSource(DataSource):
            get_coordinates_called = 0

            def get_coordinates(self):
                self.get_coordinates_called += 1
                return Coordinates([])

        node = MyDataSource()
        assert node.get_coordinates_called == 0
        assert isinstance(node.coordinates, Coordinates)
        assert node.get_coordinates_called == 1
        assert isinstance(node.coordinates, Coordinates)
        assert node.get_coordinates_called == 1

        # can't set
        with pytest.raises(AttributeError, match="can't set attribute"):
            node.coordinates = Coordinates([])

    def test_cache_coordinates(self):
        class MyDataSource(DataSource):
            get_coordinates_called = 0

            def get_coordinates(self):
                self.get_coordinates_called += 1
                return Coordinates([])

        a = MyDataSource(cache_coordinates=True, cache_ctrl=["ram"])
        b = MyDataSource(cache_coordinates=True, cache_ctrl=["ram"])
        c = MyDataSource(cache_coordinates=False, cache_ctrl=["ram"])
        d = MyDataSource(cache_coordinates=True, cache_ctrl=[])

        a.rem_cache("*")
        b.rem_cache("*")
        c.rem_cache("*")
        d.rem_cache("*")

        # get_coordinates called once
        assert not a.has_cache("coordinates")
        assert a.get_coordinates_called == 0
        assert isinstance(a.coordinates, Coordinates)
        assert a.get_coordinates_called == 1
        assert isinstance(a.coordinates, Coordinates)
        assert a.get_coordinates_called == 1

        # coordinates is cached to a, b, and c
        assert a.has_cache("coordinates")
        assert b.has_cache("coordinates")
        assert c.has_cache("coordinates")
        assert not d.has_cache("coordinates")

        # b: use cache, get_coordinates not called
        assert b.get_coordinates_called == 0
        assert isinstance(b.coordinates, Coordinates)
        assert b.get_coordinates_called == 0

        # c: don't use cache, get_coordinates called
        assert c.get_coordinates_called == 0
        assert isinstance(c.coordinates, Coordinates)
        assert c.get_coordinates_called == 1

        # d: use cache but there is no ram cache for this node, get_coordinates is called
        assert d.get_coordinates_called == 0
        assert isinstance(d.coordinates, Coordinates)
        assert d.get_coordinates_called == 1

    def test_set_coordinates(self):
        node = MockDataSource()
        node.set_coordinates(Coordinates([]))
        assert node.coordinates == Coordinates([])
        assert node.coordinates != node.get_coordinates()

        # don't overwrite
        node = MockDataSource()
        node.coordinates
        node.set_coordinates(Coordinates([]))
        assert node.coordinates != Coordinates([])
        assert node.coordinates == node.get_coordinates()

    def test_invalid_interpolation(self):
        with pytest.raises(tl.TraitError):
            DataSource(interpolation="myowninterp")

    def test_invalid_nan_vals(self):
        with pytest.raises(tl.TraitError):
            DataSource(nan_vals={})

        with pytest.raises(tl.TraitError):
            DataSource(nan_vals=10)

    def test_repr(self):
        node = DataSource()
        repr(node)

    def test_interpolation_class(self):
        node = DataSource(interpolation="max")
        assert node.interpolation_class
        assert isinstance(node.interpolation_class, Interpolation)
        assert node.interpolation_class.definition == "max"
        assert isinstance(node.interpolation_class.config, OrderedDict)
        assert ("default",) in node.interpolation_class.config

    def test_interpolators(self):
        node = MockDataSource()
        node.eval(node.coordinates)

        assert isinstance(node.interpolators, OrderedDict)

        # when no interpolation happens, this returns as an empty ordered dict
        assert not node.interpolators

        # when interpolation happens, this is filled
        node.eval(Coordinates([clinspace(-11, 11, 7), clinspace(-11, 11, 7)], dims=["lat", "lon"]))
        assert "lat" in list(node.interpolators.keys())[0]
        assert "lon" in list(node.interpolators.keys())[0]
        assert isinstance(list(node.interpolators.values())[0], Interpolator)

    def test_evaluate_at_coordinates(self):
        """evaluate node at coordinates"""

        node = MockDataSource()
        output = node.eval(node.coordinates)

        assert isinstance(output, UnitsDataArray)
        assert output.shape == (11, 11)
        assert output[0, 0] == 10
        assert output.lat.shape == (11,)
        assert output.lon.shape == (11,)

        # assert coordinates
        assert isinstance(output.coords, DataArrayCoordinates)
        assert output.coords.dims == ("lat", "lon")

        # assert attributes
        assert isinstance(output.attrs["layer_style"], Style)

    def test_evaluate_with_output(self):
        node = MockDataSource()

        # initialize a large output array
        fullcoords = Coordinates([crange(20, 30, 1), crange(20, 30, 1)], dims=["lat", "lon"])
        output = node.create_output_array(fullcoords)

        # evaluate a subset of the full coordinates
        coords = Coordinates([fullcoords["lat"][3:8], fullcoords["lon"][3:8]])

        # after evaluation, the output should be
        # - the same where it was not evaluated
        # - NaN where it was evaluated but doesn't intersect with the data source
        # - 1 where it was evaluated and does intersect with the data source (because this datasource is all 0)
        expected = output.copy()
        expected[3:8, 3:8] = np.nan
        expected[3:8, 3:8] = 1.0

        # evaluate the subset coords, passing in the cooresponding slice of the initialized output array
        # TODO: discuss if we should be using the same reference to output slice?
        output[3:8, 3:8] = node.eval(coords, output=output[3:8, 3:8])

        np.testing.assert_equal(output.data, expected.data)

    def test_evaluate_with_output_different_crs(self):

        # default crs EPSG:4193
        node = MockDataSource()
        c = Coordinates([crange(20, 30, 1), crange(20, 30, 1)], dims=["lat", "lon"])
        c_x = Coordinates([crange(20, 30, 1), crange(20, 30, 1)], dims=["lat", "lon"], crs="EPSG:2193")

        # this will not throw an error because the requested coordinates will be transformed before request
        output = node.create_output_array(c)
        with pytest.warns(UserWarning, match="transformation of coordinate segment lengths not yet implemented"):
            node.eval(c_x, output=output)

        # this will throw an error because output is not in the same crs as node
        output = node.create_output_array(c_x)
        with pytest.raises(ValueError, match="does not match"):
            node.eval(c, output=output)

    def test_evaluate_with_output_no_intersect(self):
        # there is a shortcut if there is no intersect, so we test that here
        node = MockDataSource()
        coords = Coordinates([clinspace(30, 40, 10), clinspace(30, 40, 10)], dims=["lat", "lon"])
        output = UnitsDataArray(np.ones(coords.shape), coords=coords.coords, dims=coords.dims)
        node.eval(coords, output=output)
        np.testing.assert_equal(output.data, np.full(output.shape, np.nan))

    def test_evaluate_with_output_transpose(self):
        # initialize coords with dims=[lon, lat]
        lat = clinspace(10, 20, 11)
        lon = clinspace(10, 15, 6)
        coords = Coordinates([lat, lon], dims=["lat", "lon"])

        # evaluate with dims=[lat, lon], passing in the output
        node = MockDataSource()
        output = node.create_output_array(coords.transpose("lon", "lat"))
        returned_output = node.eval(coords, output=output)

        # returned output should match the requested coordinates
        assert returned_output.dims == ("lat", "lon")

        # dims should stay in the order of the output, rather than the order of the requested coordinates
        assert output.dims == ("lon", "lat")

        # output data and returned output data should match
        np.testing.assert_equal(output.transpose("lat", "lon").data, returned_output.data)
        np.testing.assert_equal(output.transpose("lat", "lon").data, returned_output.data)

    def test_evaluate_with_crs_transform(self):
        # grid coords
        grid_coords = Coordinates([np.linspace(-10, 10, 21), np.linspace(-10, -10, 21)], dims=["lat", "lon"])
        with pytest.warns(UserWarning, match="transformation of coordinate segment lengths not yet implemented"):
            grid_coords = grid_coords.transform("EPSG:2193")

        node = MockDataSource()
        with pytest.warns(UserWarning, match="transformation of coordinate segment lengths not yet implemented"):
            out = node.eval(grid_coords)

        assert round(out.coords["lat"].values[0, 0]) == -8889021.0
        assert round(out.coords["lon"].values[0, 0]) == 1928929.0

        # stacked coords
        stack_coords = Coordinates(
            [(np.linspace(-10, 10, 21), np.linspace(-10, -10, 21)), np.linspace(0, 10, 10)], dims=["lat_lon", "time"]
        )
        stack_coords = stack_coords.transform("EPSG:2193")

        node = MockDataSource()
        out = node.eval(stack_coords)

        assert "lat_lon" in out.coords
        assert round(out.coords["lat"].values[0]) == -8889021.0
        assert round(out.coords["lon"].values[0]) == 1928929.0

    def test_evaluate_extra_dims(self):
        # drop extra unstacked dimension
        class MyDataSource(DataSource):
            coordinates = Coordinates([1, 11], dims=["lat", "lon"])

            def get_data(self, coordinates, coordinates_index):
                return self.create_output_array(coordinates)

        node = MyDataSource()
        coords = Coordinates([1, 11, "2018-01-01"], dims=["lat", "lon", "time"])
        output = node.eval(coords)
        assert output.dims == ("lat", "lon")  # time dropped

        # drop extra stacked dimension if none of its dimensions are needed
        class MyDataSource(DataSource):
            coordinates = Coordinates(["2018-01-01"], dims=["time"])

            def get_data(self, coordinates, coordinates_index):
                return self.create_output_array(coordinates)

        node = MyDataSource()
        coords = Coordinates([[1, 11], "2018-01-01"], dims=["lat_lon", "time"])
        output = node.eval(coords)
        assert output.dims == ("time",)  # lat_lon dropped

        # but don't drop extra stacked dimension if any of its dimensions are needed
        # output = node.eval(Coordinates([[1, 11, '2018-01-01']], dims=['lat_lon_time']))
        # assert output.dims == ('lat_lon_time') # lat and lon not dropped

    def test_evaluate_missing_dims(self):
        # missing unstacked dimension
        node = MockDataSource()

        with pytest.raises(ValueError, match="Cannot evaluate these coordinates.*"):
            node.eval(Coordinates([1], dims=["lat"]))
        with pytest.raises(ValueError, match="Cannot evaluate these coordinates.*"):
            node.eval(Coordinates([11], dims=["lon"]))
        with pytest.raises(ValueError, match="Cannot evaluate these coordinates.*"):
            node.eval(Coordinates(["2018-01-01"], dims=["time"]))

        # missing any part of stacked dimension
        node = MockDataSourceStacked()

        with pytest.raises(ValueError, match="Cannot evaluate these coordinates.*"):
            node.eval(Coordinates([1], dims=["time"]))

        with pytest.raises(ValueError, match="Cannot evaluate these coordinates.*"):
            node.eval(Coordinates([1], dims=["lat"]))

    def test_evaluate_no_overlap(self):
        """evaluate node with coordinates that do not overlap"""

        node = MockDataSource()
        coords = Coordinates([clinspace(-55, -45, 20), clinspace(-55, -45, 20)], dims=["lat", "lon"])
        output = node.eval(coords)

        assert np.all(np.isnan(output))

    def test_evaluate_extract_output(self):
        class MyMultipleDataSource(DataSource):
            outputs = ["a", "b", "c"]
            coordinates = Coordinates([[0, 1, 2, 3], [10, 11]], dims=["lat", "lon"])

            def get_data(self, coordinates, coordinates_index):
                return self.create_output_array(coordinates, data=1)

        # don't extract when no output field is requested
        node = MyMultipleDataSource()
        o = node.eval(node.coordinates)
        assert o.shape == (4, 2, 3)
        np.testing.assert_array_equal(o.dims, ["lat", "lon", "output"])
        np.testing.assert_array_equal(o["output"], ["a", "b", "c"])
        np.testing.assert_array_equal(o, 1)

        # do extract when an output field is requested
        node = MyMultipleDataSource(output="b")

        o = node.eval(node.coordinates)  # get_data case
        assert o.shape == (4, 2)
        np.testing.assert_array_equal(o.dims, ["lat", "lon"])
        np.testing.assert_array_equal(o, 1)

        o = node.eval(Coordinates([[100, 200], [1000, 2000, 3000]], dims=["lat", "lon"]))  # no intersection case
        assert o.shape == (2, 3)
        np.testing.assert_array_equal(o.dims, ["lat", "lon"])
        np.testing.assert_array_equal(o, np.nan)

        # should still work if the node has already extracted it
        class MyMultipleDataSource2(MyMultipleDataSource):
            def get_data(self, coordinates, coordinates_index):
                out = self.create_output_array(coordinates, data=1)
                return out.sel(output=self.output)

        node = MyMultipleDataSource2(output="b")
        o = node.eval(node.coordinates)
        assert o.shape == (4, 2)
        np.testing.assert_array_equal(o.dims, ["lat", "lon"])
        np.testing.assert_array_equal(o, 1)

    def test_nan_vals(self):
        """ evaluate note with nan_vals """

        # none
        node = MockDataSource()
        output = node.eval(node.coordinates)
        assert np.sum(np.isnan(output)) == 1
        assert np.isnan(output[1, 1])

        # one value
        node = MockDataSource(nan_vals=[10])
        output = node.eval(node.coordinates)
        assert np.sum(np.isnan(output)) == 2
        assert np.isnan(output[0, 0])
        assert np.isnan(output[1, 1])

        # multiple values
        node = MockDataSource(nan_vals=[10, 5])
        output = node.eval(node.coordinates)
        assert np.sum(np.isnan(output)) == 3
        assert np.isnan(output[0, 0])
        assert np.isnan(output[1, 1])
        assert np.isnan(output[1, 0])

    def test_get_data_np_array(self):
        class MockDataSourceReturnsArray(MockDataSource):
            def get_data(self, coordinates, coordinates_index):
                return self.data[coordinates_index]

        node = MockDataSourceReturnsArray()
        output = node.eval(node.coordinates)

        assert isinstance(output, UnitsDataArray)
        assert node.coordinates["lat"].coordinates[4] == output.coords["lat"].values[4]

    def test_get_data_DataArray(self):
        class MockDataSourceReturnsDataArray(MockDataSource):
            def get_data(self, coordinates, coordinates_index):
                return xr.DataArray(self.data[coordinates_index])

        node = MockDataSourceReturnsDataArray()
        output = node.eval(node.coordinates)

        assert isinstance(output, UnitsDataArray)
        assert node.coordinates["lat"].coordinates[4] == output.coords["lat"].values[4]

    def test_find_coordinates(self):
        node = MockDataSource()
        l = node.find_coordinates()
        assert isinstance(l, list)
        assert len(l) == 1
        assert l[0] == node.coordinates


class TestInterpolateData(object):
    """test default generic interpolation defaults"""

    def test_one_data_point(self):
        """ test when there is only one data point """
        # TODO: as this is currently written, this would never make it to the interpolater
        pass

    def test_interpolate_time(self):
        """ for now time uses nearest neighbor """

        class MyDataSource(DataSource):
            coordinates = Coordinates([clinspace(0, 10, 5)], dims=["time"])

            def get_data(self, coordinates, coordinates_index):
                return self.create_output_array(coordinates)

        node = MyDataSource()
        coords = Coordinates([clinspace(1, 11, 5)], dims=["time"])
        output = node.eval(coords)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.time.values == coords.coords["time"])

    def test_interpolate_lat_time(self):
        """interpolate with n dims and time"""
        pass

    def test_interpolate_alt(self):
        """ for now alt uses nearest neighbor """

        class MyDataSource(DataSource):
            coordinates = Coordinates([clinspace(0, 10, 5)], dims=["alt"], crs="+proj=merc +vunits=m")

            def get_data(self, coordinates, coordinates_index):
                return self.create_output_array(coordinates)

        coords = Coordinates([clinspace(1, 11, 5)], dims=["alt"], crs="+proj=merc +vunits=m")

        node = MyDataSource()
        output = node.eval(coords)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.alt.values == coords.coords["alt"])
