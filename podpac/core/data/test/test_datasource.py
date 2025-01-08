"""
Test podpac.core.data.datasource module
"""

# from collections import OrderedDict

import pytest

import numpy as np
import traitlets as tl
import xarray as xr
from xarray.core.coordinates import DataArrayCoordinates

import podpac
from podpac.core.units import UnitsDataArray
from podpac.core.node import COMMON_NODE_DOC
from podpac.core.style import Style
from podpac.core.coordinates import Coordinates, clinspace
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


class MockMultipleDataSource(DataSource):
    outputs = ["a", "b", "c"]
    coordinates = Coordinates([[0, 1, 2, 3], [10, 11]], dims=["lat", "lon"])

    def get_data(self, coordinates, coordinates_index):
        return self.create_output_array(coordinates, data=1)


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

    def test_repr(self):
        node = DataSource()
        repr(node)

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

        # make sure get_coordinates gets called only once
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

        # can't set coordinates attribute
        with pytest.raises(AttributeError):
            node.coordinates = Coordinates([])

    def test_dims(self):
        class MyDataSource(DataSource):
            def get_coordinates(self):
                return Coordinates([0, 0], dims=["lat", "lon"])

        node = MyDataSource()
        assert node.dims == ("lat", "lon")

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
        node.set_coordinates(Coordinates([]), force=True)
        assert node.coordinates == Coordinates([])
        assert node.coordinates != node.get_coordinates()

        # don't overwrite
        node = MockDataSource()
        node.coordinates
        node.set_coordinates(Coordinates([]))
        assert node.coordinates != Coordinates([])
        assert node.coordinates == node.get_coordinates()

    def test_boundary(self):
        # default
        node = DataSource()
        assert node.boundary == {}

        # none
        node = DataSource(boundary={})

        # centered
        node = DataSource(boundary={"lat": 0.25, "lon": 2.0})
        node = DataSource(boundary={"time": "1,D"})

        # box (not necessary centered)
        with pytest.raises(NotImplementedError, match="Non-centered boundary not yet supported"):
            node = DataSource(boundary={"lat": [-0.2, 0.3], "lon": [-2.0, 2.0]})

        with pytest.raises(NotImplementedError, match="Non-centered boundary not yet supported"):
            node = DataSource(boundary={"time": ["-1,D", "2,D"]})

        # polygon
        with pytest.raises(NotImplementedError, match="Non-centered boundary not yet supported"):
            node = DataSource(boundary={"lat": [0.0, -0.5, 0.0, 0.5], "lon": [-0.5, 0.0, 0.5, 0.0]})  # diamond

        # array of boundaries (one for each coordinate)
        with pytest.raises(NotImplementedError, match="Non-uniform boundary not yet supported"):
            node = DataSource(boundary={"lat": [[-0.1, 0.4], [-0.2, 0.3], [-0.3, 0.2]], "lon": 0.5})

        with pytest.raises(NotImplementedError, match="Non-uniform boundary not yet supported"):
            node = DataSource(boundary={"time": [["-1,D", "1,D"], ["-2,D", "1,D"]]})

        # invalid
        with pytest.raises(tl.TraitError):
            node = DataSource(boundary=0.5)

        with pytest.raises(ValueError, match="Invalid dimension"):
            node = DataSource(boundary={"other": 0.5})

        with pytest.raises(TypeError, match="Invalid coordinate delta"):
            node = DataSource(boundary={"lat": {}})

        with pytest.raises(ValueError, match="Invalid boundary"):
            node = DataSource(boundary={"lat": -0.25, "lon": 2.0})  # negative

        with pytest.raises(ValueError, match="Invalid boundary"):
            node = DataSource(boundary={"time": "-2,D"})  # negative

        with pytest.raises(ValueError, match="Invalid boundary"):
            node = DataSource(boundary={"time": "2018-01-01"})  # not a delta

    def test_invalid_nan_vals(self):
        with pytest.raises(tl.TraitError):
            DataSource(nan_vals={})

        with pytest.raises(tl.TraitError):
            DataSource(nan_vals=10)

    def test_find_coordinates(self):
        node = MockDataSource()
        l = node.find_coordinates()
        assert isinstance(l, list)
        assert len(l) == 1
        assert l[0] == node.coordinates

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

    def test_evaluate_at_coordinates_with_output(self):
        node = MockDataSource()
        output = node.create_output_array(node.coordinates)
        node.eval(node.coordinates, output=output)

        assert output.shape == (11, 11)
        assert output[0, 0] == 10

    def test_evaluate_no_overlap(self):
        """evaluate node with coordinates that do not overlap"""

        node = MockDataSource()
        coords = Coordinates([clinspace(-55, -45, 20), clinspace(-55, -45, 20)], dims=["lat", "lon"])
        output = node.eval(coords)

        assert np.all(np.isnan(output))

    def test_evaluate_no_overlap_with_output(self):
        # there is a shortcut if there is no intersect, so we test that here
        node = MockDataSource()
        coords = Coordinates([clinspace(30, 40, 10), clinspace(30, 40, 10)], dims=["lat", "lon"])
        output = UnitsDataArray.create(coords, data=1)
        node.eval(coords, output=output)
        np.testing.assert_equal(output.data, np.full(output.shape, np.nan))

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

        # TODO
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

    def test_evaluate_crs_transform(self):
        node = MockDataSource()

        coords = node.coordinates.transform("EPSG:2193")
        out = node.eval(coords)

        # test data
        np.testing.assert_array_equal(out.data, node.data)

    def test_evaluate_selector(self):
        def selector(rsc, coordinates, index_type=None):
            """mock selector that just strides by 2"""
            new_rsci = tuple(slice(None, None, 2) for dim in rsc.dims)
            new_rsc = rsc[new_rsci]
            return new_rsc, new_rsci

        node = MockDataSource()
        output = node.eval(node.coordinates, _selector=selector)
        assert output.shape == (6, 6)
        np.testing.assert_array_equal(output["lat"].data, node.coordinates["lat"][::2].coordinates)
        np.testing.assert_array_equal(output["lon"].data, node.coordinates["lon"][::2].coordinates)

    def test_nan_vals(self):
        """evaluate note with nan_vals"""

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

    def test_get_data_invalid(self):
        class MockDataSourceReturnsInvalid(MockDataSource):
            def get_data(self, coordinates, coordinates_index):
                return self.data[coordinates_index].tolist()

        node = MockDataSourceReturnsInvalid()
        with pytest.raises(TypeError, match="Unknown data type"):
            output = node.eval(node.coordinates)

    def test_evaluate_debug_attributes(self):
        with podpac.settings:
            podpac.settings["DEBUG"] = True

            node = MockDataSource()

            assert node._evaluated_coordinates is None
            assert node._requested_coordinates is None
            assert node._requested_source_coordinates is None
            assert node._requested_source_coordinates_index is None
            assert node._requested_source_boundary is None
            assert node._requested_source_data is None

            node.eval(node.coordinates)

            assert node._evaluated_coordinates is not None
            assert node._requested_coordinates is not None
            assert node._requested_source_coordinates is not None
            assert node._requested_source_coordinates_index is not None
            assert node._requested_source_boundary is not None
            assert node._requested_source_data is not None

    def test_evaluate_debug_attributes_no_overlap(self):
        with podpac.settings:
            podpac.settings["DEBUG"] = True

            node = MockDataSource()

            assert node._evaluated_coordinates is None
            assert node._requested_coordinates is None
            assert node._requested_source_coordinates is None
            assert node._requested_source_coordinates_index is None
            assert node._requested_source_boundary is None
            assert node._requested_source_data is None

            coords = Coordinates([clinspace(-55, -45, 20), clinspace(-55, -45, 20)], dims=["lat", "lon"])
            node.eval(coords)

            assert node._evaluated_coordinates is not None
            assert node._requested_coordinates is not None
            assert node._requested_source_coordinates is not None
            assert node._requested_source_coordinates_index is not None
            assert node._requested_source_boundary is None  # still none in this case
            assert node._requested_source_data is None  # still none in this case

    def test_get_boundary(self):
        # disable boundary validation (until non-centered and non-uniform boundaries are fully implemented)
        class MockDataSourceNoBoundaryValidation(MockDataSource):
            @tl.validate("boundary")
            def _validate_boundary(self, d):
                return d["value"]

        index = (slice(3, 9, 2), [3, 4, 6])

        # points
        node = MockDataSourceNoBoundaryValidation(boundary={})
        boundary = node._get_boundary(index)
        assert boundary == {}

        # uniform centered
        node = MockDataSourceNoBoundaryValidation(boundary={"lat": 0.1, "lon": 0.2})
        boundary = node._get_boundary(index)
        assert boundary == {"lat": 0.1, "lon": 0.2}

        # uniform polygon
        node = MockDataSourceNoBoundaryValidation(boundary={"lat": [-0.1, 0.1], "lon": [-0.1, 0.0, 0.1]})
        boundary = node._get_boundary(index)
        assert boundary == {"lat": [-0.1, 0.1], "lon": [-0.1, 0.0, 0.1]}

        # non-uniform
        lat_boundary = np.vstack([-np.arange(11), np.arange(11)]).T
        lon_boundary = np.vstack([-2 * np.arange(11), 2 * np.arange(11)]).T
        node = MockDataSourceNoBoundaryValidation(boundary={"lat": lat_boundary, "lon": lon_boundary})
        boundary = node._get_boundary(index)
        np.testing.assert_array_equal(boundary["lat"], lat_boundary[index[0]])
        np.testing.assert_array_equal(boundary["lon"], lon_boundary[index[1]])

    def test_get_boundary_stacked(self):
        # disable boundary validation (until non-centered and non-uniform boundaries are fully implemented)
        class MockDataSourceStackedNoBoundaryValidation(MockDataSourceStacked):
            @tl.validate("boundary")
            def _validate_boundary(self, d):
                return d["value"]

        index = (slice(3, 9, 2),)

        # points
        node = MockDataSourceStackedNoBoundaryValidation(boundary={})
        boundary = node._get_boundary(index)
        assert boundary == {}

        # uniform centered
        node = MockDataSourceStackedNoBoundaryValidation(boundary={"lat": 0.1, "lon": 0.1})
        boundary = node._get_boundary(index)
        assert boundary == {"lat": 0.1, "lon": 0.1}

        # uniform polygon
        node = MockDataSourceStackedNoBoundaryValidation(boundary={"lat": [-0.1, 0.1], "lon": [-0.1, 0.0, 0.1]})
        boundary = node._get_boundary(index)
        assert boundary == {"lat": [-0.1, 0.1], "lon": [-0.1, 0.0, 0.1]}

        # non-uniform
        lat_boundary = np.vstack([-np.arange(11), np.arange(11)]).T
        lon_boundary = np.vstack([-2 * np.arange(11), 2 * np.arange(11)]).T
        node = MockDataSourceStackedNoBoundaryValidation(boundary={"lat": lat_boundary, "lon": lon_boundary})
        boundary = node._get_boundary(index)
        np.testing.assert_array_equal(boundary["lat"], lat_boundary[index])
        np.testing.assert_array_equal(boundary["lon"], lon_boundary[index])

    def test_eval_get_cache_extra_dims(self):
        with podpac.settings:
            podpac.settings["DEFAULT_CACHE"] = ["ram"]

            node = podpac.data.Array(
                source=np.ones((3, 4)),
                coordinates=podpac.Coordinates([range(3), range(4)], ["lat", "lon"]),
                cache_ctrl=["ram"],
            )
            coords1 = podpac.Coordinates([range(3), range(4), "2012-05-19"], ["lat", "lon", "time"])
            coords2 = podpac.Coordinates([range(3), range(4), "2019-09-10"], ["lat", "lon", "time"])

            # retrieve from cache on the second evaluation
            node.eval(coords1)
            assert not node._from_cache

            node.eval(coords1)
            assert node._from_cache

            # also retrieve from cache with different time coordinates
            node.eval(coords2)
            assert node._from_cache

    def test_eval_get_cache_transform_crs(self):
        with podpac.settings:
            podpac.settings["DEFAULT_CACHE"] = ["ram"]
            podpac.settings["DEFAULT_CRS"] = podpac.core.settings.DEFAULT_SETTINGS["DEFAULT_CRS"]

            node = podpac.core.data.array_source.Array(
                source=np.ones((5, 4)),
                coordinates=podpac.Coordinates([range(5), range(4)], ["lat", "lon"], crs="EPSG:4326"),
                cache_ctrl=["ram"],
            )

            # retrieve from cache on the second evaluation
            node.eval(node.coordinates)
            assert not node._from_cache

            node.eval(node.coordinates)
            assert node._from_cache

            # also retrieve from cache with different crs
            node.eval(node.coordinates.transform("EPSG:4326"))
            assert node._from_cache

    def test_get_source_data(self):
        node = podpac.data.Array(
            source=np.ones((3, 4)),
            coordinates=podpac.Coordinates([range(3), range(4)], ["lat", "lon"]),
        )

        data = node.get_source_data()
        np.testing.assert_array_equal(data, node.source)

    def test_get_source_data_with_bounds(self):
        node = podpac.data.Array(
            source=np.ones((3, 4)),
            coordinates=podpac.Coordinates([range(3), range(4)], ["lat", "lon"]),
        )

        data = node.get_source_data({"lon": (1.5, 4.5)})
        np.testing.assert_array_equal(data, node.source[:, 2:])

    def test_get_bounds(self):
        node = podpac.data.Array(
            source=np.ones((3, 4)),
            coordinates=podpac.Coordinates([range(3), range(4)], ["lat", "lon"], crs="EPSG:2193"),
        )

        with podpac.settings:
            podpac.settings["DEFAULT_CRS"] = "EPSG:4326"

            # specify crs
            bounds, crs = node.get_bounds(crs="EPSG:3857")
            expected = {
                "lat": (-13291827.558247397, -13291815.707967814),
                "lon": (9231489.26794932, 9231497.142754894),
            }
            for k in expected:
                np.testing.assert_almost_equal(bounds[k], expected[k])
            assert crs == "EPSG:3857"

            # native/source crs
            bounds, crs = node.get_bounds(crs="source")
            assert bounds == {"lat": (0, 2), "lon": (0, 3)}
            assert crs == "EPSG:2193"

            # default crs
            bounds, crs = node.get_bounds()
            assert bounds == {
                "lat": (-75.81365382984804, -75.81362774074242),
                "lon": (82.92787904584206, 82.92794978642414),
            }
            assert crs == "EPSG:4326"


class TestDataSourceWithMultipleOutputs(object):
    def test_evaluate_no_overlap_with_output_extract_output(self):
        class MockMultipleDataSource(DataSource):
            outputs = ["a", "b", "c"]
            coordinates = Coordinates([[0, 1, 2, 3], [10, 11]], dims=["lat", "lon"])

            def get_data(self, coordinates, coordinates_index):
                return self.create_output_array(coordinates, data=1)

        node = MockMultipleDataSource(output="a")
        coords = Coordinates([clinspace(-55, -45, 20), clinspace(-55, -45, 20)], dims=["lat", "lon"])
        output = node.eval(coords)

        assert np.all(np.isnan(output))

    def test_evaluate_extract_output(self):
        # don't extract when no output field is requested
        node = MockMultipleDataSource()
        o = node.eval(node.coordinates)
        assert o.shape == (4, 2, 3)
        np.testing.assert_array_equal(o.dims, ["lat", "lon", "output"])
        np.testing.assert_array_equal(o["output"], ["a", "b", "c"])
        np.testing.assert_array_equal(o, 1)

        # do extract when an output field is requested
        node = MockMultipleDataSource(output="b")

        o = node.eval(node.coordinates)  # get_data case
        assert o.shape == (4, 2)
        np.testing.assert_array_equal(o.dims, ["lat", "lon"])
        np.testing.assert_array_equal(o, 1)

        # no intersection case
        o = node.eval(Coordinates([[100, 200], [1000, 2000, 3000]], dims=["lat", "lon"]))
        assert o.shape == (0, 0)
        np.testing.assert_array_equal(o.dims, ["lat", "lon"])
        np.testing.assert_array_equal(o, np.nan)

    def test_evaluate_output_already_extracted(self):
        # should still work if the node has already extracted it
        class ExtractedMultipleDataSource(MockMultipleDataSource):
            def get_data(self, coordinates, coordinates_index):
                out = self.create_output_array(coordinates, data=1)
                return out.sel(output=self.output)

        node = ExtractedMultipleDataSource(output="b")
        o = node.eval(node.coordinates)
        assert o.shape == (4, 2)
        np.testing.assert_array_equal(o.dims, ["lat", "lon"])
        np.testing.assert_array_equal(o, 1)
