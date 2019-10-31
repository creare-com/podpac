"""
Test interpolation methods


"""
# pylint: disable=C0111,W0212,R0903

from collections import OrderedDict
from copy import deepcopy

import pytest
import traitlets as tl
import numpy as np

from podpac.core.units import UnitsDataArray
from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.data.types import rasterio
from podpac.core.data import datasource
from podpac.core.data.interpolation import (
    Interpolation,
    InterpolationException,
    INTERPOLATORS,
    INTERPOLATION_METHODS,
    INTERPOLATION_DEFAULT,
    INTERPOLATORS_DICT,
    INTERPOLATION_METHODS_DICT,
)

from podpac.core.data.interpolator import Interpolator, InterpolatorException
from podpac.core.data.interpolators import NearestNeighbor, NearestPreview, Rasterio, ScipyGrid, ScipyPoint

# test fixtures
from podpac.core.data.test.test_datasource import MockArrayDataSource


class TestInterpolation(object):
    """ Test interpolation class and support methods"""

    def test_allow_missing_modules(self):
        """TODO: Allow user to be missing rasterio and scipy"""
        pass

    def test_interpolation_methods(self):
        assert len(set(INTERPOLATION_METHODS) & set(INTERPOLATION_METHODS_DICT.keys())) == len(INTERPOLATION_METHODS)

    def test_interpolator_init_type(self):
        """test constructor
        """

        # should throw an error if definition is not str, dict, or Interpolator
        with pytest.raises(TypeError):
            Interpolation(5)

    def test_str_definition(self):
        # should throw an error if string input is not one of the INTERPOLATION_METHODS
        with pytest.raises(InterpolationException):
            Interpolation("test")

        interp = Interpolation("nearest")
        assert interp.config[("default",)]
        assert isinstance(interp.config[("default",)], dict)
        assert interp.config[("default",)]["method"] == "nearest"
        assert isinstance(interp.config[("default",)]["interpolators"][0], Interpolator)

    def test_dict_definition(self):

        # should handle a default definition without any dimensions specified as keys
        interp = Interpolation({"method": "nearest", "params": {"spatial_tolerance": 1}})
        assert isinstance(interp.config[("default",)], dict)
        assert interp.config[("default",)]["method"] == "nearest"
        assert isinstance(interp.config[("default",)]["interpolators"][0], Interpolator)
        assert interp.config[("default",)]["params"] == {"spatial_tolerance": 1}

        # should throw an error on _parse_interpolation_method(definition)
        # if definition is not in INTERPOLATION_METHODS
        with pytest.raises(InterpolationException):
            Interpolation({("lat", "lon"): "test"})

        # handle string methods
        interp = Interpolation({("lat", "lon"): "nearest"})
        assert isinstance(interp.config[("lat", "lon")], dict)
        assert interp.config[("lat", "lon")]["method"] == "nearest"
        assert isinstance(interp.config[("default",)]["interpolators"][0], Interpolator)
        assert interp.config[("default",)]["params"] == {}

        # handle dict methods

        # should throw an error if method is not in dict
        with pytest.raises(InterpolationException):
            Interpolation({("lat", "lon"): {"test": "test"}})

        # should throw an error if method is not a string
        with pytest.raises(InterpolationException):
            Interpolation({("lat", "lon"): {"method": 5}})

        # should throw an error if method is not one of the INTERPOLATION_METHODS and no interpolators defined
        with pytest.raises(InterpolationException):
            Interpolation({("lat", "lon"): {"method": "myinter"}})

        # should throw an error if params is not a dict
        with pytest.raises(TypeError):
            Interpolation({("lat", "lon"): {"method": "nearest", "params": "test"}})

        # should throw an error if interpolators is not a list
        with pytest.raises(TypeError):
            Interpolation({("lat", "lon"): {"method": "nearest", "interpolators": "test"}})

        # should throw an error if interpolators are not Interpolator classes
        with pytest.raises(TypeError):
            Interpolation({("lat", "lon"): {"method": "nearest", "interpolators": [NearestNeighbor, "test"]}})

        # should throw an error if dimension is defined twice
        with pytest.raises(InterpolationException):
            Interpolation({("lat", "lon"): "nearest", "lat": "bilinear"})

        # should handle standard INTEPROLATION_SHORTCUTS
        interp = Interpolation({("lat", "lon"): {"method": "nearest"}})
        assert isinstance(interp.config[("lat", "lon")], dict)
        assert interp.config[("lat", "lon")]["method"] == "nearest"
        assert isinstance(interp.config[("lat", "lon")]["interpolators"][0], Interpolator)
        assert interp.config[("lat", "lon")]["params"] == {}

        # should not allow custom methods if interpolators can't support
        with pytest.raises(InterpolatorException):
            interp = Interpolation(
                {("lat", "lon"): {"method": "myinter", "interpolators": [NearestNeighbor, NearestPreview]}}
            )

        # should allow custom methods if interpolators can support
        class MyInterp(Interpolator):
            methods_supported = ["myinter"]

        interp = Interpolation({("lat", "lon"): {"method": "myinter", "interpolators": [MyInterp]}})
        assert interp.config[("lat", "lon")]["method"] == "myinter"
        assert isinstance(interp.config[("lat", "lon")]["interpolators"][0], MyInterp)

        # should allow params to be set
        interp = Interpolation(
            {("lat", "lon"): {"method": "myinter", "interpolators": [MyInterp], "params": {"spatial_tolerance": 5}}}
        )
        assert interp.config[("lat", "lon")]["params"] == {"spatial_tolerance": 5}

        # set default equal to empty tuple
        interp = Interpolation({"lat": "bilinear"})
        assert interp.config[("default",)]["method"] == INTERPOLATION_DEFAULT

        # use default with override if not all dimensions are supplied
        interp = Interpolation({"lat": "bilinear", "default": "nearest"})
        assert interp.config[("default",)]["method"] == "nearest"

        # make sure default is always the last key in the ordered config dict
        interp = Interpolation({"default": "nearest", "lat": "bilinear"})
        assert list(interp.config.keys())[-1] == ("default",)

    def test_init_interpolators(self):

        # should set method
        interp = Interpolation("nearest")
        assert interp.config[("default",)]["interpolators"][0].method == "nearest"

        # Interpolation init should init all interpolators in the list
        interp = Interpolation({"default": {"method": "nearest", "params": {"spatial_tolerance": 1}}})
        assert interp.config[("default",)]["interpolators"][0].spatial_tolerance == 1

        # should throw TraitErrors defined by Interpolator
        with pytest.raises(tl.TraitError):
            Interpolation({"default": {"method": "nearest", "params": {"spatial_tolerance": "tol"}}})

        # should not allow undefined params
        with pytest.warns(DeprecationWarning):  # eventually, Traitlets will raise an exception here
            interp = Interpolation({"default": {"method": "nearest", "params": {"myarg": 1}}})
        with pytest.raises(AttributeError):
            assert interp.config[("default",)]["interpolators"][0].myarg == "tol"

    def test_select_interpolator_queue(self):

        reqcoords = Coordinates([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], dims=["lat", "lon", "time", "alt"])
        srccoords = Coordinates([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], dims=["lat", "lon", "time", "alt"])

        # create a few dummy interpolators that handle certain dimensions
        # (can_select is defined by default to look at dims_supported)
        class TimeLat(Interpolator):
            methods_supported = ["myinterp"]
            dims_supported = ["time", "lat"]

            def can_select(self, udims, source_coordinates, eval_coordinates):
                return self._filter_udims_supported(udims)

            def can_interpolate(self, udims, source_coordinates, eval_coordinates):
                return self._filter_udims_supported(udims)

        class LatLon(Interpolator):
            methods_supported = ["myinterp"]
            dims_supported = ["lat", "lon"]

            def can_select(self, udims, source_coordinates, eval_coordinates):
                return self._filter_udims_supported(udims)

            def can_interpolate(self, udims, source_coordinates, eval_coordinates):
                return self._filter_udims_supported(udims)

        class Lon(Interpolator):
            methods_supported = ["myinterp"]
            dims_supported = ["lon"]

            def can_select(self, udims, source_coordinates, eval_coordinates):
                return self._filter_udims_supported(udims)

            def can_interpolate(self, udims, source_coordinates, eval_coordinates):
                return self._filter_udims_supported(udims)

        # set up a strange interpolation definition
        # we want to interpolate (lat, lon) first, then after (time, alt)
        interp = Interpolation(
            {
                ("lat", "lon"): {"method": "myinterp", "interpolators": [LatLon, TimeLat]},
                ("time", "alt"): {"method": "myinterp", "interpolators": [TimeLat, Lon]},
            }
        )

        # default = 'nearest', which will return NearestPreview for can_select
        interpolator_queue = interp._select_interpolator_queue(srccoords, reqcoords, "can_select")
        assert isinstance(interpolator_queue, OrderedDict)
        assert isinstance(interpolator_queue[("lat", "lon")], LatLon)
        assert ("time", "alt") not in interpolator_queue and ("alt", "time") not in interpolator_queue

        # should throw an error if strict is set and not all dimensions can be handled
        with pytest.raises(InterpolationException):
            interp_copy = deepcopy(interp)
            del interp_copy.config[("default",)]
            interpolator_queue = interp_copy._select_interpolator_queue(srccoords, reqcoords, "can_select", strict=True)

        # default = Nearest, which can handle all dims for can_interpolate
        interpolator_queue = interp._select_interpolator_queue(srccoords, reqcoords, "can_interpolate")
        assert isinstance(interpolator_queue, OrderedDict)
        assert isinstance(interpolator_queue[("lat", "lon")], LatLon)

        if ("alt", "time") in interpolator_queue:
            assert isinstance(interpolator_queue[("alt", "time")], NearestNeighbor)
        else:
            assert isinstance(interpolator_queue[("time", "alt")], NearestNeighbor)

    def test_select_coordinates(self):

        reqcoords = Coordinates([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], dims=["lat", "lon", "time", "alt"])
        srccoords = Coordinates([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], dims=["lat", "lon", "time", "alt"])

        # create a few dummy interpolators that handle certain dimensions
        # (can_select is defined by default to look at dims_supported)
        class TimeLat(Interpolator):
            methods_supported = ["myinterp"]
            dims_supported = ["time", "lat"]

            def select_coordinates(self, udims, srccoords, srccoords_idx, reqcoords):
                return srccoords, srccoords_idx

        class LatLon(Interpolator):
            methods_supported = ["myinterp"]
            dims_supported = ["lat", "lon"]

            def select_coordinates(self, udims, srccoords, srccoords_idx, reqcoords):
                return srccoords, srccoords_idx

        class Lon(Interpolator):
            methods_supported = ["myinterp"]
            dims_supported = ["lon"]

            def select_coordinates(self, udims, srccoords, srccoords_idx, reqcoords):
                return srccoords, srccoords_idx

        # set up a strange interpolation definition
        # we want to interpolate (lat, lon) first, then after (time, alt)
        interp = Interpolation(
            {
                ("lat", "lon"): {"method": "myinterp", "interpolators": [LatLon, TimeLat]},
                ("time", "alt"): {"method": "myinterp", "interpolators": [TimeLat, Lon]},
            }
        )

        coords, cidx = interp.select_coordinates(srccoords, [], reqcoords)

        assert len(coords) == len(srccoords)
        assert len(coords["lat"]) == len(srccoords["lat"])
        assert cidx == ()

    def test_interpolate(self):
        class TestInterp(Interpolator):
            dims_supported = ["lat", "lon"]

            def interpolate(self, udims, source_coordinates, source_data, eval_coordinates, output_data):
                output_data = source_data
                return output_data

        # test basic functionality
        reqcoords = Coordinates([[-0.5, 1.5, 3.5], [0.5, 2.5, 4.5]], dims=["lat", "lon"])
        srccoords = Coordinates([[0, 2, 4], [0, 3, 4]], dims=["lat", "lon"])
        srcdata = UnitsDataArray(
            np.random.rand(3, 3), coords=[srccoords[c].coordinates for c in srccoords], dims=srccoords.dims
        )
        outdata = UnitsDataArray(
            np.zeros(srcdata.shape), coords=[reqcoords[c].coordinates for c in reqcoords], dims=reqcoords.dims
        )

        interp = Interpolation({("lat", "lon"): {"method": "myinterp", "interpolators": [TestInterp]}})
        outdata = interp.interpolate(srccoords, srcdata, reqcoords, outdata)

        assert np.all(outdata == srcdata)

        # test if data is size 1
        class TestFakeInterp(Interpolator):
            dims_supported = ["lat"]

            def interpolate(self, udims, source_coordinates, source_data, eval_coordinates, output_data):
                return None

        reqcoords = Coordinates([[1]], dims=["lat"])
        srccoords = Coordinates([[1]], dims=["lat"])
        srcdata = UnitsDataArray(
            np.random.rand(1), coords=[srccoords[c].coordinates for c in srccoords], dims=srccoords.dims
        )
        outdata = UnitsDataArray(
            np.zeros(srcdata.shape), coords=[reqcoords[c].coordinates for c in reqcoords], dims=reqcoords.dims
        )

        interp = Interpolation({("lat", "lon"): {"method": "myinterp", "interpolators": [TestFakeInterp]}})
        outdata = interp.interpolate(srccoords, srcdata, reqcoords, outdata)

        assert np.all(outdata == srcdata)


class TestInterpolators(object):
    class TestInterpolator(object):
        """Test abstract interpolator class"""

        def test_can_select(self):
            class CanAlwaysSelect(Interpolator):
                def can_select(self, udims, reqcoords, srccoords):
                    return udims

            class CanNeverSelect(Interpolator):
                def can_select(self, udims, reqcoords, srccoords):
                    return tuple()

            interp = CanAlwaysSelect(method="method")
            can_select = interp.can_select(("time", "lat"), None, None)
            assert "lat" in can_select and "time" in can_select

            interp = CanNeverSelect(method="method")
            can_select = interp.can_select(("time", "lat"), None, None)
            assert not can_select

        def test_dim_in(self):
            interpolator = Interpolator(methods_supported=["test"], method="test")

            coords = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
            assert interpolator._dim_in("lat", coords)
            assert interpolator._dim_in("lat", coords, unstacked=True)
            assert not interpolator._dim_in("time", coords)

            coords_two = Coordinates([clinspace(0, 10, 5)], dims=["lat"])
            assert interpolator._dim_in("lat", coords, coords_two)
            assert not interpolator._dim_in("lon", coords, coords_two)

            coords_three = Coordinates([(np.linspace(0, 10, 5), np.linspace(0, 10, 5))], dims=["lat_lon"])
            assert not interpolator._dim_in("lat", coords, coords_two, coords_three)
            assert interpolator._dim_in("lat", coords, coords_two, coords_three, unstacked=True)

    class TestNearest(object):
        def test_nearest_preview_select(self):

            # test straight ahead functionality
            reqcoords = Coordinates([[-0.5, 1.5, 3.5], [0.5, 2.5, 4.5]], dims=["lat", "lon"])
            srccoords = Coordinates([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dims=["lat", "lon"])

            interp = Interpolation("nearest_preview")

            srccoords, srccoords_index = srccoords.intersect(reqcoords, outer=True, return_indices=True)
            coords, cidx = interp.select_coordinates(srccoords, srccoords_index, reqcoords)

            assert len(coords) == len(srccoords) == len(cidx)
            assert len(coords["lat"]) == len(reqcoords["lat"])
            assert len(coords["lon"]) == len(reqcoords["lon"])
            assert np.all(coords["lat"].coordinates == np.array([0, 2, 4]))

            # test when selection is applied serially
            # this is equivalent to above
            reqcoords = Coordinates([[-0.5, 1.5, 3.5], [0.5, 2.5, 4.5]], dims=["lat", "lon"])
            srccoords = Coordinates([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dims=["lat", "lon"])

            interp = Interpolation({"lat": "nearest_preview", "lon": "nearest_preview"})

            srccoords, srccoords_index = srccoords.intersect(reqcoords, outer=True, return_indices=True)
            coords, cidx = interp.select_coordinates(srccoords, srccoords_index, reqcoords)

            assert len(coords) == len(srccoords) == len(cidx)
            assert len(coords["lat"]) == len(reqcoords["lat"])
            assert len(coords["lon"]) == len(reqcoords["lon"])
            assert np.all(coords["lat"].coordinates == np.array([0, 2, 4]))

            # test when coordinates are stacked and unstacked
            # TODO: how to handle stacked/unstacked coordinate asynchrony?
            # reqcoords = Coordinates([[-.5, 1.5, 3.5], [.5, 2.5, 4.5]], dims=['lat', 'lon'])
            # srccoords = Coordinates([([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])], dims=['lat_lon'])

            # interp = Interpolation('nearest_preview')

            # srccoords, srccoords_index = srccoords.intersect(reqcoords, outer=True, return_indices=True)
            # coords, cidx = interp.select_coordinates(reqcoords, srccoords, srccoords_index)

            # assert len(coords) == len(srcoords) == len(cidx)
            # assert len(coords['lat']) == len(reqcoords['lat'])
            # assert len(coords['lon']) == len(reqcoords['lon'])
            # assert np.all(coords['lat'].coordinates == np.array([0, 2, 4]))

        def test_interpolation(self):

            for interpolation in ["nearest", "nearest_preview"]:

                # unstacked 1D
                source = np.random.rand(5)
                coords_src = Coordinates([np.linspace(0, 10, 5)], dims=["lat"])
                node = MockArrayDataSource(source=source, native_coordinates=coords_src, interpolation=interpolation)

                coords_dst = Coordinates([[1, 1.2, 1.5, 5, 9]], dims=["lat"])
                output = node.eval(coords_dst)

                assert isinstance(output, UnitsDataArray)
                assert np.all(output.lat.values == coords_dst.coords["lat"])
                assert output.values[0] == source[0] and output.values[1] == source[0] and output.values[2] == source[1]

                # unstacked N-D
                source = np.random.rand(5, 5)
                coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
                coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=["lat", "lon"])

                node = MockArrayDataSource(source=source, native_coordinates=coords_src, interpolation=interpolation)
                output = node.eval(coords_dst)

                assert isinstance(output, UnitsDataArray)
                assert np.all(output.lat.values == coords_dst.coords["lat"])
                assert output.values[0, 0] == source[1, 1]

                # stacked
                # TODO: implement stacked handling
                source = np.random.rand(5)
                coords_src = Coordinates([(np.linspace(0, 10, 5), np.linspace(0, 10, 5))], dims=["lat_lon"])
                node = MockArrayDataSource(source=source, native_coordinates=coords_src)
                node.interpolation = {"method": "nearest", "interpolators": [NearestNeighbor]}
                coords_dst = Coordinates([(np.linspace(1, 9, 3), np.linspace(1, 9, 3))], dims=["lat_lon"])

                with pytest.raises(InterpolationException):
                    output = node.eval(coords_dst)

                # TODO: implement stacked handling
                # source = stacked, dest = unstacked
                source = np.random.rand(5)
                coords_src = Coordinates([(np.linspace(0, 10, 5), np.linspace(0, 10, 5))], dims=["lat_lon"])
                node = MockArrayDataSource(source=source, native_coordinates=coords_src)
                node.interpolation = {"method": "nearest", "interpolators": [NearestNeighbor]}
                coords_dst = Coordinates([np.linspace(1, 9, 3), np.linspace(1, 9, 3)], dims=["lat", "lon"])

                with pytest.raises(InterpolationException):
                    output = node.eval(coords_dst)

                # TODO: implement stacked handling
                # source = unstacked, dest = stacked
                source = np.random.rand(5, 5)
                coords_src = Coordinates([np.linspace(0, 10, 5), np.linspace(0, 10, 5)], dims=["lat", "lon"])
                node = MockArrayDataSource(source=source, native_coordinates=coords_src)
                node.interpolation = {"method": "nearest", "interpolators": [NearestNeighbor]}
                coords_dst = Coordinates([(np.linspace(1, 9, 3), np.linspace(1, 9, 3))], dims=["lat_lon"])

                with pytest.raises(InterpolationException):
                    output = node.eval(coords_dst)

        def test_spatial_tolerance(self):

            # unstacked 1D
            source = np.random.rand(5)
            coords_src = Coordinates([np.linspace(0, 10, 5)], dims=["lat"])
            node = MockArrayDataSource(
                source=source,
                native_coordinates=coords_src,
                interpolation={"default": {"method": "nearest", "params": {"spatial_tolerance": 1.1}}},
            )

            coords_dst = Coordinates([[1, 1.2, 1.5, 5, 9]], dims=["lat"])
            output = node.eval(coords_dst)

            print(output)
            print(source)
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords["lat"])
            assert output.values[0] == source[0] and np.isnan(output.values[1]) and output.values[2] == source[1]

        def test_time_tolerance(self):

            # unstacked 1D
            source = np.random.rand(5, 5)
            coords_src = Coordinates(
                [np.linspace(0, 10, 5), clinspace("2018-01-01", "2018-01-09", 5)], dims=["lat", "time"]
            )
            node = MockArrayDataSource(
                source=source,
                native_coordinates=coords_src,
                interpolation={
                    "default": {
                        "method": "nearest",
                        "params": {"spatial_tolerance": 1.1, "time_tolerance": np.timedelta64(1, "D")},
                    }
                },
            )

            coords_dst = Coordinates(
                [[1, 1.2, 1.5, 5, 9], clinspace("2018-01-01", "2018-01-09", 3)], dims=["lat", "time"]
            )
            output = node.eval(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords["lat"])
            assert (
                output.values[0, 0] == source[0, 0]
                and output.values[0, 1] == source[0, 2]
                and np.isnan(output.values[1, 0])
                and np.isnan(output.values[1, 1])
                and output.values[2, 0] == source[1, 0]
                and output.values[2, 1] == source[1, 2]
            )

    class TestInterpolateRasterio(object):
        """test interpolation functions"""

        def test_interpolate_rasterio(self):
            """ regular interpolation using rasterio"""

            assert rasterio is not None

            source = np.arange(0, 15)
            source.resize((3, 5))

            coords_src = Coordinates([clinspace(0, 10, 3), clinspace(0, 10, 5)], dims=["lat", "lon"])
            coords_dst = Coordinates([clinspace(1, 11, 3), clinspace(1, 11, 5)], dims=["lat", "lon"])

            # try one specific rasterio case to measure output
            node = MockArrayDataSource(source=source, native_coordinates=coords_src)
            node.interpolation = {"method": "min", "interpolators": [Rasterio]}
            output = node.eval(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords["lat"])
            assert output.data[0, 3] == 3.0
            assert output.data[0, 4] == 4.0

            node.interpolation = {"method": "max", "interpolators": [Rasterio]}
            output = node.eval(coords_dst)
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords["lat"])
            assert output.data[0, 3] == 9.0
            assert output.data[0, 4] == 9.0

            node.interpolation = {"method": "bilinear", "interpolators": [Rasterio]}
            output = node.eval(coords_dst)
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords["lat"])
            assert int(output.data[0, 0]) == 1
            assert int(output.data[0, 4]) == 5

        def test_interpolate_rasterio_descending(self):
            """should handle descending"""

            source = np.random.rand(5, 5)
            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
            coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=["lat", "lon"])

            node = MockArrayDataSource(
                source=source,
                native_coordinates=coords_src,
                interpolation={"method": "nearest", "interpolators": [Rasterio]},
            )
            output = node.eval(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords["lat"])
            assert np.all(output.lon.values == coords_dst.coords["lon"])

    class TestInterpolateScipyGrid(object):
        """test interpolation functions"""

        def test_interpolate_scipy_grid(self):

            source = np.arange(0, 25)
            source.resize((5, 5))

            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
            coords_dst = Coordinates([clinspace(1, 11, 5), clinspace(1, 11, 5)], dims=["lat", "lon"])

            # try one specific rasterio case to measure output
            node = MockArrayDataSource(source=source, native_coordinates=coords_src)
            node.interpolation = {"method": "nearest", "interpolators": [ScipyGrid]}
            output = node.eval(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords["lat"])
            print(output)
            assert output.data[0, 0] == 0.0
            assert output.data[0, 3] == 3.0
            assert output.data[1, 3] == 8.0
            assert np.isnan(output.data[0, 4])  # TODO: how to handle outside bounds

            node.interpolation = {"method": "cubic_spline", "interpolators": [ScipyGrid]}
            output = node.eval(coords_dst)
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords["lat"])
            assert int(output.data[0, 0]) == 2
            assert int(output.data[2, 4]) == 16

            node.interpolation = {"method": "bilinear", "interpolators": [ScipyGrid]}
            output = node.eval(coords_dst)
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords["lat"])
            assert int(output.data[0, 0]) == 2
            assert int(output.data[3, 3]) == 20
            assert np.isnan(output.data[4, 4])  # TODO: how to handle outside bounds

        def test_interpolate_irregular_arbitrary_2dims(self):
            """ irregular interpolation """

            # try >2 dims
            source = np.random.rand(5, 5, 3)
            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5), [2, 3, 5]], dims=["lat", "lon", "time"])
            coords_dst = Coordinates([clinspace(1, 11, 5), clinspace(1, 11, 5), [2, 3, 5]], dims=["lat", "lon", "time"])

            node = MockArrayDataSource(
                source=source,
                native_coordinates=coords_src,
                interpolation={"method": "nearest", "interpolators": [ScipyGrid]},
            )
            output = node.eval(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords["lat"])
            assert np.all(output.lon.values == coords_dst.coords["lon"])
            assert np.all(output.time.values == coords_dst.coords["time"])

            # assert output.data[0, 0] == source[]

        def test_interpolate_irregular_arbitrary_descending(self):
            """should handle descending"""

            source = np.random.rand(5, 5)
            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
            coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=["lat", "lon"])

            node = MockArrayDataSource(
                source=source,
                native_coordinates=coords_src,
                interpolation={"method": "nearest", "interpolators": [ScipyGrid]},
            )
            output = node.eval(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords["lat"])
            assert np.all(output.lon.values == coords_dst.coords["lon"])

        def test_interpolate_irregular_arbitrary_swap(self):
            """should handle descending"""

            source = np.random.rand(5, 5)
            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
            coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=["lat", "lon"])

            node = MockArrayDataSource(
                source=source,
                native_coordinates=coords_src,
                interpolation={"method": "nearest", "interpolators": [ScipyGrid]},
            )
            output = node.eval(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords["lat"])
            assert np.all(output.lon.values == coords_dst.coords["lon"])

        def test_interpolate_irregular_lat_lon(self):
            """ irregular interpolation """

            source = np.random.rand(5, 5)
            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
            coords_dst = Coordinates([[[0, 2, 4, 6, 8, 10], [0, 2, 4, 5, 6, 10]]], dims=["lat_lon"])

            node = MockArrayDataSource(
                source=source,
                native_coordinates=coords_src,
                interpolation={"method": "nearest", "interpolators": [ScipyGrid]},
            )
            output = node.eval(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat_lon.values == coords_dst.coords["lat_lon"])
            assert output.values[0] == source[0, 0]
            assert output.values[1] == source[1, 1]
            assert output.values[-1] == source[-1, -1]

    class TestInterpolateScipyPoint(object):
        def test_interpolate_scipy_point(self):
            """ interpolate point data to nearest neighbor with various coords_dst"""

            source = np.random.rand(6)
            coords_src = Coordinates([[[0, 2, 4, 6, 8, 10], [0, 2, 4, 5, 6, 10]]], dims=["lat_lon"])
            coords_dst = Coordinates([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]], dims=["lat_lon"])
            node = MockArrayDataSource(
                source=source,
                native_coordinates=coords_src,
                interpolation={"method": "nearest", "interpolators": [ScipyPoint]},
            )

            output = node.eval(coords_dst)
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat_lon.values == coords_dst.coords["lat_lon"])
            assert output.values[0] == source[0]
            assert output.values[-1] == source[3]

            coords_dst = Coordinates([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dims=["lat", "lon"])
            output = node.eval(coords_dst)
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords["lat"])
            assert output.values[0, 0] == source[0]
            assert output.values[-1, -1] == source[3]
