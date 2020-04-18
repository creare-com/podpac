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
from podpac.core.coordinates import Coordinates
from podpac.core.interpolation.interpolation import Interpolation, InterpolationException
from podpac.core.interpolation.interpolation import (
    INTERPOLATION_METHODS,
    INTERPOLATION_DEFAULT,
    INTERPOLATION_METHODS_DICT,
)
from podpac.core.interpolation.interpolator import Interpolator, InterpolatorException
from podpac.core.interpolation.interpolators import NearestNeighbor, NearestPreview


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

        # should handle a default definition without any dimensions
        interp = Interpolation({"method": "nearest", "params": {"spatial_tolerance": 1}})
        assert isinstance(interp.config[("default",)], dict)
        assert interp.config[("default",)]["method"] == "nearest"
        assert isinstance(interp.config[("default",)]["interpolators"][0], Interpolator)
        assert interp.config[("default",)]["params"] == {"spatial_tolerance": 1}

        # handle string methods
        interp = Interpolation({"method": "nearest", "dims": ["lat", "lon"]})
        print(interp.config)
        assert isinstance(interp.config[("lat", "lon")], dict)
        assert interp.config[("lat", "lon")]["method"] == "nearest"
        assert isinstance(interp.config[("default",)]["interpolators"][0], Interpolator)
        assert interp.config[("default",)]["params"] == {}

        # handle dict methods

        # should throw an error if method is not in dict
        with pytest.raises(InterpolationException):
            Interpolation([{"test": "test", "dims": ["lat", "lon"]}])

        # should throw an error if method is not a string
        with pytest.raises(InterpolationException):
            Interpolation([{"method": 5, "dims": ["lat", "lon"]}])

        # should throw an error if method is not one of the INTERPOLATION_METHODS and no interpolators defined
        with pytest.raises(InterpolationException):
            Interpolation([{"method": "myinter", "dims": ["lat", "lon"]}])

        # should throw an error if params is not a dict
        with pytest.raises(TypeError):
            Interpolation([{"method": "nearest", "dims": ["lat", "lon"], "params": "test"}])

        # should throw an error if interpolators is not a list
        with pytest.raises(TypeError):
            Interpolation([{"method": "nearest", "interpolators": "test", "dims": ["lat", "lon"]}])

        # should throw an error if interpolators are not Interpolator classes
        with pytest.raises(TypeError):
            Interpolation([{"method": "nearest", "interpolators": [NearestNeighbor, "test"], "dims": ["lat", "lon"]}])

        # should throw an error if dimension is defined twice
        with pytest.raises(InterpolationException):
            Interpolation([{"method": "nearest", "dims": ["lat", "lon"]}, {"method": "bilinear", "dims": ["lat"]}])

        # should throw an error if dimension is not a list
        with pytest.raises(TypeError):
            Interpolation([{"method": "nearest", "dims": "lat"}])

        # should handle standard INTEPROLATION_SHORTCUTS
        interp = Interpolation([{"method": "nearest", "dims": ["lat", "lon"]}])
        assert isinstance(interp.config[("lat", "lon")], dict)
        assert interp.config[("lat", "lon")]["method"] == "nearest"
        assert isinstance(interp.config[("lat", "lon")]["interpolators"][0], Interpolator)
        assert interp.config[("lat", "lon")]["params"] == {}

        # should not allow custom methods if interpolators can't support
        with pytest.raises(InterpolatorException):
            interp = Interpolation(
                [{"method": "myinter", "interpolators": [NearestNeighbor, NearestPreview], "dims": ["lat", "lon"]}]
            )

        # should allow custom methods if interpolators can support
        class MyInterp(Interpolator):
            methods_supported = ["myinter"]

        interp = Interpolation([{"method": "myinter", "interpolators": [MyInterp], "dims": ["lat", "lon"]}])
        assert interp.config[("lat", "lon")]["method"] == "myinter"
        assert isinstance(interp.config[("lat", "lon")]["interpolators"][0], MyInterp)

        # should allow params to be set
        interp = Interpolation(
            [
                {
                    "method": "myinter",
                    "interpolators": [MyInterp],
                    "params": {"spatial_tolerance": 5},
                    "dims": ["lat", "lon"],
                }
            ]
        )

        assert interp.config[("lat", "lon")]["params"] == {"spatial_tolerance": 5}

        # set default equal to empty tuple
        interp = Interpolation([{"method": "bilinear", "dims": ["lat"]}])
        assert interp.config[("default",)]["method"] == INTERPOLATION_DEFAULT

        # use default with override if not all dimensions are supplied
        interp = Interpolation([{"method": "bilinear", "dims": ["lat"]}, "nearest"])
        assert interp.config[("default",)]["method"] == "nearest"

        # make sure default is always the last key in the ordered config dict
        interp = Interpolation(["nearest", {"method": "bilinear", "dims": ["lat"]}])
        assert list(interp.config.keys())[-1] == ("default",)

        # should sort the dims keys
        interp = Interpolation(["nearest", {"method": "bilinear", "dims": ["lon", "lat"]}])
        assert interp.config[("lat", "lon")]["method"] == "bilinear"

    def test_init_interpolators(self):

        # should set method
        interp = Interpolation("nearest")
        assert interp.config[("default",)]["interpolators"][0].method == "nearest"

        # Interpolation init should init all interpolators in the list
        interp = Interpolation([{"method": "nearest", "params": {"spatial_tolerance": 1}}])
        assert interp.config[("default",)]["interpolators"][0].spatial_tolerance == 1

        # should throw TraitErrors defined by Interpolator
        with pytest.raises(tl.TraitError):
            Interpolation([{"method": "nearest", "params": {"spatial_tolerance": "tol"}}])

        # should not allow undefined params
        with pytest.warns(DeprecationWarning):  # eventually, Traitlets will raise an exception here
            interp = Interpolation([{"method": "nearest", "params": {"myarg": 1}}])
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
            [
                {"method": "myinterp", "interpolators": [LatLon, TimeLat], "dims": ["lat", "lon"]},
                {"method": "myinterp", "interpolators": [TimeLat, Lon], "dims": ["time", "alt"]},
            ]
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

        reqcoords = Coordinates(
            [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], dims=["lat", "lon", "time", "alt"], crs="+proj=merc +vunits=m"
        )
        srccoords = Coordinates(
            [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], dims=["lat", "lon", "time", "alt"], crs="+proj=merc +vunits=m"
        )

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
            [
                {"method": "myinterp", "interpolators": [LatLon, TimeLat], "dims": ["lat", "lon"]},
                {"method": "myinterp", "interpolators": [TimeLat, Lon], "dims": ["time", "alt"]},
            ]
        )

        coords, cidx = interp.select_coordinates(srccoords, [], reqcoords)

        assert len(coords) == len(srccoords)
        assert len(coords["lat"]) == len(srccoords["lat"])
        assert cidx == ()

    def test_interpolate(self):
        class TestInterp(Interpolator):
            dims_supported = ["lat", "lon"]

            def interpolate(
                self, udims, source_coordinates, source_boundary, source_data, eval_coordinates, output_data
            ):
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

        interp = Interpolation({"method": "myinterp", "interpolators": [TestInterp], "dims": ["lat", "lon"]})
        outdata = interp.interpolate(srccoords, {}, srcdata, reqcoords, outdata)

        assert np.all(outdata == srcdata)

        # test if data is size 1
        class TestFakeInterp(Interpolator):
            dims_supported = ["lat"]

            def interpolate(
                self, udims, source_coordinates, source_boundary, source_data, eval_coordinates, output_data
            ):
                return None

        reqcoords = Coordinates([[1]], dims=["lat"])
        srccoords = Coordinates([[1]], dims=["lat"])
        srcdata = UnitsDataArray(
            np.random.rand(1), coords=[srccoords[c].coordinates for c in srccoords], dims=srccoords.dims
        )
        outdata = UnitsDataArray(
            np.zeros(srcdata.shape), coords=[reqcoords[c].coordinates for c in reqcoords], dims=reqcoords.dims
        )

        interp = Interpolation({"method": "myinterp", "interpolators": [TestFakeInterp], "dims": ["lat", "lon"]})
        outdata = interp.interpolate(srccoords, {}, srcdata, reqcoords, outdata)

        assert np.all(outdata == srcdata)

    def test_interpolate_boundary_not_implemented(self):
        class TestInterp(Interpolator):
            dims_supported = ["lat", "lon"]

            def interpolate(
                self, udims, source_coordinates, source_boundary, source_data, eval_coordinates, output_data
            ):
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

        interp = Interpolation({"method": "myinterp", "interpolators": [TestInterp], "dims": ["lat", "lon"]})
        outdata = interp.interpolate(srccoords, {}, srcdata, reqcoords, outdata)

        assert np.all(outdata == srcdata)

        # uniform centered boundary is fine
        outdata = interp.interpolate(srccoords, {"lat": -0.1}, srcdata, reqcoords, outdata)

        # but non-uniform and non-centered boundaries not yet supported
        with pytest.raises(NotImplementedError, match="Non-centered coordinate boundary not yet supported"):
            outdata = interp.interpolate(srccoords, {"lat": [-0.1, 0.2]}, srcdata, reqcoords, outdata)

        with pytest.raises(NotImplementedError, match="Non-uniform coordinate boundary not yet supported"):
            outdata = interp.interpolate(srccoords, {"lat": [[-0.1, 0.2]]}, srcdata, reqcoords, outdata)
