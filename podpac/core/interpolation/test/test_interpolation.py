"""
Test interpolation methods


"""
# pylint: disable=C0111,W0212,R0903


import pytest
import traitlets as tl
import numpy as np

import podpac
from podpac.core.units import UnitsDataArray
from podpac.core.coordinates import Coordinates
from podpac.core.interpolation.interpolation import Interpolation, InterpolationMixin
from podpac.core.data.array_source import Array, ArrayBase


class TestInterpolationMixin(object):
    def test_interpolation_mixin(self):
        class InterpArray(InterpolationMixin, ArrayBase):
            pass

        data = np.random.rand(4, 5)
        native_coords = Coordinates([np.linspace(0, 3, 4), np.linspace(0, 4, 5)], ["lat", "lon"])
        coords = Coordinates([np.linspace(0, 3, 7), np.linspace(0, 4, 9)], ["lat", "lon"])

        iarr_src = InterpArray(source=data, coordinates=native_coords, interpolation="bilinear")
        arr_src = Array(source=data, coordinates=native_coords, interpolation="bilinear")
        arrb_src = ArrayBase(source=data, coordinates=native_coords)

        iaso = iarr_src.eval(coords)
        aso = arr_src.eval(coords)
        abso = arrb_src.eval(coords)

        np.testing.assert_array_equal(iaso.data, aso.data)
        np.testing.assert_array_equal(abso.data, data)


class TestInterpolation(object):
    def test_basic_interpolation(self):
        # This JUST tests the interface, tests for the actual value of the interpolation is left
        # to the test_interpolation_manager.py file
        data = np.random.rand(4, 5)
        native_coords = Coordinates([np.linspace(0, 3, 4), np.linspace(0, 4, 5)], ["lat", "lon"])
        arrb_src = ArrayBase(source=data, coordinates=native_coords)
        interp = Interpolation(interpolation="nearest")

        coords = Coordinates([np.linspace(0, 3, 7), np.linspace(0, 4, 9)], ["lat", "lon"])
        o = interp.eval(coords)

        assert o.shape == [7, 9]

    def test_interpolation_definition(self):
        pass
