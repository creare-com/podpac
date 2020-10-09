"""
Test interpolation methods


"""
# pylint: disable=C0111,W0212,R0903


import pytest
import traitlets as tl
import numpy as np

import podpac
from podpac.core.units import UnitsDataArray
from podpac.core.node import Node
from podpac.core.coordinates import Coordinates
from podpac.core.interpolation.interpolation import Interpolation, InterpolationMixin
from podpac.core.data.array_source import Array, ArrayBase
from podpac.core.compositor.ordered_compositor import OrderedCompositor


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
    s1 = ArrayBase(
        source=np.random.rand(9, 15),
        coordinates=Coordinates([np.linspace(0, 8, 9), np.linspace(0, 14, 15)], ["lat", "lon"]),
    )
    s2 = ArrayBase(
        source=np.random.rand(9, 15),
        coordinates=Coordinates([np.linspace(9, 17, 9), np.linspace(0, 14, 15)], ["lat", "lon"]),
    )
    interp = Interpolation(source=s1, interpolation="nearest")
    coords = Coordinates([np.linspace(0, 8, 17), np.linspace(0, 14, 29)], ["lat", "lon"])
    coords2 = Coordinates([np.linspace(0, 17, 18), np.linspace(0, 14, 29)], ["lat", "lon"])
    coords2c = Coordinates([np.linspace(0.1, 16.8, 5), np.linspace(0.1, 13.8, 3)], ["lat", "lon"])

    def test_basic_interpolation(self):
        # This JUST tests the interface, tests for the actual value of the interpolation is left
        # to the test_interpolation_manager.py file

        o = self.interp.eval(self.coords)

        assert o.shape == [7, 9]

    def test_interpolation_definition(self):
        node = Node.from_json(self.interp.json)
        o1 = node.eval(self.coords)
        o2 = self.interp.eval(self.coords)
        np.testing.assert_array_equal(o1.data, o2.data)
        assert node.json == self.interp.json

    def test_compositor_chain(self):
        oc = OrderedCompositor(sources=[self.s2, self.s1])
        o = oc.eval(self.coords2)

        np.testing.assert_array_equal(o.data, np.concatenate([self.s1.data, self.s2.data], axis=1))

    def test_compositor_chain_optimized_find_coordinates(self):
        oc = OrderedCompositor(sources=[self.s2, self.s1])
        node = Interpolation(source=oc, interpolation="nearest")

        # This section now emulates what essentially will happen inside the eval
        # so this is a bit of a bootstrap test, and some of this might be moved
        # to the datasource
        # First let's assign a selector to the input coordinates
        self.coords2c.set_selector(node._interpolation.select_coordinates)

        # Now the intersection function inside the Datasource will only return the needed coordinates
        s1c, s1ci = self.s1.coordinates.intersect(self.coords2c, outer=True, return_indices=True)
        s2c, s2ci = self.s2.coordinates.intersect(self.coords2c, outer=True, return_indices=True)

        assert s1c.shape == (3, 3)
        assert s2c.shape == (2, 3)
