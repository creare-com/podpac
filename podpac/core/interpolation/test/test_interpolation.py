"""
Test interpolation methods


"""
# pylint: disable=C0111,W0212,R0903


import pytest
import traitlets as tl
import numpy as np
from numpy.testing import assert_array_equal

import podpac
from podpac.core.units import UnitsDataArray
from podpac.core.node import Node
from podpac.core.coordinates import Coordinates
from podpac.core.interpolation.interpolation import Interpolate, InterpolationMixin
from podpac.core.data.array_source import Array, ArrayBase
from podpac.core.compositor.data_compositor import DataCompositor
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
    interp = Interpolate(source=s1, interpolation="nearest")
    coords = Coordinates([np.linspace(0, 8, 17), np.linspace(0, 14, 29)], ["lat", "lon"])
    coords2 = Coordinates([np.linspace(0, 17, 18), np.linspace(0, 14, 15)], ["lat", "lon"])
    coords2c = Coordinates([np.linspace(0.1, 16.8, 5), np.linspace(0.1, 13.8, 3)], ["lat", "lon"])

    def test_basic_interpolation(self):
        # This JUST tests the interface, tests for the actual value of the interpolation is left
        # to the test_interpolation_manager.py file

        o = self.interp.eval(self.coords)

        assert o.shape == (17, 29)

    def test_interpolation_definition(self):
        node = Node.from_json(self.interp.json)
        o1 = node.eval(self.coords)
        o2 = self.interp.eval(self.coords)
        np.testing.assert_array_equal(o1.data, o2.data)
        assert node.json == self.interp.json

    def test_compositor_chain(self):
        dc = DataCompositor(sources=[self.s2, self.s1])
        node = Interpolate(source=dc, interpolation="nearest")
        o = node.eval(self.coords2)

        np.testing.assert_array_equal(o.data, np.concatenate([self.s1.source, self.s2.source], axis=0))


class TestInterpolationBehavior(object):
    def test_linear_1D_issue411and413(self):
        data = [0, 1, 2]
        raw_coords = data.copy()
        raw_e_coords = [0, 0.5, 1, 0.6, 2]

        for dim in ["lat", "lon", "alt", "time"]:
            ec = Coordinates([raw_e_coords], [dim])

            arrb = ArrayBase(source=data, coordinates=Coordinates([raw_coords], [dim]))
            node = Interpolate(source=arrb, interpolation="linear")
            o = node.eval(ec)

            assert np.all(o.data == raw_e_coords)

        # Do time interpolation explicitly
        raw_coords = ["2020-11-01", "2020-11-03", "2020-11-05"]
        raw_et_coords = ["2020-11-01", "2020-11-02", "2020-11-03", "2020-11-04", "2020-11-05"]
        ec = Coordinates([raw_et_coords], ["time"])

        arrb = ArrayBase(source=data, coordinates=Coordinates([raw_coords], ["time"]))
        node = Interpolate(source=arrb, interpolation="linear")
        o = node.eval(ec)

        assert np.all(o.data == raw_e_coords)

    def test_stacked_coords_with_partial_dims_issue123(self):
        node = Array(
            source=[0, 1, 2],
            coordinates=Coordinates(
                [[[0, 2, 1], [10, 12, 11], ["2018-01-01", "2018-01-02", "2018-01-03"]]], dims=["lat_lon_time"]
            ),
            interpolation="nearest",
        )

        # unstacked or and stacked requests without time
        o1 = node.eval(Coordinates([[0.5, 1.5], [10.5, 11.5]], dims=["lat", "lon"]))
        o2 = node.eval(Coordinates([[[0.5, 1.5], [10.5, 11.5]]], dims=["lat_lon"]))

        assert_array_equal(o1.data, [[0, 2], [2, 1]])
        assert_array_equal(o2.data, [0, 1])

        # request without lat or lon
        o3 = node.eval(Coordinates(["2018-01-01"], dims=["time"]))
        assert o3.data[0] == 0

    def test_ignored_interpolation_params_issue340(self, caplog):
        node = Array(
            source=[0, 1, 2],
            coordinates=Coordinates([[0, 2, 1]], dims=["time"]),
            interpolation={
                "method": "nearest",
                "params": {
                    "fake_param": 1.1,
                    "spatial_tolerance": 1,
                },
            },
        )
        node.eval(Coordinates([[0.5, 1.5]], ["time"]))
        assert "interpolation parameter 'fake_param' was ignored" in caplog.text
        assert "interpolation parameter 'spatial_tolerance' was ignored" not in caplog.text

    def test_silent_nearest_neighbor_interp_bug_issue412(self):
        node = podpac.data.Array(
            source=[0, 1, 2],
            coordinates=podpac.Coordinates([[1, 5, 9]], dims=["lat"]),
            interpolation=[{"method": "bilinear", "dims": ["lat"]}],
        )
        o = node.eval(podpac.Coordinates([podpac.crange(1, 9, 1)], dims=["lat"]))
        assert_array_equal(o.data, np.linspace(0, 2, 9))
