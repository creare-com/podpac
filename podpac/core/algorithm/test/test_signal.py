from __future__ import division, unicode_literals, print_function, absolute_import

import pytest

import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import traitlets as tl

import podpac
from podpac import Coordinates, clinspace, crange
from podpac.algorithm import Arange
from podpac.data import Array
from podpac.core.algorithm.signal import Convolution


class TestConvolution(object):
    def test_init_kernel(self):
        node = Convolution(source=Arange(), kernel=[1, 2, 1], kernel_dims=["lat"])
        assert_equal(node.kernel, [1, 2, 1])

        node = Convolution(source=Arange(), kernel_type="mean, 5", kernel_dims=["lat", "lon"])
        assert node.kernel.shape == (5, 5)
        assert np.all(node.kernel == 0.04)

        node = Convolution(source=Arange(), kernel_type="mean, 5", kernel_dims=["lat", "lon", "time"])
        assert node.kernel.shape == (5, 5, 5)
        assert np.all(node.kernel == 0.008)

        node = Convolution(source=Arange(), kernel_type="gaussian, 3, 1", kernel_dims=["lat", "lon"])
        assert node.kernel.shape == (3, 3)

        # kernel and kernel_type invalid
        with pytest.raises(TypeError, match="Convolution expected 'kernel' or 'kernel_type', not both"):
            Convolution(source=Arange(), kernel=[1, 2, 1], kernel_type="mean, 5", kernel_dims=["lat", "lon"])

        # kernel or kernel_type required
        with pytest.raises(TypeError, match="Convolution requires 'kernel' array or 'kernel_type' string"):
            Convolution(source=Arange(), kernel_dims=["lat", "lon"])

        # kernel_dims required
        with pytest.raises(
            TypeError, match="Convolution expected 'kernel_dims' to be specified when giving a 'kernel' array"
        ):
            Convolution(source=Arange(), kernel_type="mean, 5")

        # kernel_dims correct number of entries
        with pytest.raises(
            TypeError,
            match="The kernel_dims should contain the same number of dimensions as the number of axes in 'kernel', but ",
        ):
            Convolution(source=Arange(), kernel=[[[1, 2]]], kernel_dims=["lat"])

    def test_eval(self):
        lat = clinspace(45, 66, 30, name="lat")
        lon = clinspace(-80, 70, 40, name="lon")
        time = crange("2017-09-01", "2017-10-31", "1,D", name="time")

        kernel1d = [1, 2, 1]
        kernel2d = [[1, 2, 1]]
        kernel3d = [[[1, 2, 1]]]

        node1d = Convolution(source=Arange(), kernel=kernel1d, kernel_dims=["time"])
        node2d = Convolution(source=Arange(), kernel=kernel2d, kernel_dims=["lat", "lon"])
        node3d = Convolution(source=Arange(), kernel=kernel3d, kernel_dims=["lon", "lat", "time"])

        o = node1d.eval(Coordinates([time]))
        o = node2d.eval(Coordinates([lat, lon]))
        o = node3d.eval(Coordinates([lat, lon, time]))

    def test_eval_multiple_outputs(self):

        lat = clinspace(45, 66, 30, name="lat")
        lon = clinspace(-80, 70, 40, name="lon")
        kernel = [[1, 2, 1]]
        coords = Coordinates([lat, lon])
        multi = Array(source=np.random.random(coords.shape + (2,)), coordinates=coords, outputs=["a", "b"])
        node = Convolution(source=multi, kernel=kernel, kernel_dims=["lat", "lon"])
        o1 = node.eval(Coordinates([lat, lon]))

        kernel = [[[1, 2]]]
        coords = Coordinates([lat, lon])
        multi = Array(source=np.random.random(coords.shape + (2,)), coordinates=coords, outputs=["a", "b"])
        node1 = Convolution(source=multi, kernel=kernel, kernel_dims=["lat", "lon", "output"], force_eval=True)
        node2 = Convolution(source=multi, kernel=kernel[0], kernel_dims=["lat", "lon"], force_eval=True)
        o1 = node1.eval(Coordinates([lat, lon]))
        o2 = node2.eval(Coordinates([lat, lon]))

        assert np.any(o2.data != o1.data)

    def test_eval_nan(self):
        lat = clinspace(45, 66, 30, name="lat")
        lon = clinspace(-80, 70, 40, name="lon")
        coords = Coordinates([lat, lon])

        data = np.ones(coords.shape)
        data[10, 10] = np.nan
        source = Array(source=data, coordinates=coords)
        node = Convolution(source=source, kernel=[[1, 2, 1]], kernel_dims=["lat", "lon"])

        o = node.eval(coords[8:12, 7:13])

    def test_eval_with_output_argument(self):
        lat = clinspace(45, 66, 30, name="lat")
        lon = clinspace(-80, 70, 40, name="lon")
        coords = Coordinates([lat, lon])

        node = Convolution(source=Arange(), kernel=[[1, 2, 1]], kernel_dims=["lat", "lon"])

        a = node.create_output_array(coords)
        o = node.eval(coords, output=a)
        assert_array_equal(a, o)

    def test_debuggable_source(self):
        with podpac.settings:
            podpac.settings["DEBUG"] = False
            lat = clinspace(45, 66, 30, name="lat")
            lon = clinspace(-80, 70, 40, name="lon")
            coords = Coordinates([lat, lon])

            # normal version
            a = Arange()
            node = Convolution(source=a, kernel=[[1, 2, 1]], kernel_dims=["lat", "lon"])
            node.eval(coords)

            assert node.source is a

            # debuggable
            podpac.settings["DEBUG"] = True

            a = Arange()
            node = Convolution(source=a, kernel=[[1, 2, 1]], kernel_dims=["lat", "lon"])
            node.eval(coords)

            assert node.source is not a
            assert node._requested_coordinates == coords
            assert node.source._requested_coordinates is not None
            assert node.source._requested_coordinates != coords
            assert a._requested_coordinates is None

    def test_extra_kernel_dims(self):
        lat = clinspace(45, 66, 8, name="lat")
        lon = clinspace(-80, 70, 16, name="lon")
        coords = Coordinates([lat, lon])

        node = Convolution(source=Arange(), kernel=[[[1, 2, 1]]], kernel_dims=["time", "lat", "lon"])
        o = node.eval(coords)

    def test_extra_coord_dims(self):
        lat = clinspace(-0.25, 1.25, 7, name="lat")
        lon = clinspace(-0.125, 1.125, 11, name="lon")
        time = ["2012-05-19", "2016-01-31", "2018-06-20"]
        coords = Coordinates([lat, lon, time], dims=["lat", "lon", "time"])

        source = Array(source=np.random.random(coords.drop("time").shape), coordinates=coords.drop("time"))
        node = Convolution(source=source, kernel=[[-1, 2, -1]], kernel_dims=["lat", "lon"], force_eval=True)
        o = node.eval(coords)
        assert np.all([d in ["lat", "lon"] for d in o.dims])

    def test_coords_order(self):
        lat = clinspace(-0.25, 1.25, 7, name="lat")
        lon = clinspace(-0.125, 1.125, 11, name="lon")
        coords = Coordinates([lat, lon])

        lat = clinspace(0, 1, 5, name="lat")
        lon = clinspace(0, 1, 9, name="lon")
        coords1 = Coordinates([lat, lon])
        coords2 = Coordinates([lon, lat])

        source = Array(source=np.random.random(coords.shape), coordinates=coords)
        node = Convolution(source=source, kernel=[[-1, 2, -1]], kernel_dims=["lat", "lon"], force_eval=True)
        o1 = node.eval(coords1)
        o2 = node.eval(coords2)
        assert np.all(o2.data == o1.data.T)

    def test_missing_source_dims(self):
        """ When the kernel has more dimensions than the source, sum out the kernel for the missing dim"""
        lat = clinspace(-0.25, 1.25, 7, name="lat")
        lon = clinspace(-0.125, 1.125, 11, name="lon")
        time = ["2012-05-19", "2016-01-31", "2018-06-20"]
        coords = Coordinates([lat, lon, time], dims=["lat", "lon", "time"])
        coords2 = Coordinates([lat[[1, 2, 4]], lon, time], dims=["lat", "lon", "time"])

        source = Array(source=np.random.random(coords.drop("time").shape), coordinates=coords.drop("time"))
        node = Convolution(
            source=source, kernel=[[[-1], [2], [-1]]], kernel_dims=["lat", "lon", "time"], force_eval=True
        )
        o = node.eval(coords[:, 1:-1, :])
        expected = source.source[:, 1:-1] * 2 - source.source[:, 2:] - source.source[:, :-2]
        assert np.abs(o.data - expected).max() < 1e-14

        # Check when request has an ArrayCoordinates1d
        node = Convolution(source=source, kernel_type="mean,3", kernel_dims=["lat", "lon", "time"], force_eval=True)
        o = node.eval(coords2[:, 1:-1])
        expected = (
            source.source[[1, 2, 4], 1:-1]
            + source.source[[0, 1, 2], 1:-1]
            + source.source[[2, 4, 6], 1:-1]
            + source.source[[1, 2, 4], :-2]
            + source.source[[0, 1, 2], :-2]
            + source.source[[2, 4, 6], :-2]
            + source.source[[1, 2, 4], 2:]
            + source.source[[0, 1, 2], 2:]
            + source.source[[2, 4, 6], 2:]
        ) / 9
        assert np.abs(o.data - expected).max() < 1e-14

    def test_partial_source_convolution(self):
        lat = clinspace(-0.25, 1.25, 7, name="lat")
        lon = clinspace(-0.125, 1.125, 11, name="lon")
        time = ["2012-05-19", "2016-01-31", "2018-06-20"]
        coords = Coordinates([lat, lon, time], dims=["lat", "lon", "time"])

        source = Array(source=np.random.random(coords.shape), coordinates=coords)
        node = Convolution(source=source, kernel=[[-1, 2, -1]], kernel_dims=["lat", "lon"], force_eval=True)
        o = node.eval(coords[:, 1:-1, :])
        expected = source.source[:, 1:-1] * 2 - source.source[:, 2:] - source.source[:, :-2]

        assert np.abs(o.data - expected).max() < 1e-14
