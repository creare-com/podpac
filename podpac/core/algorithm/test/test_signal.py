from __future__ import division, unicode_literals, print_function, absolute_import

import pytest

import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import traitlets as tl

import podpac
from podpac import Coordinates, clinspace, crange
from podpac.algorithm import Arange
from podpac.data import Array
from podpac.core.algorithm.signal import Convolution, TimeConvolution, SpatialConvolution


class TestConvolution(object):
    def test_init_kernel(self):
        node = Convolution(source=Arange(), kernel=[1, 2, 1])
        assert_equal(node.kernel, [1, 2, 1])

        node = Convolution(source=Arange(), kernel_type="mean, 5", kernel_ndim=2)
        assert node.kernel.shape == (5, 5)
        assert np.all(node.kernel == 0.04)

        node = Convolution(source=Arange(), kernel_type="mean, 5", kernel_ndim=3)
        assert node.kernel.shape == (5, 5, 5)
        assert np.all(node.kernel == 0.008)

        node = Convolution(source=Arange(), kernel_type="gaussian, 3, 1", kernel_ndim=2)
        assert node.kernel.shape == (3, 3)

        # kernel and kernel_type invalid
        with pytest.raises(TypeError, match="Convolution expected 'kernel' or 'kernel_type', not both"):
            Convolution(source=Arange(), kernel=[1, 2, 1], kernel_type="mean, 5")

        # kernel or kernel_type required
        with pytest.raises(TypeError, match="Convolution requires 'kernel' array or 'kernel_type' string"):
            Convolution(source=Arange())

        # ndim required
        with pytest.raises(TypeError, match="Convolution requires 'kernel_ndim'"):
            Convolution(source=Arange(), kernel_type="mean, 5")

    def test_eval(self):
        lat = clinspace(45, 66, 30, name="lat")
        lon = clinspace(-80, 70, 40, name="lon")
        time = crange("2017-09-01", "2017-10-31", "1,D", name="time")

        kernel1d = [1, 2, 1]
        kernel2d = [[1, 2, 1]]
        kernel3d = [[[1, 2, 1]]]

        node1d = Convolution(source=Arange(), kernel=kernel1d)
        node2d = Convolution(source=Arange(), kernel=kernel2d)
        node3d = Convolution(source=Arange(), kernel=kernel3d)

        o = node1d.eval(Coordinates([time]))
        o = node2d.eval(Coordinates([lat, lon]))
        o = node3d.eval(Coordinates([lat, lon, time]))

        with pytest.raises(ValueError, match="Cannot evaluate coordinates, kernel and coordinates ndims mismatch"):
            node2d.eval(Coordinates([lat, lon, time]))

    def test_eval_multiple_outputs(self):

        lat = clinspace(45, 66, 30, name="lat")
        lon = clinspace(-80, 70, 40, name="lon")
        kernel = [[1, 2, 1]]
        coords = Coordinates([lat, lon])
        multi = Array(source=np.random.random(coords.shape + (2,)), native_coordinates=coords, outputs=["a", "b"])
        node = Convolution(source=multi, kernel=kernel)
        o = node.eval(Coordinates([lat, lon]))

    def test_eval_nan(self):
        lat = clinspace(45, 66, 30, name="lat")
        lon = clinspace(-80, 70, 40, name="lon")
        coords = Coordinates([lat, lon])

        data = np.ones(coords.shape)
        data[10, 10] = np.nan
        source = Array(source=data, native_coordinates=coords)
        node = Convolution(source=source, kernel=[[1, 2, 1]])

        o = node.eval(coords[8:12, 7:13])

    def test_eval_with_output_argument(self):
        lat = clinspace(45, 66, 30, name="lat")
        lon = clinspace(-80, 70, 40, name="lon")
        coords = Coordinates([lat, lon])

        node = Convolution(source=Arange(), kernel=[[1, 2, 1]], kernel_ndim=2)

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
            node = Convolution(source=a, kernel=[[1, 2, 1]], kernel_ndim=2)
            node.eval(coords)

            assert node.source is a

            # debuggable
            podpac.settings["DEBUG"] = True

            a = Arange()
            node = Convolution(source=a, kernel=[[1, 2, 1]], kernel_ndim=2)
            node.eval(coords)

            assert node.source is not a
            assert node._requested_coordinates == coords
            assert node.source._requested_coordinates is not None
            assert node.source._requested_coordinates != coords
            assert a._requested_coordinates is None


class TestSpatialConvolution(object):
    def test_init_kernel(self):
        node = SpatialConvolution(source=Arange(), kernel=[[1, 2, 1]])
        assert_equal(node.kernel, [[1, 2, 1]])

        node = SpatialConvolution(source=Arange(), kernel_type="mean, 5")
        assert node.kernel.shape == (5, 5)
        assert np.all(node.kernel == 0.04)

        node = SpatialConvolution(source=Arange(), kernel_type="gaussian, 3, 1")
        assert node.kernel.shape == (3, 3)

        # kernel and kernel_type invalid
        with pytest.raises(TypeError, match="Convolution expected 'kernel' or 'kernel_type', not both"):
            SpatialConvolution(source=Arange(), kernel=[[1, 2, 1]], kernel_type="mean, 5")

        # kernel or kernel_type required
        with pytest.raises(TypeError, match="Convolution requires 'kernel' array or 'kernel_type' string"):
            SpatialConvolution(source=Arange())

        # kernel ndim must be 2
        with pytest.raises(tl.TraitError):
            SpatialConvolution(source=Arange(), kernel=[1, 2, 1])

        with pytest.raises(tl.TraitError):
            SpatialConvolution(source=Arange(), kernel=[[[1, 2, 1]]])

    def test_eval(self):
        lat = clinspace(45, 66, 30, name="lat")
        lon = clinspace(-80, 70, 40, name="lon")
        time = crange("2017-09-01", "2017-10-31", "1,D", name="time")

        node = SpatialConvolution(source=Arange(), kernel=[[1, 2, 1]])
        o = node.eval(Coordinates([lat, lon]))
        o = node.eval(Coordinates([lat, lon, time]))

        with pytest.raises(ValueError, match="cannot compute spatial convolution with coordinate dims"):
            node.eval(Coordinates([time]))


class TestTimeConvolution(object):
    def test_init_kernel(self):
        node = TimeConvolution(source=Arange(), kernel=[1, 2, 1])
        assert_equal(node.kernel, [1, 2, 1])

        node = TimeConvolution(source=Arange(), kernel_type="mean, 5")
        assert node.kernel.shape == (5,)
        assert np.all(node.kernel == 0.2)

        node = TimeConvolution(source=Arange(), kernel_type="gaussian, 3, 1")
        assert node.kernel.shape == (3,)

        # kernel and kernel_type invalid
        with pytest.raises(TypeError, match="Convolution expected 'kernel' or 'kernel_type', not both"):
            TimeConvolution(source=Arange(), kernel=[1, 2, 1], kernel_type="mean, 5")

        # kernel or kernel_type required
        with pytest.raises(TypeError, match="Convolution requires 'kernel' array or 'kernel_type' string"):
            TimeConvolution(source=Arange())

        # kernel ndim must be 2
        with pytest.raises(tl.TraitError):
            TimeConvolution(source=Arange(), kernel=[[1, 2, 1]])

        with pytest.raises(tl.TraitError):
            TimeConvolution(source=Arange(), kernel=[[[1, 2, 1]]])

    def test_eval(self):
        lat = clinspace(45, 66, 30, name="lat")
        lon = clinspace(-80, 70, 40, name="lon")
        time = crange("2017-09-01", "2017-10-31", "1,D", name="time")

        node = TimeConvolution(source=Arange(), kernel=[1, 2, 1])
        o = node.eval(Coordinates([time]))
        o = node.eval(Coordinates([lat, lon, time]))

        with pytest.raises(ValueError, match="cannot compute time convolution with time-independent coordinates"):
            node.eval(Coordinates([lat, lon]))
