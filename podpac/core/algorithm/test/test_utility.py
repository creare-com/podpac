from __future__ import division, unicode_literals, print_function, absolute_import

import pytest
import numpy as np

import podpac
from podpac.core.algorithm.utility import Arange, CoordData, SinCoords


class TestArange(object):
    def test_Arange(self):
        coords = podpac.Coordinates([[0, 1, 2], [0, 1, 2, 3, 4]], dims=["lat", "lon"])
        node = Arange()
        output = node.eval(coords)
        assert output.shape == coords.shape


class TestCoordData(object):
    def test_CoordData(self):
        coords = podpac.Coordinates([[0, 1, 2], [0, 1, 2, 3, 4]], dims=["lat", "lon"])

        node = CoordData(coord_name="lat")
        np.testing.assert_array_equal(node.eval(coords), coords.coords["lat"])

        node = CoordData(coord_name="lon")
        np.testing.assert_array_equal(node.eval(coords), coords.coords["lon"])

    def test_invalid_dimension(self):
        coords = podpac.Coordinates([[0, 1, 2], [0, 1, 2, 3, 4]], dims=["lat", "lon"])
        node = CoordData(coord_name="time")
        with pytest.raises(ValueError):
            node.eval(coords)


class TestSinCoords(object):
    def test_SinCoords(self):
        coords = podpac.Coordinates(
            [podpac.crange(-90, 90, 1.0), podpac.crange("2018-01-01", "2018-01-30", "1,D")], dims=["lat", "time"]
        )
        node = SinCoords()
        output = node.eval(coords)
        assert output.shape == coords.shape
