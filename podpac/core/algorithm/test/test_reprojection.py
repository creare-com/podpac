from __future__ import division, unicode_literals, print_function, absolute_import

import pytest

import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import traitlets as tl

import podpac
from podpac import Coordinates, clinspace
from podpac.data import Array
from podpac.core.algorithm.reprojection import Reproject


class TestReprojection(object):
    source_coords = Coordinates([clinspace(0, 8, 9, "lat"), clinspace(0, 8, 9, "lon")])
    coarse_coords = Coordinates([clinspace(0, 8, 3, "lat"), clinspace(0, 8, 3, "lon")])
    source = Array(source=np.arange(81).reshape(9, 9), coordinates=source_coords, interpolation="nearest")
    source_coarse = Array(
        source=[[0, 4, 8], [36, 40, 44], [72, 76, 80]], coordinates=coarse_coords, interpolation="bilinear"
    )

    def test_reprojection_Coordinates(self):
        reproject = Reproject(source=self.source, interpolation="bilinear", coordinates=self.coarse_coords)
        o1 = reproject.eval(self.source_coords)
        o2 = self.source_coarse.eval(self.source_coords)

        assert_array_equal(o1.data, o2.data)

        node = podpac.Node.from_json(reproject.json)
        o3 = node.eval(self.source_coords)
        assert_array_equal(o1.data, o3.data)

    def test_reprojection_source_coords(self):
        reproject = Reproject(source=self.source, interpolation="bilinear", coordinates=self.source_coarse)
        o1 = reproject.eval(self.coarse_coords)
        o2 = self.source_coarse.eval(self.coarse_coords)

        assert_array_equal(o1.data, o2.data)

        node = podpac.Node.from_json(reproject.json)
        o3 = node.eval(self.coarse_coords)
        assert_array_equal(o1.data, o3.data)

    def test_reprojection_source_dict(self):
        reproject = Reproject(source=self.source, interpolation="bilinear", coordinates=self.coarse_coords.definition)
        o1 = reproject.eval(self.coarse_coords)
        o2 = self.source_coarse.eval(self.coarse_coords)

        assert_array_equal(o1.data, o2.data)

        node = podpac.Node.from_json(reproject.json)
        o3 = node.eval(self.coarse_coords)
        assert_array_equal(o1.data, o3.data)

    def test_reprojection_source_str(self):
        reproject = Reproject(source=self.source, interpolation="bilinear", coordinates=self.coarse_coords.json)
        o1 = reproject.eval(self.coarse_coords)
        o2 = self.source_coarse.eval(self.coarse_coords)

        assert_array_equal(o1.data, o2.data)

        node = podpac.Node.from_json(reproject.json)
        o3 = node.eval(self.coarse_coords)
        assert_array_equal(o1.data, o3.data)
