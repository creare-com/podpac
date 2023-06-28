from __future__ import division, unicode_literals, print_function, absolute_import

import pytest

import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_almost_equal
import traitlets as tl

import podpac
from podpac import Coordinates, clinspace
from podpac.data import Array
from podpac.core.algorithm.reprojection import Reproject


class TestReprojection(object):
    source_coords = Coordinates([clinspace(0, 8, 9, "lat"), clinspace(0, 8, 9, "lon")])
    coarse_coords = Coordinates([clinspace(0, 8, 3, "lat"), clinspace(0, 8, 3, "lon")])
    source = Array(source=np.arange(81).reshape(9, 9), coordinates=source_coords).interpolate(interpolation="nearest")
    source_coarse = Array(source=[[0, 4, 8], [36, 40, 44], [72, 76, 80]], coordinates=coarse_coords).interpolate(
        interpolation="bilinear"
    )
    source_coarse2 = Array(
        source=[[0, 4, 8], [36, 40, 44], [72, 76, 80]],
        coordinates=coarse_coords.transform("EPSG:3857").transform("EPSG:4326").transform("EPSG:3857"),
    ).interpolate(interpolation={"method": "bilinear", "params": {"fill_value": "extrapolate"}})

    def test_reprojection_Coordinates(self):
        reproject = Reproject(source=self.source, coordinates=self.coarse_coords, interpolation="bilinear")
        o1 = reproject.eval(self.source_coords)
        o2 = self.source_coarse.eval(self.source_coords)

        assert_array_equal(o1.data, o2.data)

        node = podpac.Node.from_json(reproject.json)
        o3 = node.eval(self.source_coords)
        assert_array_equal(o1.data, o3.data)

    def test_reprojection_source_coords(self):
        reproject = Reproject(source=self.source, coordinates=self.source_coarse, interpolation="bilinear")
        o1 = reproject.eval(self.coarse_coords)
        o2 = self.source_coarse.eval(self.coarse_coords)

        assert_array_equal(o1.data, o2.data)

        node = podpac.Node.from_json(reproject.json)
        o3 = node.eval(self.coarse_coords)
        assert_array_equal(o1.data, o3.data)

    def test_reprojection_source_dict(self):
        reproject = Reproject(source=self.source, coordinates=self.coarse_coords.definition, interpolation="bilinear")
        o1 = reproject.eval(self.coarse_coords)
        o2 = self.source_coarse.eval(self.coarse_coords)

        assert_array_equal(o1.data, o2.data)

        node = podpac.Node.from_json(reproject.json)
        o3 = node.eval(self.coarse_coords)
        assert_array_equal(o1.data, o3.data)

    def test_reprojection_source_str(self):
        reproject = Reproject(source=self.source, coordinates=self.coarse_coords.json, interpolation="bilinear")
        o1 = reproject.eval(self.coarse_coords)
        o2 = self.source_coarse.eval(self.coarse_coords)

        assert_array_equal(o1.data, o2.data)

        node = podpac.Node.from_json(reproject.json)
        o3 = node.eval(self.coarse_coords)
        assert_array_equal(o1.data, o3.data)

    def test_reprojection_Coordinates_crs(self): # not sure why this is failing?
        # same eval and source but different reproject
        reproject = Reproject(
            source=self.source,
            coordinates=self.coarse_coords.transform("EPSG:3857"),
            interpolation={"method": "bilinear", "params": {"fill_value": "extrapolate"}},
        )
        o1 = reproject.eval(self.source_coords)
        # We have to use a second source here because the reprojected source
        # gets interpreted as having it's source coordinates in EPSG:3857
        # and when being subsampled, there's a warping effect...
        o2 = self.source_coarse2.eval(self.source_coords)
        assert_almost_equal(o1.data, o2.data, decimal=13)

        node = podpac.Node.from_json(reproject.json)
        o3 = node.eval(self.source_coords)
        assert_array_equal(o1.data, o3.data)

        # same eval and reproject but different source
        o1 = reproject.eval(self.source_coords.transform("EPSG:3857"))
        o2 = self.source_coarse2.eval(self.source_coords.transform("EPSG:3857"))
        assert_almost_equal(o1.data, o2.data, decimal=13)

        # same source and reproject but different eval
        reproject = Reproject(source=self.source, coordinates=self.coarse_coords, interpolation="bilinear")
        o1 = reproject.eval(self.source_coords.transform("EPSG:3857"))
        o2 = self.source_coarse.eval(self.source_coords.transform("EPSG:3857"))
        assert_almost_equal(o1.data, o2.data, decimal=13)
