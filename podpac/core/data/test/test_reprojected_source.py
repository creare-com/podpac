import pytest

import numpy as np
from traitlets import TraitError

from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.units import UnitsDataArray
from podpac.core.node import Node
from podpac.core.algorithm.utility import Arange
from podpac.core.data.datasource import DataSource
from podpac.core.data.array_source import Array
from podpac.core.data.reprojection import ReprojectedSource


class TestReprojectedSource(object):

    """Test Reprojected Source
    TODO: this needs to be reworked with real examples
    """

    data = np.random.rand(11, 11)
    coordinates = Coordinates([clinspace(-25, 25, 11), clinspace(-25, 25, 11)], dims=["lat", "lon"])
    reprojected_coordinates = Coordinates([clinspace(-25, 50, 11), clinspace(-25, 50, 11)], dims=["lat", "lon"])

    def test_init(self):
        """test basic init of class"""

        node = ReprojectedSource(source=Node(), reprojected_coordinates=self.reprojected_coordinates)
        assert isinstance(node, ReprojectedSource)

    def test_coordinates(self):
        """test coordinates"""

        # source has no coordinates, just use reprojected_coordinates
        node = ReprojectedSource(source=Node(), reprojected_coordinates=self.reprojected_coordinates)
        assert node.coordinates == self.reprojected_coordinates

        # source has coordinates
        source = Array(coordinates=self.coordinates)
        node = ReprojectedSource(source=source, reprojected_coordinates=self.reprojected_coordinates)
        assert node.coordinates == self.reprojected_coordinates

    def test_get_data(self):
        """test get data from reprojected source"""
        source = Array(source=self.data, coordinates=self.coordinates)
        node = ReprojectedSource(source=source, reprojected_coordinates=source.coordinates)
        output = node.eval(node.coordinates)

    def test_base_ref(self):
        """test base ref"""

        node = ReprojectedSource(source=Node(), reprojected_coordinates=self.reprojected_coordinates)
        assert "_reprojected" in node.base_ref

    def test_deserialize_reprojected_coordinates(self):
        node1 = ReprojectedSource(source=Node(), reprojected_coordinates=self.reprojected_coordinates)
        node2 = ReprojectedSource(source=Node(), reprojected_coordinates=self.reprojected_coordinates.definition)
        node3 = ReprojectedSource(source=Node(), reprojected_coordinates=self.reprojected_coordinates.json)

        assert node1.reprojected_coordinates == self.reprojected_coordinates
        assert node2.reprojected_coordinates == self.reprojected_coordinates
        assert node3.reprojected_coordinates == self.reprojected_coordinates
