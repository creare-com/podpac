import pytest

import numpy as np
from traitlets import TraitError

from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.units import UnitsDataArray
from podpac.core.node import Node
from podpac.core.data.array_source import Array
from podpac.core.data.reprojection import ReprojectedSource


class TestReprojectedSource(object):

    """Test Reprojected Source
    TODO: this needs to be reworked with real examples
    """

    source = Node()
    data = np.random.rand(11, 11)
    native_coordinates = Coordinates([clinspace(-25, 25, 11), clinspace(-25, 25, 11)], dims=["lat", "lon"])
    reprojected_coordinates = Coordinates([clinspace(-25, 50, 11), clinspace(-25, 50, 11)], dims=["lat", "lon"])

    def test_init(self):
        """test basic init of class"""

        node = ReprojectedSource(source=self.source)
        assert isinstance(node, ReprojectedSource)

    def test_traits(self):
        """ check each of the s3 traits """

        ReprojectedSource(source=self.source)
        with pytest.raises(TraitError):
            ReprojectedSource(source=5)

        ReprojectedSource(source_interpolation="bilinear")
        with pytest.raises(TraitError):
            ReprojectedSource(source_interpolation=5)

        ReprojectedSource(reprojected_coordinates=self.reprojected_coordinates)
        with pytest.raises(TraitError):
            ReprojectedSource(reprojected_coordinates=5)

    def test_native_coordinates(self):
        """test native coordinates"""

        # error if no source has coordinates
        with pytest.raises(Exception):
            node = ReprojectedSource(source=Node())
            node.native_coordinates

        # source as Node
        node = ReprojectedSource(source=self.source, reprojected_coordinates=self.reprojected_coordinates)
        assert isinstance(node.native_coordinates, Coordinates)
        assert node.native_coordinates["lat"].coordinates[0] == self.reprojected_coordinates["lat"].coordinates[0]

    def test_get_data(self):
        """test get data from reprojected source"""
        datanode = Array(source=self.data, native_coordinates=self.native_coordinates)
        node = ReprojectedSource(source=datanode, reprojected_coordinates=datanode.native_coordinates)
        output = node.eval(node.native_coordinates)
        assert isinstance(output, UnitsDataArray)

    def test_base_ref(self):
        """test base ref"""

        node = ReprojectedSource(source=self.source, reprojected_coordinates=self.reprojected_coordinates)
        ref = node.base_ref
        assert "_reprojected" in ref

    def test_base_definition(self):
        """test definition"""

        node = ReprojectedSource(source=self.source, reprojected_coordinates=self.reprojected_coordinates)
        d = node.base_definition
        assert d["attrs"]["reprojected_coordinates"] == self.reprojected_coordinates

    def test_deserialize_reprojected_coordinates(self):
        node1 = ReprojectedSource(source=self.source, reprojected_coordinates=self.reprojected_coordinates)
        node2 = ReprojectedSource(source=self.source, reprojected_coordinates=self.reprojected_coordinates.definition)
        node3 = ReprojectedSource(source=self.source, reprojected_coordinates=self.reprojected_coordinates.json)

        assert node1.reprojected_coordinates == self.reprojected_coordinates
        assert node2.reprojected_coordinates == self.reprojected_coordinates
        assert node3.reprojected_coordinates == self.reprojected_coordinates
