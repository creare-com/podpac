from __future__ import division, unicode_literals, print_function, absolute_import

import pytest
import numpy as np
from pint.errors import DimensionalityError
from pint import UnitRegistry
ureg = UnitRegistry()

from podpac.core.node import *
from podpac.core.units import UnitsDataArray

class TestStyleCreation(object):
    def test_basic_creation(self):
        s = Style()
    def test_create_with_node(self):
        s = Style(Node())
    def test_get_default_cmap(self):
        Style().cmap
        

class TestNodeProperties(object):
    @classmethod
    def setup_class(cls):
        from podpac import Coordinate
        cls.crds = [Coordinate(lat_lon=((0, 1), (0, 1), 10), time=(0, 1, 2),
                            order=['lat_lon', 'time']),
                    Coordinate(lat_lon=((0.5, 1.5), (0.1, 1.1), 15))
                    ]

    def test_shape_no_nc(self):
        n = Node()
        for crd in self.crds:
            np.testing.assert_array_equal(crd.shape, n.get_output_shape(crd))
    
    def test_shape_with_nc(self):
        crd1 = Coordinate(lat_lon=((0.5, 1.5), (0.1, 1.1), 5))
        n = Node(native_coordinates=Coordinate(lat_lon=((0.5, 1.5), (0.1, 1.1), 15)))
        np.testing.assert_array_equal(crd1.shape,
                                      n.get_output_shape(crd1))
        crd2 = Coordinate(time=(0, 1, 3))
        n.native_coordinates = crd1 + crd2
        # WE SHOULD FIX THE SPEC: This really should be [3, 5]
        np.testing.assert_array_equal([5, 3],
                                      n.get_output_shape(crd2))
        np.testing.assert_array_equal(n.native_coordinates.shape,
                                      n.shape)


class TestNodeOutputArrayCreation(object):
    @classmethod
    def setup_class(cls):
        from podpac import Coordinate
        cls.c1 = Coordinate(lat_lon=((0, 1), (0, 1), 10), time=(0, 1, 2),
                            order=['lat_lon', 'time'])
        cls.c2 = Coordinate(lat_lon=((0.5, 1.5), (0.1, 1.1), 15))
    
    def test_default_output_native_coordinates(self):
        n = Node(native_coordinates=self.c1)
        o = n.output
        np.testing.assert_array_equal(o.shape, self.c1.shape)
        assert(np.all(np.isnan(o)))

    def test_default_output_evaluated_coordinates(self):
        n = Node(evaluated_coordinates=self.c1)
        o = n.output
        np.testing.assert_array_equal(o.shape, self.c1.shape)
        assert(np.all(np.isnan(o)))

    def test_output_creation_stacked_native(self):
        n = Node(native_coordinates=self.c1)
        s1 = n.initialize_output_array().shape
        assert((10, 2) == s1)
        n.evaluated_coordinates = self.c2
        s2 = n.initialize_output_array().shape
        assert((15, 2) == s2)
        n.evaluated_coordinates = self.c2.unstack()
        s3 = n.initialize_output_array().shape
        assert((15, 15, 2) == s3)
        
    def test_output_creation_unstacked_native(self):
        n = Node(native_coordinates=self.c1.unstack())
        s1 = n.initialize_output_array().shape
        assert((10, 10, 2) == s1)
        n.evaluated_coordinates = self.c2
        s2 = n.initialize_output_array().shape
        assert((15, 2) == s2)
        n.evaluated_coordinates = self.c2.unstack()
        s3 = n.initialize_output_array().shape
        assert((15, 15, 2) == s3)    


# TODO: remove this - this is currently a placeholder test until we actually have integration tests (pytest will exit with code 5 if no tests found)
@pytest.mark.integration
def tests_node_integration():
    assert True
