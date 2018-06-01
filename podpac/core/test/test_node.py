from __future__ import division, unicode_literals, print_function, absolute_import

import pytest
import numpy as np
from pint.errors import DimensionalityError
from pint import UnitRegistry
ureg = UnitRegistry()

import podpac
from podpac.core.node import Node
from podpac.core.units import UnitsDataArray

class TestNodeOutputArrayCreation(object):
    @classmethod
    def setup_class(cls):
        cls.c1 = podpac.coordinate(lat_lon=((0, 1), (0, 1), 10), time=(0, 1, 2), order=['lat_lon', 'time'])
        cls.c2 = podpac.coordinate(lat_lon=((0.5, 1.5), (0.1, 1.1), 15))
        
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
        