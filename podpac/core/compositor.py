from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
import numpy as np
import traitlets as tl
import matplotlib.colors, matplotlib.cm
import matplotlib.pyplot as plt
from pint import UnitRegistry
ureg = UnitRegistry()

# Internal imports
from podpac.core.coordinate import Coordinate
from podpac.core.node import Node, UnitsDataArray

class Compositor(Node):
    pass
        

class GridCompositor(Compositor):
    shared_coordinates = tl.Instance(Coordinate)
    unique_file_coordinates = tl.Instance(Coordinate)
    
    @tl.default('native_coordinates')
    def set_native_coordinates(self):
        pass
    
    def execute(self, coordinates, params=None, output=None):
        pass
    