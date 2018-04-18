from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np
import xarray as xr
from pint.errors import DimensionalityError
from pint import UnitRegistry
ureg = UnitRegistry()

from podpac.core.coordinate import Coordinate
from podpac.core.algorithm.algorithm import Algorithm, SinCoords, Arithmetic

class TestAlgorithm(object):
    def TestAlgorithmConstructor(self):
        a = Algorithm()
        
    def TestSinCoords(self):
        a = SinCoords()
        coords = Coordinate(lat=[-90, 90, 1.], lon=[0, 360, 2.])
        o = a.execute(coords)
        
    