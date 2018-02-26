
import pytest
import traitlets as tl

from podpac.core.coordinate import Coord, MonotonicCoord, UniformCoord
from podpac.core.coordinate import coord_linspace

class TestCoord(object):
    pass

class TestMonotonicCoord(object):
    def test_floating_point_error(self):
        c = coord_linspace(50.619, 50.62795, 30)
        assert(c.size == 30)

class TestUniformCoord(object):
    pass
 
class TestCoordLinspace(object):
    pass