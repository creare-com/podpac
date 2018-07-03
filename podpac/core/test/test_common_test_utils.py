from __future__ import division, unicode_literals, print_function, absolute_import

import pytest

import podpac.core.common_test_utils as ctu
import podpac.core.coordinate as pcoord

class TestMakeCoordinates(object):
    def test_default_creation(self):
        # Just make sure it runs
        coords = ctu.make_coordinate_combinations()
        assert(len(coords) > 0)
        assert(len(coords) == 168)
        
    def test_custom_creation_no_stack(self):
        coord1d_type = pcoord.UniformCoord

        kwargs = {}
        kwargs['lat'] = coord1d_type(start=0, stop=2, delta=1.0)
        kwargs['lon'] = coord1d_type(start=2, stop=6, delta=1.0)
        kwargs['alt'] = coord1d_type(start=6, stop=12, delta=1.0)
        kwargs['time'] = coord1d_type(start='2018-01-01T00:00:00', stop='2018-02-01T00:00:00', delta='1,M')
        coords = ctu.make_coordinate_combinations(**kwargs)
        assert(len(coords) > 0)
        assert(len(coords) == 48)        
        
    def test_custom_creation_latlon_stack(self):
        coord1d_type = pcoord.UniformCoord

        kwargs = {}
        kwargs['alt'] = coord1d_type(start=6, stop=12, delta=1.0)
        kwargs['time'] = coord1d_type(start='2018-01-01T00:00:00', stop='2018-02-01T00:00:00', delta='1,M')
        coords = ctu.make_coordinate_combinations(**kwargs)
        assert(len(coords) > 0)
        assert(len(coords) == 70)
        
    def test_custom_creation_mixed_type_1d(self):
        coords = ctu.make_coordinate_combinations(lat=pcoord.MonotonicCoord([0.0, 1.0, 2.0, 4.0]))
        assert(len(coords) > 0)
        assert(len(coords) == 84)
        
    
