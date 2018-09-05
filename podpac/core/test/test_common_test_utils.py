from __future__ import division, unicode_literals, print_function, absolute_import

import pytest

import podpac.core.common_test_utils as ctu
from podpac.core.coordinates import UniformCoordinates1d, ArrayCoordinates1d

@pytest.mark.skip('various')
class TestMakeCoordinates(object):
    def test_default_creation(self):
        # Just make sure it runs
        coords = ctu.make_coordinate_combinations()
        assert(len(coords) > 0)
        assert(len(coords) == 168)
        
    def test_custom_creation_no_stack(self):
        kwargs = {}
        kwargs['lat'] = UniformCoordinates1d(0, 2, 1.0)
        kwargs['lon'] = UniformCoordinates1d(2, 6, 1.0)
        kwargs['alt'] = UniformCoordinates1d(6, 12, 1.0)
        kwargs['time'] = UniformCoordinates1d('2018-01-01T00:00:00', '2018-02-01T00:00:00', '1,M')
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
        coords = ctu.make_coordinate_combinations(lat=ArrayCoordinates1d([0.0, 1.0, 2.0, 4.0]))
        assert(len(coords) > 0)
        assert(len(coords) == 84)
        
    
