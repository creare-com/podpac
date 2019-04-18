
from __future__ import division, unicode_literals, print_function, absolute_import

import json

import pytest

import podpac
from podpac.core.coordinates.coordinates import Coordinates
from podpac.core.coordinates.group_coordinates import GroupCoordinates

class TestGroupCoordinates(object):
    def test_init(self):
        # empty
        g = GroupCoordinates([])
        
        # same dims, unstacked
        c1 = Coordinates([[0, 1], [0, 1]], dims=['lat', 'lon'])
        c2 = Coordinates([[10, 11], [10, 11]], dims=['lat', 'lon'])
        g = GroupCoordinates([c1, c2])

        # same dims, stacked
        c2 = Coordinates([[[0, 1], [0, 1]]], dims=['lat_lon'])
        c2 = Coordinates([[[10, 11], [10, 11]]], dims=['lat_lon'])
        g = GroupCoordinates([c1, c2])

        # different order
        c1 = Coordinates([[0, 1], [0, 1]], dims=['lat', 'lon'])
        c2 = Coordinates([[10, 11], [10, 11]], dims=['lon', 'lat'])
        g = GroupCoordinates([c1, c2])

        # different stacking
        c1 = Coordinates([[0, 1], [0, 1]], dims=['lat', 'lon'])
        c2 = Coordinates([[[10, 11], [10, 11]]], dims=['lat_lon'])
        g = GroupCoordinates([c1, c2])

    def test_init_mismatching_dims(self):
        # mismatching dims
        c1 = Coordinates([[0, 1], [0, 1]], dims=['lat', 'lon'])
        c2 = Coordinates([[10, 11], [10, 11], '2018-01-01'], dims=['lat', 'lon', 'time'])

        with pytest.raises(ValueError, match='Mismatching dims'):
            GroupCoordinates([c1, c2])

    def test_properties(self):
        g = GroupCoordinates([])
        assert len(g) == 0
        assert g.udims == set()

        c1 = Coordinates([[0, 1], [0, 1]], dims=['lat', 'lon'])
        c2 = Coordinates([[[10, 11], [10, 11]]], dims=['lat_lon'])

        g = GroupCoordinates([c1, c2])
        assert len(g) == 2
        assert g.udims == set(['lat', 'lon'])

    def test_iter(self):
        c1 = Coordinates([[0, 1], [0, 1]], dims=['lat', 'lon'])
        c2 = Coordinates([[10, 11], [10, 11]], dims=['lat', 'lon'])
        g = GroupCoordinates([c1, c2])

        for c in g:
            assert isinstance(c, Coordinates)

    def test_append(self):
        c1 = Coordinates([[0, 1], [0, 1]], dims=['lat', 'lon'])
        c2 = Coordinates([[10, 11], [10, 11]], dims=['lat', 'lon'])
        c3 = Coordinates(['2018-01-01'], dims=['time'])
        
        g = GroupCoordinates([])
        assert len(g) == 0

        g.append(c1)
        assert len(g) == 1

        g.append(c2)
        assert len(g) == 2

        with pytest.raises(TypeError):
            g.append(10)

        with pytest.raises(ValueError):
            g.append(c3)

        assert g._items[0] is c1
        assert g._items[1] is c2

    def test_add(self):
        c1 = Coordinates([[0, 1], [0, 1]], dims=['lat', 'lon'])
        c2 = Coordinates([[10, 11], [10, 11]], dims=['lat', 'lon'])
        c3 = Coordinates(['2018-01-01'], dims=['time'])

        g1 = GroupCoordinates([c1])
        g2 = GroupCoordinates([c2])
        g3 = GroupCoordinates([c3])

        g = g1 + g2

        assert len(g1) == 1
        assert len(g2) == 1
        assert len(g) == 2
        assert g._items[0] is c1
        assert g._items[1] is c2

        with pytest.raises(ValueError):
            g1 + g3

        with pytest.raises(TypeError):
            g1 + c1

    def test_iadd(self):
        c1 = Coordinates([[0, 1], [0, 1]], dims=['lat', 'lon'])
        c2 = Coordinates([[10, 11], [10, 11]], dims=['lat', 'lon'])
        c3 = Coordinates(['2018-01-01'], dims=['time'])

        g1 = GroupCoordinates([c1])
        g2 = GroupCoordinates([c2])
        g3 = GroupCoordinates([c3])

        g1 += g2

        with pytest.raises(ValueError):
            g1 += g3

        with pytest.raises(TypeError):
            g1 += c1

        assert len(g1) == 2
        assert g1._items[0] is c1
        assert g1._items[1] is c2

        assert len(g2) == 1
        assert g2._items[0] is c2

    def test_repr(self):
        #empty
        g = GroupCoordinates([])
        repr(g)

        # nonempty
        c1 = Coordinates([[0, 1], [0, 1]], dims=['lat', 'lon'])
        c2 = Coordinates([[[10, 11], [10, 11]]], dims=['lat_lon'])
        g = GroupCoordinates([c1, c2])
        repr(g)

    def test_intersect(self):
        c1 = Coordinates([[0, 1, 2], [0, 1, 2]], dims=['lat', 'lon'])
        c2 = Coordinates([[10, 11], [10, 11]], dims=['lat', 'lon'])
        c3 = Coordinates([[0.5, 1.5, 2.5]], dims=['lat'])
        
        g = GroupCoordinates([c1, c2])

        g2 = g.intersect(c3)
        g2 = g.intersect(c3, outer=True)
        g2, I = g.intersect(c3, return_indices=True)

    def test_definition(self):
        c1 = Coordinates([[0, 1], [0, 1]], dims=['lat', 'lon'])
        c2 = Coordinates([[10, 11], [10, 11]], dims=['lat', 'lon'])
        g = GroupCoordinates([c1, c2])

        d = g.definition
        json.dumps(d, cls=podpac.core.utils.JSONEncoder)
        g2 = GroupCoordinates.from_definition(d)

    def test_json(self):
        c1 = Coordinates([[0, 1], [0, 1]], dims=['lat', 'lon'])
        c2 = Coordinates([[10, 11], [10, 11]], dims=['lat', 'lon'])
        g = GroupCoordinates([c1, c2])

        s = g.json
        g2 = GroupCoordinates.from_json(s)

    def test_hash(self):
        c1 = Coordinates([[0, 1], [0, 1]], dims=['lat', 'lon'])
        c2 = Coordinates([[10, 11], [10, 11]], dims=['lat', 'lon'])
        c3 = Coordinates([[10, 11], [10, 11]], dims=['lat', 'lon'])
        c4 = Coordinates([[10, 12], [10, 11]], dims=['lat', 'lon'])
        
        g1 = GroupCoordinates([c1, c2])
        g2 = GroupCoordinates([c1, c2])
        g3 = GroupCoordinates([c1, c3])
        g4 = GroupCoordinates([c1, c4])

        assert g1.hash == g1.hash
        assert g1.hash == g2.hash
        assert g1.hash == g3.hash
        assert g1.hash != g4.hash