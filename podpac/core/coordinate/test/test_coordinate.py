
import pytest
import traitlets as tl

from podpac.core.coordinate import Coordinate, CoordinateGroup

class TestCoordinate(object):
    pass

class TestCoordinateGroup(object):
    def test_init(self):
        c1 = Coordinate(
            lat=(0, 10, 5),
            lon=(0, 20, 5),
            time='2018-01-01',
            order=('lat', 'lon', 'time'))

        c2 = Coordinate(
            lat=(10, 20, 15),
            lon=(10, 20, 15),
            time='2018-01-01',
            order=('lat', 'lon', 'time'))

        c3 = Coordinate(
            lat=(10, 20, 15),
            lon=(10, 20, 15),
            time='2018-01-01',
            order=('time', 'lat', 'lon'))

        c4 = Coordinate(
            lat_lon=((0, 10, 5), (0, 20, 5)),
            time='2018-01-01',
            order=('lat_lon', 'time'))

        c5 = Coordinate(
            lat=(0, 20, 15),
            lon=(0, 20, 15),
            order=('lat', 'lon'))

        c6 = Coordinate(
            lat=(10, 20, 15),
            lon=(10, 20, 15),
            order=('lat', 'lon'))

        # empty init
        g = CoordinateGroup()
        g = CoordinateGroup([])
        
        # basic init (with mismatched stacking, ordering, shapes)
        g = CoordinateGroup([c1])
        g = CoordinateGroup([c1, c2, c3, c4])
        g = CoordinateGroup([c5, c6])
        
        # list is required
        with pytest.raises(tl.TraitError):
            CoordinateGroup(c1)
        
        # Coord objects not valid
        with pytest.raises(tl.TraitError):
            CoordinateGroup([c1['lat']])

        # CoordinateGroup objects not valid (no nesting)
        g = CoordinateGroup([c1, c2])
        with pytest.raises(tl.TraitError):
            CoordinateGroup([g])

        # dimensions must match
        with pytest.raises(ValueError):
            CoordinateGroup([c1, c5])

    def test_len(self):
        c1 = Coordinate(
            lat=(0, 10, 5),
            lon=(0, 20, 5),
            time='2018-01-01',
            order=('lat', 'lon', 'time'))

        c2 = Coordinate(
            lat=(10, 20, 15),
            lon=(10, 20, 15),
            time='2018-01-01',
            order=('lat', 'lon', 'time'))

        g = CoordinateGroup()
        assert len(g) == 0
        
        g = CoordinateGroup([])
        assert len(g) == 0
        
        g = CoordinateGroup([c1])
        assert len(g) == 1
        
        g = CoordinateGroup([c1, c2])
        assert len(g) == 2

    def test_dims(self):
        c1 = Coordinate(
            lat=(0, 10, 5),
            lon=(0, 20, 5),
            time='2018-01-01',
            order=('lat', 'lon', 'time'))

        c2 = Coordinate(
            lat=(10, 20, 15),
            lon=(10, 20, 15),
            time='2018-01-01',
            order=('lat', 'lon', 'time'))

        c3 = Coordinate(
            lat=(10, 20, 15),
            lon=(10, 20, 15),
            time='2018-01-01',
            order=('time', 'lat', 'lon'))

        c4 = Coordinate(
            lat_lon=((0, 10, 5), (0, 20, 5)),
            time='2018-01-01',
            order=('lat_lon', 'time'))

        c5 = Coordinate(
            lat=(0, 20, 15),
            lon=(0, 20, 15),
            order=('lat', 'lon'))

        c6 = Coordinate(
            lat=(10, 20, 15),
            lon=(10, 20, 15),
            order=('lat', 'lon'))

        g = CoordinateGroup()
        assert len(g.dims) == 0

        g = CoordinateGroup([c1])
        assert g.dims == {'lat', 'lon', 'time'}

        g = CoordinateGroup([c1, c2, c3, c4])
        assert g.dims == {'lat', 'lon', 'time'}

        g = CoordinateGroup([c4])
        assert g.dims == {'lat', 'lon', 'time'}

        g = CoordinateGroup([c5, c6])
        assert g.dims == {'lat', 'lon'}

    def test_iter(self):
        pass

    def test_getitem(self):
        pass

    def test_intersect(self):
        pass

    def test_add(self):
        pass

    def test_iadd(self):
        pass

    def test_append(self):
        pass

    def test_stack(self):
        pass

    def test_unstack(self):
        pass

    def test_iterchunks(self):
        pass

