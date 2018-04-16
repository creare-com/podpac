
import pytest
import traitlets as tl
import numpy as np

from podpac.core.coordinate import Coordinate, CoordinateGroup

class TestCoordinate(object):
    pass

class TestCoordIntersection(object):
    @pytest.mark.skip(reason="coordinate refactor")
    def test_regular(self):
        coord = Coord(coords=(1, 10, 10))
        coord_left = Coord(coords=(-2, 7, 10))
        coord_right = Coord(coords=(4, 13, 10))
        coord_cent = Coord(coords=(4, 7, 4))
        coord_cover = Coord(coords=(-2, 13, 15))
        
        c = coord.intersect(coord).coordinates
        np.testing.assert_array_equal(c, coord.coordinates)
        c = coord.intersect(coord_cover).coordinates
        np.testing.assert_array_equal(c, coord.coordinates)        
        
        c = coord.intersect(coord_left).coordinates
        np.testing.assert_array_equal(c, coord.coordinates[:8])                
        c = coord.intersect(coord_right).coordinates
        np.testing.assert_array_equal(c, coord.coordinates[2:])
        c = coord.intersect(coord_cent).coordinates
        np.testing.assert_array_equal(c, coord.coordinates[2:8])
        
class TestCoordinateCreation(object):
    @pytest.mark.skip(reason="coordinate refactor")
    def test_single_coord(self):
        coord = Coordinate(lat=0.25, lon=0.3, 
                           order=['lat', 'lon'])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
    
    @pytest.mark.skip(reason="coordinate refactor")
    def test_single_stacked_coord(self):
        coord = Coordinate(lat=[(0.25, 0.5, 1.2)], lon=[(0.25, 0.5, 1.2)],
                           order=['lat', 'lon'])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
    
    @pytest.mark.skip(reason="coordinate refactor")
    def test_unstacked_regular(self):
        coord = Coordinate(lat=(0, 1, 4), lon=(0, 1, 4), 
                           order=['lat', 'lon'])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        coord = Coordinate(lat=[0, 1, 4], lon=[0, 1, 4], 
                           order=['lat', 'lon'])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        coord = Coordinate(lat=(0, 1, 1/4), lon=(0, 1, 1/4), 
                           order=['lat', 'lon'])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        coord = Coordinate(lat=[0, 1, 1/4], lon=[0, 1, 1/4], 
                           order=['lat', 'lon'])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        
    @pytest.mark.skip(reason="coordinate refactor")
    def test_unstacked_irregular(self):
        coord = Coordinate(lat=np.linspace(0, 1, 4), lon=np.linspace(0, 1, 4),
                           order=['lat', 'lon'])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        
    @pytest.mark.skip(reason="coordinate refactor")
    def test_unstacked_dependent(self):
        coord = Coordinate(
            lat=xr.DataArray(
                np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0], 
                dims=['lat', 'lon']),
            lon=xr.DataArray(
                np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0], 
                dims=['lat', 'lon']),
            order=['lat', 'lon'])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        
    @pytest.mark.skip(reason="coordinate refactor")
    def test_stacked_regular(self):
        coord = Coordinate(lat=((0, 0), (1, -1), 4), lon=((0, 0), (1, -1), 4),
                           order=['lat', 'lon'])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        coord = Coordinate(lat=[(0, 0), (1, -1), 4], lon=[(0, 0), (1, -1), 4],
                           order=['lat', 'lon'])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        coord = Coordinate(lat=((0, 0), (1, -1), 1/4), lon=((0, 0), (1, -1), 1/4),
                           order=['lat', 'lon'])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        coord = Coordinate(lat=[(0, 0), (1, -1), 1/4], lon=[(0, 0), (1, -1), 1/4],
                           order=['lat', 'lon'])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        
    @pytest.mark.skip(reason="coordinate refactor")
    def test_stacked_irregular(self):
        coord = Coordinate(lat=np.column_stack((np.linspace(0, 1, 4),
                                              np.linspace(0, -1, 4))),
                           lon=np.column_stack((np.linspace(0, 1, 4),
                                              np.linspace(0, -1, 4))),
                           order=['lat', 'lon'])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        
    @pytest.mark.skip(reason="coordinate refactor")
    def test_stacked_dependent(self):
        coord = Coordinate(
            lat=[
                xr.DataArray(
                         np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0],
                         dims=['lat-lon', 'time']), 
                xr.DataArray(
                    np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[1],
                             dims=['lat-lon', 'time'])        
                ], 
            lon=[
                xr.DataArray(
                    np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0],
                    dims=['lat-lon', 'time']), 
                xr.DataArray(
                    np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[1],
                    dims=['lat-lon', 'time']),
                
                ], 
            order=['lat', 'lon'])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        coord = Coordinate(
            lat=xr.DataArray(np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5)),
                             dims=['stack', 'lat-lon', 'time']), 
            lon=xr.DataArray(np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5)),
                         dims=['stack', 'lat-lon', 'time']), 
            order=['lat', 'lon']
        )
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))

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

