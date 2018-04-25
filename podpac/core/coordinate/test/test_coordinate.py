
import sys
from collections import OrderedDict

import pytest
import traitlets as tl
import numpy as np
from six import string_types

from podpac.core.coordinate import Coordinate, CoordinateGroup
from podpac.core.coordinate import BaseCoordinate, Coord

def allclose_structured(a, b):
    return all(np.allclose(a[name], b) for name in a.dtype.names)

class TestBaseCoordinate(object):
    def test_abstract_methods(self):
        c = BaseCoordinate()

        with pytest.raises(NotImplementedError):
            c.stack([])

        with pytest.raises(NotImplementedError):
            c.unstack()

        with pytest.raises(NotImplementedError):
            c.intersect(c)

class TestCoordinate(object):
    @pytest.mark.skipif(sys.version >= '3.6', reason="Python <3.6 compatibility")
    def test_order(self):
        # required
        with pytest.raises(TypeError):
            Coordinate(lat=0.25, lon=0.3)

        # not required
        Coordinate(lat=0.25)
        
        # not required
        Coordinate(coords=OrderedDict(lat=0.25, lon=0.3))

        # invalid
        with pytest.raises(ValueError):
            Coordinate(lon=0.3, lat=0.25, order=['lat', 'lon', 'time'])

        # invalid
        with pytest.raises(ValueError):
            Coordinate(lon=0.3, lat=0.25, order=['lat'])

        # valid
        c = Coordinate(lon=0.3, lat=0.25, order=['lat', 'lon'])
        assert c.dims == ['lat', 'lon']
        
        c = Coordinate(lon=0.3, lat=0.25, order=['lon', 'lat'])
        assert c.dims == ['lon', 'lat']

    @pytest.mark.skipif(sys.version < '3.6', reason="Python >=3.6 required")
    def test_order_detect(self):
        coord = Coordinate(lat=0.25, lon=0.3)
        assert coord.dims == ['lat', 'lon']

        coord = Coordinate(lon=0.3, lat=0.25)
        assert coord.dims == ['lon', 'lat']

        # in fact, ignore order
        coord = Coordinate(lon=0.3, lat=0.25, order=['lat', 'lon'])
        assert coord.dims == ['lon', 'lat']

    def _common_checks(self, coord, expected_dims, expected_shape, stacked):
        # note: check stacked dims_map manually

        assert coord.dims == expected_dims
        assert coord.shape == expected_shape
        assert coord.is_stacked == stacked
        
        # dims map and coords
        assert isinstance(coord.dims_map, dict)
        assert isinstance(coord.coords, OrderedDict)
        for dim in expected_dims:
            if not coord.is_stacked:
                assert coord.dims_map[dim] == dim
            assert isinstance(coord.coords[dim], np.ndarray)

        # additional properties
        assert isinstance(coord.kwargs, dict)
        assert isinstance(coord.latlon_bounds_str, string_types)

    def test_coords_empty(self):
        coord = Coordinate()
        
        self._common_checks(coord, [], (), False)
        assert len(coord.dims_map.keys()) == 0
        assert len(coord.coords.keys()) == 0

    def test_coords_single_latlon(self):
        coord = Coordinate(lat=0.25, lon=0.3, order=['lat', 'lon'])
        
        self._common_checks(coord, ['lat', 'lon'], (1, 1), False)
        np.testing.assert_allclose(coord.coords['lat'], [0.25])
        np.testing.assert_allclose(coord.coords['lon'], [0.3])

    def test_coords_single_datetime(self):
        coord = Coordinate(time='2018-01-01')

        self._common_checks(coord, ['time'], (1,), False)
        np.testing.assert_array_equal(coord.coords['time'], np.datetime64('2018-01-01'))

    def test_coords_single_time_dependent(self):
        coord = Coordinate(lat=0.25, lon=0.3, time='2018-01-01', order=['lat', 'lon', 'time'])
        
        self._common_checks(coord, ['lat', 'lon', 'time'], (1, 1, 1), False)
        np.testing.assert_allclose(coord.coords['lat'], [0.25])
        np.testing.assert_allclose(coord.coords['lon'], [0.3])
        np.testing.assert_array_equal(coord.coords['time'], np.datetime64('2018-01-01'))

    def test_coords_single_latlon_stacked(self):
        coord = Coordinate(lat_lon=[0.25, 0.3])
        
        self._common_checks(coord, ['lat_lon'], (1,), True)
        assert coord.dims_map['lat'] == 'lat_lon'
        assert coord.dims_map['lon'] == 'lat_lon'
        np.testing.assert_allclose(coord.coords['lat_lon']['lat'], [0.25])
        np.testing.assert_allclose(coord.coords['lat_lon']['lon'], [0.3])

    def test_coords_single_time_dependent_stacked(self):
        coord = Coordinate(lat_lon=[0.25, 0.3], time='2018-01-01', order=['lat_lon', 'time'])
        
        self._common_checks(coord, ['lat_lon', 'time'], (1, 1), True)
        assert coord.dims_map['lat'] == 'lat_lon'
        assert coord.dims_map['lon'] == 'lat_lon'
        assert coord.dims_map['time'] == 'time'
        np.testing.assert_allclose(coord.coords['lat_lon']['lat'], [0.25])
        np.testing.assert_allclose(coord.coords['lat_lon']['lon'], [0.3])
        np.testing.assert_array_equal(coord.coords['time'], np.datetime64('2018-01-01'))

    def test_coords_latlon_coord(self):
        coord = Coordinate(lat=[0.2, 0.4, 0.5], lon=[0.3, -0.1], order=['lat', 'lon'])
        
        self._common_checks(coord, ['lat', 'lon'], (3, 2), False)
        np.testing.assert_allclose(coord.coords['lat'], [0.2, 0.4, 0.5])
        np.testing.assert_allclose(coord.coords['lon'], [0.3, -0.1])

    def test_coords_latlon_coord_stacked(self):
        coord = Coordinate(lat_lon=[[0.2, 0.4, 0.5], [0.3, -0.1, 0.2]])
        
        self._common_checks(coord, ['lat_lon'], (3,), True)
        assert coord.dims_map['lat'] == 'lat_lon'
        assert coord.dims_map['lon'] == 'lat_lon'
        np.testing.assert_allclose(coord.coords['lat_lon']['lat'], [0.2, 0.4, 0.5])
        np.testing.assert_allclose(coord.coords['lat_lon']['lon'], [0.3, -0.1, 0.2])

        # length must match
        with pytest.raises(ValueError):
            Coordinate(lat_lon=[[0.2, 0.4, 0.5], [0.3, -0.1]])

    def test_coords_datetime_coord(self):
        coord = Coordinate(time=['2018-01-01', '2018-01-02', '2018-02-01'])

        self._common_checks(coord, ['time'], (3,), False)
        np.testing.assert_array_equal(
            coord.coords['time'],
            np.array(['2018-01-01', '2018-01-02', '2018-02-01']).astype(np.datetime64))

    def test_coords_time_dependent_coord_stacked(self):
        coord = Coordinate(
            lat_lon=[[0.2, 0.4, 0.5], [0.3, -0.1, 0.2]],
            time=['2018-01-01', '2018-01-02', '2018-02-01'],
            order=['lat_lon', 'time'])
        
        self._common_checks(coord, ['lat_lon', 'time'], (3, 3), True)
        assert coord.dims_map['lat'] == 'lat_lon'
        assert coord.dims_map['lon'] == 'lat_lon'
        assert coord.dims_map['time'] == 'time'
        np.testing.assert_allclose(coord.coords['lat_lon']['lat'], [0.2, 0.4, 0.5])
        np.testing.assert_allclose(coord.coords['lat_lon']['lon'], [0.3, -0.1, 0.2])
        np.testing.assert_array_equal(
            coord.coords['time'],
            np.array(['2018-01-01', '2018-01-02', '2018-02-01']).astype(np.datetime64))

    def test_coords_latlon_uniform_num(self):
        coord = Coordinate(lat=(0.1, 0.4, 4), lon=(-0.3, -0.1, 3), order=['lat', 'lon'])
        
        self._common_checks(coord, ['lat', 'lon'], (4, 3), False)
        np.testing.assert_allclose(coord.coords['lat'], [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(coord.coords['lon'], [-0.3, -0.2, -0.1])
        
    def test_coords_latlon_uniform_step(self):
        coord = Coordinate(lat=(0.1, 0.4, 0.1), lon=(-0.3, -0.1, 0.1), order=['lat', 'lon'])
        
        self._common_checks(coord, ['lat', 'lon'], (4, 3), False)
        np.testing.assert_allclose(coord.coords['lat'], [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(coord.coords['lon'], [-0.3, -0.2, -0.1])

    def test_coords_latlon_uniform_stacked(self):
        coord = Coordinate(lat_lon=[(0.1, 0.4), (-0.3, -0.0), 4])
        
        self._common_checks(coord, ['lat_lon'], (4,), True)
        assert coord.dims_map['lat'] == 'lat_lon'
        assert coord.dims_map['lon'] == 'lat_lon'
        np.testing.assert_allclose(coord.coords['lat_lon']['lat'], [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(coord.coords['lat_lon']['lon'], [-0.3, -0.2, -0.1, 0.0])

        # step not allowed
        with pytest.raises(TypeError):
            Coordinate(lat_lon=[(0.1, 0.4), (-0.3, -0.0), 0.1])

        # timedelta not allowed
        with pytest.raises(TypeError):
            Coordinate(lat_lon=[(0.1, 0.4), (-0.3, -0.0), np.timedelta64(1, 'D')])

        # size required
        with pytest.raises(ValueError):
            Coordinate(lat_lon=[(0.1, 0.4), (-0.3, -0.0)])

    def test_coords_datetime_uniform_num(self):
        coord = Coordinate(time=('2018-01-01', '2018-01-03', 3))

        self._common_checks(coord, ['time'], (3,), False)
        np.testing.assert_array_equal(
            coord.coords['time'],
            np.array(['2018-01-01', '2018-01-02', '2018-01-03']).astype(np.datetime64))

    def test_coords_datetime_uniform_step(self):
        coord = Coordinate(time=('2018-01-01', '2018-01-03', '1,D'))

        self._common_checks(coord, ['time'], (3,), False)
        np.testing.assert_array_equal(
            coord.coords['time'],
            np.array(['2018-01-01', '2018-01-02', '2018-01-03']).astype(np.datetime64))

    def test_coords_explicit_coords(self):
        c1 = Coordinate(lat=[0.25, 0.35], lon=[0.3, 0.4], order=['lat', 'lon'])
        
        coords = OrderedDict()
        coords['lat'] = [0.25, 0.35]
        coords['lon'] = Coord([0.3, 0.4])
        c2 = Coordinate(coords=coords)
        
        assert c1.dims == c2.dims
        np.testing.assert_allclose(c1.coords['lat'], c2.coords['lat'])
        np.testing.assert_allclose(c1.coords['lon'], c2.coords['lon'])
        
        with pytest.raises(TypeError):
            Coordinate(coords=[0.25])

        with pytest.raises(TypeError):
            Coordinate(coords={'lat': 0.25, 'lon': 0.3})

        # invalid value
        # TODO how to trigger the TypeError in _validate_val

    def test_coords_explicit_coord(self):
        coord = Coordinate(lat=Coord([0.2, 0.4, 0.5]), lon=[0.3, -0.1], order=['lat', 'lon'])
        
        self._common_checks(coord, ['lat', 'lon'], (3, 2), False)
        np.testing.assert_allclose(coord.coords['lat'], [0.2, 0.4, 0.5])
        np.testing.assert_allclose(coord.coords['lon'], [0.3, -0.1])

    def test_coords_explicit_coord_stacked(self):
        coord = Coordinate(lat_lon=[Coord([0.2, 0.4, 0.5]), Coord([0.3, -0.1, 0.2])])
        
        self._common_checks(coord, ['lat_lon'], (3,), True)
        assert coord.dims_map['lat'] == 'lat_lon'
        assert coord.dims_map['lon'] == 'lat_lon'
        np.testing.assert_allclose(coord.coords['lat_lon']['lat'], [0.2, 0.4, 0.5])
        np.testing.assert_allclose(coord.coords['lat_lon']['lon'], [0.3, -0.1, 0.2])

        # length must match
        with pytest.raises(ValueError):
            coord = Coordinate(lat_lon=[Coord([0.2, 0.4, 0.5]), Coord([0.3, -0.1])])

    def test_coords_invalid_dims(self):
        # invalid dimension
        with pytest.raises(ValueError):
            coord = Coordinate(abc=[0.2, 0.4, 0.5])

        # repeated dimension
        # TODO how to trigger the repeated dim ValueError in _validate_dim
        
    @pytest.mark.skip(reason="unsupported (deprecated or future feature)")
    def test_unstacked_dependent(self):
        coord = Coordinate(
            lat=xr.DataArray(
                np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0], 
                dims=['lat', 'lon']),
            lon=xr.DataArray(
                np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0], 
                dims=['lat', 'lon']),
            order=['lat', 'lon'])
        np.testing.assert_allclose(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))     
        
    @pytest.mark.skip(reason="unsupported (deprecated or future feature)")
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
        np.testing.assert_allclose(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        coord = Coordinate(
            lat=xr.DataArray(np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5)),
                             dims=['stack', 'lat-lon', 'time']), 
            lon=xr.DataArray(np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5)),
                         dims=['stack', 'lat-lon', 'time']), 
            order=['lat', 'lon']
        )
        np.testing.assert_allclose(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))

    def test_ctype(self):
        # default
        coord = Coordinate()
        coord.ctype == 'segment'
        
        # init
        coord = Coordinate(ctype='segment')
        coord.ctype == 'segment'

        coord = Coordinate(ctype='point')
        coord.ctype == 'point'

        with pytest.raises(tl.TraitError):
            Coordinate(ctype='abc')

        # propagation
        coord = Coordinate(lat=[0.2, 0.4])
        coord._coords['lat'].ctype == 'segment'

        coord = Coordinate(lat=[0.2, 0.4], ctype='segment')
        coord._coords['lat'].ctype == 'segment'

        coord = Coordinate(lat=[0.2, 0.4], ctype='point')
        coord._coords['lat'].ctype == 'point'

    def test_segment_position(self):
        # default
        coord = Coordinate()
        coord.segment_position == 0.5
        
        # init
        coord = Coordinate(segment_position=0.3)
        coord.segment_position == 0.3

        with pytest.raises(tl.TraitError):
            Coordinate(segment_position='abc')

        # propagation
        coord = Coordinate(lat=[0.2, 0.4])
        coord._coords['lat'].segment_position == 0.5

        coord = Coordinate(lat=[0.2, 0.4], segment_position=0.3)
        coord._coords['lat'].segment_position == 0.3
        
    def test_coord_ref_sys(self):
        # default
        coord = Coordinate()
        assert coord.coord_ref_sys == 'WGS84'
        assert coord.gdal_crs == 'EPSG:4326'

        # init
        coord = Coordinate(coord_ref_sys='SPHER_MERC')
        assert coord.coord_ref_sys == 'SPHER_MERC'
        assert coord.gdal_crs == 'EPSG:3857'

        # propagation
        coord = Coordinate(lat=[0.2, 0.4])
        coord._coords['lat'].coord_ref_sys == 'WGS84'

        coord = Coordinate(lat=[0.2, 0.4], coord_ref_sys='SPHER_MERC')
        coord._coords['lat'].coord_ref_sys == 'SPHER_MERC'

    def test_get_shape(self):
        pass

    def test_intersect(self):
        pass

    def test_drop_dims(self):
        pass

    def test_transpose(self):
        pass

    def test_iterchunks(self):
        pass

    def test_add(self):
        pass

    def test_add_unique(self):
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
        np.testing.assert_allclose(c, coord.coordinates)
        c = coord.intersect(coord_cover).coordinates
        np.testing.assert_allclose(c, coord.coordinates)        
        
        c = coord.intersect(coord_left).coordinates
        np.testing.assert_allclose(c, coord.coordinates[:8])                
        c = coord.intersect(coord_right).coordinates
        np.testing.assert_allclose(c, coord.coordinates[2:])
        c = coord.intersect(coord_cent).coordinates
        np.testing.assert_allclose(c, coord.coordinates[2:8])

@pytest.mark.skip(reason="new/experimental feature; spec uncertain")
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
            lat_lon=((0, 10), (0, 20), 5),
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
            lat_lon=((0, 10), (0, 20), 5),
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

    @pytest.mark.skip(reason="unwritten test")
    def test_iter(self):
        pass

    @pytest.mark.skip(reason="unwritten test")
    def test_getitem(self):
        pass

    @pytest.mark.skip(reason="unwritten test")
    def test_intersect(self):
        pass

    @pytest.mark.skip(reason="unwritten test")
    def test_add(self):
        pass

    @pytest.mark.skip(reason="unwritten test")
    def test_iadd(self):
        pass

    @pytest.mark.skip(reason="unwritten test")
    def test_append(self):
        pass

    @pytest.mark.skip(reason="unwritten test")
    def test_stack(self):
        pass

    @pytest.mark.skip(reason="unwritten test")
    def test_unstack(self):
        pass

    @pytest.mark.skip(reason="unwritten test")
    def test_iterchunks(self):
        pass

