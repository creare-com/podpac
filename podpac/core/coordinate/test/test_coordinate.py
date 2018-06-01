
import sys
from collections import OrderedDict

import pytest
import traitlets as tl
import numpy as np
from six import string_types

from podpac.core.coordinate.coordinate import BaseCoordinate, Coordinate, CoordinateGroup, coordinate
from podpac.core.coordinate.coord import Coord, MonotonicCoord, UniformCoord, coord_linspace

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
    def _common_checks(self, coord, expected_dims, expected_shape, stacked):
        assert coord.dims == expected_dims
        assert coord.shape == expected_shape
        assert coord.is_stacked == stacked
        assert isinstance(coord.dims_map, dict)
        assert isinstance(coord.coords, OrderedDict)

        # additional properties
        assert isinstance(coord.latlon_bounds_str, string_types)
        assert isinstance(repr(coord), string_types)

    def _unstacked_checks(self, coord, expected_dims, expected_coords):
        for dim, c in zip(expected_dims, expected_coords):
            assert coord.dims_map[dim] == dim
            assert coord[dim] is c
            assert isinstance(coord.coords[dim], np.ndarray)
            if c.is_datetime:
                np.testing.assert_array_equal(coord.coords[dim], c.coordinates)
            else:
                np.testing.assert_allclose(coord.coords[dim], c.coordinates)

    def _stacked_checks(self, coord, key, expected_dims, expected_coords):
        for dim, c in zip(expected_dims, expected_coords):
            assert coord.dims_map[dim] == key
            coord[dim] is c
            assert isinstance(coord.coords[key][dim], np.ndarray)
            if c.is_datetime:
                np.testing.assert_array_equal(coord.coords[key][dim], c.coordinates)
            else:
                np.testing.assert_allclose(coord.coords[key][dim], c.coordinates)

    def test_coords_empty(self):
        coord = Coordinate()
        
        assert coord.dims == []
        assert coord.shape == ()
        assert coord.is_stacked == False
        assert isinstance(coord.dims_map, dict)

        self._common_checks(coord, [], (), False)
        assert len(coord.dims_map.keys()) == 0
        assert len(coord.coords.keys()) == 0

    def test_coords_single(self):
        lat = coord_linspace(0, 5, 10)

        d = OrderedDict()
        d['lat'] = lat
        coord = Coordinate(d)

        self._common_checks(coord, ['lat'], (10,), False)
        self._unstacked_checks(coord, ['lat'], [lat])

    def test_coords_unstacked(self):
        lat = coord_linspace(0, 5, 10)
        lon = coord_linspace(10, 20, 100)
        time = coord_linspace('2018-01-01', '2018-01-05', 5)

        d = OrderedDict()
        d['lat'] = lat
        d['lon'] = lon
        d['time'] = time
        coord = Coordinate(d)

        self._common_checks(coord, ['lat', 'lon', 'time'], (10, 100, 5), False)
        self._unstacked_checks(coord, ['lat', 'lon', 'time'], [lat, lon, time])

    def test_coords_stacked(self):
        lat = coord_linspace(0, 5, 5)
        lon = coord_linspace(10, 20, 5)
        time = coord_linspace('2018-01-01', '2018-01-05', 5)

        d = OrderedDict()
        d['lat_lon_time'] = (lat, lon, time)
        coord = Coordinate(d)

        self._common_checks(coord, ['lat_lon_time'], (5,), True)
        self._stacked_checks(coord, 'lat_lon_time', ['lat', 'lon', 'time'], [lat, lon, time])

    def test_coords_stacked_partial(self):
        lat = coord_linspace(0, 5, 15)
        lon = coord_linspace(10, 20, 15)
        time = coord_linspace('2018-01-01', '2018-01-05', 5)

        d = OrderedDict()
        d['lat_lon'] = (lat, lon)
        d['time'] = time
        coord = Coordinate(d)

        self._common_checks(coord, ['lat_lon', 'time'], (15, 5,), True)
        self._stacked_checks(coord, 'lat_lon', ['lat', 'lon'], [lat, lon])
        self._unstacked_checks(coord, ['time'], [time])

    @pytest.mark.skipif(sys.version >= '3.6', reason="Python <3.6 compatibility")
    def test_coords_invalid_ordereddict(self):
        # invalid type (must be OrderedDict)
        with pytest.raises(TypeError):
            Coordinate({})

    def test_coords_invalid(self):
        # invalid type (must be dict)
        with pytest.raises(TypeError):
            Coordinate([])

    def test_coords_stacked_invalid(self):
        lat = coord_linspace(0, 5, 5)
        lon = coord_linspace(10, 20, 5)
        time = coord_linspace('2018-01-01', '2018-01-05', 5)

        # invalid stacking
        d = OrderedDict()
        d['lat_lon'] = (lat, lon, time)
        with pytest.raises(ValueError):
            Coordinate(d)

        d = OrderedDict()
        d['lat_lon_time'] = (lat, lon)
        with pytest.raises(ValueError):
            Coordinate(d)

        # dimension repeated
        d = OrderedDict()
        d['lat_lon'] = (lat, lon)
        d['lat'] = lat
        with pytest.raises(ValueError):
            Coordinate(d)

        d = OrderedDict()
        d['lat_lat'] = (lat, lon)
        d['time'] = time
        with pytest.raises(ValueError):
            Coordinate(d)

        d = OrderedDict()
        d['lat_time'] = (lat, time)
        d['lon_time'] = (lon, time)
        with pytest.raises(ValueError):
            Coordinate(d)

    def test_coords_invalid_dim(self):
        lat = coord_linspace(0, 5, 5)
        
        # invalid dimension
        d = OrderedDict()
        d['latitude'] = lat
        with pytest.raises(ValueError):
            Coordinate(d)

    def test_coords_invalid_value(self):
        
        # invalid value, must be BaseCoord
        d = OrderedDict()
        d['lat'] = np.linspace(0, 5, 5)
        with pytest.raises(TypeError):
            Coordinate(d)
    
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

    def test_segment_position(self):
        # default
        coord = Coordinate()
        coord.segment_position == 0.5
        
        # init
        coord = Coordinate(segment_position=0.3)
        coord.segment_position == 0.3

        with pytest.raises(tl.TraitError):
            Coordinate(segment_position='abc')
        
    def test_coord_ref_sys(self):
        # default
        coord = Coordinate()
        assert coord.coord_ref_sys == 'WGS84'
        assert coord.gdal_crs == 'EPSG:4326'

        # init
        coord = Coordinate(coord_ref_sys='SPHER_MERC')
        assert coord.coord_ref_sys == 'SPHER_MERC'
        assert coord.gdal_crs == 'EPSG:3857'

    def test_kwargs(self):
        # default
        coord = Coordinate()
        assert isinstance(coord.kwargs, dict)
        assert coord.ctype == 'segment'
        assert coord.segment_position == 0.5
        assert coord.kwargs['coord_ref_sys'] == 'WGS84'
        
        # init
        coord = Coordinate(ctype='point', segment_position=0.3, coord_ref_sys='SPHER_MERC')
        assert isinstance(coord.kwargs, dict)
        assert coord.ctype == 'point'
        assert coord.segment_position == 0.3
        assert coord.kwargs['coord_ref_sys'] == 'SPHER_MERC'

    # def test_get_shape(self):
    #     pass

    # def test_intersect(self):
    #     pass

    # def test_drop_dims(self):
    #     pass

    # def test_transpose(self):
    #     pass

    # def test_iterchunks(self):
    #     pass

    # def test_add(self):
    #     pass

    # def test_add_unique(self):
    #     pass

# class TestCoordIntersection(object):
#     @pytest.mark.skip(reason="coordinate refactor")
#     def test_regular(self):
#         coord = Coord(coords=(1, 10, 10))
#         coord_left = Coord(coords=(-2, 7, 10))
#         coord_right = Coord(coords=(4, 13, 10))
#         coord_cent = Coord(coords=(4, 7, 4))
#         coord_cover = Coord(coords=(-2, 13, 15))
        
#         c = coord.intersect(coord).coordinates
#         np.testing.assert_allclose(c, coord.coordinates)
#         c = coord.intersect(coord_cover).coordinates
#         np.testing.assert_allclose(c, coord.coordinates)        
        
#         c = coord.intersect(coord_left).coordinates
#         np.testing.assert_allclose(c, coord.coordinates[:8])                
#         c = coord.intersect(coord_right).coordinates
#         np.testing.assert_allclose(c, coord.coordinates[2:])
#         c = coord.intersect(coord_cent).coordinates
#         np.testing.assert_allclose(c, coord.coordinates[2:8])

# @pytest.mark.skip(reason="new/experimental feature; spec uncertain")
# class TestCoordinateGroup(object):
#     def test_init(self):
#         c1 = Coordinate(
#             lat=(0, 10, 5),
#             lon=(0, 20, 5),
#             time='2018-01-01',
#             order=('lat', 'lon', 'time'))

#         c2 = Coordinate(
#             lat=(10, 20, 15),
#             lon=(10, 20, 15),
#             time='2018-01-01',
#             order=('lat', 'lon', 'time'))

#         c3 = Coordinate(
#             lat=(10, 20, 15),
#             lon=(10, 20, 15),
#             time='2018-01-01',
#             order=('time', 'lat', 'lon'))

#         c4 = Coordinate(
#             lat_lon=((0, 10), (0, 20), 5),
#             time='2018-01-01',
#             order=('lat_lon', 'time'))

#         c5 = Coordinate(
#             lat=(0, 20, 15),
#             lon=(0, 20, 15),
#             order=('lat', 'lon'))

#         c6 = Coordinate(
#             lat=(10, 20, 15),
#             lon=(10, 20, 15),
#             order=('lat', 'lon'))

#         # empty init
#         g = CoordinateGroup()
#         g = CoordinateGroup([])
        
#         # basic init (with mismatched stacking, ordering, shapes)
#         g = CoordinateGroup([c1])
#         g = CoordinateGroup([c1, c2, c3, c4])
#         g = CoordinateGroup([c5, c6])
        
#         # list is required
#         with pytest.raises(tl.TraitError):
#             CoordinateGroup(c1)
        
#         # Coord objects not valid
#         with pytest.raises(tl.TraitError):
#             CoordinateGroup([c1['lat']])

#         # CoordinateGroup objects not valid (no nesting)
#         g = CoordinateGroup([c1, c2])
#         with pytest.raises(tl.TraitError):
#             CoordinateGroup([g])

#         # dimensions must match
#         with pytest.raises(ValueError):
#             CoordinateGroup([c1, c5])

#     def test_len(self):
#         c1 = Coordinate(
#             lat=(0, 10, 5),
#             lon=(0, 20, 5),
#             time='2018-01-01',
#             order=('lat', 'lon', 'time'))

#         c2 = Coordinate(
#             lat=(10, 20, 15),
#             lon=(10, 20, 15),
#             time='2018-01-01',
#             order=('lat', 'lon', 'time'))

#         g = CoordinateGroup()
#         assert len(g) == 0
        
#         g = CoordinateGroup([])
#         assert len(g) == 0
        
#         g = CoordinateGroup([c1])
#         assert len(g) == 1
        
#         g = CoordinateGroup([c1, c2])
#         assert len(g) == 2

#     def test_dims(self):
#         c1 = Coordinate(
#             lat=(0, 10, 5),
#             lon=(0, 20, 5),
#             time='2018-01-01',
#             order=('lat', 'lon', 'time'))

#         c2 = Coordinate(
#             lat=(10, 20, 15),
#             lon=(10, 20, 15),
#             time='2018-01-01',
#             order=('lat', 'lon', 'time'))

#         c3 = Coordinate(
#             lat=(10, 20, 15),
#             lon=(10, 20, 15),
#             time='2018-01-01',
#             order=('time', 'lat', 'lon'))

#         c4 = Coordinate(
#             lat_lon=((0, 10), (0, 20), 5),
#             time='2018-01-01',
#             order=('lat_lon', 'time'))

#         c5 = Coordinate(
#             lat=(0, 20, 15),
#             lon=(0, 20, 15),
#             order=('lat', 'lon'))

#         c6 = Coordinate(
#             lat=(10, 20, 15),
#             lon=(10, 20, 15),
#             order=('lat', 'lon'))

#         g = CoordinateGroup()
#         assert len(g.dims) == 0

#         g = CoordinateGroup([c1])
#         assert g.dims == {'lat', 'lon', 'time'}

#         g = CoordinateGroup([c1, c2, c3, c4])
#         assert g.dims == {'lat', 'lon', 'time'}

#         g = CoordinateGroup([c4])
#         assert g.dims == {'lat', 'lon', 'time'}

#         g = CoordinateGroup([c5, c6])
#         assert g.dims == {'lat', 'lon'}

#     @pytest.mark.skip(reason="unwritten test")
#     def test_iter(self):
#         pass

#     @pytest.mark.skip(reason="unwritten test")
#     def test_getitem(self):
#         pass

#     @pytest.mark.skip(reason="unwritten test")
#     def test_intersect(self):
#         pass

#     @pytest.mark.skip(reason="unwritten test")
#     def test_add(self):
#         pass

#     @pytest.mark.skip(reason="unwritten test")
#     def test_iadd(self):
#         pass

#     @pytest.mark.skip(reason="unwritten test")
#     def test_append(self):
#         pass

#     @pytest.mark.skip(reason="unwritten test")
#     def test_stack(self):
#         pass

#     @pytest.mark.skip(reason="unwritten test")
#     def test_unstack(self):
#         pass

#     @pytest.mark.skip(reason="unwritten test")
#     def test_iterchunks(self):
#         pass

class TestCoordinateInitialization(object):
    def test_coord(self):
        coord = coordinate(lat=Coord([1., 2.]))

    def test_value(self):
        coord = coordinate(
            lat=1.,
            lon=10.,
            time='2018-01-02',
            order=['lat', 'lon', 'time'])

        assert coord.dims == ['lat', 'lon', 'time']
        assert coord.is_stacked == False
        assert isinstance(coord['lat'], Coord)
        assert isinstance(coord['lon'], Coord)
        assert isinstance(coord['time'], Coord)
        np.testing.assert_allclose(coord.coords['lat'], [1.])
        np.testing.assert_allclose(coord.coords['lon'], [10.])
        np.testing.assert_array_equal(coord.coords['time'], np.array(['2018-01-02']).astype('datetime64'))
    
    def test_array(self):
        coord = coordinate(
            lat=[1., 3., 2.],
            lon=[10., 30., 20.],
            time=['2018-01-01', '2018-01-03', '2018-01-02'],
            order=['lat', 'lon', 'time'])

        assert coord.dims == ['lat', 'lon', 'time']
        assert coord.is_stacked == False
        assert isinstance(coord['lat'], Coord)
        assert isinstance(coord['lon'], Coord)
        assert isinstance(coord['time'], Coord)
        np.testing.assert_allclose(coord.coords['lat'], [1., 3., 2.,])
        np.testing.assert_allclose(coord.coords['lon'], [10., 30., 20.])
        np.testing.assert_array_equal(
            coord.coords['time'],
            np.array(['2018-01-01', '2018-01-03', '2018-01-02']).astype('datetime64'))

    def test_monotonic(self):
        coord = coordinate(
            lat=[1., 2., 3.],
            lon=[10., 20., 30.],
            time=['2018-01-01', '2018-01-02', '2018-01-03'],
            order=['lat', 'lon', 'time'])

        assert coord.dims == ['lat', 'lon', 'time']
        assert coord.is_stacked == False
        assert isinstance(coord['lat'], MonotonicCoord)
        assert isinstance(coord['lon'], MonotonicCoord)
        assert isinstance(coord['time'], MonotonicCoord)
        np.testing.assert_allclose(coord.coords['lat'], [1., 2., 3.,])
        np.testing.assert_allclose(coord.coords['lon'], [10., 20., 30.])
        np.testing.assert_array_equal(
            coord.coords['time'],
            np.array(['2018-01-01', '2018-01-02', '2018-01-03']).astype('datetime64'))

    def test_uniform(self):
        coord = coordinate(
            lat=(1., 5., 0.5),
            lon=(10., 50., 0.5),
            time=('2018-01-01', '2018-01-05', '1,D'),
            order=['lat', 'lon', 'time'])

        assert coord.dims == ['lat', 'lon', 'time']
        assert coord.is_stacked == False
        assert isinstance(coord['lat'], UniformCoord)
        assert isinstance(coord['lon'], UniformCoord)
        assert isinstance(coord['time'], UniformCoord)
        np.testing.assert_allclose(coord.coords['lat'], np.arange(1., 5.1, 0.5))
        np.testing.assert_allclose(coord.coords['lon'], np.arange(10., 50.1, 0.5))
        np.testing.assert_array_equal(
            coord.coords['time'],
            np.arange(np.datetime64('2018-01-01'), np.datetime64('2018-01-06'), np.timedelta64(1, 'D')))

    def test_coord_linspace(self):
        coord = coordinate(
            lat=(1., 5., 10),
            lon=(10., 50., 20),
            time=('2018-01-01', '2018-01-05', 5),
            order=['lat', 'lon', 'time'])

        assert coord.dims == ['lat', 'lon', 'time']
        assert coord.is_stacked == False
        assert isinstance(coord['lat'], UniformCoord)
        assert isinstance(coord['lon'], UniformCoord)
        assert isinstance(coord['time'], UniformCoord)
        np.testing.assert_allclose(coord.coords['lat'], np.linspace(1., 5., 10))
        np.testing.assert_allclose(coord.coords['lon'], np.linspace(10., 50., 20))
        np.testing.assert_array_equal(
            coord.coords['time'],
            np.arange(np.datetime64('2018-01-01'), np.datetime64('2018-01-06'), np.timedelta64(1, 'D')))

    def test_stacked(self):
        coord = coordinate(
            lat_lon=([1, 2, 3], [10, 20, 30]),
            time='2018-01-01',
            order=['lat_lon', 'time'])

        assert coord.dims == ['lat_lon', 'time']
        assert coord.is_stacked == True
        assert isinstance(coord['lat'], Coord)
        assert isinstance(coord['lon'], Coord)
        assert isinstance(coord['time'], Coord)
        np.testing.assert_allclose(coord.coords['lat_lon']['lat'], [1., 2., 3.])
        np.testing.assert_allclose(coord.coords['lat_lon']['lon'], [10., 20., 30.])
        np.testing.assert_array_equal(coord.coords['time'], np.array(['2018-01-01']).astype('datetime64'))

        coord = coordinate(
            lat_lon_time=([1, 2, 3], [10, 20, 30], ['2018-01-01', '2018-01-02', '2018-01-03']))

        assert coord.dims == ['lat_lon_time']
        assert coord.is_stacked == True
        assert isinstance(coord['lat'], Coord)
        assert isinstance(coord['lon'], Coord)
        assert isinstance(coord['time'], Coord)
        np.testing.assert_allclose(coord.coords['lat_lon_time']['lat'], [1., 2., 3.])
        np.testing.assert_allclose(coord.coords['lat_lon_time']['lon'], [10., 20., 30.])
        np.testing.assert_array_equal(
            coord.coords['lat_lon_time']['time'],
            np.array(['2018-01-01', '2018-01-02', '2018-01-03']).astype('datetime64'))

        # mismatched size
        with pytest.raises(ValueError):
            coordinate(
                lat_lon=([1, 2, 3], [10, 20, 30, 40, 50]))

        # missing dim / extra value
        with pytest.raises(ValueError):
            coordinate(
                lat_lon=([1, 2, 3], [10, 20, 30], ['2018-01-01', '2018-01-02', '2018-01-03']))

        # extra dim / missing value
        with pytest.raises(ValueError):
            coordinate(
                lat_lon_time=([1, 2, 3], [10, 20, 30]))

    def test_stacked_linspace(self):
        coord = coordinate(
            lat_lon=((1., 5.), (10., 50.), 30),
            time='2018-01-01',
            order=['lat_lon', 'time'])

        assert coord.dims == ['lat_lon', 'time']
        assert coord.is_stacked == True
        assert isinstance(coord['lat'], UniformCoord)
        assert isinstance(coord['lon'], UniformCoord)
        assert isinstance(coord['time'], Coord)
        np.testing.assert_allclose(coord.coords['lat_lon']['lat'], np.linspace(1., 5., 30))
        np.testing.assert_allclose(coord.coords['lat_lon']['lon'], np.linspace(10., 50., 30.))
        np.testing.assert_array_equal(coord.coords['time'], np.array(['2018-01-01']).astype('datetime64'))

        coord = coordinate(
            lat_lon_time=((1., 5.), (10., 50.), ('2018-01-01', '2018-01-05'), 5))

        assert coord.dims == ['lat_lon_time']
        assert coord.is_stacked == True
        assert isinstance(coord['lat'], UniformCoord)
        assert isinstance(coord['lon'], UniformCoord)
        assert isinstance(coord['time'], UniformCoord)
        np.testing.assert_allclose(coord.coords['lat_lon_time']['lat'], np.linspace(1., 5., 5))
        np.testing.assert_allclose(coord.coords['lat_lon_time']['lon'], np.linspace(10., 50., 5))
        np.testing.assert_array_equal(
            coord.coords['lat_lon_time']['time'],
            np.arange(np.datetime64('2018-01-01'), np.datetime64('2018-01-06'), np.timedelta64(1, 'D')))

        # linspace requires integer
        with pytest.raises(TypeError):
            coordinate(
                lat_lon=((1., 5.), (10., 50.), 0.5))

        # missing dim / extra value
        with pytest.raises(ValueError):
            coordinate(
                lat_lon=((1., 5.), (10., 50.), ('2018-01-01', '2018-01-05'), 5))

        # extra dim / missing value
        with pytest.raises(ValueError):
            coord = coordinate(
                lat_lon_time=((1., 5.), (10., 50.), 5))

    def test_order(self):
        c = coordinate(lon=0.3, lat=0.25, order=['lat', 'lon'])
        assert c.dims == ['lat', 'lon']
        
        c = coordinate(lon=0.3, lat=0.25, order=['lon', 'lat'])
        assert c.dims == ['lon', 'lat']

        # extra or missing dimensions
        with pytest.raises(ValueError):
            coordinate(lon=0.3, lat=0.25, order=['lat', 'lon', 'time'])

        with pytest.raises(ValueError):
            coordinate(lon=0.3, lat=0.25, order=['lat'])
    
    @pytest.mark.skipif(sys.version >= '3.6', reason="Python <3.6 compatibility")
    def test_order_required(self):
        # not required
        coordinate(lat=0.25)

        # required
        with pytest.raises(TypeError):
            coordinate(lat=0.25, lon=0.3)

    @pytest.mark.skipif(sys.version < '3.6', reason="Python >=3.6 required")
    def test_order_detect(self):
        coord = coordinate(lat=0.25, lon=0.3)
        assert coord.dims == ['lat', 'lon']

        coord = coordinate(lon=0.3, lat=0.25)
        assert coord.dims == ['lon', 'lat']