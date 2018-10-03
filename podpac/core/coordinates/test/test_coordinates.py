
import sys
# from collections import OrderedDict

import pytest
import numpy as np
import xarray as xr
import pandas as pd
# from six import string_types

from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.uniform_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates
from podpac.core.coordinates.coordinates import Coordinates

# class TestBaseCoordinate(object):
#     def test_abstract_methods(self):
#         c = BaseCoordinate()

#         with pytest.raises(NotImplementedError):
#             c.stack([])

#         with pytest.raises(NotImplementedError):
#             c.unstack()

#         with pytest.raises(NotImplementedError):
#             c.intersect(c)

class TestCoordinateCreation(object):
    def test_empty(self):
        c = Coordinates()
        assert c.dims == tuple()
        assert c.shape == tuple()
        assert c.size == 0

    def test_single_dim(self):
        # single value
        date = '2018-01-01'

        c = Coordinates([date], dims=['time'])
        assert c.dims == ('time',)
        assert c.shape == (1,)

        # array
        dates = ['2018-01-01', '2018-01-02']

        c = Coordinates([dates], dims=['time'])
        assert c.dims == ('time',)
        assert c.shape == (2,)

        c = Coordinates([np.array(dates).astype(np.datetime64)], dims=['time'])
        assert c.dims == ('time',)
        assert c.shape == (2,)

        c = Coordinates([xr.DataArray(dates).astype(np.datetime64)], dims=['time'])
        assert c.dims == ('time',)
        assert c.shape == (2,)
        
        # use DataArray name, but dims overrides the DataArray name
        c = Coordinates([xr.DataArray(dates, name='time').astype(np.datetime64)])
        assert c.dims == ('time',)
        assert c.shape == (2,)

        c = Coordinates([xr.DataArray(dates, name='a').astype(np.datetime64)], dims=['time'])
        assert c.dims == ('time',)
        assert c.shape == (2,)

    def test_unstacked(self):
        # single value
        c = Coordinates([0, 10], dims=['lat', 'lon'])
        assert c.dims == ('lat', 'lon')
        assert c.shape == (1, 1)

        # arrays
        lat = [0, 1, 2]
        lon = [10, 20, 30, 40]

        c = Coordinates([lat, lon], dims=['lat', 'lon'])
        assert c.dims == ('lat', 'lon')
        assert c.shape == (3, 4)

        # use DataArray names
        c = Coordinates([xr.DataArray(lat, name='lat'), xr.DataArray(lon, name='lon')])
        assert c.dims == ('lat', 'lon')
        assert c.shape == (3, 4)

        # dims overrides the DataArray names
        c = Coordinates([xr.DataArray(lat, name='a'), xr.DataArray(lon, name='b')], dims=['lat', 'lon'])
        assert c.dims == ('lat', 'lon')
        assert c.shape == (3, 4)

    def test_stacked(self):
        # single value
        c = Coordinates([[0, 10]], dims=['lat_lon'])
        assert c.dims == ('lat_lon',)
        assert c.shape == (1,)

        # arrays
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        c = Coordinates([[lat, lon]], dims=['lat_lon'])
        assert c.dims == ('lat_lon',)
        assert c.shape == (3,)

        # TODO lat_lon MultiIndex

    def test_mixed(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ['2018-01-01', '2018-01-02']

        c = Coordinates([[lat, lon], dates], dims=['lat_lon', 'time'])
        assert c.dims == ('lat_lon', 'time')
        assert c.shape == (3, 2)

    def test_invalid_dims(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ['2018-01-01', '2018-01-02']

        with pytest.raises(TypeError):
            Coordinates([dates], dims='time')

        with pytest.raises(ValueError):
            Coordinates(dates, dims=['time'])

        with pytest.raises(ValueError):
            Coordinates([[lat, lon]], dims=['lat'])

        with pytest.raises(ValueError):
            Coordinates([lat, lon, dates], dims=['lat_lon', 'time'])

        with pytest.raises(ValueError):
            Coordinates([[lat, lon], dates], dims=['lat', 'lon', 'dates'])

        with pytest.raises(ValueError):
            Coordinates([lat, lon], dims=['lat_lon'])

        with pytest.raises(ValueError):
            Coordinates([[lat, lon]], dims=['lat', 'lon'])

        with pytest.raises(ValueError):
            Coordinates([lat, lon], dims=['lat_lon'])

    def test_Coordinates1d(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ['2018-01-01', '2018-01-02']

        c = Coordinates([
            StackedCoordinates([
                ArrayCoordinates1d(lat, name='lat'),
                ArrayCoordinates1d(lon, name='lon')]),
            ArrayCoordinates1d(dates, name='time')])
        assert c.dims == ('lat_lon', 'time')
        assert c.shape == (3, 2)

        # TODO default and overridden properties

    def test_grid(self):
        # array
        lat = [0, 1, 2]
        lon = [10, 20, 30, 40]
        dates = ['2018-01-01', '2018-01-02']
            
        c = Coordinates.grid(lat=lat, lon=lon, time=dates, order=['time', 'lat', 'lon'])
        assert c.dims == ('time', 'lat', 'lon')
        assert c.shape == (2, 3, 4)

        # size
        lat = (0, 1, 3)
        lon = (10, 40, 4)
        dates = ('2018-01-01', '2018-01-05', 5)

        c = Coordinates.grid(lat=lat, lon=lon, time=dates, order=['time', 'lat', 'lon'])
        assert c.dims == ('time', 'lat', 'lon')
        assert c.shape == (5, 3, 4)

        # step
        lat = (0, 1, 0.5)
        lon = (10, 40, 10.0)
        dates = ('2018-01-01', '2018-01-05', '1,D')
        
        c = Coordinates.grid(lat=lat, lon=lon, time=dates, order=['time', 'lat', 'lon'])
        assert c.dims == ('time', 'lat', 'lon')
        assert c.shape == (5, 3, 4)

    def test_points(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ['2018-01-01', '2018-01-02', '2018-01-03']

        c = Coordinates.points(lat=lat, lon=lon, time=dates, order=['time', 'lat', 'lon'])
        assert c.dims == ('time_lat_lon',)
        assert c.shape == (3,)

        # TODO
        # with pytest.raises(ValueError):
        #     Coordinates.points(lat=lat, lon=lon, time=dates[:2], order=['time', 'lat', 'lon'])

    def test_grid_points_order(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30, 40]
        dates = ['2018-01-01', '2018-01-02']

        with pytest.raises(ValueError):
            Coordinates.grid(lat=lat, lon=lon, time=dates, order=['lat', 'lon'])

        with pytest.raises(ValueError):
            Coordinates.grid(lat=lat, lon=lon, order=['lat', 'lon', 'time'])

        if sys.version < '3.6':
            with pytest.raises(TypeError):
                Coordinates.grid(lat=lat, lon=lon, time=dates)
        else:
            Coordinates.grid(lat=lat, lon=lon, time=dates)

class TestCoordinateXarray(object):
    def test_to_xarray(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ['2018-01-01', '2018-01-02']

        c = Coordinates([
            StackedCoordinates([
                ArrayCoordinates1d(lat, name='lat'),
                ArrayCoordinates1d(lon, name='lon')]),
            ArrayCoordinates1d(dates, name='time')])
        
        assert isinstance(c.coords, xr.core.coordinates.DataArrayCoordinates)
        assert c.coords.dims == ('lat_lon', 'time')
        np.testing.assert_equal(c.coords['lat'].data, np.array(lat, dtype=float))
        np.testing.assert_equal(c.coords['lon'].data, np.array(lon, dtype=float))
        np.testing.assert_equal(c.coords['time'].data, np.array(dates).astype(np.datetime64))

    def test_from_xarray(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ['2018-01-01', '2018-01-02']

        c = Coordinates([
            StackedCoordinates([
                ArrayCoordinates1d(lat, name='lat'),
                ArrayCoordinates1d(lon, name='lon')]),
            ArrayCoordinates1d(dates, name='time')])

        # from xarray
        c2 = Coordinates.from_xarray(c.coords)
        assert c2.dims == c.dims
        assert c2.shape == c.shape
        assert isinstance(c2['lat_lon'], StackedCoordinates)
        assert isinstance(c2['time'], Coordinates1d)
        np.testing.assert_equal(c2.coords['lat'].data, np.array(lat, dtype=float))
        np.testing.assert_equal(c2.coords['lon'].data, np.array(lon, dtype=float))
        np.testing.assert_equal(c2.coords['time'].data, np.array(dates).astype(np.datetime64))

# class TestCoordinate(object):
#     @pytest.mark.skipif(sys.version >= '3.6', reason="Python <3.6 compatibility")
#     def test_order(self):
#         # required
#         with pytest.raises(TypeError):
#             Coordinate(lat=0.25, lon=0.3)

#         # not required
#         Coordinate(lat=0.25)
        
#         # not required
#         Coordinate(coords=OrderedDict(lat=0.25, lon=0.3))

#         # invalid
#         with pytest.raises(ValueError):

#         # invalid
#         with pytest.raises(ValueError):
#             Coordinate(lon=0.3, lat=0.25, order=['lat'])

#         # valid
#         c = Coordinate(lon=0.3, lat=0.25, order=['lat', 'lon'])
#         assert c.dims == ['lat', 'lon']
        
#         c = Coordinate(lon=0.3, lat=0.25, order=['lon', 'lat'])
#         assert c.dims == ['lon', 'lat']

#     @pytest.mark.skipif(sys.version < '3.6', reason="Python >=3.6 required")
#     def test_order_detect(self):
#         coord = Coordinate(lat=0.25, lon=0.3)
#         assert coord.dims == ['lat', 'lon']

#         coord = Coordinate(lon=0.3, lat=0.25)
#         assert coord.dims == ['lon', 'lat']

#         # in fact, ignore order
#         coord = Coordinate(lon=0.3, lat=0.25, order=['lat', 'lon'])
#         assert coord.dims == ['lon', 'lat']

#     def _common_checks(self, coord, expected_dims, expected_shape, stacked):
#         # note: check stacked dims_map manually

#         assert coord.dims == expected_dims
#         assert coord.shape == expected_shape
#         assert coord.is_stacked == stacked
        
#         # dims map and coords
#         assert isinstance(coord.dims_map, dict)
#         assert isinstance(coord.coords, OrderedDict)
#         for dim in expected_dims:
#             if not coord.is_stacked:
#                 assert coord.dims_map[dim] == dim
#             assert isinstance(coord.coords[dim], np.ndarray)

#         # additional properties
#         assert isinstance(coord.kwargs, dict)
#         assert isinstance(coord.latlon_bounds_str, string_types)

#     def test_coords_empty(self):
#         coord = Coordinate()
        
#         self._common_checks(coord, [], (), False)
#         assert len(coord.dims_map.keys()) == 0
#         assert len(coord.coords.keys()) == 0

#     def test_coords_single_latlon(self):
#         coord = Coordinate(lat=0.25, lon=0.3, order=['lat', 'lon'])
        
#         self._common_checks(coord, ['lat', 'lon'], (1, 1), False)
#         np.testing.assert_allclose(coord.coords['lat'], [0.25])
#         np.testing.assert_allclose(coord.coords['lon'], [0.3])

#     def test_coords_single_datetime(self):
#         coord = Coordinate(time='2018-01-01')

#         self._common_checks(coord, ['time'], (1,), False)
#         np.testing.assert_array_equal(coord.coords['time'], np.datetime64('2018-01-01'))

#     def test_coords_single_time_dependent(self):
#         coord = Coordinate(lat=0.25, lon=0.3, time='2018-01-01', order=['lat', 'lon', 'time'])
        
#         self._common_checks(coord, ['lat', 'lon', 'time'], (1, 1, 1), False)
#         np.testing.assert_allclose(coord.coords['lat'], [0.25])
#         np.testing.assert_allclose(coord.coords['lon'], [0.3])
#         np.testing.assert_array_equal(coord.coords['time'], np.datetime64('2018-01-01'))

#     def test_coords_single_latlon_stacked(self):
#         coord = Coordinate(lat_lon=[0.25, 0.3])
        
#         self._common_checks(coord, ['lat_lon'], (1,), True)
#         assert coord.dims_map['lat'] == 'lat_lon'
#         assert coord.dims_map['lon'] == 'lat_lon'
#         np.testing.assert_allclose(coord.coords['lat_lon']['lat'], [0.25])
#         np.testing.assert_allclose(coord.coords['lat_lon']['lon'], [0.3])

#     def test_coords_single_time_dependent_stacked(self):
#         coord = Coordinate(lat_lon=[0.25, 0.3], time='2018-01-01', order=['lat_lon', 'time'])
        
#         self._common_checks(coord, ['lat_lon', 'time'], (1, 1), True)
#         assert coord.dims_map['lat'] == 'lat_lon'
#         assert coord.dims_map['lon'] == 'lat_lon'
#         assert coord.dims_map['time'] == 'time'
#         np.testing.assert_allclose(coord.coords['lat_lon']['lat'], [0.25])
#         np.testing.assert_allclose(coord.coords['lat_lon']['lon'], [0.3])
#         np.testing.assert_array_equal(coord.coords['time'], np.datetime64('2018-01-01'))

#     def test_coords_latlon_coord(self):
#         coord = Coordinate(lat=[0.2, 0.4, 0.5], lon=[0.3, -0.1], order=['lat', 'lon'])
        
#         self._common_checks(coord, ['lat', 'lon'], (3, 2), False)
#         np.testing.assert_allclose(coord.coords['lat'], [0.2, 0.4, 0.5])
#         np.testing.assert_allclose(coord.coords['lon'], [0.3, -0.1])

#     def test_coords_latlon_coord_stacked(self):
#         coord = Coordinate(lat_lon=[[0.2, 0.4, 0.5], [0.3, -0.1, 0.2]])
        
#         self._common_checks(coord, ['lat_lon'], (3,), True)
#         assert coord.dims_map['lat'] == 'lat_lon'
#         assert coord.dims_map['lon'] == 'lat_lon'
#         np.testing.assert_allclose(coord.coords['lat_lon']['lat'], [0.2, 0.4, 0.5])
#         np.testing.assert_allclose(coord.coords['lat_lon']['lon'], [0.3, -0.1, 0.2])

#         # length must match
#         with pytest.raises(ValueError):
#             Coordinate(lat_lon=[[0.2, 0.4, 0.5], [0.3, -0.1]])

#     def test_coords_datetime_coord(self):
#         coord = Coordinate(time=['2018-01-01', '2018-01-02', '2018-02-01'])

#         self._common_checks(coord, ['time'], (3,), False)
#         np.testing.assert_array_equal(
#             coord.coords['time'],
#             np.array(['2018-01-01', '2018-01-02', '2018-02-01']).astype(np.datetime64))

#     def test_coords_time_dependent_coord_stacked(self):
#         coord = Coordinate(
#             lat_lon=[[0.2, 0.4, 0.5], [0.3, -0.1, 0.2]],
#             time=['2018-01-01', '2018-01-02', '2018-02-01'],
#             order=['lat_lon', 'time'])
        
#         self._common_checks(coord, ['lat_lon', 'time'], (3, 3), True)
#         assert coord.dims_map['lat'] == 'lat_lon'
#         assert coord.dims_map['lon'] == 'lat_lon'
#         assert coord.dims_map['time'] == 'time'
#         np.testing.assert_allclose(coord.coords['lat_lon']['lat'], [0.2, 0.4, 0.5])
#         np.testing.assert_allclose(coord.coords['lat_lon']['lon'], [0.3, -0.1, 0.2])
#         np.testing.assert_array_equal(
#             coord.coords['time'],
#             np.array(['2018-01-01', '2018-01-02', '2018-02-01']).astype(np.datetime64))

#     def test_coords_latlon_uniform_num(self):
#         coord = Coordinate(lat=(0.1, 0.4, 4), lon=(-0.3, -0.1, 3), order=['lat', 'lon'])
        
#         self._common_checks(coord, ['lat', 'lon'], (4, 3), False)
#         np.testing.assert_allclose(coord.coords['lat'], [0.1, 0.2, 0.3, 0.4])
#         np.testing.assert_allclose(coord.coords['lon'], [-0.3, -0.2, -0.1])
        
#     def test_coords_latlon_uniform_step(self):
#         coord = Coordinate(lat=(0.1, 0.4, 0.1), lon=(-0.3, -0.1, 0.1), order=['lat', 'lon'])
        
#         self._common_checks(coord, ['lat', 'lon'], (4, 3), False)
#         np.testing.assert_allclose(coord.coords['lat'], [0.1, 0.2, 0.3, 0.4])
#         np.testing.assert_allclose(coord.coords['lon'], [-0.3, -0.2, -0.1])

#     def test_coords_latlon_uniform_stacked(self):
#         coord = Coordinate(lat_lon=[(0.1, 0.4), (-0.3, -0.0), 4])
        
#         self._common_checks(coord, ['lat_lon'], (4,), True)
#         assert coord.dims_map['lat'] == 'lat_lon'
#         assert coord.dims_map['lon'] == 'lat_lon'
#         np.testing.assert_allclose(coord.coords['lat_lon']['lat'], [0.1, 0.2, 0.3, 0.4])
#         np.testing.assert_allclose(coord.coords['lat_lon']['lon'], [-0.3, -0.2, -0.1, 0.0])

#         # step not allowed
#         with pytest.raises(TypeError):
#             Coordinate(lat_lon=[(0.1, 0.4), (-0.3, -0.0), 0.1])

#         # timedelta not allowed
#         with pytest.raises(TypeError):
#             Coordinate(lat_lon=[(0.1, 0.4), (-0.3, -0.0), np.timedelta64(1, 'D')])

#         # size required
#         with pytest.raises(ValueError):
#             Coordinate(lat_lon=[(0.1, 0.4), (-0.3, -0.0)])

#     def test_coords_datetime_uniform_num(self):
#         coord = Coordinate(time=('2018-01-01', '2018-01-03', 3))

#         self._common_checks(coord, ['time'], (3,), False)
#         np.testing.assert_array_equal(
#             coord.coords['time'],
#             np.array(['2018-01-01', '2018-01-02', '2018-01-03']).astype(np.datetime64))

#     def test_coords_datetime_uniform_step(self):
#         coord = Coordinate(time=('2018-01-01', '2018-01-03', '1,D'))

#         self._common_checks(coord, ['time'], (3,), False)
#         np.testing.assert_array_equal(
#             coord.coords['time'],
#             np.array(['2018-01-01', '2018-01-02', '2018-01-03']).astype(np.datetime64))

#     def test_coords_explicit_coords(self):
#         c1 = Coordinate(lat=[0.25, 0.35], lon=[0.3, 0.4], order=['lat', 'lon'])
        
#         coords = OrderedDict()
#         coords['lat'] = [0.25, 0.35]
#         coords['lon'] = Coord([0.3, 0.4])
#         c2 = Coordinate(coords=coords)
        
#         assert c1.dims == c2.dims
#         np.testing.assert_allclose(c1.coords['lat'], c2.coords['lat'])
#         np.testing.assert_allclose(c1.coords['lon'], c2.coords['lon'])
        
#         with pytest.raises(TypeError):
#             Coordinate(coords=[0.25])

#         with pytest.raises(TypeError):
#             Coordinate(coords={'lat': 0.25, 'lon': 0.3})

#         # invalid value
#         # TODO how to trigger the TypeError in _validate_val

#     def test_coords_explicit_coord(self):
#         coord = Coordinate(lat=Coord([0.2, 0.4, 0.5]), lon=[0.3, -0.1], order=['lat', 'lon'])
        
#         self._common_checks(coord, ['lat', 'lon'], (3, 2), False)
#         np.testing.assert_allclose(coord.coords['lat'], [0.2, 0.4, 0.5])
#         np.testing.assert_allclose(coord.coords['lon'], [0.3, -0.1])

#     def test_coords_explicit_coord_stacked(self):
#         coord = Coordinate(lat_lon=[Coord([0.2, 0.4, 0.5]), Coord([0.3, -0.1, 0.2])])
        
#         self._common_checks(coord, ['lat_lon'], (3,), True)
#         assert coord.dims_map['lat'] == 'lat_lon'
#         assert coord.dims_map['lon'] == 'lat_lon'
#         np.testing.assert_allclose(coord.coords['lat_lon']['lat'], [0.2, 0.4, 0.5])
#         np.testing.assert_allclose(coord.coords['lat_lon']['lon'], [0.3, -0.1, 0.2])

#         # length must match
#         with pytest.raises(ValueError):
#             coord = Coordinate(lat_lon=[Coord([0.2, 0.4, 0.5]), Coord([0.3, -0.1])])

#     def test_coords_invalid_dims(self):
#         # invalid dimension
#         with pytest.raises(ValueError):
#             coord = Coordinate(abc=[0.2, 0.4, 0.5])

#         # repeated dimension
#         # TODO how to trigger the repeated dim ValueError in _validate_dim
        
#     @pytest.mark.skip(reason="unsupported (deprecated or future feature)")
#     def test_unstacked_dependent(self):
#         coord = Coordinate(
#             lat=xr.DataArray(
#                 np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0], 
#                 dims=['lat', 'lon']),
#             lon=xr.DataArray(
#                 np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0], 
#                 dims=['lat', 'lon']),
#             order=['lat', 'lon'])
#         np.testing.assert_allclose(np.array(coord.intersect(coord)._coords['lat'].bounds),
#                                           np.array(coord._coords['lat'].bounds))     
        
#     @pytest.mark.skip(reason="unsupported (deprecated or future feature)")
#     def test_stacked_dependent(self):
#         coord = Coordinate(
#             lat=[
#                 xr.DataArray(
#                          np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0],
#                          dims=['lat-lon', 'time']), 
#                 xr.DataArray(
#                     np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[1],
#                              dims=['lat-lon', 'time'])        
#                 ], 
#             lon=[
#                 xr.DataArray(
#                     np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0],
#                     dims=['lat-lon', 'time']), 
#                 xr.DataArray(
#                     np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[1],
#                     dims=['lat-lon', 'time']),
                
#                 ], 
#             order=['lat', 'lon'])
#         np.testing.assert_allclose(np.array(coord.intersect(coord)._coords['lat'].bounds),
#                                           np.array(coord._coords['lat'].bounds))        
#         coord = Coordinate(
#             lat=xr.DataArray(np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5)),
#                              dims=['stack', 'lat-lon', 'time']), 
#             lon=xr.DataArray(np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5)),
#                          dims=['stack', 'lat-lon', 'time']), 
#             order=['lat', 'lon']
#         )
#         np.testing.assert_allclose(np.array(coord.intersect(coord)._coords['lat'].bounds),
#                                           np.array(coord._coords['lat'].bounds))

#     def test_ctype(self):
#         # default
#         coord = Coordinate()
#         coord.ctype == 'segment'
        
#         # init
#         coord = Coordinate(ctype='segment')
#         coord.ctype == 'segment'

#         coord = Coordinate(ctype='point')
#         coord.ctype == 'point'

#         with pytest.raises(tl.TraitError):
#             Coordinate(ctype='abc')

#         # propagation
#         coord = Coordinate(lat=[0.2, 0.4])
#         coord._coords['lat'].ctype == 'segment'

#         coord = Coordinate(lat=[0.2, 0.4], ctype='segment')
#         coord._coords['lat'].ctype == 'segment'

#         coord = Coordinate(lat=[0.2, 0.4], ctype='point')
#         coord._coords['lat'].ctype == 'point'

#     def test_segment_position(self):
#         # default
#         coord = Coordinate()
#         coord.segment_position == 0.5
        
#         # init
#         coord = Coordinate(segment_position=0.3)
#         coord.segment_position == 0.3

#         with pytest.raises(tl.TraitError):
#             Coordinate(segment_position='abc')

#         # propagation
#         coord = Coordinate(lat=[0.2, 0.4])
#         coord._coords['lat'].segment_position == 0.5

#         coord = Coordinate(lat=[0.2, 0.4], segment_position=0.3)
#         coord._coords['lat'].segment_position == 0.3
        
#     def test_coord_ref_sys(self):
#         # default
#         coord = Coordinate()
#         assert coord.coord_ref_sys == 'WGS84'
#         assert coord.gdal_crs == 'EPSG:4326'

#         # init
#         coord = Coordinate(coord_ref_sys='SPHER_MERC')
#         assert coord.coord_ref_sys == 'SPHER_MERC'
#         assert coord.gdal_crs == 'EPSG:3857'

#         # propagation
#         coord = Coordinate(lat=[0.2, 0.4])
#         coord._coords['lat'].coord_ref_sys == 'WGS84'

#         coord = Coordinate(lat=[0.2, 0.4], coord_ref_sys='SPHER_MERC')
#         coord._coords['lat'].coord_ref_sys == 'SPHER_MERC'

#     def test_dims_map(self):
#         coord = Coordinate(lat=[0.2, 0.4, 0.5], lon=[0.3, -0.1], order=['lat', 'lon'])
#         coord.get_dims_map()

#     def test_stack_dict(self):
#         coord = Coordinate(
#             lat_lon=[[0.2, 0.4, 0.5], [0.3, -0.1, 0.5]],
#             time=['2018-01-01', '2018-01-02'],
#             order=['lat_lon', 'time'])
#         d = coord.stack_dict()

#         assert isinstance(d, OrderedDict)
#         assert isinstance(d['lat_lon'], list)
#         assert len(d['lat_lon']) == 2
#         assert isinstance(d['lat_lon'][0], BaseCoord)
#         assert isinstance(d['lat_lon'][1], BaseCoord)
#         assert isinstance(d['time'], BaseCoord)

#     def test_stack(self):
#         coord = Coordinate(
#             lat=[0.2, 0.4, 0.5],
#             lon=[0.3, -0.1, 0.5],
#             time=['2018-01-01', '2018-01-02'],
#             order=['lat', 'lon', 'time'])
        
#         stacked = coord.stack(['lat', 'lon'], copy=True)
#         assert isinstance(stacked, Coordinate)
#         assert coord.dims == ['lat', 'lon', 'time']
#         # assert stacked.dims == ['lat_lon', 'time'] # TODO python 3.5 doesn't preserve order, bug?)

#         coord.stack(['lat', 'lon'], copy=False)
#         # assert coord.dims == ['lat_lon', 'time'] # TODO python 3.5 doesn't preserve order, bug?)

#     def test_unstack(self):
#         coord = Coordinate(
#             lat_lon=[[0.2, 0.4, 0.5], [0.3, -0.1, 0.5]],
#             time=['2018-01-01', '2018-01-02'],
#             order=['lat_lon', 'time'])
        
#         unstacked = coord.unstack(copy=True)
#         assert isinstance(unstacked, Coordinate)
#         assert coord.dims == ['lat_lon', 'time']
#         # assert unstacked.dims == ['lat', 'lon', 'time'] # TODO python 3.5 doesn't preserve order, bug?)

#         coord.unstack(copy=False)
#         # assert coord.dims == ['lat', 'lon', 'time'] # TODO python 3.5 doesn't preserve order, bug?)

#     def test_delta(self):
#         coord = Coordinate(lat=[0.2, 0.4, 0.5], lon=[0.3, -0.1], order=['lat', 'lon'])
#         coord.delta

#     def test_intersect(self):
#         coord1 = Coordinate(
#             lat=[0.2, 0.4, 0.5],
#             lon=[0.3, -0.1, 0.5],
#             time=['2018-01-01', '2018-01-02'],
#             order=['lat', 'lon', 'time'])

#         coord2 = Coordinate(
#             lat_lon=[(0.2, 0.5), (0.2, 0.5), 10],
#             time=['2018-01-01', '2018-01-02'],
#             order=['lat_lon', 'time'])

#         coord3 = Coordinate(
#             lat_lon=[(0.3, 0.6), (0.1, 0.4), 5],
#             order=['lat_lon'])

#         coord1.intersect(coord2)
#         coord1.intersect(coord2, ind=True)
        
#         coord1.intersect(coord3)
#         coord1.intersect(coord3, ind=True)
        
#         coord2.intersect(coord1)
#         coord2.intersect(coord1, ind=True)

#     def test_replace_coords(self):
#         coord = Coordinate(
#             lat=[0.2, 0.4, 0.5],
#             lon=[0.3, -0.1, 0.5],
#             time=['2018-01-01', '2018-01-02'],
#             order=['lat', 'lon', 'time'])

#         other = Coordinate(
#             lat=[1, 2, 3],
#             lon=[0.5],
#             order=['lat', 'lon'])

#         replaced = coord.replace_coords(other, copy=True)
#         coord.replace_coords(other, copy=False)

#         # TODO check coordinates

#     def test_drop_dims(self):
#         coord = Coordinate(
#             lat=[0.2, 0.4, 0.5],
#             lon=[0.3, -0.1, 0.5],
#             time=['2018-01-01', '2018-01-02'],
#             order=['lat', 'lon', 'time'])

#         coord.drop_dims('time', 'alt')
#         assert coord.dims == ['lat', 'lon']

#         # TODO this isn't working
#         # coord = Coordinate(
#         #     lat_lon=([0.2, 0.4, 0.5], [0.3, -0.1, 0.5]),
#         #     time=['2018-01-01', '2018-01-02'],
#         #     order=['lat_lon', 'time'])

#         # coord.drop_dims('lat', 'lon')
#         # assert coord.dims == ['time']

#     def test_get_shape(self):
#         coord1 = Coordinate(
#             lat=[0.2, 0.4, 0.5],
#             lon=[0.3, -0.1, 0.5],
#             order=['lat', 'lon'])

#         coord2 = Coordinate(
#             lat=[0.2, 0.4],
#             lon=[0.3, -0.1],
#             time=['2018-01-01', '2018-01-02'],
#             order=['lat', 'lon', 'time'])

#         coord3 = Coordinate(
#             lat_lon=([0.2, 0.4], [0.3, -0.1]),
#             time=['2018-01-01', '2018-01-02'],
#             order=['lat_lon', 'time'])

#         assert coord1.get_shape() == (3, 3)
#         assert coord2.get_shape() == (2, 2, 2)
#         assert coord3.get_shape() == (2, 2)

#         assert coord1.get_shape(coord2) == (2, 2)
#         assert coord1.get_shape(coord3) == (2,)

#         assert coord2.get_shape(coord1) == (3, 3, 2)
#         assert coord2.get_shape(coord3) == (2, 2)

#         assert coord3.get_shape(coord1) == (3, 3, 2)
#         assert coord3.get_shape(coord2) == (2, 2, 2)

#     def test_add(self):
#         coord1 = Coordinate(
#             lat=[0.2, 0.4, 0.5],
#             lon=[0.3, -0.1, 0.5],
#             order=['lat', 'lon'])

#         coord2 = Coordinate(
#             lat=[0.2, 0.3],
#             lon=[0.3, 0.0],
#             time=['2018-01-01', '2018-01-02'],
#             order=['lat', 'lon', 'time'])

#         coord3 = Coordinate(
#             lat_lon=([0.2, 0.3], [0.3, 0.0]),
#             order=['lat_lon'])

#         coord = coord1 + coord2
#         assert coord.shape == (5, 5, 2)

#         # TODO not working?
#         # coord = coord1.add_unique(coord2)
#         # assert coord.shape == (4, 4, 2)

#         with pytest.raises(TypeError):
#             coord1 + [1, 2]

#         with pytest.raises(ValueError):
#             coord1 + coord3

#     def test_iterchunks(self):
#         coord = Coordinate(
#             lat=(0, 1, 100),
#             lon=(0, 1, 200),
#             time=['2018-01-01', '2018-01-02'],
#             order=['lat', 'lon', 'time'])
        
#         for chunk in coord.iterchunks(shape=(10, 10, 10)):
#             assert chunk.shape == (10, 10, 2)

#         for chunk, slices in coord.iterchunks(shape=(10, 10, 10), return_slices=True):
#             assert isinstance(slices, tuple)
#             assert len(slices) == 3
#             assert isinstance(slices[0], slice)
#             assert isinstance(slices[1], slice)
#             assert isinstance(slices[2], slice)
#             assert chunk.shape == (10, 10, 2)

#     def test_transpose(self):
#         coord = Coordinate(
#             lat=[0.2, 0.4],
#             lon=[0.3, -0.1],
#             time=['2018-01-01', '2018-01-02'],
#             order=['lat', 'lon', 'time'])

#         transposed = coord.transpose('lon', 'lat', 'time', inplace=False)
#         assert coord.dims == ['lat', 'lon', 'time']
#         assert transposed.dims == ['lon', 'lat', 'time']

#         transposed = coord.transpose(inplace=False)
#         assert coord.dims == ['lat', 'lon', 'time']
#         assert transposed.dims == ['time', 'lon', 'lat']

#         transposed = coord.transpose('lon', 'lat', 'time')
#         assert coord.dims == ['lat', 'lon', 'time']
#         assert transposed.dims == ['lon', 'lat', 'time']

#         # TODO not working
#         # coord.transpose('lon', 'lat', 'time', inplace=True)
#         # assert coord.dims == ['lon', 'lat', 'time']

#         # TODO check not implemented yet
#         # with pytest.raises(ValueError):
#         #     coord.transpose('lon', 'lat')

#         # TODO check not implemented yet
#         # with pytest.raises(ValueError):
#         #     coord.transpose('lon', 'lat', inplace=True)

#     @pytest.mark.skip(reason='errors')
#     def test_leftovers(self):
#         from podpac.core.coordinate import coord_linspace
#         coord = coord_linspace(1, 10, 10)
#         coord_cent = coord_linspace(4, 7, 4)
        
#         c = Coordinate(lat=coord, lon=coord, order=('lat', 'lon'))
#         c_s = Coordinate(lat_lon=(coord, coord))
#         c_cent = Coordinate(lat=coord_cent, lon=coord_cent, order=('lat', 'lon'))
#         c_cent_s = Coordinate(lon_lat=(coord_cent, coord_cent))

#         print(c.intersect(c_cent))
#         print(c.intersect(c_cent_s))
#         print(c_s.intersect(c_cent))
#         print(c_s.intersect(c_cent_s))
        
#         try:
#             c = Coordinate(lat_lon=((0, 1, 10), (0, 1, 11)))
#         except ValueError as e:
#             print(e)
#         else:
#             raise Exception('expceted exception')
        
#         c = Coordinate(lat_lon=((0, 1, 10), (0, 1, 10)), time=(0, 1, 2), order=('lat_lon', 'time'))
#         c2 = Coordinate(lat_lon=((0.5, 1.5, 15), (0.1, 1.1, 15)))
#         print (c.shape)
#         print (c.unstack().shape)
#         print (c.get_shape(c2))
#         print (c.get_shape(c2.unstack()))
#         print (c.unstack().get_shape(c2))
#         print (c.unstack().get_shape(c2.unstack()))
        
#         c = Coordinate(lat=(0, 1, 10), lon=(0, 1, 10), time=(0, 1, 2), order=('lat', 'lon', 'time'))
#         print(c.stack(['lat', 'lon']))
#         try:
#             c.stack(['lat','time'])
#         except Exception as e:
#             print(e)
#         else:
#             raise Exception('expected exception')

#         try:
#             c.stack(['lat','time'], copy=False)
#         except Exception as e:
#             print(e)
#         else:
#             raise Exception('expected exception')

#         c = Coordinate(lat_lon=((0, 1, 10), (0, 1, 10)), time=(0, 1, 2), order=('lat_lon', 'time'))
#         c2 = Coordinate(lat_lon=((0.5, 1.5, 15), (0.1, 1.1, 15)))

#         print (c.replace_coords(c2))
#         print (c.replace_coords(c2.unstack()))
#         print (c.unstack().replace_coords(c2))
#         print (c.unstack().replace_coords(c2.unstack()))  
        
#         c = UniformCoord(1, 10, 2)
#         np.testing.assert_equal(c.coordinates, np.arange(1., 10, 2))
        
#         c = UniformCoord(10, 1, -2)
#         np.testing.assert_equal(c.coordinates, np.arange(10., 1, -2))    

#         try:
#             c = UniformCoord(10, 1, 2)
#             raise Exception
#         except ValueError as e:
#             print(e)
        
#         try:
#             c = UniformCoord(1, 10, -2)
#             raise Exception
#         except ValueError as e:
#             print(e)
        
#         c = UniformCoord('2015-01-01', '2015-01-04', '1,D')
#         c2 = UniformCoord('2015-01-01', '2015-01-04', '2,D')
        
#         print('Done')

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

#     def test_iter(self):
#         pass

#     def test_getitem(self):
#         pass

#     def test_intersect(self):
#         pass

#     def test_add(self):
#         pass

#     def test_iadd(self):
#         pass

#     def test_append(self):
#         pass

#     def test_stack(self):
#         pass

#     def test_unstack(self):
#         pass

#     def test_iterchunks(self):
#         pass

# def test_convert_xarray_to_podpac():
#     from podpac.core.algorithm.algorithm import Arange
#     node = Arange()

#     coords = Coordinate(lat=[3, 4], lon=[10, 30], order=['lat', 'lon'])
#     output = node.eval(coords)
#     outcoords = convert_xarray_to_podpac(output.coords)
#     assert outcoords.shape == coords.shape
#     assert outcoords.dims == coords.dims

#     coords = Coordinate(lat_lon=[[3, 4], [10, 30]], order=['lat_lon'])
#     output = node.eval(coords)
#     outcoords = convert_xarray_to_podpac(output.coords)
#     assert outcoords.shape == coords.shape
#     assert outcoords.dims == coords.dims

#     with pytest.raises(TypeError):
#         convert_xarray_to_podpac(output)