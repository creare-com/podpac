
import sys
from copy import deepcopy
import json

import pytest
import numpy as np
import xarray as xr
import pandas as pd
from numpy.testing import assert_equal
import pyproj

import podpac
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates
from podpac.core.coordinates.cfunctions import crange, clinspace
from podpac.core.coordinates.coordinates import Coordinates
from podpac.core.coordinates.coordinates import concat, union, merge_dims
from podpac.core.settings import settings

class TestCoordinateCreation(object):
    def test_empty(self):
        c = Coordinates([])
        assert c.dims == tuple()
        assert c.udims == tuple()
        assert c.shape == tuple()
        assert c.ndim == 0
        assert c.size == 0

    def test_single_dim(self):
        # single value
        date = '2018-01-01'

        c = Coordinates([date], dims=['time'])
        assert c.dims == ('time',)
        assert c.udims == ('time',)
        assert c.shape == (1,)
        assert c.ndim == 1
        assert c.size == 1

        # array
        dates = ['2018-01-01', '2018-01-02']

        c = Coordinates([dates], dims=['time'])
        assert c.udims == ('time',)
        assert c.dims == ('time',)
        assert c.shape == (2,)
        assert c.ndim == 1
        assert c.size == 2

        c = Coordinates([np.array(dates).astype(np.datetime64)], dims=['time'])
        assert c.dims == ('time',)
        assert c.udims == ('time',)
        assert c.shape == (2,)
        assert c.ndim == 1

        c = Coordinates([xr.DataArray(dates).astype(np.datetime64)], dims=['time'])
        assert c.dims == ('time',)
        assert c.udims == ('time',)
        assert c.shape == (2,)
        assert c.ndim == 1
        assert c.size == 2
        
        # use DataArray name, but dims overrides the DataArray name
        c = Coordinates([xr.DataArray(dates, name='time').astype(np.datetime64)])
        assert c.dims == ('time',)
        assert c.udims == ('time',)
        assert c.shape == (2,)
        assert c.ndim == 1
        assert c.size == 2

        c = Coordinates([xr.DataArray(dates, name='a').astype(np.datetime64)], dims=['time'])
        assert c.dims == ('time',)
        assert c.udims == ('time',)
        assert c.shape == (2,)
        assert c.ndim == 1
        assert c.size == 2

    def test_unstacked(self):
        # single value
        c = Coordinates([0, 10], dims=['lat', 'lon'])
        assert c.dims == ('lat', 'lon')
        assert c.udims == ('lat', 'lon',)
        assert c.shape == (1, 1)
        assert c.ndim == 2
        assert c.size == 1

        # arrays
        lat = [0, 1, 2]
        lon = [10, 20, 30, 40]

        c = Coordinates([lat, lon], dims=['lat', 'lon'])
        assert c.dims == ('lat', 'lon')
        assert c.udims == ('lat', 'lon',)
        assert c.shape == (3, 4)
        assert c.ndim == 2
        assert c.size == 12

        # use DataArray names
        c = Coordinates([xr.DataArray(lat, name='lat'), xr.DataArray(lon, name='lon')])
        assert c.dims == ('lat', 'lon')
        assert c.udims == ('lat', 'lon',)
        assert c.shape == (3, 4)
        assert c.ndim == 2
        assert c.size == 12

        # dims overrides the DataArray names
        c = Coordinates([xr.DataArray(lat, name='a'), xr.DataArray(lon, name='b')], dims=['lat', 'lon'])
        assert c.dims == ('lat', 'lon')
        assert c.udims == ('lat', 'lon',)
        assert c.shape == (3, 4)
        assert c.ndim == 2
        assert c.size == 12

    def test_stacked(self):
        # single value
        c = Coordinates([[0, 10]], dims=['lat_lon'])
        assert c.dims == ('lat_lon',)
        assert c.udims == ('lat', 'lon',)
        assert c.shape == (1,)
        assert c.ndim == 1
        assert c.size == 1

        # arrays
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        c = Coordinates([[lat, lon]], dims=['lat_lon'])
        assert c.dims == ('lat_lon',)
        assert c.udims == ('lat', 'lon',)
        assert c.shape == (3,)
        assert c.ndim == 1
        assert c.size == 3

        # TODO lat_lon MultiIndex

    def test_mixed(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ['2018-01-01', '2018-01-02']

        c = Coordinates([[lat, lon], dates], dims=['lat_lon', 'time'])
        assert c.dims == ('lat_lon', 'time')
        assert c.udims == ('lat', 'lon', 'time')
        assert c.shape == (3, 2)
        assert c.ndim == 2
        assert c.size == 6

    def test_invalid_dims(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ['2018-01-01', '2018-01-02']

        with pytest.raises(TypeError, match="Invalid dims type"):
            Coordinates([dates], dims='time')

        with pytest.raises(ValueError, match="coords and dims size mismatch"):
            Coordinates(dates, dims=['time'])

        with pytest.raises(ValueError, match="coords and dims size mismatch"):
            Coordinates([lat, lon, dates], dims=['lat_lon', 'time'])
        
        with pytest.raises(ValueError, match="coords and dims size mismatch"):
            Coordinates([[lat, lon], dates], dims=['lat', 'lon', 'dates'])

        with pytest.raises(ValueError, match="coords and dims size mismatch"):
            Coordinates([lat, lon], dims=['lat_lon'])
        
        with pytest.raises(ValueError, match="coords and dims size mismatch"):
            Coordinates([[lat, lon]], dims=['lat', 'lon'])
        
        with pytest.raises(ValueError, match="coords and dims size mismatch"):
            Coordinates([lat, lon], dims=['lat_lon'])
        
        with pytest.raises(ValueError, match="Invalid coordinate values"):
            Coordinates([[lat, lon]], dims=['lat'])

        with pytest.raises(TypeError, match="Cannot get dim for coordinates at position"):
            # this doesn't work because lat and lon are not named BaseCoordinates/xarray objects
            Coordinates([lat, lon])

        with pytest.raises(ValueError, match="Duplicate dimension name"):
            Coordinates([lat, lon], dims=['lat', 'lat'])

        with pytest.raises(ValueError, match="Duplicate dimension name"):
            Coordinates([[lat, lon], lon], dims=['lat_lon', 'lat'])

    def test_dims_mismatch(self):
        c1d = ArrayCoordinates1d([0, 1, 2], name='lat')

        with pytest.raises(ValueError, match="Dimension name mismatch"):
            Coordinates([c1d], dims=['lon'])

    def test_invalid_coords(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ['2018-01-01', '2018-01-02']

        with pytest.raises(TypeError, match="Invalid coords"):
            Coordinates({'lat': lat, 'lon': lon})

    def test_base_coordinates(self):
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
            
        c = Coordinates.grid(lat=lat, lon=lon, time=dates, dims=['time', 'lat', 'lon'])
        assert c.dims == ('time', 'lat', 'lon')
        assert c.udims == ('time', 'lat', 'lon')
        assert c.shape == (2, 3, 4)
        assert c.ndim == 3
        assert c.size == 24

        # size
        lat = (0, 1, 3)
        lon = (10, 40, 4)
        dates = ('2018-01-01', '2018-01-05', 5)

        c = Coordinates.grid(lat=lat, lon=lon, time=dates, dims=['time', 'lat', 'lon'])
        assert c.dims == ('time', 'lat', 'lon')
        assert c.udims == ('time', 'lat', 'lon')
        assert c.shape == (5, 3, 4)
        assert c.ndim == 3
        assert c.size == 60

        # step
        lat = (0, 1, 0.5)
        lon = (10, 40, 10.0)
        dates = ('2018-01-01', '2018-01-05', '1,D')
        
        c = Coordinates.grid(lat=lat, lon=lon, time=dates, dims=['time', 'lat', 'lon'])
        assert c.dims == ('time', 'lat', 'lon')
        assert c.udims == ('time', 'lat', 'lon')
        assert c.shape == (5, 3, 4)
        assert c.ndim == 3
        assert c.size == 60

    def test_points(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ['2018-01-01', '2018-01-02', '2018-01-03']

        c = Coordinates.points(lat=lat, lon=lon, time=dates, dims=['time', 'lat', 'lon'])
        assert c.dims == ('time_lat_lon',)
        assert c.udims == ('time', 'lat', 'lon')
        assert c.shape == (3,)
        assert c.ndim == 1
        assert c.size == 3

    def test_grid_points_order(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30, 40]
        dates = ['2018-01-01', '2018-01-02']

        with pytest.raises(ValueError):
            Coordinates.grid(lat=lat, lon=lon, time=dates, dims=['lat', 'lon'])

        with pytest.raises(ValueError):
            Coordinates.grid(lat=lat, lon=lon, dims=['lat', 'lon', 'time'])

        if sys.version < '3.6':
            with pytest.raises(TypeError):
                Coordinates.grid(lat=lat, lon=lon, time=dates)
        else:
            Coordinates.grid(lat=lat, lon=lon, time=dates)

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

        # invalid
        with pytest.raises(TypeError, match="Coordinates.from_xarray expects xarray DataArrayCoordinates"):
            Coordinates.from_xarray([0, 10])

    def test_coord_ref_sys(self):
        # assign
        lat = ArrayCoordinates1d([0, 1, 2])
        lon = ArrayCoordinates1d([0, 1, 2])
        
        c = Coordinates([lat, lon], dims=['lat', 'lon'], coord_ref_sys='SPHER_MERC')
        assert c['lat'].coord_ref_sys == 'SPHER_MERC'
        assert c['lon'].coord_ref_sys == 'SPHER_MERC'

        # don't overwrite
        lat = ArrayCoordinates1d([0, 1, 2], coord_ref_sys='WGS84')
        lon = ArrayCoordinates1d([0, 1, 2], coord_ref_sys='WGS84')

        with pytest.raises(ValueError, match='coord_ref_sys mismatch'):
             Coordinates([lat, lon], dims=['lat', 'lon'], coord_ref_sys='SPHER_MERC')
        
        # but just repeating is okay
        lat = ArrayCoordinates1d([0, 1, 2], coord_ref_sys='WGS84')
        lon = ArrayCoordinates1d([0, 1, 2], coord_ref_sys='WGS84')
        c = Coordinates([lat, lon], dims=['lat', 'lon'], coord_ref_sys='WGS84')

        # mismatch
        lat = ArrayCoordinates1d([0, 1, 2], coord_ref_sys='WGS84')
        lon = ArrayCoordinates1d([0, 1, 2], coord_ref_sys='SPHER_MERC')
        with pytest.raises(ValueError, match='coord_ref_sys mismatch'):
             Coordinates([lat, lon], dims=['lat', 'lon'])

    def test_ctype(self):
        # assign
        lat = ArrayCoordinates1d([0, 1, 2])
        lon = ArrayCoordinates1d([0, 1, 2])
        
        c = Coordinates([lat, lon], dims=['lat', 'lon'], ctype='left')
        assert c['lat'].ctype == 'left'
        assert c['lon'].ctype == 'left'

        # don't overwrite
        lat = ArrayCoordinates1d([0, 1, 2], ctype='right')
        lon = ArrayCoordinates1d([0, 1, 2])
        
        c = Coordinates([lat, lon], dims=['lat', 'lon'], ctype='left')
        assert c['lat'].ctype == 'right'
        assert c['lon'].ctype == 'left'

    def test_distance_units(self):
        lat = ArrayCoordinates1d([0, 1, 2])
        lon = ArrayCoordinates1d([0, 1, 2])
        time = ArrayCoordinates1d('2018-01-01')

        units = podpac.core.units.Units()
        c = Coordinates([lat, lon, time], dims=['lat', 'lon', 'time'], distance_units=units)

        assert c['lat'].units is units
        assert c['lon'].units is units
        assert c['time'].units is not units

class TestCoordinatesSerialization(object):
    def test_definition(self):
        c = Coordinates(
            [[[0, 1, 2], [10, 20, 30]], ['2018-01-01', '2018-01-02'], crange(0, 10, 0.5)],
            dims=['lat_lon', 'time', 'alt'])

        d = c.definition
        
        json.dumps(d, cls=podpac.core.utils.JSONEncoder)

        c2 = Coordinates.from_definition(d)
        assert c2 == c
        
    def test_invalid_definition(self):
        with pytest.raises(TypeError, match="Could not parse coordinates definition of type"):
            Coordinates.from_definition({'lat': [0, 1, 2]})

        with pytest.raises(ValueError, match="Could not parse coordinates definition item"):
            Coordinates.from_definition([{"data": [0, 1, 2]}])

    def test_json(self):
        c = Coordinates(
            [[[0, 1, 2], [10, 20, 30]], ['2018-01-01', '2018-01-02'], crange(0, 10, 0.5)],
            dims=['lat_lon', 'time', 'alt'])

        s = c.json

        json.loads(s)
        
        c2 = Coordinates.from_json(s)
        assert c2 == c

class TestCoordinatesProperties(object):
    def test_xarray_coords(self):
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
    
    def test_coord_ref_sys(self):
        # empty
        c = Coordinates([])
        assert c.coord_ref_sys == None
        assert c.crs == None

        # default
        c = Coordinates([[0, 1, 2]], dims=['lat'])
        assert c.coord_ref_sys == settings['DEFAULT_CRS']
        assert c.crs == settings['DEFAULT_CRS']

        # set
        c = Coordinates([[0, 1, 2]], dims=['lat'], coord_ref_sys='EPSG:2193')
        assert c.coord_ref_sys == 'EPSG:2193'
        assert c.crs == 'EPSG:2193'

class TestCoordinatesDict(object):
    coords = Coordinates([[[0, 1, 2], [10, 20, 30]], ['2018-01-01', '2018-01-02']], dims=['lat_lon', 'time'])

    def test_keys(self):
        assert [dim for dim in self.coords.keys()] == ['lat_lon', 'time']

    def test_values(self):
        assert [c for c in self.coords.values()] == [self.coords['lat_lon'], self.coords['time']]

    def test_items(self):
        assert [dim for dim, c in self.coords.items()] == ['lat_lon', 'time']
        assert [c for dim, c in self.coords.items()] == [self.coords['lat_lon'], self.coords['time']]

    def test_iter(self):
        assert [dim for dim in self.coords] == ['lat_lon', 'time']

    def test_getitem(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon')
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02'], name='time')
        lat_lon = StackedCoordinates([lat, lon])
        coords = Coordinates([lat_lon, time])

        assert coords['lat_lon'] == lat_lon
        assert coords['time'] == time
        assert coords['lat'] == lat
        assert coords['lon'] == lon

        with pytest.raises(KeyError, match="Dimension 'alt' not found in Coordinates"):
            coords['alt']

    def test_get(self):
        assert self.coords.get('lat_lon') is self.coords['lat_lon']
        assert self.coords.get('lat') is self.coords['lat']
        assert self.coords.get('alt') == None
        assert self.coords.get('alt', 'DEFAULT') == 'DEFAULT'

    def test_setitem(self):
        coords = deepcopy(self.coords)
        
        coords['time'] = [1, 2, 3]
        coords['time'] = ArrayCoordinates1d([1, 2, 3])
        coords['time'] = ArrayCoordinates1d([1, 2, 3], name='time')
        coords['time'] = Coordinates([[1, 2, 3]], dims=['time'])

        # coords['lat_lon'] = [np.linspace(0, 10, 5), np.linspace(0, 10, 5)]
        coords['lat_lon'] = clinspace((0, 1), (10, 20), 5)
        coords['lat_lon'] = (np.linspace(0, 10, 5), np.linspace(0, 10, 5))
        coords['lat_lon'] = Coordinates([(np.linspace(0, 10, 5), np.linspace(0, 10, 5))], dims=['lat_lon'])

        # update a single stacked dimension
        coords['lat'] = np.linspace(5, 20, 5)
        assert coords['lat'] == ArrayCoordinates1d(np.linspace(5, 20, 5), name='lat')
        
        coords = deepcopy(self.coords)
        coords['lat_lon']['lat'] = np.linspace(5, 20, 3)
        assert coords['lat'] == ArrayCoordinates1d(np.linspace(5, 20, 3), name='lat')

        with pytest.raises(KeyError, match="Cannot set dimension"):
            coords['alt'] = ArrayCoordinates1d([1, 2, 3], name='alt')

        with pytest.raises(KeyError, match="Cannot set dimension"):
            coords['alt'] = ArrayCoordinates1d([1, 2, 3], name='lat')
        
        with pytest.raises(ValueError, match="Dimension name mismatch"):
            coords['time'] = ArrayCoordinates1d([1, 2, 3], name='alt')

        with pytest.raises(ValueError, match="coord_ref_sys mismatch"):
            coords['time'] = ArrayCoordinates1d([1, 2, 3], coord_ref_sys='SPHER_MERC')

        with pytest.raises(KeyError, match="not found in Coordinates"):
            coords['lat_lon'] = Coordinates([(np.linspace(0, 10, 5), np.linspace(0, 10, 5))], dims=['lon_lat'])

        with pytest.raises(ValueError, match="Dimension name mismatch"):
            coords['lat_lon'] = clinspace((0, 1), (10, 20), 5, name='lon_lat')

        with pytest.raises(ValueError, match="Size mismatch"):
            coords['lat'] = np.linspace(5, 20, 5)

        with pytest.raises(ValueError, match="Duplicate dimension"):
            coords['lat'] = clinspace(0, 10, 3, name='lon')

        with pytest.raises(ValueError, match="coord_ref_sys mismatch"):
            coords['lat_lon'] = Coordinates([(np.linspace(0, 10, 3), np.linspace(0, 10, 3))], dims=['lat_lon'], coord_ref_sys='WGS84')
            coords['lat'] = Coordinates([np.linspace(0, 10, 3)], dims=['lat'], coord_ref_sys='SPHER_MERC') # should work

    def test_delitem(self):
        # unstacked
        coords = deepcopy(self.coords)
        del coords['time']
        assert coords.dims == ('lat_lon',)

        # stacked
        coords = deepcopy(self.coords)
        del coords['lat_lon']
        assert coords.dims == ('time',)

        # missing 
        coords = deepcopy(self.coords)
        with pytest.raises(KeyError, match="Cannot delete dimension 'alt' in Coordinates"):
            del coords['alt']

        # part of stacked dimension
        coords = deepcopy(self.coords)
        with pytest.raises(KeyError, match="Cannot delete dimension 'lat' in Coordinates"):
            del coords['lat']

    def test_update(self):
        # add a new dimension
        coords = deepcopy(self.coords)
        c = Coordinates([[100, 200, 300]], dims=['alt'])
        coords.update(c)
        assert coords.dims == ('lat_lon', 'time', 'alt')
        assert coords['lat_lon'] == self.coords['lat_lon']
        assert coords['time'] == self.coords['time']
        assert coords['alt'] == c['alt']

        # overwrite a dimension
        coords = deepcopy(self.coords)
        c = Coordinates([[100, 200, 300]], dims=['time'])
        coords.update(c)
        assert coords.dims == ('lat_lon', 'time')
        assert coords['lat_lon'] == self.coords['lat_lon']
        assert coords['time'] == c['time']

        # overwrite a stacked dimension
        coords = deepcopy(self.coords)
        c = Coordinates([clinspace((0, 1), (10, 20), 5)], dims=['lat_lon'])
        coords.update(c)
        assert coords.dims == ('lat_lon', 'time')
        assert coords['lat_lon'] == c['lat_lon']
        assert coords['time'] == self.coords['time']

        # mixed
        coords = deepcopy(self.coords)
        c = Coordinates([clinspace((0, 1), (10, 20), 5), [100, 200, 300]], dims=['lat_lon', 'alt'])
        coords.update(c)
        assert coords.dims == ('lat_lon', 'time', 'alt')
        assert coords['lat_lon'] == c['lat_lon']
        assert coords['time'] == self.coords['time']
        assert coords['alt'] == c['alt']

        # invalid
        coords = deepcopy(self.coords)
        with pytest.raises(TypeError, match="Cannot update Coordinates with object of type"):
            coords.update({'time': [1, 2, 3]})

        # duplicate dimension
        coords = deepcopy(self.coords)
        c = Coordinates([[0, 0.1, 0.2]], dims=['lat'])
        with pytest.raises(ValueError, match="Duplicate dimension name 'lat'"):
            coords.update(c)

    def test_len(self):
        assert len(self.coords) == 2

class TestCoordinatesMethods(object):
    coords = Coordinates([[[0, 1, 2], [10, 20, 30]], ['2018-01-01', '2018-01-02'], 10], dims=['lat_lon', 'time', 'alt'])
    
    def test_drop(self):
        # drop one existing dimension
        c1 = self.coords.drop('time')
        c2 = self.coords.udrop('time')
        assert c1.dims == ('lat_lon', 'alt')
        assert c2.dims == ('lat_lon', 'alt')

        # drop multiple existing dimensions
        c1 = self.coords.drop(['time', 'alt'])
        c2 = self.coords.udrop(['time', 'alt'])
        assert c1.dims == ('lat_lon',)
        assert c2.dims == ('lat_lon',)

        # drop all dimensions
        c1 = self.coords.drop(self.coords.dims)
        c2 = self.coords.udrop(self.coords.udims)
        assert c1.dims == ()
        assert c2.dims == ()

        # drop no dimensions
        c1 = self.coords.drop([])
        c2 = self.coords.udrop([])
        assert c1.dims == ('lat_lon', 'time', 'alt')
        assert c2.dims == ('lat_lon', 'time', 'alt')

        # drop a missing dimension
        c = self.coords.drop('alt')
        with pytest.raises(KeyError, match="Dimension 'alt' not found in Coordinates with dims"):
            c1 = c.drop('alt')
        with pytest.raises(KeyError, match="Dimension 'alt' not found in Coordinates with udims"):
            c2 = c.udrop('alt')

        c1 = c.drop('alt', ignore_missing=True)
        c2 = c.udrop('alt', ignore_missing=True)
        assert c1.dims == ('lat_lon', 'time')
        assert c2.dims == ('lat_lon', 'time')

        # drop a stacked dimension: drop works but udrop gives an exception
        c1 = self.coords.drop('lat_lon')
        assert c1.dims == ('time', 'alt')
        
        with pytest.raises(KeyError, match="Dimension 'lat_lon' not found in Coordinates with udims"):
            c2 = self.coords.udrop('lat_lon')

        # drop part of a stacked dimension: drop gives exception but udrop does not
        # note: two udrop cases: 'lat_lon' -> 'lon' and 'lat_lon_alt' -> 'lat_lon'
        with pytest.raises(KeyError, match="Dimension 'lat' not found in Coordinates with dims"):
            c1 = self.coords.drop('lat')
        
        c2 = self.coords.udrop('lat')
        assert c2.dims == ('lon', 'time', 'alt')
        
        coords = Coordinates([[[0, 1], [10, 20], [100, 300]]], dims=['lat_lon_alt'])
        c2 = coords.udrop('alt')
        assert c2.dims == ('lat_lon',)

        # invalid type
        with pytest.raises(TypeError, match="Invalid drop dimension type"):
            self.coords.drop(2)

        with pytest.raises(TypeError, match="Invalid drop dimension type"):
            self.coords.udrop(2)

        with pytest.raises(TypeError, match="Invalid drop dimension type"):
            self.coords.drop([2, 3])

        with pytest.raises(TypeError, match="Invalid drop dimension type"):
            self.coords.udrop([2, 3])

    def test_unique(self):
        # unstacked (numerical, datetime, and empty)
        c = Coordinates([[2, 1, 0, 1], ['2018-01-01', '2018-01-02', '2018-01-01'], []], dims=['lat', 'time', 'alt'])
        c2 = c.unique()
        assert_equal(c2['lat'].coordinates, [0, 1, 2])
        assert_equal(c2['time'].coordinates, [np.datetime64('2018-01-01'), np.datetime64('2018-01-02')])
        assert_equal(c2['alt'].coordinates, [])
        
        # stacked
        lat_lon = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 2), # duplicate
            (1, 0),
            (1, 1),
            (1, 1), # duplicate
        ]
        lat, lon = zip(*lat_lon)
        c = Coordinates([[lat, lon]], dims=['lat_lon'])
        c2 = c.unique()
        assert_equal(c2['lat'].coordinates, [0., 0., 0., 1., 1.])
        assert_equal(c2['lon'].coordinates, [0., 1., 2., 0., 1.])

    def test_unstack(self):
        c1 = Coordinates([[[0, 1], [10, 20], [100, 300]]], dims=['lat_lon_alt'])
        c2 = c1.unstack()
        assert c1.dims == ('lat_lon_alt',)
        assert c2.dims == ('lat', 'lon', 'alt')
        assert c1['lat'] == c2['lat']
        assert c1['lon'] == c2['lon']
        assert c1['alt'] == c2['alt']

        # mixed
        c1 = Coordinates([[[0, 1], [10, 20]], [100, 200, 300]], dims=['lat_lon', 'alt'])
        c2 = c1.unstack()
        assert c1.dims == ('lat_lon', 'alt',)
        assert c2.dims == ('lat', 'lon', 'alt')
        assert c1['lat'] == c2['lat']
        assert c1['lon'] == c2['lon']
        assert c1['alt'] == c2['alt']

    def test_iterchunks(self):
        c = Coordinates(
            [clinspace(0, 1, 100), clinspace(0, 1, 200), ['2018-01-01', '2018-01-02']],
            dims=['lat', 'lon', 'time'])
        
        for chunk in c.iterchunks(shape=(10, 10, 10)):
            assert chunk.shape == (10, 10, 2)

        for chunk, slices in c.iterchunks(shape=(10, 10, 10), return_slices=True):
            assert isinstance(slices, tuple)
            assert len(slices) == 3
            assert isinstance(slices[0], slice)
            assert isinstance(slices[1], slice)
            assert isinstance(slices[2], slice)
            assert chunk.shape == (10, 10, 2)

    def test_tranpose(self):
        c = Coordinates([[0, 1], [10, 20], ['2018-01-01', '2018-01-02']], dims=['lat', 'lon', 'time'])

        # transpose
        t = c.transpose('lon', 'lat', 'time')
        assert c.dims == ('lat', 'lon', 'time')
        assert t.dims == ('lon', 'lat', 'time')

        # default: full transpose
        t = c.transpose()
        assert c.dims == ('lat', 'lon', 'time')
        assert t.dims == ('time', 'lon', 'lat')

        # in place
        t = c.transpose('lon', 'lat', 'time', in_place=False)
        assert c.dims == ('lat', 'lon', 'time')
        assert t.dims == ('lon', 'lat', 'time')

        c.transpose('lon', 'lat', 'time', in_place=True)
        assert c.dims == ('lon', 'lat', 'time')

        with pytest.raises(ValueError, match="Invalid transpose dimensions"):
            c.transpose('lon', 'lat')

    def test_transform(self):
        c = Coordinates([[0, 1], [10, 20], ['2018-01-01', '2018-01-02'], [0, 1, 2]], \
                        dims=['lat', 'lon', 'time', 'alt'])
        c1 = Coordinates([[[0, 1], [10, 20]], [100, 200, 300]], dims=['lat_lon', 'alt'])

        # default crs
        assert c.crs == settings['DEFAULT_CRS']

        # transform
        c_trans = c.transform('EPSG:2193')
        assert c.crs == settings['DEFAULT_CRS']
        assert c_trans.crs == 'EPSG:2193'
        assert round(c_trans['lat'].values[0]) == 29995930.0

        # support proj4 strings
        proj = '+proj=merc +lat_ts=56.5 +ellps=GRS80'
        c_trans = c.transform(proj)
        assert c.crs == settings['DEFAULT_CRS']
        assert c_trans.crs == pyproj.CRS(proj).srs
        assert round(c_trans['lat'].values[0]) == 615849.0

        # support stacked coordinates
        proj = '+proj=merc +lat_ts=56.5 +ellps=GRS80'
        c1_trans = c1.transform(proj)
        assert c1.crs == settings['DEFAULT_CRS']
        assert c_trans.crs == pyproj.CRS(proj).srs
        assert round(c1_trans['lat'].values[0]) == 615849.0

        # support altitude unit transformations
        proj = '+proj=merc +vunits=us-ft'
        c_trans = c.transform(proj)
        assert round(c_trans['lat'].values[0]) == 1113195.0
        assert round(c_trans['alt'].values[1]) == 3.0
        assert round(c_trans['alt'].values[2]) == 7.0
        assert '+vunits=us-ft' in c_trans.crs
        c1_trans = c1.transform(proj)
        assert round(c1_trans['lat'].values[0]) == 1113195.0
        assert round(c1_trans['alt'].values[0]) == 328.0
        assert round(c1_trans['alt'].values[1]) == 656.0
        assert '+vunits=us-ft' in c1_trans.crs

        # make sure vunits can be overwritten appropriately
        c2_trans = c1_trans.transform(alt_units='m')
        assert round(c2_trans['alt'].values[0]) == 100.0
        assert '+vunits=m' in c2_trans.crs and '+vunits=us-ft' not in c2_trans.crs

        # alt_units parameter
        c_trans = c.transform('EPSG:2193', alt_units='us-ft')
        assert round(c_trans['alt'].values[1]) == 3.0
        assert round(c_trans['alt'].values[2]) == 7.0
        assert '+vunits=us-ft' in c_trans.crs
        c_trans = c.transform('EPSG:2193', alt_units='km')
        assert c_trans['alt'].values[1] == 0.001
        assert c_trans['alt'].values[2] == 0.002
        assert '+vunits=km' in c_trans.crs


    def test_intersect(self):
        # TODO: add additional testing
        
        # should change the other coordinates crs into the native coordinates crs for intersect
        c = Coordinates([np.linspace(0, 10, 11), np.linspace(0, 10, 11), ['2018-01-01', '2018-01-02']], \
                        dims=['lat', 'lon', 'time'])
        o = Coordinates([np.linspace(28000000, 29500000, 20), np.linspace(-280000, 400000, 20), ['2018-01-01', '2018-01-02']], \
                dims=['lat', 'lon', 'time'], coord_ref_sys='EPSG:2193')

        c_int = c.intersect(o)
        assert c_int.crs == settings['DEFAULT_CRS']
        assert np.all(c_int['lat'].bounds == np.array([5., 10.]))
        assert np.all(c_int['lon'].bounds == np.array([4., 10.]))
        assert np.all(c_int['time'].values == c['time'].values)

class TestCoordinatesSpecial(object):
    def test_repr(self):
        repr(Coordinates([[0, 1], [10, 20], ['2018-01-01', '2018-01-02']], dims=['lat', 'lon', 'time']))
        repr(Coordinates([[[0, 1], [10, 20]], ['2018-01-01', '2018-01-02']], dims=['lat_lon', 'time']))
        repr(Coordinates([0, 10, []], dims=['lat', 'lon', 'time'], ctype='point'))
        repr(Coordinates([crange(0, 10, 0.5)], dims=['alt']))
        repr(Coordinates([]))

    def test_eq(self):
        c1 = Coordinates([[[0, 1, 2], [10, 20, 30]], ['2018-01-01', '2018-01-02']], dims=['lat_lon', 'time'])
        c2 = Coordinates([[[0, 1, 2], [10, 20, 30]], ['2018-01-01', '2018-01-02']], dims=['lat_lon', 'time'])
        c3 = Coordinates([[[0, 1, 2], [10, 20, 30]], ['2018-01-01', '2018-01-02']], dims=['lat_lon', 'time'], ctype='point')
        c4 = Coordinates([[[0, 2, 1], [10, 20, 30]], ['2018-01-01', '2018-01-02']], dims=['lat_lon', 'time'])
        c5 = Coordinates([[[0, 1, 2], [10, 20, 30]], ['2018-01-01', '2018-01-02']], dims=['lon_lat', 'time'])
        c6 = Coordinates([[[0, 1, 2], [10, 20, 30]], ['2018-01-01']], dims=['lat_lon', 'time'])
        c7 = Coordinates([[0, 1, 2], [10, 20, 30], ['2018-01-01', '2018-01-02']], dims=['lat', 'lon', 'time'])

        assert c1 == c1
        assert c1 == c2
        assert c1 == deepcopy(c1)

        assert c1 != c3
        assert c1 != c4
        assert c1 != c5
        assert c1 != c6
        assert c1 != c7
        assert c1 != None

    def test_hash(self):
        c1 = Coordinates([[[0, 1, 2], [10, 20, 30]], ['2018-01-01', '2018-01-02']], dims=['lat_lon', 'time'])
        c2 = Coordinates([[[0, 1, 2], [10, 20, 30]], ['2018-01-01', '2018-01-02']], dims=['lat_lon', 'time'], ctype='point')
        c3 = Coordinates([[[0, 2, 1], [10, 20, 30]], ['2018-01-01', '2018-01-02']], dims=['lat_lon', 'time'])
        c4 = Coordinates([[[0, 1, 2], [10, 20, 30]], ['2018-01-01', '2018-01-02']], dims=['lon_lat', 'time'])
        c5 = Coordinates([[[0, 1, 2], [10, 20, 30]], ['2018-01-01']], dims=['lat_lon', 'time'])
        c6 = Coordinates([[0, 1, 2], [10, 20, 30], ['2018-01-01', '2018-01-02']], dims=['lat', 'lon', 'time'])
        
        assert c1.hash == c1.hash
        assert c1.hash == deepcopy(c1).hash
        
        assert c1.hash != c2.hash
        assert c1.hash != c3.hash
        assert c1.hash != c4.hash
        assert c1.hash != c5.hash
        assert c1.hash != c6.hash

def test_merge_dims():
    ctime = Coordinates([['2018-01-01', '2018-01-02']], dims=['time'])
    clatlon = Coordinates([[2, 4, 5], [3, -1, 5]], dims=['lat', 'lon'])
    clatlon_stacked = Coordinates([[[2, 4, 5], [3, -1, 5]]], dims=['lat_lon'])
    clat = Coordinates([[2, 4, 5]], dims=['lat'])

    c = merge_dims([clatlon, ctime])
    assert c.dims == ('lat', 'lon', 'time')

    c = merge_dims([ctime, clatlon])
    assert c.dims == ('time', 'lat', 'lon')

    c = merge_dims([clatlon_stacked, ctime])
    assert c.dims == ('lat_lon', 'time')

    c = merge_dims([ctime, clatlon_stacked])
    assert c.dims == ('time', 'lat_lon')

    with pytest.raises(ValueError, match="Duplicate dimension name 'lat'"):
        merge_dims([clatlon, clat])
    
    with pytest.raises(ValueError, match="Duplicate dimension name 'lat'"):
        merge_dims([clatlon_stacked, clat])

    with pytest.raises(TypeError, match="Cannot merge"):
        merge_dims([clat, 0])

def test_concat_and_union():
    c1 = Coordinates([[2, 4, 5], [3, -1, 5]], dims=['lat', 'lon'])
    c2 = Coordinates([[2, 3], [3, 0], ['2018-01-01', '2018-01-02']], dims=['lat', 'lon', 'time'])
    c3 = Coordinates([[[2, 3], [3, 0]]], dims=['lat_lon'])

    c = concat([c1, c2])
    assert c.shape == (5, 5, 2)

    c = union([c1, c2])
    assert c.shape == (4, 4, 2)

    with pytest.raises(TypeError, match="Cannot concat"):
        concat([c1, [1, 2]])

    with pytest.raises(ValueError, match="Duplicate dimension name 'lat' in dims"):
        concat([c1, c3])

def test_concat_stacked_datetimes():
    c1 = Coordinates([[0, 0.5, '2018-01-01']], dims=['lat_lon_time'])
    c2 = Coordinates([[1, 1.5, '2018-01-02']], dims=['lat_lon_time'])
    c = concat([c1, c2])
    np.testing.assert_array_equal(c['lat'].coordinates, np.array([0.0, 1.0]))
    np.testing.assert_array_equal(c['lon'].coordinates, np.array([0.5, 1.5]))
    np.testing.assert_array_equal(
        c['time'].coordinates,
        np.array(['2018-01-01', '2018-01-02']).astype(np.datetime64))

    c1 = Coordinates([[0, 0.5, '2018-01-01T01:01:01']], dims=['lat_lon_time'])
    c2 = Coordinates([[1, 1.5, '2018-01-01T01:01:02']], dims=['lat_lon_time'])
    c = concat([c1, c2])
    np.testing.assert_array_equal(c['lat'].coordinates, np.array([0.0, 1.0]))
    np.testing.assert_array_equal(c['lon'].coordinates, np.array([0.5, 1.5]))
    np.testing.assert_array_equal(
        c['time'].coordinates,
        np.array(['2018-01-01T01:01:01', '2018-01-01T01:01:02']).astype(np.datetime64))
