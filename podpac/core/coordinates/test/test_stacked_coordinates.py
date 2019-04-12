
from datetime import datetime
import json

import pytest
import traitlets as tl
import numpy as np
import pandas as pd
import xarray as xr
from numpy.testing import assert_equal

import podpac
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates

class TestStackedCoordinatesCreation(object):
    def test_init_Coordinates1d(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon')
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03'], name='time')
        c = StackedCoordinates([lat, lon, time])
        assert c.dims == ('lat', 'lon', 'time')
        assert c.udims == ('lat', 'lon', 'time')
        assert c.name == 'lat_lon_time'
        repr(c)

        # un-named
        lat = ArrayCoordinates1d([0, 1, 2])
        lon = ArrayCoordinates1d([10, 20, 30])
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03'])
        c = StackedCoordinates([lat, lon, time])
        assert c.dims == (None, None, None)
        assert c.udims == (None, None, None)
        assert c.name is None

        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        c = StackedCoordinates([lat, lon, time])
        assert c.dims == ('lat', None, None)
        assert c.udims == ('lat', None, None)
        assert c.name == 'lat_?_?'

        repr(c)

    def test_ctype(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat', ctype='left')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon')
        c = StackedCoordinates([lat, lon], ctype='right')

        # lon ctype set by StackedCoordinates
        assert c['lon'].ctype == 'right'

        # but lat is left by StackedCoordinates because it was already explicitly set
        assert c['lat'].ctype == 'left'

    def test_coord_ref_sys(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon')
        c = StackedCoordinates([lat, lon], coord_ref_sys='SPHER_MERC')

        assert c['lat'].coord_ref_sys == 'SPHER_MERC'
        assert c['lon'].coord_ref_sys == 'SPHER_MERC'

        # must match
        lat = ArrayCoordinates1d([0, 1, 2], name='lat', coord_ref_sys='WGS84')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon', coord_ref_sys='SPHER_MERC')
        with pytest.raises(ValueError, match="coord_ref_sys mismatch"):
            StackedCoordinates([lat, lon])

        lat = ArrayCoordinates1d([0, 1, 2], name='lat', coord_ref_sys='WGS84')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon', coord_ref_sys='WGS84')
        with pytest.raises(ValueError, match="coord_ref_sys mismatch"):
            StackedCoordinates([lat, lon], coord_ref_sys='SPHER_MERC')

    def test_distance_units(self):
        lat = ArrayCoordinates1d([0, 1], name='lat')
        lon = ArrayCoordinates1d([0, 1], name='lon')
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02'], name='time')

        units = podpac.core.units.Units()
        c = StackedCoordinates([lat, lon, time], distance_units=units)

        assert c['lat'].units is units
        assert c['lon'].units is units
        assert c['time'].units is not units

    def test_StackedCoordinates(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon')
        c = StackedCoordinates([lat, lon])
        c2 = StackedCoordinates(c)

    def test_invalid_coords(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([0, 1, 2, 3], name='lon')
        c = ArrayCoordinates1d([0, 1, 2])

        with pytest.raises(TypeError, match="Unrecognized coords type"):
            StackedCoordinates({})

        with pytest.raises(ValueError, match="Stacked coords must have at least 2 coords"):
            StackedCoordinates([lat])

        with pytest.raises(ValueError, match="Size mismatch in stacked coords"):
            StackedCoordinates([lat, lon])

        with pytest.raises(ValueError, match="Duplicate dimension name"):
            StackedCoordinates([lat, lat])

        # but duplicate None name is okay
        StackedCoordinates([c, c])

        with pytest.raises(TypeError, match='Invalid coordinates'):
            StackedCoordinates([[0, 1, 2], [10, 20, 30]])

    def test_from_xarray(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon')
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03'], name='time')
        xcoords = StackedCoordinates([lat, lon, time]).coords

        c2 = StackedCoordinates.from_xarray(xcoords)
        assert c2.dims == ('lat', 'lon', 'time')
        assert_equal(c2['lat'].coordinates, lat.coordinates)
        assert_equal(c2['lon'].coordinates, lon.coordinates)
        assert_equal(c2['time'].coordinates, time.coordinates)

    def test_copy(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon')
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03'], name='time')
        c = StackedCoordinates([lat, lon, time])

        c2 = c.copy()
        assert c2 is not c
        assert c2 == c

class TestStackedCoordinatesEq(object):
    def test_eq_type(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon')
        c = StackedCoordinates([lat, lon])
        assert c != [[0, 1, 2], [10, 20, 30]]

    def test_eq_size_shortcut(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon')
        c1 = StackedCoordinates([lat, lon])
        c2 = StackedCoordinates([lat[:2], lon[:2]])
        assert c1 != c2

    def test_eq_dims_shortcut(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon')
        c1 = StackedCoordinates([lat, lon])
        c2 = StackedCoordinates([lon, lat])
        assert c1 != c2

    def test_eq_coordinates(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon')
        c1 = StackedCoordinates([lat, lon])
        c2 = StackedCoordinates([lat, lon])
        c3 = StackedCoordinates([lat[::-1], lon])
        c4 = StackedCoordinates([lat, lon[::-1]])
        
        assert c1 == c2
        assert c1 != c3
        assert c1 != c4

class TestStackedCoordinatesSerialization(object):
    def test_definition(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon')
        time = UniformCoordinates1d('2018-01-01', '2018-01-03', '1,D', name='time')
        c = StackedCoordinates([lat, lon, time])
        d = c.definition
        
        assert isinstance(d, list)
        json.dumps(d, cls=podpac.core.utils.JSONEncoder) # test serializable
        c2 = StackedCoordinates.from_definition(d)
        assert c2 == c

    def test_invalid_definition(self):
        with pytest.raises(ValueError, match="Could not parse coordinates definition with keys"):
            StackedCoordinates.from_definition([{'apple': 10}, {}])

class TestStackedCoordinatesProperties(object):
    def test_set_name(self):
        lat = ArrayCoordinates1d([0, 1, 2])
        lon = ArrayCoordinates1d([10, 20, 30])
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03'])
        c = StackedCoordinates([lat, lon, time])
        c.name = 'lat_lon_time'

        assert c.dims == ('lat', 'lon', 'time')
        assert c.udims == ('lat', 'lon', 'time')
        assert c.name == 'lat_lon_time'
        
        # also sets the Coordinates1d objects:
        assert lat.name == 'lat'
        assert lon.name == 'lon'
        assert time.name == 'time'

        with pytest.raises(ValueError, match="Invalid name"):
            c.name = 'lat_lon'

        with pytest.raises(ValueError, match="Duplicate dimension name"):
            c.name = 'lat_lat_time'

    def test_size(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3])
        lon = ArrayCoordinates1d([10, 20, 30, 40])
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'])
        c = StackedCoordinates([lat, lon, time])

        assert c.size == 4

    def test_coordinates(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30, 40], name='lon')
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'], name='time')
        c = StackedCoordinates([lat, lon, time])

        assert isinstance(c.coordinates, pd.MultiIndex)
        assert c.coordinates.size == 4
        assert c.coordinates.names == ['lat', 'lon', 'time']
        assert c.coordinates[0] == (0.0, 10, np.datetime64('2018-01-01'))
        assert c.coordinates[1] == (1.0, 20, np.datetime64('2018-01-02'))
        assert c.coordinates[2] == (2.0, 30, np.datetime64('2018-01-03'))
        assert c.coordinates[3] == (3.0, 40, np.datetime64('2018-01-04'))

    def test_coords(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30, 40], name='lon')
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'], name='time')
        c = StackedCoordinates([lat, lon, time])

        assert isinstance(c.coords, xr.core.coordinates.DataArrayCoordinates)
        assert c.coords.dims == ('lat_lon_time',)
        assert_equal(c.coords['lat'], c['lat'].coordinates)
        assert_equal(c.coords['lon'], c['lon'].coordinates)
        assert_equal(c.coords['time'], c['time'].coordinates)

    def test_coord_ref_sys(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30, 40], name='lon')
        c = StackedCoordinates([lat, lon], coord_ref_sys='SPHER_MERC')
        assert c.coord_ref_sys == 'SPHER_MERC'

class TestStackedCoordinatesIndexing(object):
    def test_get_dim(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30, 40], name='lon')
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'], name='time')
        c = StackedCoordinates([lat, lon, time])

        assert c['lat'] is lat
        assert c['lon'] is lon
        assert c['time'] is time
        with pytest.raises(KeyError, match="Dimension 'other' not found in dims"):
            c['other']

    def test_get_index(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30, 40], name='lon')
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'], name='time')
        c = StackedCoordinates([lat, lon, time])

        # integer index
        assert isinstance(c[0], StackedCoordinates)
        assert c[0].size == 1
        assert c[0].dims == c.dims
        assert_equal(c[0]['lat'].coordinates, c['lat'].coordinates[0])

        # index array
        assert isinstance(c[[1, 2]], StackedCoordinates)
        assert c[[1, 2]].size == 2
        assert c[[1, 2]].dims == c.dims
        assert_equal(c[[1, 2]]['lat'].coordinates, c['lat'].coordinates[[1, 2]])

        # boolean array
        assert isinstance(c[[False, True, True, False]], StackedCoordinates)
        assert c[[False, True, True, False]].size == 2
        assert c[[False, True, True, False]].dims == c.dims
        assert_equal(c[[False, True, True, False]]['lat'].coordinates, c['lat'].coordinates[[False, True, True, False]])

        # slice
        assert isinstance(c[1:3], StackedCoordinates)
        assert c[1:3].size == 2
        assert c[1:3].dims == c.dims
        assert_equal(c[1:3]['lat'].coordinates, c['lat'].coordinates[1:3])

    def test_iter(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3])
        lon = ArrayCoordinates1d([10, 20, 30, 40])
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'])
        c = StackedCoordinates([lat, lon, time])

        for item in c:
            assert isinstance(item, Coordinates1d)

    def test_len(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3])
        lon = ArrayCoordinates1d([10, 20, 30, 40])
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'])
        c = StackedCoordinates([lat, lon, time])

        assert len(c) == 3

class TestStackedCoordinatesSelection(object):
    def test_intersect(self):
        # TODO going to test Coordinates intersect first
        pass

    @pytest.mark.skip(reason="not yet implemented")
    def test_select(self):
        pass
