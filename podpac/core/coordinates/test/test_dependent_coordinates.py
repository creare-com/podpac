
from datetime import datetime
import json

import pytest
import traitlets as tl
import numpy as np
import pandas as pd
import xarray as xr
from numpy.testing import assert_equal

import podpac
from podpac.core.units import Units
from podpac.coordinates import ArrayCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates
from podpac.core.coordinates.dependent_coordinates import DependentCoordinates, ArrayCoordinatesNd

LAT = np.linspace(0, 1, 12).reshape((3, 4))
LON = np.linspace(10, 20, 12).reshape((3, 4))

class TestDependentCoordinatesCreation(object):
    def test_init(self):
        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'])

        assert c.dims == ('lat', 'lon')
        assert c.udims == ('lat', 'lon')
        assert c.idims == ('i', 'j')
        assert c.name == 'lat,lon'
        repr(c)

        c = DependentCoordinates((LAT, LON))
        assert c.dims == (None, None)
        assert c.udims == (None, None)
        assert c.idims == ('i', 'j')
        assert c.name == '?,?'
        repr(c)

    def test_invalid(self):
        # mismatched shape
        with pytest.raises(ValueError, match="coordinates shape mismatch"):
            DependentCoordinates((LAT, LON.reshape((4, 3))))

        # invalid dims
        with pytest.raises(ValueError, match="dims and coordinates size mismatch"):
            DependentCoordinates((LAT, LON), dims=['lat'])

        with pytest.raises(ValueError, match="dims and coordinates size mismatch"):
            DependentCoordinates((LAT,), dims=['lat', 'lon'])

        with pytest.raises(ValueError, match="Duplicate dimension"):
            DependentCoordinates((LAT, LON), dims=['lat', 'lat'])

        with pytest.raises(tl.TraitError):
            DependentCoordinates((LAT, LON), dims=['lat', 'depth'])

        with pytest.raises(ValueError, match="Dependent coordinates cannot be empty"):
            DependentCoordinates([], dims=[])

    def test_set_name(self):
        # set when empty
        c = DependentCoordinates((LAT, LON))
        c._set_name('lat,lon')
        assert c.name == 'lat,lon'

        # check when setting
        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'])
        c._set_name('lat,lon')

        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'])
        with pytest.raises(ValueError, match="Dimension mismatch"):
            c._set_name('lon,lat')

    def test_ctype_and_segment_lengths(self):
        # explicit
        c = DependentCoordinates((LAT, LON), ctypes=['left', 'right'], segment_lengths=[1.0, 2.0])
        assert c.ctypes == ('left', 'right')
        assert c.segment_lengths == (1.0, 2.0)

        c = DependentCoordinates((LAT, LON), ctypes=['point', 'point'])
        assert c.ctypes == ('point', 'point')
        assert c.segment_lengths == (None, None)

        c = DependentCoordinates((LAT, LON), ctypes=['midpoint', 'point'], segment_lengths=[1.0, None])
        assert c.ctypes == ('midpoint', 'point')
        assert c.segment_lengths == (1.0, None)

        # single value
        c = DependentCoordinates((LAT, LON), ctypes='left', segment_lengths=1.0)
        assert c.ctypes == ('left', 'left')
        assert c.segment_lengths == (1.0, 1.0)

        c = DependentCoordinates((LAT, LON), ctypes='point')
        assert c.ctypes == ('point', 'point')
        assert c.segment_lengths == (None, None)
        
        # defaults
        c = DependentCoordinates((LAT, LON))
        assert c.ctypes == ('point', 'point')

        # don't overwrite
        c = DependentCoordinates((LAT, LON), ctypes='left', segment_lengths=1.0)
        c._set_ctype('right')
        assert c.ctypes == ('left', 'left')

        # size mismatch
        with pytest.raises(ValueError, match='size mismatch'):
            DependentCoordinates((LAT, LON), ctypes=['left', 'left', 'left'], segment_lengths=1.0)

        with pytest.raises(ValueError, match='segment_lengths and coordinates size mismatch'):
            DependentCoordinates((LAT, LON), ctypes='left', segment_lengths=[1.0, 1.0, 1.0])

        # segment lengths required
        with pytest.raises(TypeError, match='segment_lengths cannot be None'):
            DependentCoordinates((LAT, LON), ctypes='left')
        
        with pytest.raises(TypeError, match='segment_lengths cannot be None'):
            DependentCoordinates((LAT, LON), ctypes=['left', 'point'])
        
        with pytest.raises(TypeError, match='segment_lengths cannot be None'):
            DependentCoordinates((LAT, LON), ctypes=['left', 'point'], segment_lengths=[None, None])

        # segment lengths prohibited
        with pytest.raises(TypeError, match='segment_lengths must be None'):
            DependentCoordinates((LAT, LON), segment_lengths=1.0)
        
        with pytest.raises(TypeError, match='segment_lengths must be None'):
            DependentCoordinates((LAT, LON), ctypes='point', segment_lengths=1.0)
        
        with pytest.raises(TypeError, match='segment_lengths must be None'):
            DependentCoordinates((LAT, LON), ctypes=['left', 'point'], segment_lengths=[1.0, 1.0])

        # invalid
        with pytest.raises(tl.TraitError):
            DependentCoordinates((LAT, LON), ctypes='abc')
        
        # invalid segment_lengths
        with pytest.raises(ValueError):
            DependentCoordinates((LAT, LON), ctypes='left', segment_lengths='abc')

        with pytest.raises(ValueError, match="segment_lengths must be positive"):
            DependentCoordinates((LAT, LON), ctypes=['left', 'right'], segment_lengths=[1.0, -2.0])

    def test_coord_ref_sys(self):
        c = DependentCoordinates((LAT, LON), coord_ref_sys='SPHER_MERC')
        assert c.coord_ref_sys == 'SPHER_MERC'

        c = DependentCoordinates((LAT, LON))
        assert c.coord_ref_sys == 'WGS84'

        # check when setting
        c = DependentCoordinates((LAT, LON), coord_ref_sys='SPHER_MERC')
        c._set_coord_ref_sys('SPHER_MERC')

        c = DependentCoordinates((LAT, LON), coord_ref_sys='SPHER_MERC')
        with pytest.raises(ValueError, match="coord_ref_sys mismatch"):
            c._set_coord_ref_sys('WGS84')

    def test_units(self):
        ua = Units()
        ub = Units()

        # explicit
        c = DependentCoordinates((LAT, LON), units=[ua, ub])
        assert len(c.units) == 2
        assert c.units[0] is ua
        assert c.units[1] is ub

        # single value
        c = DependentCoordinates((LAT, LON), units=ua)
        assert len(c.units) == 2
        assert c.units[0] is ua
        assert c.units[1] is ua

        # don't overwrite
        c = DependentCoordinates((LAT, LON), units=ua)
        c._set_units(ub)
        assert len(c.units) == 2
        assert c.units[0] is ua
        assert c.units[1] is ua

        # distance only
        time = np.linspace(1, 13, 12).reshape((3, 4))
        c = DependentCoordinates((LAT, LON, time), dims=['lat', 'lon', 'time'])
        c._set_distance_units(ua)
        assert len(c.units) == 3
        assert c.units[0] is ua
        assert c.units[1] is ua
        assert c.units[2] is None

    def test_copy(self):
        c = DependentCoordinates((LAT, LON))

        c2 = c.copy()
        assert c2 is not c
        assert c2 == c

class TestDependentCoordinatesStandardMethods(object):
    def test_eq_type(self):
        c = DependentCoordinates([LAT, LON])
        assert c != [[0, 1, 2], [10, 20, 30]]

    def test_eq_shape_shortcut(self):
        c1 = DependentCoordinates([LAT, LON])
        c2 = DependentCoordinates([LAT[:2], LON[:2]])
        assert c1 != c2

    def test_eq_dims(self):
        c1 = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])
        c2 = DependentCoordinates([LAT, LON], dims=['lon', 'lat'])
        assert c1 != c2

    def test_eq_coordinates(self):
        c1 = DependentCoordinates([LAT, LON])
        c2 = DependentCoordinates([LAT, LON])
        c3 = DependentCoordinates([LAT[::-1], LON])
        c4 = DependentCoordinates([LAT, LON[::-1]])
        
        assert c1 == c2
        assert c1 != c3
        assert c1 != c4

class TestDependentCoordinatesSerialization(object):
    def test_definition(self):
        c = DependentCoordinates([LAT, LON])
        d = c.definition
        
        assert isinstance(d, dict)
        json.dumps(d, cls=podpac.core.utils.JSONEncoder) # test serializable
        c2 = DependentCoordinates.from_definition(d)
        assert c2 == c

    def test_invalid_definition(self):
        with pytest.raises(ValueError, match='DependentCoordinates definition requires "values"'):
            DependentCoordinates.from_definition({'dims':['lat', 'lon']})

class TestDependentCoordinatesProperties(object):
    def test_size(self):
        c = DependentCoordinates([LAT, LON])
        assert c.size == 12

    def test_shape(self):
        c = DependentCoordinates([LAT, LON])
        assert c.shape == (3, 4)

    def test_coords(self):
        c = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])

        assert isinstance(c.coords, dict)
        x = xr.DataArray(np.empty(c.shape), dims=c.idims, coords=c.coords)
        assert x.dims == ('i', 'j')
        assert_equal(x.coords['i'], np.arange(c.shape[0]))
        assert_equal(x.coords['j'], np.arange(c.shape[1]))
        assert_equal(x.coords['lat'], c['lat'].coordinates)
        assert_equal(x.coords['lon'], c['lon'].coordinates)
        
        c = DependentCoordinates([LAT, LON])
        with pytest.raises(ValueError, match="Cannot get coords"):
            c.coords

    def test_bounds(self):
        c = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])
        bounds = c.bounds
        assert isinstance(bounds, dict)
        assert set(bounds.keys()) == set(c.udims)
        assert_equal(bounds['lat'], c['lat'].bounds)
        assert_equal(bounds['lon'], c['lon'].bounds)

        c = DependentCoordinates([LAT, LON])
        with pytest.raises(ValueError, match="Cannot get bounds"):
            c.bounds

    def test_area_bounds(self):
        c = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])
        area_bounds = c.area_bounds
        assert isinstance(area_bounds, dict)
        assert set(area_bounds.keys()) == set(c.udims)
        assert_equal(area_bounds['lat'], c['lat'].area_bounds)
        assert_equal(area_bounds['lon'], c['lon'].area_bounds)

        c = DependentCoordinates([LAT, LON])
        with pytest.raises(ValueError, match="Cannot get area_bounds"):
            c.area_bounds

class TestDependentCoordinatesIndexing(object):
    def test_get_dim(self):
        c = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])

        lat = c['lat']
        lon = c['lon']
        assert isinstance(lat, ArrayCoordinatesNd)
        assert isinstance(lon, ArrayCoordinatesNd)
        assert lat.name == 'lat'
        assert lon.name == 'lon'
        assert_equal(lat.coordinates, LAT)
        assert_equal(lon.coordinates, LON)
        
        with pytest.raises(KeyError, match="Cannot get dimension"):
            c['other']

    def test_get_dim_with_properties(self):
        c = DependentCoordinates(
            [LAT, LON],
            dims=['lat', 'lon'],
            ctypes=['left', 'right'],
            segment_lengths=[1.0, 2.0],
            coord_ref_sys='SPHER_MERC',
            units=[Units(), Units()])

        lat = c['lat']
        assert isinstance(lat, ArrayCoordinatesNd)
        assert lat.name == c.dims[0]
        assert lat.ctype == c.ctypes[0]
        assert lat.segment_lengths == c.segment_lengths[0]
        assert lat.units is c.units[0]
        assert lat.coord_ref_sys == c.coord_ref_sys
        assert lat.shape == c.shape
        repr(lat)

        lon = c['lon']
        assert isinstance(lon, ArrayCoordinatesNd)
        assert lon.name == c.dims[1]
        assert lon.ctype == c.ctypes[1]
        assert lon.segment_lengths == c.segment_lengths[1]
        assert lon.units is c.units[1]
        assert lon.coord_ref_sys == c.coord_ref_sys
        assert lon.shape == c.shape
        repr(lon)

        # rare
        assert c._properties_at(index=0) == c._properties_at(dim='lat')
        assert c._properties_at(index=1) == c._properties_at(dim='lon')

    def test_get_index(self):
        lat = np.linspace(0, 1, 60).reshape((5, 4, 3))
        lon = np.linspace(1, 2, 60).reshape((5, 4, 3))
        c = DependentCoordinates([lat, lon])

        I = [3, 1, 2]
        J = slice(1, 3)
        K = 1
        B = lat > 0.5

        # full
        c2 = c[I, J, K]
        assert isinstance(c2, DependentCoordinates)
        assert c2.shape == (3, 2)
        assert_equal(c2.coordinates[0], lat[I, J, K])
        assert_equal(c2.coordinates[1], lon[I, J, K])

        # partial/implicit
        c2 = c[I, J]
        assert isinstance(c2, DependentCoordinates)
        assert c2.shape == (3, 2, 3)
        assert_equal(c2.coordinates[0], lat[I, J])
        assert_equal(c2.coordinates[1], lon[I, J])

        # boolean
        c2 = c[B]
        assert isinstance(c2, StackedCoordinates)
        assert c2.shape == (30,)
        assert_equal(c2._coords[0].coordinates, lat[B])
        assert_equal(c2._coords[1].coordinates, lon[B])

    def test_get_index_with_properties(self):
        c = DependentCoordinates(
            [LAT, LON],
            dims=['lat', 'lon'],
            ctypes=['left', 'right'],
            segment_lengths=[1.0, 2.0],
            coord_ref_sys='SPHER_MERC',
            units=[Units(), Units()])

        c2 = c[[1, 2]]
        assert c2.dims == c.dims
        assert c2.ctypes == c.ctypes
        assert c2.segment_lengths == c.segment_lengths
        assert c2.units == c.units
        assert c2.coord_ref_sys == c.coord_ref_sys

    def test_iter(self):
        c = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])
        a, b = iter(c)
        assert a == c['lat']
        assert b == c['lon']

class TestDependentCoordinatesSelection(object):
    def test_select_single(self):
        c = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])

        # single dimension
        bounds = {'lat': [0.25, .55]}
        E0, E1 = [0, 1, 1, 1], [3, 0, 1, 2] # expected
        
        s = c.select(bounds)
        assert isinstance(s, StackedCoordinates)
        assert s == c[E0, E1]

        s, I = c.select(bounds, return_indices=True)
        assert isinstance(s, StackedCoordinates)
        assert s == c[I]
        assert_equal(I[0], E0)
        assert_equal(I[1], E1)

        # a different single dimension
        bounds = {'lon': [12.5, 17.5]}
        E0, E1 = [0, 1, 1, 1, 1, 2], [3, 0, 1, 2, 3, 0]
        
        s = c.select(bounds)
        assert isinstance(s, StackedCoordinates)
        assert s == c[E0, E1]

        s, I = c.select(bounds, return_indices=True)
        assert isinstance(s, StackedCoordinates)
        assert s == c[I]
        assert_equal(I[0], E0)
        assert_equal(I[1], E1)

        # outer
        bounds = {'lat': [0.25, .75]}
        E0, E1 = [0, 0, 1, 1, 1, 1, 2, 2], [2, 3, 0, 1, 2, 3, 0, 1]
        
        s = c.select(bounds, outer=True)
        assert isinstance(s, StackedCoordinates)
        assert s == c[E0, E1]

        s, I = c.select(bounds, outer=True, return_indices=True)
        assert isinstance(s, StackedCoordinates)
        assert s == c[E0, E1]
        assert_equal(I[0], E0)
        assert_equal(I[1], E1)

        # no matching dimension
        bounds = {'alt': [0, 10]}
        s = c.select(bounds)
        assert s == c

        s, I = c.select(bounds, return_indices=True)
        assert s == c[I]
        assert s == c

    def test_select_multiple(self):
        c = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])

        # this should be the AND of both intersections
        bounds = {'lat': [0.25, 0.95], 'lon': [10.5, 17.5]}
        E0, E1 = [0, 1, 1, 1, 1, 2], [3, 0, 1, 2, 3, 0]
        s = c.select(bounds)
        assert s == c[E0, E1]
        
        s, I = c.select(bounds, return_indices=True)
        assert s == c[E0, E1]
        assert_equal(I[0], E0)
        assert_equal(I[1], E1)

    def test_intersect(self):
        c = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])

        
        other_lat = ArrayCoordinates1d([0.25, 0.5, .95], name='lat')
        other_lon = ArrayCoordinates1d([10.5, 15, 17.5], name='lon')

        # single other
        E0, E1 = [0, 1, 1, 1, 1, 2, 2, 2], [3, 0, 1, 2, 3, 0, 1, 2]
        s = c.intersect(other_lat)
        assert s == c[E0, E1]

        s, I = c.intersect(other_lat, return_indices=True)
        assert s == c[E0, E1]
        assert s == c[I]
        
        E0, E1 = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2], [2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        s = c.intersect(other_lat, outer=True)
        assert s == c[E0, E1]

        E0, E1 = [0, 0, 0, 1, 1, 1, 1, 2], [1, 2, 3, 0, 1, 2, 3, 0]
        s = c.intersect(other_lon)
        assert s == c[E0, E1]

        # multiple, in various ways
        E0, E1 = [0, 1, 1, 1, 1, 2], [3, 0, 1, 2, 3, 0]
        
        other = StackedCoordinates([other_lat, other_lon])
        s = c.intersect(other)
        assert s == c[E0, E1]

        other = StackedCoordinates([other_lon, other_lat])
        s = c.intersect(other)
        assert s == c[E0, E1]

        from podpac.coordinates import Coordinates
        other = Coordinates([other_lat, other_lon])
        s = c.intersect(other)
        assert s == c[E0, E1]

        # full
        other = Coordinates(['2018-01-01'], dims=['time'])
        s = c.intersect(other)
        assert s == c

        s, I = c.intersect(other, return_indices=True)
        assert s == c
        assert s == c[I]

    def test_intersect_invalid(self):
        c = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])

        with pytest.raises(TypeError, match="Cannot intersect with type"):
            c.intersect({})

        with pytest.raises(ValueError, match="Cannot intersect mismatched dtypes"):
            c.intersect(ArrayCoordinates1d(['2018-01-01'], name='lat'))

class TestArrayCoordinatesNd(object):
    def test_unavailable(self):
        with pytest.raises(RuntimeError, match="ArrayCoordinatesNd from_definition is unavailable"):
            ArrayCoordinatesNd.from_definition({})

        with pytest.raises(RuntimeError, match="ArrayCoordinatesNd from_xarray is unavailable"):
            ArrayCoordinatesNd.from_xarray(xr.DataArray([]))
        
        a = ArrayCoordinatesNd([])

        with pytest.raises(RuntimeError, match="ArrayCoordinatesNd definition is unavailable"):
            a.definition

        with pytest.raises(RuntimeError, match="ArrayCoordinatesNd coords is unavailable"):
            a.coords

        with pytest.raises(RuntimeError, match="ArrayCoordinatesNd intersect is unavailable"):
            a.intersect(a)

        with pytest.raises(RuntimeError, match="ArrayCoordinatesNd select is unavailable"):
            a.select([0, 1])