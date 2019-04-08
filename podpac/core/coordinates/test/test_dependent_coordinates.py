
from datetime import datetime
import json

import pytest
import traitlets as tl
import numpy as np
import pandas as pd
import xarray as xr
from numpy.testing import assert_equal

import podpac
from podpac.core.coordinates.dependent_coordinates import DependentCoordinates
from podpac.core.units import Units

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

    def test_invalid(self):
        # mismatched shape
        with pytest.raises(ValueError, match="coordinates shape mismatch"):
            DependentCoordinates((LAT, LON.reshape((4, 3))), dims=['lat', 'lon'])

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
        # check when setting
        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'])
        c._set_name('lat,lon')

        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'])
        with pytest.raises(ValueError, match="Dimension mismatch"):
            c._set_name('lon,lat')

    def test_ctype_and_segment_lengths(self):
        # explicit
        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'], ctypes=['left', 'right'], segment_lengths=[1.0, 2.0])
        assert c.ctypes == ('left', 'right')
        assert c.segment_lengths == (1.0, 2.0)

        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'], ctypes=['point', 'point'])
        assert c.ctypes == ('point', 'point')
        assert c.segment_lengths == (None, None)

        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'], ctypes=['midpoint', 'point'], segment_lengths=[1.0, None])
        assert c.ctypes == ('midpoint', 'point')
        assert c.segment_lengths == (1.0, None)

        # single value
        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'], ctypes='left', segment_lengths=1.0)
        assert c.ctypes == ('left', 'left')
        assert c.segment_lengths == (1.0, 1.0)

        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'], ctypes='point')
        assert c.ctypes == ('point', 'point')
        assert c.segment_lengths == (None, None)
        
        # defaults
        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'])
        assert c.ctypes == ('point', 'point')

        # don't overwrite
        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'], ctypes='left', segment_lengths=1.0)
        c._set_ctype('right')
        assert c.ctypes == ('left', 'left')

        # size mismatch
        with pytest.raises(ValueError, match='size mismatch'):
            DependentCoordinates((LAT, LON), dims=['lat', 'lon'], ctypes=['left', 'left', 'left'], segment_lengths=1.0)

        with pytest.raises(ValueError, match='segment_lengths and coordinates size mismatch'):
            DependentCoordinates((LAT, LON), dims=['lat', 'lon'], ctypes='left', segment_lengths=[1.0, 1.0, 1.0])

        # segment lengths required
        with pytest.raises(TypeError, match='segment_lengths cannot be None'):
            DependentCoordinates((LAT, LON), dims=['lat', 'lon'], ctypes='left')
        
        with pytest.raises(TypeError, match='segment_lengths cannot be None'):
            DependentCoordinates((LAT, LON), dims=['lat', 'lon'], ctypes=['left', 'point'])
        
        with pytest.raises(TypeError, match='segment_lengths cannot be None'):
            DependentCoordinates((LAT, LON), dims=['lat', 'lon'], ctypes=['left', 'point'], segment_lengths=[None, None])

        # segment lengths prohibited
        with pytest.raises(TypeError, match='segment_lengths must be None'):
            DependentCoordinates((LAT, LON), dims=['lat', 'lon'], segment_lengths=1.0)
        
        with pytest.raises(TypeError, match='segment_lengths must be None'):
            DependentCoordinates((LAT, LON), dims=['lat', 'lon'], ctypes='point', segment_lengths=1.0)
        
        with pytest.raises(TypeError, match='segment_lengths must be None'):
            DependentCoordinates((LAT, LON), dims=['lat', 'lon'], ctypes=['left', 'point'], segment_lengths=[1.0, 1.0])

        # invalid
        with pytest.raises(tl.TraitError):
            DependentCoordinates((LAT, LON), dims=['lat', 'lon'], ctypes='abc')
        
        # invalid segment_lengths
        with pytest.raises(ValueError):
            DependentCoordinates((LAT, LON), dims=['lat', 'lon'], ctypes='left', segment_lengths='abc')

        with pytest.raises(ValueError, match="segment_lengths must be positive"):
            DependentCoordinates((LAT, LON), dims=['lat', 'lon'], ctypes=['left', 'right'], segment_lengths=[1.0, -2.0])

    def test_coord_ref_sys(self):
        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'], coord_ref_sys='SPHER_MERC')
        assert c.coord_ref_sys == 'SPHER_MERC'

        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'])
        assert c.coord_ref_sys == 'WGS84'

        # check when setting
        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'], coord_ref_sys='SPHER_MERC')
        c._set_coord_ref_sys('SPHER_MERC')

        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'], coord_ref_sys='SPHER_MERC')
        with pytest.raises(ValueError, match="coord_ref_sys mismatch"):
            c._set_coord_ref_sys('WGS84')

    def test_units(self):
        ua = Units()
        ub = Units()

        # explicit
        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'], units=[ua, ub])
        assert len(c.units) == 2
        assert c.units[0] is ua
        assert c.units[1] is ub

        # single value
        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'], units=ua)
        assert len(c.units) == 2
        assert c.units[0] is ua
        assert c.units[1] is ua

        # don't overwrite
        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'], units=ua)
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
        c = DependentCoordinates((LAT, LON), dims=['lat', 'lon'])

        c2 = c.copy()
        assert c2 is not c
        assert c2 == c

class TestDependentCoordinatesStandardMethods(object):
    def test_eq_type(self):
        c = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])
        assert c != [[0, 1, 2], [10, 20, 30]]

    def test_eq_shape_shortcut(self):
        c1 = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])
        c2 = DependentCoordinates([LAT[:2], LON[:2]], dims=['lat', 'lon'])
        assert c1 != c2

    def test_eq_dims_shortcut(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c1 = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])
        c2 = DependentCoordinates([LAT, LON], dims=['lon', 'lat'])
        assert c1 != c2

    def test_eq_coordinates(self):
        c1 = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])
        c2 = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])
        c3 = DependentCoordinates([LAT[::-1], LON], dims=['lat', 'lon'])
        c4 = DependentCoordinates([LAT, LON[::-1]], dims=['lat', 'lon'])
        
        assert c1 == c2
        assert c1 != c3
        assert c1 != c4

    def test_iter(self):
        c = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])
        a, b = iter(c)
        assert a == c['lat']
        assert b == c['lon']

class TestDependentCoordinatesSerialization(object):
    def test_definition(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])
        d = c.definition
        
        assert isinstance(d, dict)
        json.dumps(d, cls=podpac.core.utils.JSONEncoder) # test serializable
        c2 = DependentCoordinates.from_definition(d)
        assert c2 == c

    def test_invalid_definition(self):
        with pytest.raises(ValueError, match='DependentCoordinates definition requires "dims"'):
            DependentCoordinates.from_definition({'values': [0, 1]})

        with pytest.raises(ValueError, match='DependentCoordinates definition requires "values"'):
            DependentCoordinates.from_definition({'dims':['lat', 'lon']})

class TestStackedCoordinatesProperties(object):
    def test_size(self):
        c = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])
        assert c.size == 12

    def test_shaped(self):
        c = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])
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

    def test_bounds(self):
        c = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])
        bounds = c.bounds
        assert isinstance(bounds, dict)
        assert set(bounds.keys()) == set(c.udims)
        assert_equal(bounds['lat'], c['lat'].bounds)
        assert_equal(bounds['lon'], c['lon'].bounds)

    def test_area_bounds(self):
        c = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])
        area_bounds = c.area_bounds
        assert isinstance(area_bounds, dict)
        assert set(area_bounds.keys()) == set(c.udims)
        assert_equal(area_bounds['lat'], c['lat'].area_bounds)
        assert_equal(area_bounds['lon'], c['lon'].area_bounds)

class TestStackedCoordinatesIndexing(object):
    def test_get_dim(self):
        c = DependentCoordinates([LAT, LON], dims=['lat', 'lon'])

        assert_equal(c['lat'].coordinates, LAT)
        assert_equal(c['lon'].coordinates, LON)
        
        with pytest.raises(KeyError, match="Cannot get dimension"):
            c['other']

#     def test_get_index(self):
#         lat = ArrayCoordinates1d([0, 1, 2, 3], name='lat')
#         lon = ArrayCoordinates1d([10, 20, 30, 40], name='lon')
#         time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'], name='time')
#         c = StackedCoordinates([lat, lon, time])

#         # integer index
#         I = 0
#         cI = c[I]
#         assert isinstance(cI, StackedCoordinates)
#         assert cI.size == 1
#         assert cI.dims == c.dims
#         assert_equal(cI['lat'].coordinates, c['lat'].coordinates[I])

#         # index array
#         I = [1, 2]
#         cI = c[I]
#         assert isinstance(cI, StackedCoordinates)
#         assert cI.size == 2
#         assert cI.dims == c.dims
#         assert_equal(cI['lat'].coordinates, c['lat'].coordinates[I])

#         # boolean array
#         I = [False, True, True, False]
#         cI = c[I]
#         assert isinstance(cI, StackedCoordinates)
#         assert cI.size == 2
#         assert cI.dims == c.dims
#         assert_equal(cI['lat'].coordinates, c['lat'].coordinates[I])

#         # slice
#         cI = c[1:3]
#         assert isinstance(cI, StackedCoordinates)
#         assert cI.size == 2
#         assert cI.dims == c.dims
#         assert_equal(cI['lat'].coordinates, c['lat'].coordinates[1:3])

#     def test_iter(self):
#         lat = ArrayCoordinates1d([0, 1, 2, 3])
#         lon = ArrayCoordinates1d([10, 20, 30, 40])
#         time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'])
#         c = StackedCoordinates([lat, lon, time])

#         for item in c:
#             assert isinstance(item, Coordinates1d)

#     def test_len(self):
#         lat = ArrayCoordinates1d([0, 1, 2, 3])
#         lon = ArrayCoordinates1d([10, 20, 30, 40])
#         time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'])
#         c = StackedCoordinates([lat, lon, time])

#         assert len(c) == 3

# class TestStackedCoordinatesSelection(object):
#     def test_select_single(self):
#         lat = ArrayCoordinates1d([0, 1, 2, 3], name='lat')
#         lon = ArrayCoordinates1d([10, 20, 30, 40], name='lon')
#         time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'], name='time')
#         c = StackedCoordinates([lat, lon, time])

#         # single dimension
#         s = c.select({'lat': [0.5, 2.5]})
#         assert s == c[1:3]

#         s, I = c.select({'lat': [0.5, 2.5]}, return_indices=True)
#         assert s == c[I]
#         assert s == c[1:3]

#         # a different single dimension
#         s = c.select({'lon': [5, 25]})
#         assert s == c[0:2]

#         s, I = c.select({'lon': [5, 25]}, return_indices=True)
#         assert s == c[I]
#         assert s == c[0:2]

#         # outer
#         s = c.select({'lat': [0.5, 2.5]}, outer=True)
#         assert s == c[0:4]

#         s, I = c.select({'lat': [0.5, 2.5]}, outer=True, return_indices=True)
#         assert s == c[I]
#         assert s == c[0:4]

#         # no matching dimension
#         s = c.select({'alt': [0, 10]})
#         assert s == c

#         s, I = c.select({'alt': [0, 10]}, return_indices=True)
#         assert s == c[I]
#         assert s == c

#     def test_select_multiple(self):
#         lat = ArrayCoordinates1d([0, 1, 2, 3, 4, 5], name='lat')
#         lon = ArrayCoordinates1d([10, 20, 30, 40, 50, 60], name='lon')
#         c = StackedCoordinates([lat, lon])

#         # this should be the AND of both intersections
#         slat = c.select({'lat': [0.5, 3.5]})
#         slon = c.select({'lon': [25, 55]})
#         s = c.select({'lat': [0.5, 3.5], 'lon': [25, 55]})
#         assert slat == c[1:4]
#         assert slon == c[2:5]
#         assert s == c[2:4]
        
#         s, I = c.select({'lat': [0.5, 3.5], 'lon': [25, 55]}, return_indices=True)
#         assert s == c[2:4]
#         assert s == c[I]

#     def test_intersect(self):
#         lat = ArrayCoordinates1d([0, 1, 2, 3, 4, 5], name='lat')
#         lon = ArrayCoordinates1d([10, 20, 30, 40, 50, 60], name='lon')
#         c = StackedCoordinates([lat, lon])

#         other_lat = ArrayCoordinates1d([0.5, 2.5, 3.5], name='lat')
#         other_lon = ArrayCoordinates1d([25, 35, 55], name='lon')

#         # single other
#         s = c.intersect(other_lat)
#         assert s == c[1:4]

#         s = c.intersect(other_lat, outer=True)
#         assert s == c[0:5]

#         s, I = c.intersect(other_lat, return_indices=True)
#         assert s == c[1:4]
#         assert s == c[I]

#         s = c.intersect(other_lon)
#         assert s == c[2:5]

#         # stacked other
#         other = StackedCoordinates([other_lat, other_lon])
#         s = c.intersect(other)
#         assert s == c[2:4]

#         other = StackedCoordinates([other_lon, other_lat])
#         s = c.intersect(other)
#         assert s == c[2:4]

#         # coordinates other
#         from podpac.coordinates import Coordinates
#         other = Coordinates([other_lat, other_lon])
#         s = c.intersect(other)
#         assert s == c[2:4]

#     def test_intersect_multiple(self):
#         lat = ArrayCoordinates1d([0, 1, 2, 3, 4, 5], name='lat')
#         lon = ArrayCoordinates1d([10, 20, 30, 40, 50, 60], name='lon')
#         c = StackedCoordinates([lat, lon])

