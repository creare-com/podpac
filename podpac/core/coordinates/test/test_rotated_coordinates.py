
from datetime import datetime
import json

import pytest
import traitlets as tl
import numpy as np
import pandas as pd
import xarray as xr
from numpy.testing import assert_equal, assert_allclose

import podpac
from podpac.core.units import Units
from podpac.coordinates import ArrayCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates
from podpac.core.coordinates.dependent_coordinates import DependentCoordinates, ArrayCoordinatesNd
from podpac.core.coordinates.rotated_coordinates import RotatedCoordinates

class TestRotatedCoordinatesCreation(object):
    def test_init_step(self):
        # positive steps
        c = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])

        assert c.shape == (3, 4)
        assert c.theta == np.pi/4
        assert_equal(c.ulc, [10, 20])
        assert_equal(c.step, [1.0, 2.0])
        assert_allclose(c.lrc, [15.656854, 17.171573])
        assert c.dims == ('lat', 'lon')
        assert c.udims == ('lat', 'lon')
        assert c.idims == ('i', 'j')
        assert c.name == 'lat,lon'
        repr(c)

        # negative steps
        c = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[-1.0, -2.0], dims=['lat', 'lon'])
        assert c.shape == (3, 4)
        assert c.theta == np.pi/4
        assert_equal(c.ulc, [10, 20])
        assert_equal(c.step, [-1.0, -2.0])
        assert_allclose(c.lrc, [4.343146, 22.828427])
        assert c.dims == ('lat', 'lon')
        assert c.udims == ('lat', 'lon')
        assert c.idims == ('i', 'j')
        assert c.name == 'lat,lon'
        repr(c)

    def test_init_lrc(self):
        c = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], lrc=[15, 17], dims=['lat', 'lon'])
        assert c.shape == (3, 4)
        assert c.theta == np.pi/4
        assert_equal(c.ulc, [10, 20])
        assert_allclose(c.step, [0.70710678, 1.88561808])
        assert_allclose(c.lrc, [15., 17.])
        assert c.dims == ('lat', 'lon')
        assert c.udims == ('lat', 'lon')
        assert c.idims == ('i', 'j')
        assert c.name == 'lat,lon'
        repr(c)

    def test_thetas(self):
        c = RotatedCoordinates(shape=(3, 4), theta=0*np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        assert_allclose(c.lrc, [12.0, 14.0])

        c = RotatedCoordinates(shape=(3, 4), theta=1*np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        assert_allclose(c.lrc, [15.656854, 17.171573])

        c = RotatedCoordinates(shape=(3, 4), theta=2*np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        assert_allclose(c.lrc, [16.0, 22.0])

        c = RotatedCoordinates(shape=(3, 4), theta=3*np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        assert_allclose(c.lrc, [12.828427, 25.656854])

        c = RotatedCoordinates(shape=(3, 4), theta=4*np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        assert_allclose(c.lrc, [8.0, 26.0])

        c = RotatedCoordinates(shape=(3, 4), theta=5*np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        assert_allclose(c.lrc, [4.3431458, 22.828427])

        c = RotatedCoordinates(shape=(3, 4), theta=6*np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        assert_allclose(c.lrc, [4.0, 18.0])

        c = RotatedCoordinates(shape=(3, 4), theta=7*np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        assert_allclose(c.lrc, [7.1715729, 14.343146])

        c = RotatedCoordinates(shape=(3, 4), theta=8*np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        assert_allclose(c.lrc, [12.0, 14.0])

        c = RotatedCoordinates(shape=(3, 4), theta=-np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        assert_allclose(c.lrc, [7.1715729, 14.343146])

    def test_ctypes(self):
        c = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'],
                               ctypes=['left', 'right'], segment_lengths=1.0)
        repr(c)
        
    def test_invalid(self):
        with pytest.raises(ValueError, match="Invalid shape"):
            RotatedCoordinates(shape=(-3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])

        with pytest.raises(ValueError, match="Invalid shape"):
            RotatedCoordinates(shape=(3, 0), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])

        with pytest.raises(ValueError, match="Invalid step"):
            RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[0, 2.0], dims=['lat', 'lon'])

        with pytest.raises(ValueError, match="RotatedCoordinates dims"):
            RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'time'])

        with pytest.raises(ValueError, match="dims and coordinates size mismatch"):
            RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat'])

        with pytest.raises(ValueError, match="Duplicate dimension"):
            RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lat'])

    def test_copy(self):
        c = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        c2 = c.copy()
        assert c2 is not c
        assert c2 == c

class TestRotatedCoordinatesGeotransform(object):
    def test_geotransform(self):
        c = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        assert_allclose(c.geotransform, (10.0, 0.7071068, -1.4142136, 20.0, 0.7071068, 1.4142136))

        c2 = RotatedCoordinates.from_geotransform(c.geotransform, c.shape, dims=['lat', 'lon'])
        assert c == c2

class TestRotatedCoordinatesStandardMethods(object):
    def test_eq_type(self):
        c = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        assert c != []

    def test_eq_shape(self):
        c1 = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        c2 = RotatedCoordinates(shape=(4, 3), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        assert c1 != c2

    def test_eq_affine(self):
        c1 = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        c2 = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        c3 = RotatedCoordinates(shape=(3, 4), theta=np.pi/3, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        c4 = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[11, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        c5 = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.1, 2.0], dims=['lat', 'lon'])
        
        assert c1 == c2
        assert c1 != c3
        assert c1 != c4
        assert c1 != c5
    
    def test_eq_dims(self):
        c1 = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        c2 = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lon', 'lat'])
        assert c1 != c2

class TestRotatedCoordinatesSerialization(object):
    def test_definition(self):
        c = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        d = c.definition
        
        assert isinstance(d, dict)
        json.dumps(d, cls=podpac.core.utils.JSONEncoder) # test serializable
        c2 = RotatedCoordinates.from_definition(d)
        assert c2 == c

    def test_from_definition_lrc(self):
        c1 = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], lrc=[15, 17], dims=['lat', 'lon'])

        d = {'shape': (3, 4), 'theta': np.pi/4, 'ulc':[10, 20], 'lrc':[15, 17], 'dims':['lat', 'lon']}
        c2 = RotatedCoordinates.from_definition(d)
        
        assert c1 == c2  

    def test_invalid_definition(self):
        d = {'theta': np.pi/4, 'ulc':[10, 20], 'step':[1.0, 2.0], 'dims':['lat', 'lon']}
        with pytest.raises(ValueError, match='RotatedCoordinates definition requires "shape"'):
            RotatedCoordinates.from_definition(d)

        d = {'shape': (3, 4), 'ulc':[10, 20], 'step':[1.0, 2.0], 'dims':['lat', 'lon']}
        with pytest.raises(ValueError, match='RotatedCoordinates definition requires "theta"'):
            RotatedCoordinates.from_definition(d)

        d = {'shape': (3, 4), 'theta': np.pi/4, 'step':[1.0, 2.0], 'dims':['lat', 'lon']}
        with pytest.raises(ValueError, match='RotatedCoordinates definition requires "ulc"'):
            RotatedCoordinates.from_definition(d)

        d = {'shape': (3, 4), 'theta': np.pi/4, 'ulc':[10, 20], 'dims':['lat', 'lon']}
        with pytest.raises(ValueError, match='RotatedCoordinates definition requires "step" or "lrc"'):
            RotatedCoordinates.from_definition(d)

        d = {'shape': (3, 4), 'theta': np.pi/4, 'ulc':[10, 20], 'step':[1.0, 2.0]}
        with pytest.raises(ValueError, match='RotatedCoordinates definition requires "dims"'):
            RotatedCoordinates.from_definition(d)

    def test_full_definition(self):
        c = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        d = c.full_definition
        
        assert isinstance(d, dict)
        assert set(d.keys()) == {'dims', 'shape', 'theta', 'ulc', 'step', 'ctypes', 'segment_lengths', 'units', 'crs'}
        json.dumps(d, cls=podpac.core.utils.JSONEncoder) # test serializable

class TestRotatedCoordinatesProperties(object):
    def test_affine(self):
        c = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        R = c.affine
        assert_allclose([R.a, R.b, R.c, R.d, R.e, R.f], [0.70710678, -1.41421356, 10.0, 0.70710678, 1.41421356, 20.0])
        
    def test_coordinates(self):
        c = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        lat, lon = c.coordinates

        assert_allclose(
            lat,
            [[10.        , 11.41421356, 12.82842712, 14.24264069],
             [10.70710678, 12.12132034, 13.53553391, 14.94974747],
             [11.41421356, 12.82842712, 14.24264069, 15.65685425]])

        assert_allclose(
            lon,
            [[20.        , 18.58578644, 17.17157288, 15.75735931],
             [20.70710678, 19.29289322, 17.87867966, 16.46446609],
             [21.41421356, 20.        , 18.58578644, 17.17157288]])


class TestRotatedCoordinatesIndexing(object):
    def test_get_dim(self):
        c = RotatedCoordinates(shape=(3, 4), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])

        lat = c['lat']
        lon = c['lon']
        assert isinstance(lat, ArrayCoordinatesNd)
        assert isinstance(lon, ArrayCoordinatesNd)
        assert lat.name == 'lat'
        assert lon.name == 'lon'
        assert_equal(lat.coordinates, c.coordinates[0])
        assert_equal(lon.coordinates, c.coordinates[1])
        
        with pytest.raises(KeyError, match="Cannot get dimension"):
            c['other']

    def test_get_index_slices(self):
        c = RotatedCoordinates(shape=(5, 7), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])

        # full
        c2 = c[1:4, 2:4]
        assert isinstance(c2, RotatedCoordinates)
        assert c2.shape == (3, 2)
        assert c2.theta == c.theta
        assert_allclose(c2.ulc, c.coordinates[0][1, 2], c.coordinates[1][1, 2])
        assert_allclose(c2.lrc, c.coordinates[0][3, 3], c.coordinates[1][3, 3])
        assert c2.dims == c.dims
        assert_allclose(c2.coordinates[0], c.coordinates[0][1:4, 2:4])
        assert_allclose(c2.coordinates[1], c.coordinates[1][1:4, 2:4])

        # partial/implicit
        c2 = c[1:4]
        assert isinstance(c2, RotatedCoordinates)
        assert c2.shape == (3, 7)
        assert c2.theta == c.theta
        assert_allclose(c2.ulc, c.coordinates[0][1, 0], c.coordinates[0][1, 0])
        assert_allclose(c2.lrc, c.coordinates[0][3, -1], c.coordinates[0][3, -1])
        assert c2.dims == c.dims
        assert_allclose(c2.coordinates[0], c.coordinates[0][1:4])
        assert_allclose(c2.coordinates[1], c.coordinates[1][1:4])

        # stepped
        c2 = c[1:4:2, 2:4]
        assert isinstance(c2, RotatedCoordinates)
        assert c2.shape == (2, 2)
        assert c2.theta == c.theta
        assert_allclose(c2.ulc, c.coordinates[0][1, 2], c.coordinates[0][1, 2])
        assert_allclose(c2.lrc, c.coordinates[0][3, 3], c.coordinates[0][3, 3])
        assert c2.dims == c.dims
        assert_allclose(c2.coordinates[0], c.coordinates[0][1:4:2, 2:4])
        assert_allclose(c2.coordinates[1], c.coordinates[1][1:4:2, 2:4])

        # reversed
        c2 = c[4:1:-1, 2:4]
        assert isinstance(c2, RotatedCoordinates)
        assert c2.shape == (3, 2)
        assert c2.theta == c.theta
        assert_allclose(c2.ulc, c.coordinates[0][1, 2], c.coordinates[0][1, 2])
        assert_allclose(c2.lrc, c.coordinates[0][3, 3], c.coordinates[0][3, 3])
        assert c2.dims == c.dims
        assert_allclose(c2.coordinates[0], c.coordinates[0][4:1:-1, 2:4])
        assert_allclose(c2.coordinates[1], c.coordinates[1][4:1:-1, 2:4])

    def test_get_index_fallback(self):
        c = RotatedCoordinates(shape=(5, 7), theta=np.pi/4, ulc=[10, 20], step=[1.0, 2.0], dims=['lat', 'lon'])
        lat, lon = c.coordinates

        I = [3, 1]
        J = slice(1, 4)
        B = lat > 12

        # int/slice/indices
        c2 = c[I, J]
        assert isinstance(c2, DependentCoordinates)
        assert c2.shape == (2, 3)
        assert c2.dims == c.dims
        assert_equal(c2['lat'].coordinates, lat[I, J])
        assert_equal(c2['lon'].coordinates, lon[I, J])

        # boolean
        c2 = c[B]
        assert isinstance(c2, StackedCoordinates)
        assert c2.shape == (31,)
        assert c2.dims == c.dims
        assert_equal(c2['lat'].coordinates, lat[B])
        assert_equal(c2['lon'].coordinates, lon[B])

# class TestRotatedCoordinatesSelection(object):
#     def test_select_single(self):
#         c = RotatedCoordinates([LAT, LON], dims=['lat', 'lon'])

#         # single dimension
#         bounds = {'lat': [0.25, .55]}
#         E0, E1 = [0, 1, 1, 1], [3, 0, 1, 2] # expected
        
#         s = c.select(bounds)
#         assert isinstance(s, StackedCoordinates)
#         assert s == c[E0, E1]

#         s, I = c.select(bounds, return_indices=True)
#         assert isinstance(s, StackedCoordinates)
#         assert s == c[I]
#         assert_equal(I[0], E0)
#         assert_equal(I[1], E1)

#         # a different single dimension
#         bounds = {'lon': [12.5, 17.5]}
#         E0, E1 = [0, 1, 1, 1, 1, 2], [3, 0, 1, 2, 3, 0]
        
#         s = c.select(bounds)
#         assert isinstance(s, StackedCoordinates)
#         assert s == c[E0, E1]

#         s, I = c.select(bounds, return_indices=True)
#         assert isinstance(s, StackedCoordinates)
#         assert s == c[I]
#         assert_equal(I[0], E0)
#         assert_equal(I[1], E1)

#         # outer
#         bounds = {'lat': [0.25, .75]}
#         E0, E1 = [0, 0, 1, 1, 1, 1, 2, 2], [2, 3, 0, 1, 2, 3, 0, 1]
        
#         s = c.select(bounds, outer=True)
#         assert isinstance(s, StackedCoordinates)
#         assert s == c[E0, E1]

#         s, I = c.select(bounds, outer=True, return_indices=True)
#         assert isinstance(s, StackedCoordinates)
#         assert s == c[E0, E1]
#         assert_equal(I[0], E0)
#         assert_equal(I[1], E1)

#         # no matching dimension
#         bounds = {'alt': [0, 10]}
#         s = c.select(bounds)
#         assert s == c

#         s, I = c.select(bounds, return_indices=True)
#         assert s == c[I]
#         assert s == c

#     def test_select_multiple(self):
#         c = RotatedCoordinates([LAT, LON], dims=['lat', 'lon'])

#         # this should be the AND of both intersections
#         bounds = {'lat': [0.25, 0.95], 'lon': [10.5, 17.5]}
#         E0, E1 = [0, 1, 1, 1, 1, 2], [3, 0, 1, 2, 3, 0]
#         s = c.select(bounds)
#         assert s == c[E0, E1]
        
#         s, I = c.select(bounds, return_indices=True)
#         assert s == c[E0, E1]
#         assert_equal(I[0], E0)
#         assert_equal(I[1], E1)

#     def test_intersect(self):
#         c = RotatedCoordinates([LAT, LON], dims=['lat', 'lon'])

        
#         other_lat = ArrayCoordinates1d([0.25, 0.5, .95], name='lat')
#         other_lon = ArrayCoordinates1d([10.5, 15, 17.5], name='lon')

#         # single other
#         E0, E1 = [0, 1, 1, 1, 1, 2, 2, 2], [3, 0, 1, 2, 3, 0, 1, 2]
#         s = c.intersect(other_lat)
#         assert s == c[E0, E1]

#         s, I = c.intersect(other_lat, return_indices=True)
#         assert s == c[E0, E1]
#         assert s == c[I]
        
#         E0, E1 = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2], [2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
#         s = c.intersect(other_lat, outer=True)
#         assert s == c[E0, E1]

#         E0, E1 = [0, 0, 0, 1, 1, 1, 1, 2], [1, 2, 3, 0, 1, 2, 3, 0]
#         s = c.intersect(other_lon)
#         assert s == c[E0, E1]

#         # multiple, in various ways
#         E0, E1 = [0, 1, 1, 1, 1, 2], [3, 0, 1, 2, 3, 0]
        
#         other = StackedCoordinates([other_lat, other_lon])
#         s = c.intersect(other)
#         assert s == c[E0, E1]

#         other = StackedCoordinates([other_lon, other_lat])
#         s = c.intersect(other)
#         assert s == c[E0, E1]

#         from podpac.coordinates import Coordinates
#         other = Coordinates([other_lat, other_lon])
#         s = c.intersect(other)
#         assert s == c[E0, E1]

#         # full
#         other = Coordinates(['2018-01-01'], dims=['time'])
#         s = c.intersect(other)
#         assert s == c

#         s, I = c.intersect(other, return_indices=True)
#         assert s == c
#         assert s == c[I]

#     def test_intersect_invalid(self):
#         c = RotatedCoordinates([LAT, LON], dims=['lat', 'lon'])

#         with pytest.raises(TypeError, match="Cannot intersect with type"):
#             c.intersect({})

#         with pytest.raises(ValueError, match="Cannot intersect mismatched dtypes"):
#             c.intersect(ArrayCoordinates1d(['2018-01-01'], name='lat'))