
from datetime import datetime

import pytest
import traitlets as tl
import numpy as np
import xarray as xr
from numpy.testing import assert_equal

from podpac.core.units import Units
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates
from podpac.core.coordinates.coordinates import Coordinates

class TestArrayCoordinatesCreation(object):
    def test_empty(self):
        c = ArrayCoordinates1d([])
        a = np.array([], dtype=float)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([np.nan, np.nan]))
        assert c.size == 0
        assert c.dtype is None
        assert c.is_monotonic is None
        assert c.is_descending is None
        assert c.is_uniform is None

    def test_numerical_singleton(self):
        a = np.array([10], dtype=float)
        c = ArrayCoordinates1d(10)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([10.0, 10.0]))
        assert c.size == 1
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending is None
        assert c.is_uniform == True

    def test_numerical_array(self):
        # unsorted
        values = [1, 6, 0, 4.]
        a = np.array(values, dtype=float)
        c = ArrayCoordinates1d(a)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([0., 6.]))
        assert c.size == 4
        assert c.dtype == float
        assert c.is_monotonic == False
        assert c.is_descending is None
        assert c.is_uniform == False

        # sorted ascending
        values = [0, 1, 4, 6]
        a = np.array(values, dtype=float)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([0., 6.]))
        assert c.size == 4
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == False

        # sorted descending
        values = [6, 4, 1, 0]
        a = np.array(values, dtype=float)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([0., 6.]))
        assert c.size == 4
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == False

        # uniform ascending
        values = [0, 2, 4, 6]
        a = np.array(values, dtype=float)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([0., 6.]))
        assert c.size == 4
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # uniform descending
        values = [6, 4, 2, 0]
        a = np.array(values, dtype=float)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([0., 6.]))
        assert c.size == 4
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_datetime_singleton(self):
        a = np.array('2018-01-01').astype(np.datetime64)
        c = ArrayCoordinates1d('2018-01-01')
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(['2018-01-01', '2018-01-01']).astype(np.datetime64))
        assert c.size == 1
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending is None
        assert c.is_uniform == True

    def test_datetime_array(self):
        # unsorted
        values = ['2018-01-01', '2019-01-01', '2017-01-01', '2018-01-02']
        a = np.array(values).astype(np.datetime64)
        c = ArrayCoordinates1d(values, ctype='point')
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        assert c.size == 4
        assert c.dtype == np.datetime64
        assert c.is_monotonic == False
        assert c.is_descending is None
        assert c.is_uniform == False

        # sorted ascending
        values = ['2017-01-01', '2018-01-01', '2018-01-02', '2019-01-01']
        a = np.array(values).astype(np.datetime64)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        assert c.size == 4
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == False

        # sorted descending
        values = ['2019-01-01', '2018-01-02', '2018-01-01', '2017-01-01']
        a = np.array(values).astype(np.datetime64)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        assert c.size == 4
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == False

        # uniform ascending
        values = ['2017-01-01', '2018-01-01', '2019-01-01']
        a = np.array(values).astype(np.datetime64)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        assert c.size == 3
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # uniform descending
        values = ['2019-01-01', '2018-01-01', '2017-01-01']
        a = np.array(values).astype(np.datetime64)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        assert c.size == 3
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_invalid_coords(self):
        c = ArrayCoordinates1d([])
        
        with pytest.raises(ValueError):
            c.coords = np.array([1, 2, 3])

        with pytest.raises(tl.TraitError):
            c.coords = np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_from_xarray(self):
        # numerical
        x = xr.DataArray([0, 1, 2], name='lat')
        c = ArrayCoordinates1d.from_xarray(x, ctype='point')
        assert c.name == 'lat'
        assert c.ctype == 'point'
        assert_equal(c.coordinates, x.data)

        # datetime
        x = xr.DataArray([np.datetime64('2018-01-01'), np.datetime64('2018-01-02')], name='time')
        c = ArrayCoordinates1d.from_xarray(x, ctype='point')
        assert c.name == 'time'
        assert c.ctype == 'point'
        assert_equal(c.coordinates, x.data)

        # unnamed
        x = xr.DataArray([0, 1, 2])
        c = ArrayCoordinates1d.from_xarray(x)
        assert c.name is None

    def test_copy(self):
        c = ArrayCoordinates1d([1, 2, 3], ctype='point', name='lat')
        c2 = c.copy()
        assert c2.name == 'lat'
        assert c2.ctype == 'point'
        assert_equal(c2.coordinates, c.coordinates)

        c3 = c.copy(name='lon', ctype='left')
        assert c3.name == 'lon'
        assert c3.ctype == 'left'
        assert_equal(c3.coordinates, c.coordinates)

    def test_name(self):
        ArrayCoordinates1d([])
        ArrayCoordinates1d([], name='lat')
        ArrayCoordinates1d([], name='lon')
        ArrayCoordinates1d([], name='alt')
        ArrayCoordinates1d([], name='time')

        with pytest.raises(tl.TraitError):
            ArrayCoordinates1d([], name='depth')

    def test_extents(self):
        c = ArrayCoordinates1d([])
        assert c.extents is None

        # numerical
        c = ArrayCoordinates1d([1, 2], extents=[0.5, 2.5])
        assert_equal(c.extents, np.array([0.5, 2.5], dtype=float))

        # datetime
        c = ArrayCoordinates1d(['2018-02-01', '2019-02-01'], extents=['2018-01-01', '2019-03-01'])
        assert_equal(c.extents, np.array(['2018-01-01', '2019-03-01']).astype(np.datetime64))

        # invalid (ctype=point)
        with pytest.raises(TypeError):
            ArrayCoordinates1d([1, 2], ctype='point', extents=[0.5, 2.5])

        # invalid (wrong dtype)
        with pytest.raises(ValueError):
            ArrayCoordinates1d(['2018-02-01', '2019-02-01'], extents=[0.5, 2.5])
        
        with pytest.raises(ValueError):
            ArrayCoordinates1d([1, 2], extents=['2018-01-01', '2019-03-01'])

        # invalid (shape)
        with pytest.raises(ValueError):
            ArrayCoordinates1d([1, 2], extents=[0.5])

class TestArrayCoordinatesDefinition(object):
    def test_from_definition(self):
        # numerical
        d = {
            'values': [0, 1, 2],
            'name': 'lat',
            'ctype': 'point'
        }
        c = ArrayCoordinates1d.from_definition(d)
        assert c.name == 'lat'
        assert c.ctype == 'point'
        assert_equal(c.coordinates, [0, 1, 2])

        # datetime
        d = {
            'values': ['2018-01-01', '2018-01-02'],
            'name': 'time',
            'ctype': 'point'
        }
        c = ArrayCoordinates1d.from_definition(d)
        assert c.name == 'time'
        assert c.ctype == 'point'
        assert_equal(c.coordinates, np.array(['2018-01-01', '2018-01-02']).astype(np.datetime64))

        # incorrect definition
        d = {'coords': [0, 1, 2]}
        with pytest.raises(ValueError, match='ArrayCoordinates1d definition requires "values" property'):
            ArrayCoordinates1d.from_definition(d)

    def test_definition(self):
        # numerical
        c = ArrayCoordinates1d([0, 1, 2], name="lat", ctype="point")
        d = c.definition
        assert isinstance(d, dict)
        assert_equal(d['values'], c.coordinates)
        assert d['name'] == c.name
        assert d['ctype'] == c.ctype

        c2 = ArrayCoordinates1d.from_definition(d)
        assert c2.name == c.name
        assert c2.ctype == c.ctype
        assert_equal(c2.coordinates, c.coordinates)

        # datetimes
        c = ArrayCoordinates1d(['2018-01-01', '2018-01-02'], name="lat", ctype="point")
        d = c.definition
        assert isinstance(d, dict)
        assert_equal(d['values'], c.coordinates.astype(str))
        assert d['name'] == c.name
        assert d['ctype'] == c.ctype

        c2 = ArrayCoordinates1d.from_definition(d)
        assert c2.name == c.name
        assert c2.ctype == c.ctype
        assert_equal(c2.coordinates, c.coordinates)

class TestArrayCoordinatesProperties(object):
    def test_properties(self):
        c = ArrayCoordinates1d([])
        assert isinstance(c.properties, dict)
        assert set(c.properties.keys()) == set(['ctype', 'coord_ref_sys'])

        c = ArrayCoordinates1d([], name='lat')
        assert isinstance(c.properties, dict)
        assert set(c.properties.keys()) == set(['ctype', 'coord_ref_sys', 'name'])

        c = ArrayCoordinates1d([], units=Units())
        assert isinstance(c.properties, dict)
        assert set(c.properties.keys()) == set(['ctype', 'coord_ref_sys', 'units'])

        c = ArrayCoordinates1d([], extents=[0, 1])
        assert isinstance(c.properties, dict)
        assert set(c.properties.keys()) == set(['ctype', 'coord_ref_sys', 'extents'])

    def test_dims(self):
        c = ArrayCoordinates1d([], name='lat')
        assert c.dims == ['lat']
        assert c.udims == ['lat']

        c = ArrayCoordinates1d([])
        with pytest.raises(TypeError, match="cannot access dims property of unnamed Coordinates1d"):
            c.dims

        with pytest.raises(TypeError, match="cannot access dims property of unnamed Coordinates1d"):
            c.udims

    def test_area_bounds_point(self):
        # numerical
        values = [0.0, 1.0, 4.0, 6.0]
        c = ArrayCoordinates1d(values, ctype='point')
        assert_equal(c.area_bounds, np.array([0.0, 6.0], dtype=float))
        c = ArrayCoordinates1d(values[::-1], ctype='point')
        assert_equal(c.area_bounds, np.array([0.0, 6.0], dtype=float))

        # datetime
        values = ['2017-01-01', '2017-01-02', '2018-01-01', '2019-01-01']
        c = ArrayCoordinates1d(values, ctype='point')
        assert_equal(c.area_bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        c = ArrayCoordinates1d(values[::-1], ctype='point')
        assert_equal(c.area_bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))

    def test_area_bounds_explicit_extents(self):
        # numerical
        values = [0.0, 1.0, 4.0, 6.0]
        c = ArrayCoordinates1d(values, extents=[-10, 10])
        assert_equal(c.area_bounds, np.array([-10.0, 10.0], dtype=float))
        c = ArrayCoordinates1d(values[::-1], extents=[-10, 10])
        assert_equal(c.area_bounds, np.array([-10.0, 10.0], dtype=float))

        # datetime
        values = ['2017-01-01', '2018-01-01', '2018-01-02', '2019-01-01']
        c = ArrayCoordinates1d(values, extents=['2016-01-01', '2021-01-01'])
        assert_equal(c.area_bounds, np.array(['2016-01-01', '2021-01-01']).astype(np.datetime64))
        c = ArrayCoordinates1d(values[::-1], extents=['2016-01-01', '2021-01-01'])
        assert_equal(c.area_bounds, np.array(['2016-01-01', '2021-01-01']).astype(np.datetime64))

    def test_area_bounds_left(self):
        # numerical
        values = [0.0, 1.0, 4.0, 6.0]
        c = ArrayCoordinates1d(values, ctype='left')
        assert_equal(c.area_bounds, np.array([0.0, 8.0], dtype=float))
        c = ArrayCoordinates1d(values[::-1], ctype='left')
        assert_equal(c.area_bounds, np.array([0.0, 8.0], dtype=float))

        # datetime
        values = ['2017-01-01', '2017-01-02', '2018-01-01', '2019-01-01']
        c = ArrayCoordinates1d(values, ctype='left')
        assert_equal(c.area_bounds, np.array(['2017-01-01', '2020-01-01']).astype(np.datetime64))
        c = ArrayCoordinates1d(values[::-1], ctype='left')
        assert_equal(c.area_bounds, np.array(['2017-01-01', '2020-01-01']).astype(np.datetime64))

    def test_area_bounds_right(self):
        # numerical
        values = [0.0, 1.0, 4.0, 6.0]
        c = ArrayCoordinates1d(values, ctype='right')
        assert_equal(c.area_bounds, np.array([-1.0, 6.0], dtype=float))
        c = ArrayCoordinates1d(values[::-1], ctype='right')
        assert_equal(c.area_bounds, np.array([-1.0, 6.0], dtype=float))

        # datetime
        values = ['2017-01-01', '2017-01-02', '2018-01-01', '2019-01-01']
        c = ArrayCoordinates1d(values, ctype='right')
        assert_equal(c.area_bounds, np.array(['2016-12-31', '2019-01-01']).astype(np.datetime64))
        c = ArrayCoordinates1d(values[::-1], ctype='right')
        assert_equal(c.area_bounds, np.array(['2016-12-31', '2019-01-01']).astype(np.datetime64))

    def test_area_bounds_midpoint_numerical(self):
        values = [0.0, 1.0, 4.0, 6.0]
        c = ArrayCoordinates1d(values, ctype='midpoint')
        assert_equal(c.area_bounds, np.array([-0.5, 7.0], dtype=float))
        c = ArrayCoordinates1d(values[::-1], ctype='midpoint')
        assert_equal(c.area_bounds, np.array([-0.5, 7.0], dtype=float))

    @pytest.mark.skip('TODO')
    def test_area_bounds_midpoint_datetime(self):
        values = ['2017-01-01', '2017-01-02', '2018-01-01', '2019-01-01']
        c = ArrayCoordinates1d(values, ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2016-12-31 12:00:00', '2019-06-01']).astype(np.datetime64))
        c = ArrayCoordinates1d(values[::-1], ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2016-12-31 12:00:00', '2019-07-02']).astype(np.datetime64))

    def test_area_bounds_segment_singleton(self):
        # numerical
        value = 10.0
        c = ArrayCoordinates1d(value, ctype='left')
        assert_equal(c.area_bounds, np.array([value, value], dtype=float))
        c = ArrayCoordinates1d(value, ctype='right')
        assert_equal(c.area_bounds, np.array([value, value], dtype=float))
        c = ArrayCoordinates1d(value, ctype='midpoint')
        assert_equal(c.area_bounds, np.array([value, value], dtype=float))

        # datetime
        value = '2018-01-01'
        c = ArrayCoordinates1d(value, ctype='left')
        assert_equal(c.area_bounds, np.array([value, value]).astype(np.datetime64))
        c = ArrayCoordinates1d(value, ctype='right')
        assert_equal(c.area_bounds, np.array([value, value]).astype(np.datetime64))
        c = ArrayCoordinates1d(value, ctype='midpoint')
        assert_equal(c.area_bounds, np.array([value, value]).astype(np.datetime64))

class TestArrayCoordinatesIndexing(object):
    def test_len(self):
        c = ArrayCoordinates1d([])
        assert len(c) == 0

        c = ArrayCoordinates1d([0, 1, 2])
        assert len(c) == 3

    def test_index(self):
        c = ArrayCoordinates1d([20, 50, 60, 90, 40, 10], name='lat', ctype='point')
        
        # int
        c2 = c[2]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, np.array([60], dtype=float))

        c2 = c[-2]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, np.array([40], dtype=float))

        # slice
        c2 = c[:2]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, np.array([20, 50], dtype=float))
        
        c2 = c[::2]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, np.array([20, 60, 40], dtype=float))
        
        c2 = c[1:-1]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, np.array([50, 60, 90, 40], dtype=float))
        
        c2 = c[::-1]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, np.array([10, 40, 90, 60, 50, 20], dtype=float))
        
        # array
        c2 = c[[0, 3, 1]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, np.array([20, 90, 50], dtype=float))

        # boolean array
        c2 = c[[True, True, True, False, True, False]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, np.array([20, 50, 60, 40], dtype=float))

        # invalid
        with pytest.raises(IndexError):
            c[0.3]

        with pytest.raises(IndexError):
            c[10]

class TestArrayCoordinatesSelection(object):
    def test_select(self):
        c = ArrayCoordinates1d([20., 50., 60., 90., 40., 10.], ctype='point')

        # full selection
        s = c.select([0, 100])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, c.coordinates)

        # none, above
        s = c.select([100, 200])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        # none, below
        s = c.select([0, 5])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        # partial, above
        s = c.select([50, 100])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [50., 60., 90.])

        # partial, below
        s = c.select([0, 50])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [20., 50., 40., 10.])

        # partial, inner
        s = c.select([30., 55.])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [50., 40.])

        # partial, very inner (none)
        s = c.select([52, 55])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        # partial, inner exact
        s = c.select([40., 60.])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [50., 60., 40.])

        # partial, backwards bounds
        s = c.select([70, 30])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

    def test_select_empty(self):
        c = ArrayCoordinates1d([], ctype='point')
        s = c.select([0, 1])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

    def test_select_ind(self):
        c = ArrayCoordinates1d([20., 50., 60., 90., 40., 10.], ctype='point')
        
        # full selection
        s, I = c.select([0, 100], return_indices=True)
        assert_equal(c.coordinates[I], c.coordinates)
        assert_equal(s.coordinates, c.coordinates[I])

        # none, above
        s, I = c.select([100, 200], return_indices=True)
        assert_equal(c.coordinates[I], [])
        assert_equal(s.coordinates, c.coordinates[I])

        # none, below
        s, I = c.select([0, 5], return_indices=True)
        assert_equal(c.coordinates[I], [])
        assert_equal(s.coordinates, c.coordinates[I])

        # partial, above
        s, I = c.select([50, 100], return_indices=True)
        assert_equal(c.coordinates[I], [50., 60., 90.])
        assert_equal(s.coordinates, c.coordinates[I])

        # partial, below
        s, I = c.select([0, 50], return_indices=True)
        assert_equal(c.coordinates[I], [20., 50., 40., 10.])
        assert_equal(s.coordinates, c.coordinates[I])

        # partial, inner
        s, I = c.select([30., 55.], return_indices=True)
        assert_equal(c.coordinates[I], [50., 40.])
        assert_equal(s.coordinates, c.coordinates[I])

        # partial, very inner (none)
        s, I = c.select([52, 55], return_indices=True)
        assert_equal(c.coordinates[I], [])
        assert_equal(s.coordinates, c.coordinates[I])
        
        # partial, inner exact
        s, I = c.select([40., 60.], return_indices=True)
        assert_equal(c.coordinates[I], [50., 60., 40.])
        assert_equal(s.coordinates, c.coordinates[I])

        # partial, backwards bounds
        s, I = c.select([70, 30], return_indices=True)
        assert_equal(c.coordinates[I], [])
        assert_equal(s.coordinates, c.coordinates[I])

    def test_select_empty_ind(self):
        c = ArrayCoordinates1d([])        
        s, I = c.select([0, 1], return_indices=True)
        assert_equal(c.coordinates[I], [])
        assert_equal(s.coordinates, c.coordinates[I])

    def test_select_outer_ascending(self):
        c = ArrayCoordinates1d([10., 20., 40., 50., 60., 90.])
        
        # partial, above
        s = c.select([50, 100], outer=True)
        assert_equal(s.coordinates, [50., 60., 90.])

        # partial, below
        s = c.select([0, 50], outer=True)
        assert_equal(s.coordinates, [10., 20., 40., 50.])

        # partial, inner
        s = c.select([30., 55.], outer=True)
        assert_equal(s.coordinates, [20, 40., 50., 60.])

        # partial, very inner
        s = c.select([52, 55], outer=True)
        assert_equal(s.coordinates, [50, 60])

        # partial, inner exact
        s = c.select([40., 60.], outer=True)
        assert_equal(s.coordinates, [40., 50., 60.])

        # partial, backwards bounds
        s = c.select([70, 30], outer=True)
        assert_equal(s.coordinates, [])

    def test_select_outer_descending(self):
        c = ArrayCoordinates1d([90., 60., 50., 40., 20., 10.])
        
        # partial, above
        s = c.select([50, 100], outer=True)
        assert_equal(s.coordinates, [90., 60., 50.])

        # partial, below
        s = c.select([0, 50], outer=True)
        assert_equal(s.coordinates, [50., 40., 20., 10.])

        # partial, inner
        s = c.select([30., 55.], outer=True)
        assert_equal(s.coordinates, [60., 50., 40., 20.])

        # partial, very inner
        s = c.select([52, 55], outer=True)
        assert_equal(s.coordinates, [60, 50])

        # partial, inner exact
        s = c.select([40., 60.], outer=True)
        assert_equal(s.coordinates, [60., 50., 40.])

        # partial, backwards bounds
        s = c.select([70, 30], outer=True)
        assert_equal(s.coordinates, [])

    def test_select_outer_ascending_ind(self):
        c = ArrayCoordinates1d([10., 20., 40., 50., 60., 90.])
        
        # partial, above
        s, I = c.select([50, 100], outer=True, return_indices=True)
        assert_equal(c.coordinates[I], [50., 60., 90.])
        assert_equal(s.coordinates, c.coordinates[I])

        # partial, below
        s, I = c.select([0, 50], outer=True, return_indices=True)
        assert_equal(c.coordinates[I], [10., 20., 40., 50.])
        assert_equal(s.coordinates, c.coordinates[I])

        # partial, inner
        s, I = c.select([30., 55.], outer=True, return_indices=True)
        assert_equal(c.coordinates[I], [20., 40., 50., 60.])
        assert_equal(s.coordinates, c.coordinates[I])

        # partial, very inner
        s, I = c.select([52, 55], outer=True, return_indices=True)
        assert_equal(c.coordinates[I], [50., 60.])
        assert_equal(s.coordinates, c.coordinates[I])
        
        # partial, inner exact
        s, I = c.select([40., 60.], outer=True, return_indices=True)
        assert_equal(c.coordinates[I], [40., 50., 60.])
        assert_equal(s.coordinates, c.coordinates[I])

        # partial, backwards bounds
        s, I = c.select([70, 30], outer=True, return_indices=True)
        assert_equal(c.coordinates[I], [])
        assert_equal(s.coordinates, c.coordinates[I])

    def test_intersect(self):
        a = ArrayCoordinates1d([20., 50., 60., 10.], ctype='point')
        b = ArrayCoordinates1d([55., 65., 95., 45.], ctype='point')
        c = ArrayCoordinates1d([80., 70., 90.], ctype='point')
        e = ArrayCoordinates1d([], ctype='point')
        u = UniformCoordinates1d(45, 95, 10)
        
        # overlap, in both directions
        ab = a.intersect(b)
        assert_equal(ab.coordinates, [50., 60.])
        
        ba = b.intersect(a)
        assert_equal(ba.coordinates, [55., 45.])

        # no overlap
        ac = a.intersect(c)
        assert_equal(ac.coordinates, [])

        # empty self
        ea = e.intersect(a)
        assert_equal(ea.coordinates, [])

        # empty other
        ae = a.intersect(e)
        assert_equal(ae.coordinates, [])

        # UniformCoordinates1d other
        au = a.intersect(u)
        assert_equal(au.coordinates, [50., 60.])

    def test_intersect_ind(self):
        a = ArrayCoordinates1d([20., 50., 60., 10.], ctype='point')
        b = ArrayCoordinates1d([55., 65., 95., 45.], ctype='point')
        c = ArrayCoordinates1d([80., 70., 90.], ctype='point')
        e = ArrayCoordinates1d([], ctype='point')
        u = UniformCoordinates1d(45, 95, 10)
        
        # overlap, both directions
        intersection, I = a.intersect(b, return_indices=True)
        assert_equal(a.coordinates[I], [50., 60.])
        assert_equal(a.coordinates[I], intersection.coordinates)
        
        intersection, I = b.intersect(a, return_indices=True)
        assert_equal(b.coordinates[I], [55., 45.])
        assert_equal(b.coordinates[I], intersection.coordinates)

        # no overlap
        intersection, I = a.intersect(c, return_indices=True)
        assert_equal(a.coordinates[I], [])
        assert_equal(a.coordinates[I], intersection.coordinates)

        # empty self
        intersection, I = e.intersect(a, return_indices=True)
        assert_equal(e.coordinates[I], [])
        assert_equal(e.coordinates[I], intersection.coordinates)

        # empty other
        intersection, I = a.intersect(e, return_indices=True)
        assert_equal(a.coordinates[I], [])
        assert_equal(a.coordinates[I], intersection.coordinates)

        # UniformCoordinates1d other
        intersection, I = a.intersect(u, return_indices=True)
        assert_equal(a.coordinates[I], [50., 60.])
        assert_equal(a.coordinates[I], intersection.coordinates)

    def test_intersect_stacked(self):
        lat = ArrayCoordinates1d([55., 65., 95., 45.], name='lat')
        lon = ArrayCoordinates1d([ 1.,  2.,  3.,  4.], name='lon')
        stacked = StackedCoordinates([lat, lon])
        
        # intersect correct dimension, or all coordinates if missing
        a = ArrayCoordinates1d([50., 60., 10.], ctype='point', name='lat')
        b = ArrayCoordinates1d([2.5, 3.5, 4.5], ctype='point', name='lon')
        c = ArrayCoordinates1d([100., 200., 300.], ctype='point', name='alt')

        ai = a.intersect(stacked)
        bi = b.intersect(stacked)
        ci = c.intersect(stacked)

        assert_equal(ai.coordinates, [50., 60.])
        assert_equal(bi.coordinates, [2.5, 3.5])
        assert_equal(ci.coordinates, [100., 200., 300.])

    def test_intersect_multi(self):
        coords = Coordinates([[55., 65., 95., 45.], [1., 2., 3., 4.]], dims=['lat', 'lon'])
        
        # intersect correct dimension
        a = ArrayCoordinates1d([50., 60., 10.], ctype='point', name='lat')
        b = ArrayCoordinates1d([2.5, 3.5, 4.5], ctype='point', name='lon')
        c = ArrayCoordinates1d([100., 200., 300.], ctype='point', name='alt')

        ai = a.intersect(coords)
        bi = b.intersect(coords)
        ci = c.intersect(coords)

        assert_equal(ai.coordinates, [50., 60.])
        assert_equal(bi.coordinates, [2.5, 3.5])
        assert_equal(ci.coordinates, [100., 200., 300.])        

    def test_intersect_invalid(self):
        a = ArrayCoordinates1d([20., 50., 60., 10.], ctype='point')
        b = [55., 65., 95., 45.]

        with pytest.raises(TypeError, match="Cannot intersect with type"):
            a.intersect(b)

    def test_intersect_name_mismatch(self):
        a = ArrayCoordinates1d([20., 50., 60., 10.], name='lat')
        b = ArrayCoordinates1d([55., 65., 95., 45.], name='lon')

        with pytest.raises(ValueError, match="Cannot intersect mismatched dimensions"):
            a.intersect(b)

    def test_intersect_dtype_mismatch(self):
        a = ArrayCoordinates1d([1., 2., 3., 4.], name='time')
        b = ArrayCoordinates1d(['2018-01-01', '2018-01-02'], name='time')

        with pytest.raises(ValueError, match="Cannot intersect mismatched dtypes"):
            a.intersect(b)

    def test_intersect_units_mismatch(self):
        pass
