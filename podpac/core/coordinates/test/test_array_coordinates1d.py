
from datetime import datetime
import json

import pytest
import traitlets as tl
import numpy as np
import xarray as xr
from numpy.testing import assert_equal

import podpac
from podpac.core.units import Units
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates
from podpac.core.coordinates.coordinates import Coordinates

class TestArrayCoordinatesInit(object):
    def test_empty(self):
        c = ArrayCoordinates1d([])
        a = np.array([], dtype=float)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [np.nan, np.nan])
        assert c.size == 0
        assert c.dtype is None
        assert c.ctype == 'point'
        assert c.is_monotonic is None
        assert c.is_descending is None
        assert c.is_uniform is None
        repr(c)

    def test_numerical_singleton(self):
        a = np.array([10], dtype=float)
        c = ArrayCoordinates1d(10)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [10.0, 10.0])
        assert c.size == 1
        assert c.dtype == float
        assert c.ctype == 'point'
        assert c.is_monotonic == True
        assert c.is_descending is None
        assert c.is_uniform == True
        repr(c)

    def test_numerical_array(self):
        # unsorted
        values = [1, 6, 0, 4.]
        a = np.array(values, dtype=float)
        c = ArrayCoordinates1d(a)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [0., 6.])
        assert c.size == 4
        assert c.dtype == float
        assert c.ctype == 'point'
        assert c.is_monotonic == False
        assert c.is_descending is None
        assert c.is_uniform == False
        repr(c)

        # sorted ascending
        values = [0, 1, 4, 6]
        a = np.array(values, dtype=float)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [0., 6.])
        assert c.size == 4
        assert c.dtype == float
        assert c.ctype == 'midpoint'
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == False
        repr(c)

        # sorted descending
        values = [6, 4, 1, 0]
        a = np.array(values, dtype=float)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [0., 6.])
        assert c.size == 4
        assert c.dtype == float
        assert c.ctype == 'midpoint'
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == False
        repr(c)

        # uniform ascending
        values = [0, 2, 4, 6]
        a = np.array(values, dtype=float)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [0., 6.])
        assert c.size == 4
        assert c.dtype == float
        assert c.ctype == 'midpoint'
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True
        repr(c)

        # uniform descending
        values = [6, 4, 2, 0]
        a = np.array(values, dtype=float)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [0., 6.])
        assert c.size == 4
        assert c.dtype == float
        assert c.ctype == 'midpoint'
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True
        repr(c)

    def test_datetime_singleton(self):
        a = np.array('2018-01-01').astype(np.datetime64)
        c = ArrayCoordinates1d('2018-01-01')
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(['2018-01-01', '2018-01-01']).astype(np.datetime64))
        assert c.size == 1
        assert c.dtype == np.datetime64
        assert c.ctype == 'point'
        assert c.is_monotonic == True
        assert c.is_descending is None
        assert c.is_uniform == True
        repr(c)

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
        assert c.ctype == 'point'
        assert c.is_monotonic == False
        assert c.is_descending is None
        assert c.is_uniform == False
        repr(c)

        # sorted ascending
        values = ['2017-01-01', '2018-01-01', '2018-01-02', '2019-01-01']
        a = np.array(values).astype(np.datetime64)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        assert c.size == 4
        assert c.dtype == np.datetime64
        assert c.ctype == 'point'
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == False
        repr(c)

        # sorted descending
        values = ['2019-01-01', '2018-01-02', '2018-01-01', '2017-01-01']
        a = np.array(values).astype(np.datetime64)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        assert c.size == 4
        assert c.dtype == np.datetime64
        assert c.ctype == 'point'
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == False
        repr(c)

        # uniform ascending
        values = ['2017-01-01', '2018-01-01', '2019-01-01']
        a = np.array(values).astype(np.datetime64)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        assert c.size == 3
        assert c.dtype == np.datetime64
        assert c.ctype == 'point'
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True
        repr(c)

        # uniform descending
        values = ['2019-01-01', '2018-01-01', '2017-01-01']
        a = np.array(values).astype(np.datetime64)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        assert c.size == 3
        assert c.dtype == np.datetime64
        assert c.ctype == 'point'
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True
        repr(c)

    def test_invalid_coords(self):
        with pytest.raises(ValueError, match="Invalid coordinate values"):
            ArrayCoordinates1d([1, 2, '2018-01'])

        with pytest.raises(ValueError, match="Invalid coordinate values"):
            ArrayCoordinates1d([[1.0, 2.0], [3.0, 4.0]])

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
        c = ArrayCoordinates1d([1, 2, 3], name='lat')
        c2 = c.copy()
        assert c is not c2
        assert c == c2

        c = ArrayCoordinates1d([1, 2, 3], segment_lengths=0.5)
        c2 = c.copy()
        assert c is not c2
        assert c == c2

    def test_name(self):
        ArrayCoordinates1d([])
        ArrayCoordinates1d([], name='lat')
        ArrayCoordinates1d([], name='lon')
        ArrayCoordinates1d([], name='alt')
        ArrayCoordinates1d([], name='time')

        with pytest.raises(tl.TraitError):
            ArrayCoordinates1d([], name='depth')

        repr(ArrayCoordinates1d([], name='lat'))

    def test_segment_lenths_point(self):
        with pytest.raises(TypeError, match="segment_lengths must be None"):
            ArrayCoordinates1d([1, 2], ctype='point', segment_lengths=1.0)
        
        with pytest.raises(TypeError, match="segment_lengths must be None"):
            ArrayCoordinates1d([1, 2], ctype='point', segment_lengths=[1.0, 1.0])

    def test_segment_lengths_empty(self):
        c = ArrayCoordinates1d([])
        assert c.segment_lengths is None

    def test_segment_lengths_delta(self):
        # numeric
        c = ArrayCoordinates1d([1, 2, 3], ctype='midpoint', segment_lengths=1.0)
        assert c.segment_lengths == 1.0

        # datetime
        c = ArrayCoordinates1d(['2018-01-01', '2018-01-02'], ctype='midpoint', segment_lengths='1,D')
        assert c.segment_lengths == np.timedelta64(1, 'D')

        # mismatch
        with pytest.raises(TypeError, match="coordinates and segment_lengths type mismatch"):
            ArrayCoordinates1d([1, 2, 3], ctype='midpoint', segment_lengths='1,D')

        with pytest.raises(TypeError, match="coordinates and segment_lengths type mismatch"):
            ArrayCoordinates1d(['2018-01-01', '2018-01-02'], ctype='midpoint', segment_lengths=1.0)

    def test_segment_lengths_array(self):
        # numeric
        c = ArrayCoordinates1d([1, 2, 3], ctype='midpoint', segment_lengths=[1.0, 1.0, 1.0])
        assert_equal(c.segment_lengths, np.array([1.0, 1.0, 1.0]))

        # datetime
        c = ArrayCoordinates1d(['2018-01-01', '2018-01-02'], ctype='midpoint', segment_lengths=['1,D', '1,D'])
        assert_equal(c.segment_lengths, np.array([np.timedelta64(1, 'D'), np.timedelta64(1, 'D')]))

        # mismatch
        with pytest.raises(ValueError, match="coordinates and segment_lengths size mismatch"):
            ArrayCoordinates1d([1, 2, 3], ctype='midpoint', segment_lengths=[1.0, 1.0])

        with pytest.raises(ValueError, match="coordinates and segment_lengths dtype mismatch"):
            ArrayCoordinates1d([1, 2, 3], ctype='midpoint', segment_lengths=['1,D', '1,D', '1,D'])

        with pytest.raises(ValueError, match="coordinates and segment_lengths dtype mismatch"):
            ArrayCoordinates1d(['2018-01-01', '2018-01-02'], ctype='midpoint', segment_lengths=[1.0, 1.0])

    def test_segment_lengths_inferred(self):
        # no segment lengths for point coordinates
        c = ArrayCoordinates1d([1, 2, 3], ctype='point')
        assert c.segment_lengths is None

        c = ArrayCoordinates1d(['2018-01-01', '2018-01-02'], ctype='point')
        assert c.segment_lengths is None

        # no segment lengths for empty segment coordinates
        c = ArrayCoordinates1d([], ctype='midpoint')
        assert c.segment_lengths is None

        # segment lengths required for datetime segment coordinates
        with pytest.raises(TypeError, match="segment_lengths required"):
            ArrayCoordinates1d(['2018-01-01', '2018-01-02'], ctype='midpoint')

        # segment lengths required for singleton segment coordinates
        with pytest.raises(TypeError, match="segment_lengths required"):
            ArrayCoordinates1d([1], ctype='midpoint')

        # segment lengths required for nonmonotonic segment coordinates
        with pytest.raises(TypeError, match="segment_lengths required"):
            ArrayCoordinates1d([1, 4, 2], ctype='midpoint')

        values = [1, 2, 4, 7]
        
        # left
        c = ArrayCoordinates1d(values, ctype='left')
        assert_equal(c.segment_lengths, [1., 2., 3., 3.])

        c = ArrayCoordinates1d(values[::-1], ctype='left')
        assert_equal(c.segment_lengths, [3., 3., 2., 1.])

        # right
        c = ArrayCoordinates1d(values, ctype='right')
        assert_equal(c.segment_lengths, [1., 1., 2., 3.])
        
        c = ArrayCoordinates1d(values[::-1], ctype='right')
        assert_equal(c.segment_lengths, [3., 2., 1., 1.])

        # midpoint
        c = ArrayCoordinates1d(values, ctype='midpoint')
        assert_equal(c.segment_lengths, [1., 1.5, 2.5, 3.])
        
        c = ArrayCoordinates1d(values[::-1], ctype='midpoint')
        assert_equal(c.segment_lengths, [3, 2.5, 1.5, 1.])

        # uniform coordinates should use a single segment length
        c = ArrayCoordinates1d([1., 2., 3.], ctype='midpoint')
        assert c.segment_lengths == 1.0

    def test_segment_lengths_positive(self):
        with pytest.raises(ValueError, match="segment_lengths must be positive"):
            ArrayCoordinates1d([0, 1, 2], segment_lengths=[1.0, 0.0, 1.0])
        
        with pytest.raises(ValueError, match="segment_lengths must be positive"):
            ArrayCoordinates1d([0, 1, 2], segment_lengths=[1.0, -1.0, 1.0])
        
        with pytest.raises(ValueError, match="segment_lengths must be positive"):
            ArrayCoordinates1d([0, 1, 2], segment_lengths=0.0)

        with pytest.raises(ValueError, match="segment_lengths must be positive"):
            ArrayCoordinates1d([0, 1, 2], segment_lengths=-1.0)

    def test_coord_ref_sys(self):
        c = ArrayCoordinates1d([])
        assert c.coord_ref_sys == 'WGS84'

        c = ArrayCoordinates1d([], coord_ref_sys='SPHER_MERC')
        assert c.coord_ref_sys == 'SPHER_MERC'
        
        with pytest.raises(tl.TraitError):
            ArrayCoordinates1d([], coord_ref_sys='ABCD')

class TestArrayCoordinatesEq(object):
    def test_eq_type(self):
        c1 = ArrayCoordinates1d([0, 1, 3])
        assert c1 != [0, 1, 3]

    def test_eq_coordinates(self):
        c1 = ArrayCoordinates1d([0, 1, 3])
        c2 = ArrayCoordinates1d([0, 1, 3])
        c3 = ArrayCoordinates1d([0, 1, 3, 4])
        c4 = ArrayCoordinates1d([0, 1, 4])
        c5 = ArrayCoordinates1d([0, 3, 1])

        assert c1 == c2
        assert c1 != c3
        assert c1 != c4
        assert c1 != c5

        c1 = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-04'])
        c2 = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-04'])
        c3 = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-04', '2018-01-05'])
        c4 = ArrayCoordinates1d(['2018-01-01', '2018-01-04', '2018-01-02'])

        assert c1 == c2
        assert c1 != c3
        assert c1 != c4

    def test_eq_name(self):
        c1 = ArrayCoordinates1d([0, 1, 3], name='lat')
        c2 = ArrayCoordinates1d([0, 1, 3], name='lat')
        c3 = ArrayCoordinates1d([0, 1, 3], name='lon')
        c4 = ArrayCoordinates1d([0, 1, 3])
        
        assert c1 == c2
        assert c1 != c3
        assert c1 != c4
        
        c4.set_trait('name', 'lat')
        assert c1 == c4

    def test_eq_ctype(self):
        c1 = ArrayCoordinates1d([0, 1, 3])
        c2 = ArrayCoordinates1d([0, 1, 3], ctype='midpoint')
        c3 = ArrayCoordinates1d([0, 1, 3], ctype='left')
        
        assert c1 == c2
        assert c1 != c3
        assert c2 != c3

    def test_eq_segment_lengths(self):
        c1 = ArrayCoordinates1d([0, 1, 3], segment_lengths=[1, 1, 1])
        c2 = ArrayCoordinates1d([0, 1, 3], segment_lengths=[1, 1, 1])
        c3 = ArrayCoordinates1d([0, 1, 3], segment_lengths=[1, 2, 3])
        
        assert c1 == c2
        assert c1 != c3
        
        c1 = ArrayCoordinates1d([0, 1, 3], segment_lengths=1)
        c2 = ArrayCoordinates1d([0, 1, 3], segment_lengths=1)
        c3 = ArrayCoordinates1d([0, 1, 3], segment_lengths=2)
        
        assert c1 == c2
        assert c1 != c3
        
        # mixed segment_lengths type
        c1 = ArrayCoordinates1d([0, 1, 3], segment_lengths=[1, 1, 1])
        c2 = ArrayCoordinates1d([0, 1, 3], segment_lengths=1)
        assert c1 == c2

class TestArrayCoordinatesSerialization(object):
    def test_invalid_definition(self):
        d = {'coords': [0, 1, 2]}
        with pytest.raises(ValueError, match='ArrayCoordinates1d definition requires "values" property'):
            ArrayCoordinates1d.from_definition(d)

    def test_definition(self):
        # numerical
        c = ArrayCoordinates1d([0, 1, 2], name="lat", ctype="point")
        d = c.definition
        assert isinstance(d, dict)
        assert set(d.keys()) == set(['values', 'name', 'ctype'])
        json.dumps(d, cls=podpac.core.utils.JSONEncoder) # test serializable
        c2 = ArrayCoordinates1d.from_definition(d) # test from_definition
        assert c2 == c

        # datetimes
        c = ArrayCoordinates1d(['2018-01-01', '2018-01-02'])
        d = c.definition
        assert isinstance(d, dict)
        assert set(d.keys()) == set(['values'])
        json.dumps(d, cls=podpac.core.utils.JSONEncoder) # test serializable
        c2 = ArrayCoordinates1d.from_definition(d) #test from_definition
        assert c2 == c

    def test_definition_segment_lengths(self):
        c = ArrayCoordinates1d([0, 1, 2], segment_lengths=0.5)
        d = c.definition
        assert isinstance(d, dict)
        assert set(d.keys()) == set(['values', 'segment_lengths'])
        json.dumps(d, cls=podpac.core.utils.JSONEncoder) # test serializable
        c2 = ArrayCoordinates1d.from_definition(d) # test from_definition
        assert c2 == c

class TestArrayCoordinatesProperties(object):
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
        values = np.array([0.0, 1.0, 4.0, 6.0])
        c = ArrayCoordinates1d(values, ctype='point')
        assert_equal(c.area_bounds, [0.0, 6.0])
        c = ArrayCoordinates1d(values[::-1], ctype='point')
        assert_equal(c.area_bounds, [0.0, 6.0])
        c = ArrayCoordinates1d(values[[1, 2, 0, 3]], ctype='point')
        assert_equal(c.area_bounds, [0.0, 6.0])

        # datetime
        values = np.array(['2017-01-01', '2017-01-02', '2018-01-01', '2019-01-01']).astype(np.datetime64)
        c = ArrayCoordinates1d(values, ctype='point')
        assert_equal(c.area_bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        c = ArrayCoordinates1d(values[::-1], ctype='point')
        assert_equal(c.area_bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        c = ArrayCoordinates1d(values[[1, 2, 0, 3]], ctype='point')
        assert_equal(c.area_bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))

    def test_area_bounds_empty(self):
        c = ArrayCoordinates1d([], ctype='midpoint')
        assert np.all(np.isnan(c.area_bounds))

    def test_area_bounds_left(self):
        # numerical
        values = np.array([0.0, 1.0, 4.0, 6.0])
        c = ArrayCoordinates1d(values, ctype='left')
        assert_equal(c.area_bounds, [0.0, 8.0])
        c = ArrayCoordinates1d(values[::-1], ctype='left')
        assert_equal(c.area_bounds, [0.0, 8.0])
        c = ArrayCoordinates1d(values[[1, 0, 3, 2]], ctype='left', segment_lengths=2.0)
        assert_equal(c.area_bounds, [0.0, 8.0])
        c = ArrayCoordinates1d(values[[1, 0, 3, 2]], ctype='left', segment_lengths=[1.0, 1.0, 2.0, 1.0])
        assert_equal(c.area_bounds, [0.0, 8.0])

        # datetime
        values = np.array(['2017-01-02', '2017-01-01', '2019-01-01', '2018-01-01']).astype(np.datetime64)
        c = ArrayCoordinates1d(values, ctype='left', segment_lengths='1,D')
        assert_equal(c.area_bounds, np.array(['2017-01-01', '2019-01-02']).astype(np.datetime64))
        c = ArrayCoordinates1d(values, ctype='left', segment_lengths='1,M')
        assert_equal(c.area_bounds, np.array(['2017-01-01', '2019-02-01']).astype(np.datetime64))
        c = ArrayCoordinates1d(values, ctype='left', segment_lengths='1,Y')
        assert_equal(c.area_bounds, np.array(['2017-01-01', '2020-01-01']).astype(np.datetime64))
        c = ArrayCoordinates1d(values, ctype='left', segment_lengths=['2,D', '2,D', '1,D', '2,D'])
        assert_equal(c.area_bounds, np.array(['2017-01-01', '2019-01-02']).astype(np.datetime64))

    def test_area_bounds_right(self):
        # numerical
        values = np.array([0.0, 1.0, 4.0, 6.0])
        c = ArrayCoordinates1d(values, ctype='right')
        assert_equal(c.area_bounds, [-1.0, 6.0])
        c = ArrayCoordinates1d(values[::-1], ctype='right')
        assert_equal(c.area_bounds, [-1.0, 6.0])
        c = ArrayCoordinates1d(values[[1, 0, 3, 2]], ctype='right', segment_lengths=1.0)
        assert_equal(c.area_bounds, [-1.0, 6.0])
        c = ArrayCoordinates1d(values[[1, 0, 3, 2]], ctype='right', segment_lengths=[3.0, 1.0, 3.0, 3.0])
        assert_equal(c.area_bounds, [-1.0, 6.0])
        
        # datetime
        values = np.array(['2017-01-02', '2017-01-01', '2019-01-01', '2018-01-01']).astype(np.datetime64)
        c = ArrayCoordinates1d(values, ctype='right', segment_lengths='1,D')
        assert_equal(c.area_bounds, np.array(['2016-12-31', '2019-01-01']).astype(np.datetime64))
        c = ArrayCoordinates1d(values, ctype='right', segment_lengths='1,M')
        assert_equal(c.area_bounds, np.array(['2016-12-01', '2019-01-01']).astype(np.datetime64))
        c = ArrayCoordinates1d(values, ctype='right', segment_lengths='1,Y')
        assert_equal(c.area_bounds, np.array(['2016-01-01', '2019-01-01']).astype(np.datetime64))
        c = ArrayCoordinates1d(values, ctype='right', segment_lengths=['2,D', '1,D', '2,D', '2,D'])
        assert_equal(c.area_bounds, np.array(['2016-12-31', '2019-01-01']).astype(np.datetime64))
    
    def test_area_bounds_midpoint(self):
        # numerical
        values = np.array([0.0, 1.0, 4.0, 6.0])
        c = ArrayCoordinates1d(values, ctype='midpoint')
        assert_equal(c.area_bounds, [-0.5, 7.0])
        c = ArrayCoordinates1d(values[::-1], ctype='midpoint')
        assert_equal(c.area_bounds, [-0.5, 7.0])
        c = ArrayCoordinates1d(values[[1, 0, 3, 2]], ctype='midpoint', segment_lengths=1.0)
        assert_equal(c.area_bounds, [-0.5, 6.5])
        c = ArrayCoordinates1d(values[[1, 0, 3, 2]], ctype='midpoint', segment_lengths=[1.0, 2.0, 3.0, 4.0])
        assert_equal(c.area_bounds, [-1.0, 7.5])
        
        # datetime
        values = np.array(['2017-01-02', '2017-01-01', '2019-01-01', '2018-01-01']).astype(np.datetime64)
        c = ArrayCoordinates1d(values, ctype='midpoint', segment_lengths='2,D')
        assert_equal(c.area_bounds, np.array(['2016-12-31', '2019-01-02']).astype(np.datetime64))
        c = ArrayCoordinates1d(values, ctype='midpoint', segment_lengths='2,M')
        assert_equal(c.area_bounds, np.array(['2016-12-01', '2019-02-01']).astype(np.datetime64))
        c = ArrayCoordinates1d(values, ctype='midpoint', segment_lengths='2,Y')
        assert_equal(c.area_bounds, np.array(['2016-01-01', '2020-01-01']).astype(np.datetime64))
        c = ArrayCoordinates1d(values, ctype='midpoint', segment_lengths=['2,D', '4,D', '6,D', '8,D'])
        assert_equal(c.area_bounds, np.array(['2016-12-30', '2019-01-04']).astype(np.datetime64))

        # datetime divide_delta
        c = ArrayCoordinates1d(values, ctype='midpoint', segment_lengths='1,D')
        assert_equal(c.area_bounds, np.array(['2016-12-31 12', '2019-01-01 12']).astype(np.datetime64))

    def test_properties(self):
        c = ArrayCoordinates1d([])
        assert isinstance(c.properties, dict)
        assert set(c.properties) == set([])

        c = ArrayCoordinates1d([], name='lat')
        assert isinstance(c.properties, dict)
        assert set(c.properties) == set(['name'])

        c = ArrayCoordinates1d([], ctype='point')
        assert isinstance(c.properties, dict)
        assert set(c.properties) == set(['ctype'])

        c = ArrayCoordinates1d([], units=Units())
        assert isinstance(c.properties, dict)
        assert set(c.properties) == set(['units'])

        c = ArrayCoordinates1d([], coord_ref_sys='WGS84')
        assert isinstance(c.properties, dict)
        assert set(c.properties) == set(['coord_ref_sys'])

        # segment_lengths are not properties
        c = ArrayCoordinates1d([1, 2], segment_lengths=1)
        assert isinstance(c.properties, dict)
        assert set(c.properties) == set()

        c = ArrayCoordinates1d([1, 2], name='lat', ctype='left', segment_lengths=1)
        assert isinstance(c.properties, dict)
        assert set(c.properties) == set(['name', 'ctype'])

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
        assert_equal(c2.coords, [60])

        c2 = c[-2]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, [40])

        # slice
        c2 = c[:2]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, [20, 50])
        
        c2 = c[::2]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, [20, 60, 40])
        
        c2 = c[1:-1]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, [50, 60, 90, 40])
        
        c2 = c[::-1]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, [10, 40, 90, 60, 50, 20])
        
        # array
        c2 = c[[0, 3, 1]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, [20, 90, 50])

        # boolean array
        c2 = c[[True, True, True, False, True, False]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [20, 50, 60, 40])

        # invalid
        with pytest.raises(IndexError):
            c[0.3]

        with pytest.raises(IndexError):
            c[10]

    def test_index_segment_lengths(self):
        # array segment_lengths
        c = ArrayCoordinates1d([1, 2, 4, 5], segment_lengths=[0.1, 0.2, 0.3, 0.4])
        
        c2 = c[1]
        assert c2.segment_lengths == 0.2 or np.array_equal(c2.segment_lengths, [0.2])

        c2 = c[1:3]
        assert_equal(c2.segment_lengths, [0.2, 0.3])
        
        c2 = c[[2, 1]]
        assert_equal(c2.segment_lengths, [0.3, 0.2])

        c2 = c[[]]
        assert_equal(c2.segment_lengths, [])

        # uniform segment_lengths
        c = ArrayCoordinates1d([1, 2, 4, 5], segment_lengths=0.5)
        
        c2 = c[1]
        assert c2.segment_lengths == 0.5

        c2 = c[1:3]
        assert c2.segment_lengths == 0.5
        
        c2 = c[[2, 1]]
        assert c2.segment_lengths == 0.5

        c2 = c[[]]
        assert c2.segment_lengths == 0.5
        
        # inferred segment_lengths
        c = ArrayCoordinates1d([1, 2, 4, 7], ctype='left')
        c2 = c[1]
        assert c2.segment_lengths == 2.0 or np.array_equal(c2.segment_lengths, [2.0])

class TestArrayCoordinatesSelection(object):
    def test_select_empty_shortcut(self):
        c = ArrayCoordinates1d([])
        bounds = [0, 1]

        s = c.select(bounds)
        assert_equal(s.coordinates, [])

        s, I = c.select(bounds, return_indices=True)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

    def test_select_all_shortcut(self):
        c = ArrayCoordinates1d([20., 50., 60., 90., 40., 10.], ctype='point')
        bounds = [0, 100]

        s = c.select(bounds)
        assert_equal(s.coordinates, c.coordinates)

        s, I = c.select(bounds, return_indices=True)
        assert_equal(s.coordinates, c.coordinates)
        assert_equal(c.coordinates[I], c.coordinates)

    def test_select_none_shortcut(self):
        c = ArrayCoordinates1d([20., 50., 60., 90., 40., 10.], ctype='point')

        # above
        s = c.select([100, 200])
        assert_equal(s.coordinates, [])

        s, I = c.select([100, 200], return_indices=True)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])
        
        # below
        s = c.select([0, 5])
        assert_equal(s.coordinates, [])

        s, I = c.select([0, 5], return_indices=True)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

    def test_select(self):
        c = ArrayCoordinates1d([20., 50., 60., 90., 40., 10.], ctype='point')
        
        # inner
        s = c.select([30., 55.])
        assert_equal(s.coordinates, [50., 40.])

        s, I = c.select([30., 55.], return_indices=True)
        assert_equal(s.coordinates, [50., 40.])
        assert_equal(c.coordinates[I], [50., 40.])

        # inner with aligned bounds
        s = c.select([40., 60.])
        assert_equal(s.coordinates, [50., 60., 40.])

        s, I = c.select([40., 60.], return_indices=True)
        assert_equal(s.coordinates, [50., 60., 40.])
        assert_equal(c.coordinates[I], [50., 60., 40.])
        
        # above
        s = c.select([50, 100])
        assert_equal(s.coordinates, [50., 60., 90.])

        s, I = c.select([50, 100], return_indices=True)
        assert_equal(s.coordinates, [50., 60., 90.])
        assert_equal(c.coordinates[I], [50., 60., 90.])

        # below
        s = c.select([0, 50])
        assert_equal(s.coordinates, [20., 50., 40., 10.])

        s, I = c.select([0, 50], return_indices=True)
        assert_equal(s.coordinates, [20., 50., 40., 10.])
        assert_equal(c.coordinates[I], [20., 50., 40., 10.])

        # between coordinates
        s = c.select([52, 55])
        assert_equal(s.coordinates, [])

        s, I = c.select([52, 55], return_indices=True)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

        # backwards bounds
        s = c.select([70, 30])
        assert_equal(s.coordinates, [])

        s, I = c.select([70, 30], return_indices=True)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

    def test_select_outer_ascending(self):
        c = ArrayCoordinates1d([10., 20., 40., 50., 60., 90.])
        
        # inner
        s = c.select([30., 55.], outer=True)
        assert_equal(s.coordinates, [20, 40., 50., 60.])

        s, I = c.select([30., 55.], outer=True, return_indices=True)
        assert_equal(s.coordinates, [20, 40., 50., 60.])
        assert_equal(c.coordinates[I], [20, 40., 50., 60.])

        # inner with aligned bounds
        s = c.select([40., 60.], outer=True)
        assert_equal(s.coordinates, [40., 50., 60.])

        s, I = c.select([40., 60.], outer=True, return_indices=True)
        assert_equal(s.coordinates, [40., 50., 60.])
        assert_equal(c.coordinates[I], [40., 50., 60.])
        
        # above
        s = c.select([50, 100], outer=True)
        assert_equal(s.coordinates, [50., 60., 90.])

        s, I = c.select([50, 100], outer=True, return_indices=True)
        assert_equal(s.coordinates, [50., 60., 90.])
        assert_equal(c.coordinates[I], [50., 60., 90.])

        # below
        s = c.select([0, 50], outer=True)
        assert_equal(s.coordinates, [10., 20., 40., 50.])

        s, I = c.select([0, 50], outer=True, return_indices=True)
        assert_equal(s.coordinates, [10., 20., 40., 50.])
        assert_equal(c.coordinates[I], [10., 20., 40., 50.])

        # between coordinates
        s = c.select([52, 55], outer=True)
        assert_equal(s.coordinates, [50, 60])

        s, I = c.select([52, 55], outer=True, return_indices=True)
        assert_equal(s.coordinates, [50, 60])
        assert_equal(c.coordinates[I], [50, 60])

        # backwards bounds
        s = c.select([70, 30], outer=True)
        assert_equal(s.coordinates, [])

        s, I = c.select([70, 30], outer=True, return_indices=True)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

    def test_select_outer_descending(self):
        c = ArrayCoordinates1d([90., 60., 50., 40., 20., 10.])
        
        # inner
        s = c.select([30., 55.], outer=True)
        assert_equal(s.coordinates, [60., 50., 40., 20.])

        s, I = c.select([30., 55.], outer=True, return_indices=True)
        assert_equal(s.coordinates, [60., 50., 40., 20.])
        assert_equal(c.coordinates[I], [60., 50., 40., 20.])
        
        # inner with aligned bounds
        s = c.select([40., 60.], outer=True)
        assert_equal(s.coordinates, [60., 50., 40.])

        s, I = c.select([40., 60.], outer=True, return_indices=True)
        assert_equal(s.coordinates, [60., 50., 40.])
        assert_equal(c.coordinates[I], [60., 50., 40.])
        
        # above
        s = c.select([50, 100], outer=True)
        assert_equal(s.coordinates, [90., 60., 50.])

        s, I = c.select([50, 100], outer=True, return_indices=True)
        assert_equal(s.coordinates, [90., 60., 50.])
        assert_equal(c.coordinates[I], [90., 60., 50.])

        # below
        s = c.select([0, 50], outer=True)
        assert_equal(s.coordinates, [50., 40., 20., 10.])

        s, I = c.select([0, 50], outer=True, return_indices=True)
        assert_equal(s.coordinates, [50., 40., 20., 10.])
        assert_equal(c.coordinates[I], [50., 40., 20., 10.])

        # between coordinates
        s = c.select([52, 55], outer=True)
        assert_equal(s.coordinates, [60, 50])

        s, I = c.select([52, 55], outer=True, return_indices=True)
        assert_equal(s.coordinates, [60, 50])
        assert_equal(c.coordinates[I], [60, 50])

        # backwards bounds
        s = c.select([70, 30], outer=True)
        assert_equal(s.coordinates, [])

        s, I = c.select([70, 30], outer=True, return_indices=True)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

    def test_select_outer_nonmonotonic(self):
        c = ArrayCoordinates1d([20., 40., 60., 10., 90., 50.])
        
        # inner
        s = c.select([30., 55.], outer=True)
        assert_equal(s.coordinates, [20, 40., 60., 50.])

        s, I = c.select([30., 55.], outer=True, return_indices=True)
        assert_equal(s.coordinates, [20, 40., 60., 50.])
        assert_equal(c.coordinates[I], [20, 40., 60., 50.])

        # inner with aligned bounds
        s = c.select([40., 60.], outer=True)
        assert_equal(s.coordinates, [40., 60., 50.])

        s, I = c.select([40., 60.], outer=True, return_indices=True)
        assert_equal(s.coordinates, [40., 60., 50.])
        assert_equal(c.coordinates[I], [40., 60., 50.])
        
        # above
        s = c.select([50, 100], outer=True)
        assert_equal(s.coordinates, [60., 90., 50.])

        s, I = c.select([50, 100], outer=True, return_indices=True)
        assert_equal(s.coordinates, [60., 90., 50.])
        assert_equal(c.coordinates[I], [60., 90., 50.])

        # below
        s = c.select([0, 50], outer=True)
        assert_equal(s.coordinates, [20., 40., 10., 50.])

        s, I = c.select([0, 50], outer=True, return_indices=True)
        assert_equal(s.coordinates, [20., 40., 10., 50.])
        assert_equal(c.coordinates[I], [20., 40., 10., 50.])

        # between coordinates
        s = c.select([52, 55], outer=True)
        assert_equal(s.coordinates, [60, 50])

        s, I = c.select([52, 55], outer=True, return_indices=True)
        assert_equal(s.coordinates, [60, 50])
        assert_equal(c.coordinates[I], [60, 50])

        # backwards bounds
        s = c.select([70, 30], outer=True)
        assert_equal(s.coordinates, [])

        s, I = c.select([70, 30], outer=True, return_indices=True)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

class TestArrayCoordinatesIntersection(object):
    def test_intersect_invalid(self):
        a = ArrayCoordinates1d([20., 50., 60., 10.], ctype='point')
        b = [55., 65., 95., 45.]

        with pytest.raises(TypeError, match="Cannot intersect with type"):
            a.intersect(b)

    def test_inteserct_full_shortcut(self):
        pass

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

    def test_intersect(self):
        a = ArrayCoordinates1d([20., 50., 60., 10.], ctype='point')
        b = ArrayCoordinates1d([55., 65., 95., 45.], ctype='point')
        c = ArrayCoordinates1d([80., 70., 90.], ctype='point')
        e = ArrayCoordinates1d([], ctype='point')
        u = UniformCoordinates1d(45, 95, 10)
        
        # overlap, in both directions
        ab = a.intersect(b)
        assert_equal(ab.coordinates, [50., 60.])

        ab, I = a.intersect(b, return_indices=True)
        assert_equal(ab.coordinates, [50., 60.])
        assert_equal(a.coordinates[I], [50., 60.])
        
        ba = b.intersect(a)
        assert_equal(ba.coordinates, [55., 45.])

        ba, I = b.intersect(a, return_indices=True)
        assert_equal(ba.coordinates, [55., 45.])
        assert_equal(b.coordinates[I], [55., 45.])

        # no overlap
        ac = a.intersect(c)
        assert_equal(ac.coordinates, [])

        ac, I = a.intersect(c, return_indices=True)
        assert_equal(ac.coordinates, [])
        assert_equal(a.coordinates[I], [])

        ca = a.intersect(c)
        assert_equal(ca.coordinates, [])

        ca, I = a.intersect(c, return_indices=True)
        assert_equal(ca.coordinates, [])
        assert_equal(c.coordinates[I], [])

        # empty self
        ea = e.intersect(a)
        assert_equal(ea.coordinates, [])

        ea, I = e.intersect(a, return_indices=True)
        assert_equal(ea.coordinates, [])
        assert_equal(e.coordinates[I], [])

        # empty other
        ae = a.intersect(e)
        assert_equal(ae.coordinates, [])

        ae, I = a.intersect(e, return_indices=True)
        assert_equal(ae.coordinates, [])
        assert_equal(a.coordinates[I], [])

        # UniformCoordinates1d other
        au = a.intersect(u)
        assert_equal(au.coordinates, [50., 60.])

        au, I = a.intersect(u, return_indices=True)
        assert_equal(au.coordinates, [50., 60.])
        assert_equal(a.coordinates[I], [50., 60.])

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
