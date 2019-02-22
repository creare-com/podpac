
from datetime import datetime
import json

import pytest
import traitlets as tl
import numpy as np
from numpy.testing import assert_equal

import podpac
from podpac.core.units import Units
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d

class TestUniformCoordinatesCreation(object):
    def test_numerical(self):
        # ascending
        c = UniformCoordinates1d(0, 50, 10)
        a = np.array([0, 10, 20, 30, 40, 50], dtype=float)
        assert c.start == 0
        assert c.stop == 50
        assert c.step == 10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [0, 50])
        assert c.size == 6
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending
        c = UniformCoordinates1d(50, 0, -10)
        a = np.array([50, 40, 30, 20, 10, 0], dtype=float)
        assert c.start == 50
        assert c.stop == 0
        assert c.step == -10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [0, 50])
        assert c.size == 6
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_numerical_inexact(self):
        # ascending
        c = UniformCoordinates1d(0, 49, 10)
        a = np.array([0, 10, 20, 30, 40], dtype=float)
        assert c.start == 0
        assert c.stop == 49
        assert c.step == 10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [0, 40])
        assert c.size == 5
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending
        c = UniformCoordinates1d(50, 1, -10)
        a = np.array([50, 40, 30, 20, 10], dtype=float)
        assert c.start == 50
        assert c.stop == 1
        assert c.step == -10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [10, 50])
        assert c.dtype == float
        assert c.size == a.size
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True
        
    def test_datetime(self):
        # ascending
        c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D')
        a = np.array(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-01-04')
        assert c.step == np.timedelta64(1, 'D')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True
        
        # descending
        c = UniformCoordinates1d('2018-01-04', '2018-01-01', '-1,D')
        a = np.array(['2018-01-04', '2018-01-03', '2018-01-02', '2018-01-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-01-04')
        assert c.stop == np.datetime64('2018-01-01')
        assert c.step == np.timedelta64(-1, 'D')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_datetime_inexact(self):
        # ascending
        c = UniformCoordinates1d('2018-01-01', '2018-01-06', '2,D')
        a = np.array(['2018-01-01', '2018-01-03', '2018-01-05']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-01-06')
        assert c.step == np.timedelta64(2, 'D')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending
        c = UniformCoordinates1d('2018-01-06', '2018-01-01', '-2,D')
        a = np.array(['2018-01-06', '2018-01-04', '2018-01-02']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-01-06')
        assert c.stop == np.datetime64('2018-01-01')
        assert c.step == np.timedelta64(-2, 'D')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_datetime_month_step(self):
        # ascending
        c = UniformCoordinates1d('2018-01-01', '2018-04-01', '1,M')
        a = np.array(['2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-04-01')
        assert c.step == np.timedelta64(1, 'M')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending
        c = UniformCoordinates1d('2018-04-01', '2018-01-01', '-1,M')
        a = np.array(['2018-04-01', '2018-03-01', '2018-02-01', '2018-01-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-04-01')
        assert c.stop == np.datetime64('2018-01-01')
        assert c.step == np.timedelta64(-1, 'M')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_datetime_year_step(self):
        # ascending, exact
        c = UniformCoordinates1d('2018-01-01', '2021-01-01', '1,Y')
        a = np.array(['2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2021-01-01')
        assert c.step == np.timedelta64(1, 'Y')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending, exact
        c = UniformCoordinates1d('2021-01-01', '2018-01-01', '-1,Y')
        a = np.array(['2021-01-01', '2020-01-01', '2019-01-01', '2018-01-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2021-01-01')
        assert c.stop == np.datetime64('2018-01-01')
        assert c.step == np.timedelta64(-1, 'Y')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

        # ascending, inexact (two cases)
        c = UniformCoordinates1d('2018-01-01', '2021-04-01', '1,Y')
        a = np.array(['2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2021-04-01')
        assert c.step == np.timedelta64(1, 'Y')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        c = UniformCoordinates1d('2018-04-01', '2021-01-01', '1,Y')
        a = np.array(['2018-04-01', '2019-04-01', '2020-04-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-04-01')
        assert c.stop == np.datetime64('2021-01-01')
        assert c.step == np.timedelta64(1, 'Y')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending, inexact (two cases)
        c = UniformCoordinates1d('2021-01-01', '2018-04-01', '-1,Y')
        a = np.array(['2021-01-01', '2020-01-01', '2019-01-01', '2018-01-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2021-01-01')
        assert c.stop == np.datetime64('2018-04-01')
        assert c.step == np.timedelta64(-1, 'Y')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

        c = UniformCoordinates1d('2021-04-01', '2018-01-01', '-1,Y')
        a = np.array(['2021-04-01', '2020-04-01', '2019-04-01', '2018-04-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2021-04-01')
        assert c.stop == np.datetime64('2018-01-01')
        assert c.step == np.timedelta64(-1, 'Y')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_numerical_size(self):
        # ascending
        c = UniformCoordinates1d(0, 10, size=20)
        assert c.start == 0
        assert c.stop == 10
        assert c.step == 10/19.
        assert_equal(c.coordinates, np.linspace(0, 10, 20))
        assert_equal(c.bounds, [0, 10])
        assert c.size == 20
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending
        c = UniformCoordinates1d(10, 0, size=20)
        assert c.start == 10
        assert c.stop == 0
        assert c.step == -10/19.
        assert_equal(c.coordinates, np.linspace(10, 0, 20))
        assert_equal(c.bounds, [0, 10])
        assert c.size == 20
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_datetime_size(self):
        # ascending
        c = UniformCoordinates1d('2018-01-01', '2018-01-10', size=10)
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-01-10')
        assert_equal(c.bounds, [np.datetime64('2018-01-01'), np.datetime64('2018-01-10')])
        assert c.size == 10
        assert c.dtype == np.datetime64
        assert c.is_descending == False

        # descending
        c = UniformCoordinates1d('2018-01-10', '2018-01-01', size=10)
        assert c.start == np.datetime64('2018-01-10')
        assert c.stop == np.datetime64('2018-01-01')
        assert_equal(c.bounds, [np.datetime64('2018-01-01'), np.datetime64('2018-01-10')])
        assert c.size == 10
        assert c.dtype == np.datetime64
        assert c.is_descending == True

        # increase resolution
        c = UniformCoordinates1d('2018-01-01', '2018-01-10', size=21)
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-01-10')
        assert_equal(c.bounds, [np.datetime64('2018-01-01'), np.datetime64('2018-01-10')])
        assert c.size == 21
        assert c.dtype == np.datetime64
        assert c.is_descending == False

    def test_datetime_size_invalid(self):
        with pytest.raises(ValueError, match="Cannot divide timedelta"):
            c = UniformCoordinates1d('2018-01-01', '2018-01-10', size=20)
        
    def test_numerical_size_floating_point_error(self):
        c = UniformCoordinates1d(50.619, 50.62795, size=30)
        assert c.size == 30

    def test_numerical_singleton(self):
        # positive step
        c = UniformCoordinates1d(1, 1, 10)
        a = np.array([1], dtype=float)
        assert c.start == 1
        assert c.stop == 1
        assert c.step == 10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [1, 1])
        assert c.size == 1
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == None
        assert c.is_uniform == True

        # negative step
        c = UniformCoordinates1d(1, 1, -10)
        a = np.array([1], dtype=float)
        assert c.start == 1
        assert c.stop == 1
        assert c.step == -10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [1, 1])
        assert c.size == 1
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == None
        assert c.is_uniform == True

    def test_datetime_singleton(self):
        # positive step
        c = UniformCoordinates1d('2018-01-01', '2018-01-01', '1,D')
        a = np.array(['2018-01-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-01-01')
        assert c.step == np.timedelta64(1, 'D')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == None
        assert c.is_uniform == True
        
        # negative step
        c = UniformCoordinates1d('2018-01-01', '2018-01-01', '-1,D')
        a = np.array(['2018-01-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-01-01')
        assert c.step == np.timedelta64(-1, 'D')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == None
        assert c.is_uniform == True

    def test_from_tuple(self):
        # numerical, step
        c = UniformCoordinates1d.from_tuple((0, 10, 0.5))
        assert c.start == 0.
        assert c.stop == 10.
        assert c.step == 0.5

        # numerical, size
        c = UniformCoordinates1d.from_tuple((0, 10, 20))
        assert c.start == 0.
        assert c.stop == 10.
        assert c.size == 20

        # datetime, step
        c = UniformCoordinates1d.from_tuple(('2018-01-01', '2018-01-04', '1,D'))
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-01-04')
        assert c.step == np.timedelta64(1, 'D')

        # invalid
        with pytest.raises(ValueError, match="UniformCoordinates1d.from_tuple expects a tuple of \(start, stop, step/size\)"):
            UniformCoordinates1d.from_tuple((0, 10))

        with pytest.raises(ValueError, match="UniformCoordinates1d.from_tuple expects a tuple of \(start, stop, step/size\)"):
            UniformCoordinates1d.from_tuple(np.array([0, 10, 0.5]))

    def test_copy(self):
        c = UniformCoordinates1d(0, 10, 50, name='lat')
        c2 = c.copy()
        assert c is not c2
        assert c == c2

        c = UniformCoordinates1d(0, 10, 50, segment_lengths=0.5)
        c2 = c.copy()
        assert c is not c2
        assert c == c2

    def test_invalid_init(self):
        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 50, 0)

        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 50, -10)

        with pytest.raises(ValueError):
            UniformCoordinates1d(50, 0, 10)
        
        with pytest.raises(TypeError):
            UniformCoordinates1d(0, '2018-01-01', 10)

        with pytest.raises(TypeError):
            UniformCoordinates1d('2018-01-01', 50, 10)

        with pytest.raises(TypeError):
            UniformCoordinates1d('2018-01-01', '2018-01-02', 10)
        
        with pytest.raises(TypeError):
            UniformCoordinates1d(0., '2018-01-01', '1,D')

        with pytest.raises(TypeError):
            UniformCoordinates1d('2018-01-01', 50, '1,D')

        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 50, '1,D')

        with pytest.raises(ValueError):
            UniformCoordinates1d('a', 50, 10)

        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 'b', 10)

        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 50, 'a')

        with pytest.raises(TypeError):
            UniformCoordinates1d()

        with pytest.raises(TypeError):
            UniformCoordinates1d(0)

        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 50)

        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 50, 10, size=6)

        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 10, size=20.)
        
        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 10, size='string')

        with pytest.raises(TypeError):
            UniformCoordinates1d('2018-01-10', '2018-01-01', size='1,D')

    def test_segment_lengths(self):
        c = UniformCoordinates1d(0, 50, 10, segment_lengths=5)
        assert c.segment_lengths == 5

        c = UniformCoordinates1d(0, 50, 10, segment_lengths=[5, 5, 5, 5, 5, 5])
        assert_equal(c.segment_lengths, np.array([5., 5., 5., 5., 5., 5.]))

    def test_segment_lengths_inferred(self):
        # numerical
        c = UniformCoordinates1d(0, 50, 10)
        assert c.segment_lengths == 10

        # datetime
        c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D')
        assert c.segment_lengths == np.timedelta64(1, 'D')

        # point coordinates
        c = UniformCoordinates1d(0, 50, 10, ctype='point')
        assert c.segment_lengths is None

class TestUniformCoordinatesEq(object):
    def test_equal(self):
        c1 = UniformCoordinates1d( 0, 50,  10)
        c2 = UniformCoordinates1d( 0, 50,  10)
        c3 = UniformCoordinates1d( 0, 50,  10)
        c4 = UniformCoordinates1d( 5, 50,  10)
        c5 = UniformCoordinates1d( 0, 60,  10)
        c6 = UniformCoordinates1d( 0, 50,   5)
        c7 = UniformCoordinates1d(50,  0, -10)
        
        assert c1 == c2
        assert c1 == c3
        assert c1 != c4
        assert c1 != c5
        assert c1 != c6
        assert c1 != c7

    def test_equal_array_coordinates(self):
        c1 = UniformCoordinates1d(0, 50, 10)
        c2 = ArrayCoordinates1d([ 0, 10, 20, 30, 40, 50])
        c3 = ArrayCoordinates1d([10, 20, 30, 40, 50, 60])

        assert c1 == c2
        assert c1 != c3

    def test_equal_segment_lengths(object):
        c1 = UniformCoordinates1d(0, 50, 10)
        c2 = UniformCoordinates1d(0, 50, 10, segment_lengths=10)
        c3 = UniformCoordinates1d(0, 50, 10, segment_lengths=5)

        assert c1 == c2
        assert c1 != c3
        
class TestUniformCoordinatesSerialization(object):
    def test_definition(self):
        # numerical
        c = UniformCoordinates1d(0, 50, 10, name="lat", ctype="point")
        d = c.definition
        assert isinstance(d, dict)
        assert set(d.keys()) == set(['start', 'stop', 'step', 'name', 'ctype'])
        json.dumps(d, cls=podpac.core.utils.JSONEncoder) # test serializable
        c2 = UniformCoordinates1d.from_definition(d) # test from_definition
        assert c2 == c

        # datetimes
        c = UniformCoordinates1d('2018-01-01', '2018-01-03', '1,D')
        d = c.definition
        assert isinstance(d, dict)
        assert set(d.keys()) == set(['start', 'stop', 'step'])
        json.dumps(d, cls=podpac.core.utils.JSONEncoder) # test serializable
        c2 = UniformCoordinates1d.from_definition(d) # test from_definition
        assert c2 == c

    def test_definition_segment_lengths(self):
        c = UniformCoordinates1d(0, 50, 10, segment_lengths=0.5)
        d = c.definition
        assert isinstance(d, dict)
        assert set(d.keys()) == set(['start', 'stop', 'step', 'segment_lengths'])
        json.dumps(d, cls=podpac.core.utils.JSONEncoder) # test serializable
        c2 = UniformCoordinates1d.from_definition(d) # test from_definition
        assert c2 == c

    def test_invalid_definition(self):
        # incorrect definition
        d = {'stop': 50}
        with pytest.raises(ValueError, match='UniformCoordinates1d definition requires "start"'):
            UniformCoordinates1d.from_definition(d)

        d = {'start': 0}
        with pytest.raises(ValueError, match='UniformCoordinates1d definition requires "stop"'):
            UniformCoordinates1d.from_definition(d)

    def test_from_definition_size(self):
        # numerical
        d = {
            'start': 0,
            'stop': 50,
            'size': 6
        }
        c = UniformCoordinates1d.from_definition(d)
        assert_equal(c.coordinates, [0, 10, 20, 30, 40, 50])

        # datetime, size
        d = {
            'start': '2018-01-01',
            'stop': '2018-01-03',
            'size': 3
        }
        c = UniformCoordinates1d.from_definition(d)
        assert_equal(c.coordinates, np.array(['2018-01-01', '2018-01-02', '2018-01-03']).astype(np.datetime64))

class TestUniformCoordinatesProperties(object):
    def test_area_bounds_point(self):
        # numerical, ascending/descending and exact/inexact
        c = UniformCoordinates1d(0, 50, 10, ctype='point')
        assert_equal(c.area_bounds, [0, 50])
        c = UniformCoordinates1d(50, 0, -10, ctype='point')
        assert_equal(c.area_bounds, [0, 50])
        c = UniformCoordinates1d(0, 49, 10, ctype='point')
        assert_equal(c.area_bounds, [0, 40])
        c = UniformCoordinates1d(50, 9, -10, ctype='point')
        assert_equal(c.area_bounds, [10, 50])

        # datetime, ascending/descending and exact/inexact
        c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D', ctype='point')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-04', '2018-01-01', '-1,D', ctype='point')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-01', '2018-01-06', '2,D', ctype='point')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-05']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-06', '2018-01-01', '-2,D', ctype='point')
        assert_equal(c.area_bounds, np.array(['2018-01-02', '2018-01-06']).astype(np.datetime64))

    def test_area_bounds_left(self):
        # numerical, ascending/descending and exact/inexact/singleton
        c = UniformCoordinates1d(0, 50, 10, ctype='left')
        assert_equal(c.area_bounds, [0, 60])
        c = UniformCoordinates1d(50, 0, -10, ctype='left')
        assert_equal(c.area_bounds, [0, 60])
        c = UniformCoordinates1d(0, 49, 10, ctype='left')
        assert_equal(c.area_bounds, [0, 50.0])
        c = UniformCoordinates1d(50, 9, -10, ctype='left')
        assert_equal(c.area_bounds, [10, 60.0])
        c = UniformCoordinates1d(0, 0, 10, ctype='left')
        assert_equal(c.area_bounds, [0, 10])
        c = UniformCoordinates1d(0, 0, -10, ctype='left')
        assert_equal(c.area_bounds, [0, 10])

        # datetime, ascending/descending and exact/inexact/singleton
        c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D', ctype='left')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-05']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-04', '2018-01-01', '-1,D', ctype='left')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-05']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-01', '2018-01-06', '2,D', ctype='left')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-07']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-06', '2018-01-01', '-2,D', ctype='left')
        assert_equal(c.area_bounds, np.array(['2018-01-02', '2018-01-08']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-01', '2018-01-01', '1,D', ctype='left')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-02']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-01', '2018-01-01', '-1,D', ctype='left')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-02']).astype(np.datetime64))

    def test_area_bounds_right(self):
        # numerical, ascending/descending and exact/inexact/singleton
        c = UniformCoordinates1d(0, 50, 10, ctype='right')
        assert_equal(c.area_bounds, [-10, 50])
        c = UniformCoordinates1d(50, 0, -10, ctype='right')
        assert_equal(c.area_bounds, [-10, 50])
        c = UniformCoordinates1d(0, 49, 10, ctype='right')
        assert_equal(c.area_bounds, [-10, 40])
        c = UniformCoordinates1d(50, 9, -10, ctype='right')
        assert_equal(c.area_bounds, [0, 50])
        c = UniformCoordinates1d(0, 0, 10, ctype='right')
        assert_equal(c.area_bounds, [-10, 0])
        c = UniformCoordinates1d(0, 0, -10, ctype='right')
        assert_equal(c.area_bounds, [-10, 0])

        # datetime, ascending/descending and exact/inexact/singleton
        c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D', ctype='right')
        assert_equal(c.area_bounds, np.array(['2017-12-31', '2018-01-04']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-04', '2018-01-01', '-1,D', ctype='right')
        assert_equal(c.area_bounds, np.array(['2017-12-31', '2018-01-04']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-01', '2018-01-06', '2,D', ctype='right')
        assert_equal(c.area_bounds, np.array(['2017-12-30', '2018-01-05']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-06', '2018-01-01', '-2,D', ctype='right')
        assert_equal(c.area_bounds, np.array(['2017-12-31', '2018-01-06']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-01', '2018-01-01', '1,D', ctype='right')
        assert_equal(c.area_bounds, np.array(['2017-12-31', '2018-01-01']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-01', '2018-01-01', '-1,D', ctype='right')
        assert_equal(c.area_bounds, np.array(['2017-12-31', '2018-01-01']).astype(np.datetime64))

    def test_area_bounds_midpoint(self):
        # numerical, ascending/descending and exact/inexact/singleton
        c = UniformCoordinates1d(0, 50, 10, ctype='midpoint')
        assert_equal(c.area_bounds, [-5, 55])
        c = UniformCoordinates1d(50, 0, -10, ctype='midpoint')
        assert_equal(c.area_bounds, [-5, 55])
        c = UniformCoordinates1d(0, 49, 10, ctype='midpoint')
        assert_equal(c.area_bounds, [-5, 45])
        c = UniformCoordinates1d(50, 9, -10, ctype='midpoint')
        assert_equal(c.area_bounds, [5, 55])
        c = UniformCoordinates1d(0, 0, 10, ctype='midpoint')
        assert_equal(c.area_bounds, [-5, 5])
        c = UniformCoordinates1d(0, 0, -10, ctype='midpoint')
        assert_equal(c.area_bounds, [-5, 5])

        # datetime, ascending/descending and exact/inexact/singleton
        c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2017-12-31 12', '2018-01-04 12']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-04', '2018-01-01', '-1,D', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2017-12-31 12', '2018-01-04 12']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-01', '2018-01-06', '2,D', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2017-12-31', '2018-01-06']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-06', '2018-01-01', '-2,D', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-07']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-01', '2018-01-01', '1,D', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2017-12-31 12', '2018-01-01 12']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-01', '2018-01-01', '-1,D', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2017-12-31 12', '2018-01-01 12']).astype(np.datetime64))

class TestUniformCoordinatesIndexing(object):
    def test_len(self):
        c = UniformCoordinates1d(0, 50, 10)
        assert len(c) == 6

    def test_index(self):
        c = UniformCoordinates1d(0, 50, 10, name='lat', ctype='point')
        
        # int
        c2 = c[2]
        assert isinstance(c2, Coordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [20])

        c2 = c[-2]
        assert isinstance(c2, Coordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [40])
        
        # slice
        c2 = c[:2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 0
        assert c2.stop == 10
        assert c2.step == 10

        c2 = c[2:]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 20
        assert c2.stop == 50
        assert c2.step == 10
        
        c2 = c[::2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 0
        assert c2.stop == 50
        assert c2.step == 20
        
        c2 = c[1:-1]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 10
        assert c2.stop == 40
        assert c2.step == 10

        c2 = c[-3:5]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 30
        assert c2.stop == 40
        assert c2.step == 10
        
        c2 = c[::-1]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 50
        assert c2.stop == 0
        assert c2.step == -10
        
        # index array
        c2 = c[[0, 1, 3]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [0, 10, 30])

        c2 = c[[3, 1, 0]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [30, 10, 0])

        c2 = c[[0, 3, 1]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [0, 30, 10])

        c2 = c[[]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [])

        c2 = c[0:0]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [])

        c2 = c[[]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [])

        # boolean array
        c2 = c[[True, True, True, False, True, False]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [0, 10, 20, 40])

        # invalid
        with pytest.raises(IndexError):
            c[0.3]

        with pytest.raises(IndexError):
            c[10]

    def test_index_descending(self):
        c = UniformCoordinates1d(50, 0, -10, name='lat', ctype='point')
        
        # int
        c2 = c[2]
        assert isinstance(c2, Coordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [30])

        c2 = c[-2]
        assert isinstance(c2, Coordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [10])

        # slice
        c2 = c[:2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 50
        assert c2.stop == 40
        assert c2.step == -10

        c2 = c[2:]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 30
        assert c2.stop == 0
        assert c2.step == -10
        
        c2 = c[::2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 50
        assert c2.stop == 0
        assert c2.step == -20
        
        c2 = c[1:-1]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 40
        assert c2.stop == 10
        assert c2.step == -10

        c2 = c[-3:5]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 20
        assert c2.stop == 10
        assert c2.step == -10
        
        c2 = c[::-1]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 0
        assert c2.stop == 50
        assert c2.step == 10
        
        # index array
        c2 = c[[0, 1, 3]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [50, 40, 20])
        
        c2 = c[[3, 1, 0]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [20, 40, 50])
        
        c2 = c[[0, 3, 1]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [50, 20, 40])

        # boolean array
        c2 = c[[True, True, True, False, True, False]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [50, 40, 30, 10])

    def test_index_segment_lengths(self):
        # array of segment_lengths
        c = UniformCoordinates1d(0, 50, 10, segment_lengths=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        
        c2 = c[1]
        assert c2.segment_lengths == 0.2 or np.array_equal(c2.segment_lengths, [0.2])
        
        c2 = c[1:3]
        assert_equal(c2.segment_lengths, [0.2, 0.3])
        
        c2 = c[[1, 3]]
        assert_equal(c2.segment_lengths, [0.2, 0.4])
        
        c2 = c[[4, 1, 2]]
        assert_equal(c2.segment_lengths, [0.5, 0.2, 0.3])
        
        c2 = c[[True, True, True, False, True, False]]
        assert_equal(c2.segment_lengths, [0.1, 0.2, 0.3, 0.5])

        # uniform segment_lengths
        c = UniformCoordinates1d(0, 50, 10, segment_lengths=0.5)
        
        c2 = c[1]
        assert c2.segment_lengths == 0.5
        
        c2 = c[1:3]
        assert c2.segment_lengths == 0.5
        
        c2 = c[[1, 3]]
        assert c2.segment_lengths == 0.5
        
        c2 = c[[4, 1, 2]]
        assert c2.segment_lengths == 0.5
        
        c2 = c[[True, True, True, False, True, False]]
        assert c2.segment_lengths == 0.5

        # inferred segment_lengths
        c = UniformCoordinates1d(0, 50, 10)
        c2 = c[1]
        assert c2.segment_lengths == 10 or np.array_equal(c2.segment_lengths, [10])
        
class TestUniformCoordinatesSelection(object):
    def test_select_all_shortcut(self):
        c = UniformCoordinates1d(20., 70., 10.)
        
        s = c.select([0, 100])
        assert s.start == 20.
        assert s.stop == 70.
        assert s.step == 10.0

        s, I = c.select([0, 100], return_indices=True)
        assert s.start == 20.
        assert s.stop == 70.
        assert s.step == 10.0
        assert_equal(c[I], s)

    def test_select_none_shortcut(self):
        c = UniformCoordinates1d(20., 70., 10.)
                
        # above
        s = c.select([100, 200])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])
        
        s, I = c.select([100, 200], return_indices=True)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])
        assert c[I] == s

        # below
        s = c.select([0, 5])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        s, I = c.select([0, 5], return_indices=True)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])
        assert c[I] == s

    def test_select_ascending(self):
        c = UniformCoordinates1d(20., 70., 10.)

        # inner
        s = c.select([35., 55.])
        assert s.start == 40.
        assert s.stop == 50.
        assert s.step == 10.

        s, I = c.select([35., 55.], return_indices=True)
        assert s.start == 40.
        assert s.stop == 50.
        assert s.step == 10.
        assert c[I] == s

        # inner with aligned bounds
        s = c.select([30., 60.])
        assert s.start == 30.
        assert s.stop == 60.
        assert s.step == 10.

        s, I = c.select([30., 60.], return_indices=True)
        assert s.start == 30.
        assert s.stop == 60.
        assert s.step == 10.
        assert c[I] == s

        # above
        s = c.select([45, 100])
        assert s.start == 50.
        assert s.stop == 70.
        assert s.step == 10.

        s, I = c.select([45, 100], return_indices=True)
        assert s.start == 50.
        assert s.stop == 70.
        assert s.step == 10.
        assert c[I] == s
        
        # below
        s = c.select([5, 55])
        assert s.start == 20.
        assert s.stop == 50.
        assert s.step == 10.
        
        s, I = c.select([5, 55], return_indices=True)
        assert s.start == 20.
        assert s.stop == 50.
        assert s.step == 10.
        assert c[I] == s

        # between coordinates
        s = c.select([52, 55])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        s, I = c.select([52, 55], return_indices=True)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

        # backwards bounds
        s = c.select([70, 30])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        s, I = c.select([70, 30], return_indices=True)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

    def test_select_descending(self):
        c = UniformCoordinates1d(70., 20., -10.)

        # inner
        s = c.select([35., 55.])
        assert s.start == 50.
        assert s.stop == 40.
        assert s.step == -10.

        s, I = c.select([35., 55.], return_indices=True)
        assert s.start == 50.
        assert s.stop == 40.
        assert s.step == -10.
        assert c[I] == s

        # inner with aligned bounds
        s = c.select([30., 60.])
        assert s.start == 60.
        assert s.stop == 30.
        assert s.step == -10.

        s, I = c.select([30., 60.], return_indices=True)
        assert s.start == 60.
        assert s.stop == 30.
        assert s.step == -10.
        assert c[I] == s

        # above
        s = c.select([45, 100])
        assert s.start == 70.
        assert s.stop == 50.
        assert s.step == -10.

        s, I = c.select([45, 100], return_indices=True)
        assert s.start == 70.
        assert s.stop == 50.
        assert s.step == -10.
        assert c[I] == s
        
        # below
        s = c.select([5, 55])
        assert s.start == 50.
        assert s.stop == 20.
        assert s.step == -10.
        
        s, I = c.select([5, 55], return_indices=True)
        assert s.start == 50.
        assert s.stop == 20.
        assert s.step == -10.
        assert c[I] == s

        # between coordinates
        s = c.select([52, 55])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        s, I = c.select([52, 55], return_indices=True)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

        # backwards bounds
        s = c.select([70, 30])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        s, I = c.select([70, 30], return_indices=True)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

    def test_select_outer(self):
        c = UniformCoordinates1d(20., 70., 10.)

        # inner
        s = c.select([35., 55.], outer=True)
        assert s.start == 30.
        assert s.stop == 60.
        assert s.step == 10.

        s, I = c.select([35., 55.], outer=True, return_indices=True)
        assert s.start == 30.
        assert s.stop == 60.
        assert s.step == 10.
        assert c[I] == s

        # inner with aligned bounds
        s = c.select([30., 60.], outer=True)
        assert s.start == 30.
        assert s.stop == 60.
        assert s.step == 10.

        s, I = c.select([30., 60.], outer=True, return_indices=True)
        assert s.start == 30.
        assert s.stop == 60.
        assert s.step == 10.
        assert c[I] == s

        # above
        s = c.select([45, 100], outer=True)
        assert s.start == 40.
        assert s.stop == 70.
        assert s.step == 10.

        s, I = c.select([45, 100], outer=True, return_indices=True)
        assert s.start == 40.
        assert s.stop == 70.
        assert s.step == 10.
        assert c[I] == s
        
        # below
        s = c.select([5, 55], outer=True)
        assert s.start == 20.
        assert s.stop == 60.
        assert s.step == 10.
        
        s, I = c.select([5, 55], outer=True, return_indices=True)
        assert s.start == 20.
        assert s.stop == 60.
        assert s.step == 10.
        assert c[I] == s

        # between coordinates
        s = c.select([52, 55], outer=True)
        assert s.start == 50.
        assert s.stop == 60.
        assert s.step == 10.

        s, I = c.select([52, 55], outer=True, return_indices=True)
        assert s.start == 50.
        assert s.stop == 60.
        assert s.step == 10.
        assert c[I] == s

        # backwards bounds
        s = c.select([70, 30], outer=True)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        s, I = c.select([70, 30], outer=True, return_indices=True)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

    def test_intersect(self):
        a = ArrayCoordinates1d([40., 70., 50.,])
        u1 = UniformCoordinates1d(10., 60., 10.)
        u2 = UniformCoordinates1d(35., 85., 5.)
        
        assert isinstance(u1.intersect(a), UniformCoordinates1d)
        assert isinstance(u1.intersect(u2), UniformCoordinates1d)