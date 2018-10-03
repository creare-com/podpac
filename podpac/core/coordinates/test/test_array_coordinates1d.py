
from datetime import datetime

import pytest
import traitlets as tl
import numpy as np
from numpy.testing import assert_equal

from podpac.core.units import Units
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d

class TestArrayCoordinatesCreation(object):
    def test_empty(self):
        c = ArrayCoordinates1d()
        a = np.array([], dtype=float)
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([np.nan, np.nan]))
        assert c.size == 0
        assert c.dtype is None
        assert c.is_monotonic is None
        assert c.is_descending is None
        assert c.is_uniform is None

        c = ArrayCoordinates1d([])
        assert_equal(c.coords, a)

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
        c = ArrayCoordinates1d(a, dtype='point')
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
        c = ArrayCoordinates1d()
        
        with pytest.raises(ValueError):
            c.coords = np.array([1, 2, 3])

        with pytest.raises(ValueError):
            c.coords = np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_name(self):
        ArrayCoordinates1d()
        ArrayCoordinates1d(name='lat')
        ArrayCoordinates1d(name='lon')
        ArrayCoordinates1d(name='alt')
        ArrayCoordinates1d(name='time')

        with pytest.raises(tl.TraitError):
            ArrayCoordinates1d(name='depth')

    def test_extents(self):
        c = ArrayCoordinates1d()
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

class TestArrayCoordinatesProperties(object):
    def test_properties(self):
        c = ArrayCoordinates1d()
        assert isinstance(c.properties, dict)
        assert set(c.properties.keys()) == set(['ctype', 'coord_ref_sys'])

        c = ArrayCoordinates1d(name='lat')
        assert isinstance(c.properties, dict)
        assert set(c.properties.keys()) == set(['ctype', 'coord_ref_sys', 'name'])

        c = ArrayCoordinates1d(units=Units())
        assert isinstance(c.properties, dict)
        assert set(c.properties.keys()) == set(['ctype', 'coord_ref_sys', 'units'])

        c = ArrayCoordinates1d(extents=[0, 1])
        assert isinstance(c.properties, dict)
        assert set(c.properties.keys()) == set(['ctype', 'coord_ref_sys', 'extents'])

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

@pytest.mark.skip("needs update")
class TestArrayCoordinatesSelection(object):
    def test_select(self):
        c = ArrayCoordinates1d([20., 50., 60., 90., 40., 10.])
        
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
        s = c.select([30., 70.])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [50., 60., 40.])

        # partial, inner exact
        s = c.select([40., 60.])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [50., 60., 40.])

        # partial, none
        s = c.select([52, 55])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        # partial, backwards bounds
        s = c.select([70, 30])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        # empty coords
        c = ArrayCoordinates1d()        
        s = c.select([0, 1])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

    def test_select_ind(self):
        c = ArrayCoordinates1d([20., 50., 60., 90., 40., 10.])
        
        # full selection
        I = c.select([0, 100], ind=True)
        assert_equal(c.coordinates[I], c.coordinates)

        # none, above
        I = c.select([100, 200], ind=True)
        assert_equal(c.coordinates[I], [])

        # none, below
        I = c.select([0, 5], ind=True)
        assert_equal(c.coordinates[I], [])

        # partial, above
        I = c.select([50, 100], ind=True)
        assert_equal(c.coordinates[I], [50., 60., 90.])

        # partial, below
        I = c.select([0, 50], ind=True)
        assert_equal(c.coordinates[I], [20., 50., 40., 10.])

        # partial, inner
        I = c.select([30., 70.], ind=True)
        assert_equal(c.coordinates[I], [50., 60., 40.])

        # partial, inner exact
        I = c.select([40., 60.], ind=True)
        assert_equal(c.coordinates[I], [50., 60., 40.])

        # partial, none
        I = c.select([52, 55], ind=True)
        assert_equal(c.coordinates[I], [])

        # partial, backwards bounds
        I = c.select([70, 30], ind=True)
        assert_equal(c.coordinates[I], [])

        # empty coords
        c = ArrayCoordinates1d()        
        I = c.select([0, 1], ind=True)
        assert_equal(c.coordinates[I], [])

    def test_select_ascending(self):
        c = MonotonicCoordinates1d([10., 20., 40., 50., 60., 90.])
        
        # full and empty selection type
        assert isinstance(c.select([0, 100]), MonotonicCoordinates1d)
        assert isinstance(c.select([100, 200]), ArrayCoordinates1d)
        assert isinstance(c.select([0, 5]), ArrayCoordinates1d)
        
        # partial, above
        s = c.select([50, 100], pad=0)
        assert isinstance(s, MonotonicCoordinates1d)
        assert_equal(s.coordinates, [50., 60., 90.])

        # partial, below
        s = c.select([0, 50], pad=0)
        assert isinstance(s, MonotonicCoordinates1d)
        assert_equal(s.coordinates, [10., 20., 40., 50.])

        # partial, inner
        s = c.select([30., 70.], pad=0)
        assert isinstance(s, MonotonicCoordinates1d)
        assert_equal(s.coordinates, [40., 50., 60.])

        # partial, inner exact
        s = c.select([40., 60.], pad=0)
        assert isinstance(s, MonotonicCoordinates1d)
        assert_equal(s.coordinates, [40., 50., 60.])

        # partial, none
        s = c.select([52, 55], pad=0)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        # partial, backwards bounds
        s = c.select([70, 30], pad=0)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

    def test_select_descending(self):
        c = MonotonicCoordinates1d([90., 60., 50., 40., 20., 10.])
        
        # full and empty selection type
        assert isinstance(c.select([0, 100]), MonotonicCoordinates1d)
        assert isinstance(c.select([100, 200]), ArrayCoordinates1d)
        assert isinstance(c.select([0, 5]), ArrayCoordinates1d)
        
        # partial, above
        s = c.select([50, 100], pad=0)
        assert isinstance(s, MonotonicCoordinates1d)
        assert_equal(s.coordinates, [90., 60., 50.])

        # partial, below
        s = c.select([0, 50], pad=0)
        assert isinstance(s, MonotonicCoordinates1d)
        assert_equal(s.coordinates, [50., 40., 20., 10.])

        # partial, inner
        s = c.select([30., 70.], pad=0)
        assert isinstance(s, MonotonicCoordinates1d)
        assert_equal(s.coordinates, [60., 50., 40.])

        # partial, inner exact
        s = c.select([40., 60.], pad=0)
        assert isinstance(s, MonotonicCoordinates1d)
        assert_equal(s.coordinates, [60., 50., 40.])

        # partial, none
        s = c.select([52, 55], pad=0)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        # partial, backwards bounds
        s = c.select([70, 30], pad=0)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

    def test_select_ind(self):
        c = MonotonicCoordinates1d([10., 20., 40., 50., 60., 90.])
        
        # partial, above
        s = c.select([50, 100], ind=True, pad=0)
        assert_equal(c.coordinates[s], [50., 60., 90.])

        # partial, below
        s = c.select([0, 50], ind=True, pad=0)
        assert_equal(c.coordinates[s], [10., 20., 40., 50.])

        # partial, inner
        s = c.select([30., 70.], ind=True, pad=0)
        assert_equal(c.coordinates[s], [40., 50., 60.])

        # partial, inner exact
        s = c.select([40., 60.], ind=True, pad=0)
        assert_equal(c.coordinates[s], [40., 50., 60.])

        # partial, none
        s = c.select([52, 55], ind=True, pad=0)
        assert_equal(c.coordinates[s], [])

        # partial, backwards bounds
        s = c.select([70, 30], ind=True, pad=0)
        assert_equal(c.coordinates[s], [])

    def test_intersect(self):
        a = ArrayCoordinates1d([20., 50., 60., 10.])
        b = ArrayCoordinates1d([55., 65., 95., 45.])
        c = ArrayCoordinates1d([80., 70., 90.])
        e = ArrayCoordinates1d()
        
        # ArrayCoordinates1d other, both directions
        ab = a.intersect(b)
        assert isinstance(ab, ArrayCoordinates1d)
        assert_equal(ab.coordinates, [50., 60.])
        
        ba = b.intersect(a)
        assert isinstance(ba, ArrayCoordinates1d)
        assert_equal(ba.coordinates, [55., 45.])

        # ArrayCoordinates1d other, no overlap
        ac = a.intersect(c)
        assert isinstance(ac, ArrayCoordinates1d)
        assert_equal(ac.coordinates, [])

        # empty self
        ea = e.intersect(a)
        assert isinstance(ea, ArrayCoordinates1d)
        assert_equal(ea.coordinates, [])

        # empty other
        ae = a.intersect(e)
        assert isinstance(ae, ArrayCoordinates1d)
        assert_equal(ae.coordinates, [])

        # MonotonicCoordinates1d other
        m = MonotonicCoordinates1d([45., 55., 65., 95.])
        am = a.intersect(m)
        assert isinstance(am, ArrayCoordinates1d)
        assert_equal(am.coordinates, [50., 60.])

        # UniformCoordinates1d other
        u = UniformCoordinates1d(45, 95, 10)
        au = a.intersect(u)
        assert isinstance(au, ArrayCoordinates1d)
        assert_equal(au.coordinates, [50., 60.])

    def test_intersect_ind(self):
        a = ArrayCoordinates1d([20., 50., 60., 10.])
        b = ArrayCoordinates1d([55., 65., 95., 45.])
        c = ArrayCoordinates1d([80., 70., 90.])
        e = ArrayCoordinates1d()
        
        # ArrayCoordinates1d other, both directions
        I = a.intersect(b, ind=True)
        assert_equal(a.coordinates[I], [50., 60.])
        
        I = b.intersect(a, ind=True)
        assert_equal(b.coordinates[I], [55., 45.])

        # ArrayCoordinates1d other, no overlap
        I = a.intersect(c, ind=True)
        assert_equal(a.coordinates[I], [])

        # empty self
        I = e.intersect(a, ind=True)
        assert_equal(e.coordinates[I], [])

        # empty other
        I = a.intersect(e, ind=True)
        assert_equal(a.coordinates[I], [])

        # MonotonicCoordinates1d other
        m = MonotonicCoordinates1d([45., 55., 65., 95.])
        I = a.intersect(m, ind=True)
        assert_equal(a.coordinates[I], [50., 60.])

        # UniformCoordinates1d other
        u = UniformCoordinates1d(45, 95, 10)
        I = a.intersect(u, ind=True)
        assert_equal(a.coordinates[I], [50., 60.])

    def test_intersect(self):
        # MonotonicCoordinates1d other
        a = MonotonicCoordinates1d([10., 20., 50., 60.])
        b = MonotonicCoordinates1d([45., 55., 65., 95.])
        assert isinstance(a.intersect(b), MonotonicCoordinates1d)

        # ArrayCoordinates1d other
        c = ArrayCoordinates1d([20., 50., 60., 10.])
        assert isinstance(a.intersect(c), MonotonicCoordinates1d)
        
        # UniformCoordinates1d
        u = UniformCoordinates1d(45, 95, 10)
        assert isinstance(a.intersect(u), MonotonicCoordinates1d)