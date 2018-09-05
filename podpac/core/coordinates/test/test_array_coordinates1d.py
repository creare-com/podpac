
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

@pytest.mark.skip("TODO")
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

@pytest.mark.skip("TODO")
class TestArrayCoordinatesConcatenation(object):
    def test_concat(self):
        a = ArrayCoordinates1d([20., 50., 60., 10.])
        b = ArrayCoordinates1d([55., 65., 95., 45.])
        c = ArrayCoordinates1d([80., 70., 90.])
        e = ArrayCoordinates1d()

        t = ArrayCoordinates1d(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03'])
        
        # ArrayCoordinates1d other, both directions
        ab = a.concat(b)
        assert isinstance(ab, ArrayCoordinates1d)
        assert_equal(ab.coordinates, [20., 50., 60., 10., 55., 65., 95., 45.])
        
        ba = b.concat(a)
        assert isinstance(ba, ArrayCoordinates1d)
        assert_equal(ba.coordinates, [55., 65., 95., 45., 20., 50., 60., 10.])

        # empty self
        ea = e.concat(a)
        assert isinstance(ea, ArrayCoordinates1d)
        assert_equal(ea.coordinates, a.coordinates)

        et = e.concat(t)
        assert isinstance(et, ArrayCoordinates1d)
        assert_equal(et.coordinates, t.coordinates)

        # empty other
        ae = a.concat(e)
        assert isinstance(ae, ArrayCoordinates1d)
        assert_equal(ae.coordinates, a.coordinates)

        te = t.concat(e)
        assert isinstance(te, ArrayCoordinates1d)
        assert_equal(te.coordinates, t.coordinates)

        # MonotonicCoordinates1d other
        m = MonotonicCoordinates1d([45., 55., 65., 95.])
        am = a.concat(m)
        assert isinstance(am, ArrayCoordinates1d)
        assert_equal(am.coordinates, [20., 50., 60., 10., 45., 55., 65., 95.])

        # UniformCoordinates1d other
        u = UniformCoordinates1d(45, 95, 10)
        au = a.concat(u)
        assert isinstance(au, ArrayCoordinates1d)
        assert_equal(au.coordinates, [20., 50., 60., 10., 45., 55., 65., 75., 85., 95.])

        # type error
        with pytest.raises(TypeError):
            a.concat(5)

        with pytest.raises(TypeError):
            a.concat(t)

        with pytest.raises(TypeError):
            t.concat(a)

    def test_concat_equal(self):
        # ArrayCoordinates1d other
        c = ArrayCoordinates1d([20., 50., 60., 10.])
        c.concat(ArrayCoordinates1d([55., 65., 95., 45.]), inplace=True)
        assert_equal(c.coordinates, [20., 50., 60., 10., 55., 65., 95., 45.])
        
        # empty self
        c = ArrayCoordinates1d()
        c.concat(ArrayCoordinates1d([55., 65., 95., 45.]), inplace=True)
        assert_equal(c.coordinates, [55., 65., 95., 45.])

        c = ArrayCoordinates1d()
        c.concat(ArrayCoordinates1d(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03']), inplace=True)
        assert_equal(c.coordinates, np.array(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03']).astype(np.datetime64))

        # empty other
        c = ArrayCoordinates1d([20., 50., 60., 10.])
        c.concat(ArrayCoordinates1d(), inplace=True)
        assert_equal(c.coordinates, [20., 50., 60., 10.])

        c = ArrayCoordinates1d(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03'])
        c.concat(ArrayCoordinates1d(), inplace=True)
        assert_equal(c.coordinates, np.array(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03']).astype(np.datetime64))

        # MonotonicCoordinates1d other
        c = ArrayCoordinates1d([20., 50., 60., 10.])
        c.concat(MonotonicCoordinates1d([45., 55., 65., 95.]), inplace=True)
        assert_equal(c.coordinates, [20., 50., 60., 10., 45., 55., 65., 95.])

        # UniformCoordinates1d other
        c = ArrayCoordinates1d([20., 50., 60., 10.])
        c.concat(UniformCoordinates1d(45, 95, 10), inplace=True)
        assert_equal(c.coordinates, [20., 50., 60., 10., 45., 55., 65., 75., 85., 95.])

    def test_concat(self):
        a = MonotonicCoordinates1d([10., 20., 50., 60.])
        b = MonotonicCoordinates1d([45., 55., 65., 95.])
        c = MonotonicCoordinates1d([70., 80., 90.])
        d = MonotonicCoordinates1d([35., 25., 15.])
        e = MonotonicCoordinates1d()
        f = MonotonicCoordinates1d([90., 80., 70.])
        o = ArrayCoordinates1d([20., 50., 60., 10.])
        u = UniformCoordinates1d(45, 95, 10)
        v = UniformCoordinates1d(75, 95, 10)
        t = MonotonicCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'])
        
        # overlap, ascending
        ab = a.concat(b)
        assert isinstance(ab, ArrayCoordinates1d)
        assert not isinstance(ab, MonotonicCoordinates1d)
        assert_equal(ab.coordinates, [10., 20., 50., 60., 45., 55., 65., 95.])
        
        ba = b.concat(a)
        assert isinstance(ba, ArrayCoordinates1d)
        assert not isinstance(ba, MonotonicCoordinates1d)
        assert_equal(ba.coordinates, [45., 55., 65., 95., 10., 20., 50., 60.])

        # overlap, descending
        da = d.concat(a)
        assert isinstance(da, ArrayCoordinates1d)
        assert not isinstance(da, MonotonicCoordinates1d)
        assert_equal(da.coordinates, [35., 25., 15., 10., 20., 50., 60.])

        # ascending
        ac = a.concat(c)
        assert isinstance(ac, MonotonicCoordinates1d)
        assert_equal(ac.coordinates, [10., 20., 50., 60., 70., 80., 90])

        # ascending, reverse
        cd = c.concat(d)
        assert isinstance(cd, MonotonicCoordinates1d)
        assert_equal(cd.coordinates, [15., 25., 35., 70., 80., 90])

        # descending
        fd = f.concat(d)
        assert isinstance(fd, MonotonicCoordinates1d)
        assert_equal(fd.coordinates, [90., 80., 70, 35, 25, 15])

        # descending, reverse
        dc = d.concat(c)
        assert isinstance(cd, MonotonicCoordinates1d)
        assert_equal(cd.coordinates, [15., 25., 35., 70., 80., 90])

        # empty self
        ea = e.concat(a)
        assert isinstance(ea, MonotonicCoordinates1d)
        assert_equal(ea.coordinates, a.coordinates)

        et = e.concat(t)
        assert isinstance(et, MonotonicCoordinates1d)
        assert_equal(et.coordinates, t.coordinates)
        
        eu = e.concat(u)
        assert isinstance(eu, UniformCoordinates1d)
        assert_equal(eu.coordinates, u.coordinates)
        
        eo = e.concat(o)
        assert isinstance(eo, ArrayCoordinates1d)
        assert not isinstance(eo, MonotonicCoordinates1d)
        assert_equal(eo.coordinates, o.coordinates)

        # empty other
        ae = a.concat(e)
        assert isinstance(ae, ArrayCoordinates1d)
        assert_equal(ae.coordinates, a.coordinates)

        te = t.concat(e)
        assert isinstance(te, ArrayCoordinates1d)
        assert_equal(te.coordinates, t.coordinates)

        # ArrayCoordinates1d other
        co = c.concat(o)
        assert isinstance(co, ArrayCoordinates1d)
        assert not isinstance(co, MonotonicCoordinates1d)
        assert_equal(co.coordinates, [70., 80., 90., 20., 50., 60., 10.])

        # UniformCoordinates1d other, overlap
        au = a.concat(u)
        assert isinstance(au, ArrayCoordinates1d)
        assert not isinstance(au, MonotonicCoordinates1d)
        assert_equal(au.coordinates, [10., 20., 50., 60., 45., 55., 65., 75., 85., 95.])

        # UniformCoordinates1d other, no overlap
        av = a.concat(v)
        assert isinstance(av, MonotonicCoordinates1d)
        assert_equal(av.coordinates, [10., 20., 50., 60., 75., 85., 95.])

    def test_concat_equal(self):        
        # ascending
        c = MonotonicCoordinates1d([10., 20., 50., 60.])
        c.concat(MonotonicCoordinates1d([70., 80., 90.]), inplace=True)
        assert_equal(c.coordinates, [10., 20., 50., 60., 70., 80., 90])

        # ascending, reverse
        c = MonotonicCoordinates1d([70., 80., 90.])
        c.concat(MonotonicCoordinates1d([35., 25., 15.]), inplace=True)
        assert_equal(c.coordinates, [15., 25., 35., 70., 80., 90])

        # descending
        c = MonotonicCoordinates1d([90., 80., 70.])
        c.concat(MonotonicCoordinates1d([35., 25., 15.]), inplace=True)
        assert_equal(c.coordinates, [90., 80., 70, 35, 25, 15])

        # descending, reverse
        c = MonotonicCoordinates1d([35., 25., 15.])
        c.concat(MonotonicCoordinates1d([70., 80., 90.]), inplace=True)
        assert_equal(c.coordinates, [90.,  80.,  70.,  35.,  25.,  15.])

        # UniformCoordinates1d other, no overlap
        c = MonotonicCoordinates1d([10., 20., 50., 60.])
        c.concat(UniformCoordinates1d(75, 95, 10), inplace=True)
        assert_equal(c.coordinates, [10., 20., 50., 60., 75., 85., 95.])

        # empty self
        c = MonotonicCoordinates1d()
        c.concat(MonotonicCoordinates1d([10., 20., 50., 60.]), inplace=True)
        assert_equal(c.coordinates, [10., 20., 50., 60.])

        c = MonotonicCoordinates1d()
        c.concat(MonotonicCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04']), inplace=True)
        assert_equal(c.coordinates, np.array(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04']).astype(np.datetime64))
        
        c = MonotonicCoordinates1d()
        c.concat(UniformCoordinates1d(75, 95, 10), inplace=True)
        assert_equal(c.coordinates, [75., 85., 95.])
        
        # empty other
        c = MonotonicCoordinates1d([10., 20., 50., 60.])
        c.concat(MonotonicCoordinates1d(), inplace=True)
        assert_equal(c.coordinates, [10., 20., 50., 60.])

        c = MonotonicCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'])
        c.concat(MonotonicCoordinates1d(), inplace=True)
        assert_equal(c.coordinates, np.array(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04']).astype(np.datetime64))

        # overlap, ascending
        c = MonotonicCoordinates1d([10., 20., 50., 60.])
        with pytest.raises(ValueError):
            c.concat(MonotonicCoordinates1d([45., 55., 65., 95.]), inplace=True)
        
        # overlap, descending
        c = MonotonicCoordinates1d([35., 25., 15.])
        with pytest.raises(ValueError):
            c.concat(MonotonicCoordinates1d([10., 20., 50., 60.]), inplace=True)

        # overlap, UniformCoordinates1d
        c = MonotonicCoordinates1d([10., 20., 50., 60.])
        with pytest.raises(ValueError):
            c.concat(UniformCoordinates1d(45, 95, 10), inplace=True)

        # ArrayCoordinates1d other
        c = MonotonicCoordinates1d([45., 55., 65., 95.])
        with pytest.raises(TypeError):
            c.concat(ArrayCoordinates1d([20., 50., 60., 10.]), inplace=True)

        c = MonotonicCoordinates1d() # should fail here even when empty
        with pytest.raises(TypeError):
            c.concat(ArrayCoordinates1d([20., 50., 60, 10.]), inplace=True)

@pytest.mark.skip("TODO")
class TestArrayCoordinatesAdd(object):
    def test_add(self):
        a = ArrayCoordinates1d([20., 50., 60., 10.])
        t = ArrayCoordinates1d(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03'])
        e = ArrayCoordinates1d()

        # empty
        r = e.add(1)
        assert isinstance(r, ArrayCoordinates1d)
        assert_equal(r.coordinates, [])

        r = e.add('10,D')
        assert isinstance(r, ArrayCoordinates1d)
        assert_equal(r.coordinates, [])
        
        # standard
        r = a.add(1)
        assert isinstance(r, ArrayCoordinates1d)
        assert_equal(r.coordinates, [21., 51., 61., 11.])

        r = t.add('10,D')
        assert isinstance(r, ArrayCoordinates1d)
        assert_equal(r.coordinates, np.array(['2018-01-12', '2018-01-11', '2018-01-14', '2018-01-13']).astype(np.datetime64))

        # type error
        with pytest.raises(TypeError):
            a.add(a)

        with pytest.raises(TypeError):
            a.add('10,D')

        with pytest.raises(TypeError):
            t.add(4)

    def test_add_equal(self):
        # empty
        c = ArrayCoordinates1d()
        c.add(1, inplace=True)
        assert_equal(c.coordinates, [])

        c = ArrayCoordinates1d()
        c.add('10,D', inplace=True)
        assert_equal(c.coordinates, [])

        # standard
        c = ArrayCoordinates1d([20., 50., 60., 10.])
        c.add(1, inplace=True)
        assert_equal(c.coordinates, [21., 51., 61., 11.])

        c = ArrayCoordinates1d(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03'])
        c.add('10,D', inplace=True)
        assert_equal(c.coordinates, np.array(['2018-01-12', '2018-01-11', '2018-01-14', '2018-01-13']).astype(np.datetime64))

    def test_add(self):
        a = MonotonicCoordinates1d([10., 20., 50., 60.])
        assert isinstance(a + 1, MonotonicCoordinates1d)

@pytest.mark.skip("TODO")
class TestArrayCoordinatesMagic(object):
    def test_len(self):
        c = ArrayCoordinates1d([20., 50., 60., 10.])
        assert len(c) == 4

        e = ArrayCoordinates1d()
        assert len(e) == 0

    def test_repr(self):
        c = ArrayCoordinates1d([20., 50., 60., 10.])
        repr(c)

    def test_add(self):
        a = ArrayCoordinates1d([20., 50., 60., 10.])
        b = ArrayCoordinates1d([55., 65., 95., 45.])
        
        # concat
        r = a + b
        assert isinstance(r, ArrayCoordinates1d)
        assert_equal(r.coordinates, [20., 50., 60., 10., 55., 65., 95., 45.])
        
        r = b + a
        assert isinstance(r, ArrayCoordinates1d)
        assert_equal(r.coordinates, [55., 65., 95., 45., 20., 50., 60., 10.])

        # add
        r = a + 1
        assert isinstance(r, ArrayCoordinates1d)
        assert_equal(r.coordinates, [21., 51., 61., 11.])

    def test_iadd(self):
        # concat
        c = ArrayCoordinates1d([20., 50., 60., 10.])
        c += ArrayCoordinates1d([55., 65., 95., 45.])
        assert_equal(c.coordinates, [20., 50., 60., 10., 55., 65., 95., 45.])
        
        # add
        c = ArrayCoordinates1d([20., 50., 60., 10.])
        c += 1
        assert_equal(c.coordinates, [21., 51., 61., 11.])

    def test_sub(self):
        c = ArrayCoordinates1d([20., 50., 60., 10.])
        r = c - 1
        assert isinstance(r, ArrayCoordinates1d)
        assert_equal(r.coordinates, [19., 49., 59., 9.])

    def test_isub(self):
        c = ArrayCoordinates1d([20., 50., 60., 10.])
        c -= 1
        assert_equal(c.coordinates, [19., 49., 59., 9.])

    def test_and(self):
        a = ArrayCoordinates1d([20., 50., 60., 10.])
        b = ArrayCoordinates1d([55., 65., 95., 45.])
        
        r = a & b
        assert isinstance(r, ArrayCoordinates1d)
        assert_equal(r.coordinates, [50., 60.])
        
        r = b & a
        assert isinstance(r, ArrayCoordinates1d)
        assert_equal(r.coordinates, [55., 45.])