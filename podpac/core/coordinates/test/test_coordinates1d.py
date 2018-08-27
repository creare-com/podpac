
from datetime import datetime

import pytest
import traitlets as tl
import numpy as np
from numpy.testing import assert_equal

from podpac.core.units import Units
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.coordinates1d import MonotonicCoordinates1d
from podpac.core.coordinates.coordinates1d import UniformCoordinates1d

class TestCoordinates1d(object):
    def test_abstract(self):
        c = Coordinates1d(name='lat')
        with pytest.raises(NotImplementedError):
            c.coordinates
        with pytest.raises(NotImplementedError):
            c.bounds
        with pytest.raises(NotImplementedError):
            c.size
        with pytest.raises(NotImplementedError):
            c.dtype
        with pytest.raises(NotImplementedError):
            c.is_monotonic
        with pytest.raises(NotImplementedError):
            c.is_descending
        with pytest.raises(NotImplementedError):
            c.select([0, 1])
        with pytest.raises(NotImplementedError):
            c[0]

    def test_name(self):
        Coordinates1d(name='lat')
        Coordinates1d(name='lon')
        Coordinates1d(name='alt')
        Coordinates1d(name='time')

        with pytest.raises(tl.TraitError):
            Coordinates1d(name='depth')

        with pytest.raises(TypeError):
            Coordinates1d()

class TestArrayCoordinates1d(object):
    def test_empty(self):
        c = ArrayCoordinates1d(name='lat')
        a = np.array([], dtype=float)

        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([np.nan, np.nan]))
        assert_equal(c.area_bounds, c.bounds)
        assert c.size == 0
        assert c.dtype is None
        assert c.is_monotonic == False
        assert c.is_descending is None

        c = ArrayCoordinates1d([], name='lat')
        assert_equal(c.coords, a)

    def test_numerical_array(self):
        values = [0., 100., -3., 10.]
        a = np.array(values, dtype=float)
        
        c = ArrayCoordinates1d(a, name='lat')
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([-3., 100.]))
        assert_equal(c.area_bounds, c.bounds)
        assert c.size == 4
        assert c.dtype == float
        assert c.is_monotonic == False
        assert c.is_descending is None

    def test_numerical_singleton(self):
        a = np.array([10.0], dtype=float)

        c = ArrayCoordinates1d(10.0, name='lat')
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([10.0, 10.0]))
        assert_equal(c.area_bounds, c.bounds)
        assert c.size == 1
        assert c.dtype == float
        assert c.is_monotonic == False
        assert c.is_descending is None

    def test_datetime_array(self):
        values = ['2018-01-01', '2019-01-01', '2017-01-01', '2018-01-02']
        a = np.array(values).astype(np.datetime64)
        
        c = ArrayCoordinates1d(values, name='time')
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        assert_equal(c.area_bounds, c.bounds)
        assert c.size == 4
        assert c.dtype == np.datetime64
        assert c.is_monotonic == False
        assert c.is_descending is None

    def test_datetime_singleton(self):
        a = np.array('2018-01-01').astype(np.datetime64)

        # string input
        c = ArrayCoordinates1d('2018-01-01', name='time')
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(['2018-01-01', '2018-01-01']).astype(np.datetime64))
        assert_equal(c.area_bounds, c.bounds)
        assert c.size == 1
        assert c.dtype == np.datetime64
        assert c.is_monotonic == False
        assert c.is_descending is None

    def test_invalid_coords(self):
        c = ArrayCoordinates1d(name='lat')
        
        with pytest.raises(ValueError):
            c.coords = np.array([1, 2, 3])

        with pytest.raises(ValueError):
            c.coords = np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_ctype(self):
        # default
        assert ArrayCoordinates1d(name='lat').ctype == 'point'
        
        # valid
        ArrayCoordinates1d(name='lat', ctype='point')

        # invalid
        with pytest.raises(tl.TraitError):
            ArrayCoordinates1d(name='lat', ctype='left')
        with pytest.raises(tl.TraitError):
            ArrayCoordinates1d(name='lat', ctype='right')
        with pytest.raises(tl.TraitError):
            ArrayCoordinates1d(name='lat', ctype='midpoint')
        with pytest.raises(tl.TraitError):
            ArrayCoordinates1d(name='lat', ctype='abc')

    def test_properties(self):
        c = ArrayCoordinates1d(name='lat')
        assert isinstance(c.properties, dict)
        assert set(c.properties.keys()) == set(['units', 'ctype', 'coord_ref_sys'])

    def test_index(self):
        c = ArrayCoordinates1d([20, 50, 60, 90, 40, 10], name='lat')
        
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

#     def test_select(self):
#         c = ArrayCoordinates1d([20., 50., 60., 90., 40., 10.])
        
#         # full selection
#         s = c.select([0, 100])
#         assert isinstance(s, ArrayCoordinates1d)
#         assert_equal(s.coordinates, c.coordinates)

#         # none, above
#         s = c.select([100, 200])
#         assert isinstance(s, ArrayCoordinates1d)
#         assert_equal(s.coordinates, [])

#         # none, below
#         s = c.select([0, 5])
#         assert isinstance(s, ArrayCoordinates1d)
#         assert_equal(s.coordinates, [])

#         # partial, above
#         s = c.select([50, 100])
#         assert isinstance(s, ArrayCoordinates1d)
#         assert_equal(s.coordinates, [50., 60., 90.])

#         # partial, below
#         s = c.select([0, 50])
#         assert isinstance(s, ArrayCoordinates1d)
#         assert_equal(s.coordinates, [20., 50., 40., 10.])

#         # partial, inner
#         s = c.select([30., 70.])
#         assert isinstance(s, ArrayCoordinates1d)
#         assert_equal(s.coordinates, [50., 60., 40.])

#         # partial, inner exact
#         s = c.select([40., 60.])
#         assert isinstance(s, ArrayCoordinates1d)
#         assert_equal(s.coordinates, [50., 60., 40.])

#         # partial, none
#         s = c.select([52, 55])
#         assert isinstance(s, ArrayCoordinates1d)
#         assert_equal(s.coordinates, [])

#         # partial, backwards bounds
#         s = c.select([70, 30])
#         assert isinstance(s, ArrayCoordinates1d)
#         assert_equal(s.coordinates, [])

#         # empty coords
#         c = ArrayCoordinates1d()        
#         s = c.select([0, 1])
#         assert isinstance(s, ArrayCoordinates1d)
#         assert_equal(s.coordinates, [])

#     def test_select_ind(self):
#         c = ArrayCoordinates1d([20., 50., 60., 90., 40., 10.])
        
#         # full selection
#         I = c.select([0, 100], ind=True)
#         assert_equal(c.coordinates[I], c.coordinates)

#         # none, above
#         I = c.select([100, 200], ind=True)
#         assert_equal(c.coordinates[I], [])

#         # none, below
#         I = c.select([0, 5], ind=True)
#         assert_equal(c.coordinates[I], [])

#         # partial, above
#         I = c.select([50, 100], ind=True)
#         assert_equal(c.coordinates[I], [50., 60., 90.])

#         # partial, below
#         I = c.select([0, 50], ind=True)
#         assert_equal(c.coordinates[I], [20., 50., 40., 10.])

#         # partial, inner
#         I = c.select([30., 70.], ind=True)
#         assert_equal(c.coordinates[I], [50., 60., 40.])

#         # partial, inner exact
#         I = c.select([40., 60.], ind=True)
#         assert_equal(c.coordinates[I], [50., 60., 40.])

#         # partial, none
#         I = c.select([52, 55], ind=True)
#         assert_equal(c.coordinates[I], [])

#         # partial, backwards bounds
#         I = c.select([70, 30], ind=True)
#         assert_equal(c.coordinates[I], [])

#         # empty coords
#         c = ArrayCoordinates1d()        
#         I = c.select([0, 1], ind=True)
#         assert_equal(c.coordinates[I], [])

#     def test_intersect(self):
#         a = ArrayCoordinates1d([20., 50., 60., 10.])
#         b = ArrayCoordinates1d([55., 65., 95., 45.])
#         c = ArrayCoordinates1d([80., 70., 90.])
#         e = ArrayCoordinates1d()
        
#         # ArrayCoordinates1d other, both directions
#         ab = a.intersect(b)
#         assert isinstance(ab, ArrayCoordinates1d)
#         assert_equal(ab.coordinates, [50., 60.])
        
#         ba = b.intersect(a)
#         assert isinstance(ba, ArrayCoordinates1d)
#         assert_equal(ba.coordinates, [55., 45.])

#         # ArrayCoordinates1d other, no overlap
#         ac = a.intersect(c)
#         assert isinstance(ac, ArrayCoordinates1d)
#         assert_equal(ac.coordinates, [])

#         # empty self
#         ea = e.intersect(a)
#         assert isinstance(ea, ArrayCoordinates1d)
#         assert_equal(ea.coordinates, [])

#         # empty other
#         ae = a.intersect(e)
#         assert isinstance(ae, ArrayCoordinates1d)
#         assert_equal(ae.coordinates, [])

#         # MonotonicCoordinates1d other
#         m = MonotonicCoordinates1d([45., 55., 65., 95.])
#         am = a.intersect(m)
#         assert isinstance(am, ArrayCoordinates1d)
#         assert_equal(am.coordinates, [50., 60.])

#         # UniformCoordinates1d other
#         u = UniformCoordinates1d(45, 95, 10)
#         au = a.intersect(u)
#         assert isinstance(au, ArrayCoordinates1d)
#         assert_equal(au.coordinates, [50., 60.])

#     def test_intersect_ind(self):
#         a = ArrayCoordinates1d([20., 50., 60., 10.])
#         b = ArrayCoordinates1d([55., 65., 95., 45.])
#         c = ArrayCoordinates1d([80., 70., 90.])
#         e = ArrayCoordinates1d()
        
#         # ArrayCoordinates1d other, both directions
#         I = a.intersect(b, ind=True)
#         assert_equal(a.coordinates[I], [50., 60.])
        
#         I = b.intersect(a, ind=True)
#         assert_equal(b.coordinates[I], [55., 45.])

#         # ArrayCoordinates1d other, no overlap
#         I = a.intersect(c, ind=True)
#         assert_equal(a.coordinates[I], [])

#         # empty self
#         I = e.intersect(a, ind=True)
#         assert_equal(e.coordinates[I], [])

#         # empty other
#         I = a.intersect(e, ind=True)
#         assert_equal(a.coordinates[I], [])

#         # MonotonicCoordinates1d other
#         m = MonotonicCoordinates1d([45., 55., 65., 95.])
#         I = a.intersect(m, ind=True)
#         assert_equal(a.coordinates[I], [50., 60.])

#         # UniformCoordinates1d other
#         u = UniformCoordinates1d(45, 95, 10)
#         I = a.intersect(u, ind=True)
#         assert_equal(a.coordinates[I], [50., 60.])

#     def test_concat(self):
#         a = ArrayCoordinates1d([20., 50., 60., 10.])
#         b = ArrayCoordinates1d([55., 65., 95., 45.])
#         c = ArrayCoordinates1d([80., 70., 90.])
#         e = ArrayCoordinates1d()

#         t = ArrayCoordinates1d(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03'])
        
#         # ArrayCoordinates1d other, both directions
#         ab = a.concat(b)
#         assert isinstance(ab, ArrayCoordinates1d)
#         assert_equal(ab.coordinates, [20., 50., 60., 10., 55., 65., 95., 45.])
        
#         ba = b.concat(a)
#         assert isinstance(ba, ArrayCoordinates1d)
#         assert_equal(ba.coordinates, [55., 65., 95., 45., 20., 50., 60., 10.])

#         # empty self
#         ea = e.concat(a)
#         assert isinstance(ea, ArrayCoordinates1d)
#         assert_equal(ea.coordinates, a.coordinates)

#         et = e.concat(t)
#         assert isinstance(et, ArrayCoordinates1d)
#         assert_equal(et.coordinates, t.coordinates)

#         # empty other
#         ae = a.concat(e)
#         assert isinstance(ae, ArrayCoordinates1d)
#         assert_equal(ae.coordinates, a.coordinates)

#         te = t.concat(e)
#         assert isinstance(te, ArrayCoordinates1d)
#         assert_equal(te.coordinates, t.coordinates)

#         # MonotonicCoordinates1d other
#         m = MonotonicCoordinates1d([45., 55., 65., 95.])
#         am = a.concat(m)
#         assert isinstance(am, ArrayCoordinates1d)
#         assert_equal(am.coordinates, [20., 50., 60., 10., 45., 55., 65., 95.])

#         # UniformCoordinates1d other
#         u = UniformCoordinates1d(45, 95, 10)
#         au = a.concat(u)
#         assert isinstance(au, ArrayCoordinates1d)
#         assert_equal(au.coordinates, [20., 50., 60., 10., 45., 55., 65., 75., 85., 95.])

#         # type error
#         with pytest.raises(TypeError):
#             a.concat(5)

#         with pytest.raises(TypeError):
#             a.concat(t)

#         with pytest.raises(TypeError):
#             t.concat(a)

#     def test_concat_equal(self):
#         # ArrayCoordinates1d other
#         c = ArrayCoordinates1d([20., 50., 60., 10.])
#         c.concat(ArrayCoordinates1d([55., 65., 95., 45.]), inplace=True)
#         assert_equal(c.coordinates, [20., 50., 60., 10., 55., 65., 95., 45.])
        
#         # empty self
#         c = ArrayCoordinates1d()
#         c.concat(ArrayCoordinates1d([55., 65., 95., 45.]), inplace=True)
#         assert_equal(c.coordinates, [55., 65., 95., 45.])

#         c = ArrayCoordinates1d()
#         c.concat(ArrayCoordinates1d(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03']), inplace=True)
#         assert_equal(c.coordinates, np.array(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03']).astype(np.datetime64))

#         # empty other
#         c = ArrayCoordinates1d([20., 50., 60., 10.])
#         c.concat(ArrayCoordinates1d(), inplace=True)
#         assert_equal(c.coordinates, [20., 50., 60., 10.])

#         c = ArrayCoordinates1d(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03'])
#         c.concat(ArrayCoordinates1d(), inplace=True)
#         assert_equal(c.coordinates, np.array(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03']).astype(np.datetime64))

#         # MonotonicCoordinates1d other
#         c = ArrayCoordinates1d([20., 50., 60., 10.])
#         c.concat(MonotonicCoordinates1d([45., 55., 65., 95.]), inplace=True)
#         assert_equal(c.coordinates, [20., 50., 60., 10., 45., 55., 65., 95.])

#         # UniformCoordinates1d other
#         c = ArrayCoordinates1d([20., 50., 60., 10.])
#         c.concat(UniformCoordinates1d(45, 95, 10), inplace=True)
#         assert_equal(c.coordinates, [20., 50., 60., 10., 45., 55., 65., 75., 85., 95.])

#     def test_add(self):
#         a = ArrayCoordinates1d([20., 50., 60., 10.])
#         t = ArrayCoordinates1d(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03'])
#         e = ArrayCoordinates1d()

#         # empty
#         r = e.add(1)
#         assert isinstance(r, ArrayCoordinates1d)
#         assert_equal(r.coordinates, [])

#         r = e.add('10,D')
#         assert isinstance(r, ArrayCoordinates1d)
#         assert_equal(r.coordinates, [])
        
#         # standard
#         r = a.add(1)
#         assert isinstance(r, ArrayCoordinates1d)
#         assert_equal(r.coordinates, [21., 51., 61., 11.])

#         r = t.add('10,D')
#         assert isinstance(r, ArrayCoordinates1d)
#         assert_equal(r.coordinates, np.array(['2018-01-12', '2018-01-11', '2018-01-14', '2018-01-13']).astype(np.datetime64))

#         # type error
#         with pytest.raises(TypeError):
#             a.add(a)

#         with pytest.raises(TypeError):
#             a.add('10,D')

#         with pytest.raises(TypeError):
#             t.add(4)

#     def test_add_equal(self):
#         # empty
#         c = ArrayCoordinates1d()
#         c.add(1, inplace=True)
#         assert_equal(c.coordinates, [])

#         c = ArrayCoordinates1d()
#         c.add('10,D', inplace=True)
#         assert_equal(c.coordinates, [])

#         # standard
#         c = ArrayCoordinates1d([20., 50., 60., 10.])
#         c.add(1, inplace=True)
#         assert_equal(c.coordinates, [21., 51., 61., 11.])

#         c = ArrayCoordinates1d(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03'])
#         c.add('10,D', inplace=True)
#         assert_equal(c.coordinates, np.array(['2018-01-12', '2018-01-11', '2018-01-14', '2018-01-13']).astype(np.datetime64))

#     def test___add__(self):
#         a = ArrayCoordinates1d([20., 50., 60., 10.])
#         b = ArrayCoordinates1d([55., 65., 95., 45.])
        
#         # concat
#         r = a + b
#         assert isinstance(r, ArrayCoordinates1d)
#         assert_equal(r.coordinates, [20., 50., 60., 10., 55., 65., 95., 45.])
        
#         r = b + a
#         assert isinstance(r, ArrayCoordinates1d)
#         assert_equal(r.coordinates, [55., 65., 95., 45., 20., 50., 60., 10.])

#         # add
#         r = a + 1
#         assert isinstance(r, ArrayCoordinates1d)
#         assert_equal(r.coordinates, [21., 51., 61., 11.])

#     def test___iadd__(self):
#         # concat
#         c = ArrayCoordinates1d([20., 50., 60., 10.])
#         c += ArrayCoordinates1d([55., 65., 95., 45.])
#         assert_equal(c.coordinates, [20., 50., 60., 10., 55., 65., 95., 45.])
        
#         # add
#         c = ArrayCoordinates1d([20., 50., 60., 10.])
#         c += 1
#         assert_equal(c.coordinates, [21., 51., 61., 11.])

#     def test___sub__(self):
#         c = ArrayCoordinates1d([20., 50., 60., 10.])
#         r = c - 1
#         assert isinstance(r, ArrayCoordinates1d)
#         assert_equal(r.coordinates, [19., 49., 59., 9.])

#     def test___isub__(self):
#         c = ArrayCoordinates1d([20., 50., 60., 10.])
#         c -= 1
#         assert_equal(c.coordinates, [19., 49., 59., 9.])

#     def test___and__(self):
#         a = ArrayCoordinates1d([20., 50., 60., 10.])
#         b = ArrayCoordinates1d([55., 65., 95., 45.])
        
#         r = a & b
#         assert isinstance(r, ArrayCoordinates1d)
#         assert_equal(r.coordinates, [50., 60.])
        
#         r = b & a
#         assert isinstance(r, ArrayCoordinates1d)
#         assert_equal(r.coordinates, [55., 45.])

    def test___len__(self):
        c = ArrayCoordinates1d([20., 50., 60., 10.], name='lat')
        assert len(c) == 4

        e = ArrayCoordinates1d(name='lat')
        assert len(e) == 0

    def test___repr__(self):
        c = ArrayCoordinates1d([20., 50., 60., 10.], name='lat')
        repr(c)

class TestMonotonicCoordinates1d(object):
    """
    MonotonicCoordinates1d extends ArrayCoordinates1d, so only some properties and methods are 
    tested here::
     - coords validation
     - bounds, is_datetime, is_datetime, and is_descending properties
     - intersect and select methods
    """

    def test_empty(self):
        a = np.array([], dtype=float)

        c = MonotonicCoordinates1d(name='lat')
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([np.nan, np.nan]))
        assert_equal(c.area_bounds, np.array([np.nan, np.nan]))
        assert c.dtype is None
        assert c.is_monotonic == True
        assert c.is_descending == None

        c = MonotonicCoordinates1d([], name='lat')
        assert_equal(c.coords, a)

    def test_numerical_array(self):
        values = [0.0, 1.0, 4.0, 6.0]
        a = np.array(values, dtype=float)
        
        # ascending
        c = MonotonicCoordinates1d(values, name='lat')
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([0., 6.]))
        assert c.size == 4
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == False

        # descending
        c = MonotonicCoordinates1d(values[::-1], name='lat')
        assert_equal(c.coords, a[::-1])
        assert_equal(c.coordinates, a[::-1])
        assert_equal(c.bounds, np.array([0., 6.]))
        assert c.size == 4
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == True

        # unsorted
        with pytest.raises(ValueError):
            MonotonicCoordinates1d(a[[0, 2, 1, 3]], name='lat')

    def test_numerical_singleton(self):
        a = np.array([10.0], dtype=float)

        c = MonotonicCoordinates1d(10.0, name='lat')
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([10.0, 10.0]))
        assert c.size == 1
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending is None

    def test_datetime_array(self):
        values = ['2017-01-01', '2018-01-01', '2018-01-02', '2019-01-01']
        a = np.array(values).astype(np.datetime64)
        
        # ascending
        c = MonotonicCoordinates1d(values, name='time')
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        assert c.size == 4
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False

        # descending
        c = MonotonicCoordinates1d(values[::-1], name='time')
        assert_equal(c.coords, a[::-1])
        assert_equal(c.coordinates, a[::-1])
        assert_equal(c.bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        assert c.size == 4
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True

        # unsorted
        with pytest.raises(ValueError):
            MonotonicCoordinates1d(a[[0, 2, 1, 3]], name='time')

    def test_datetime_singleton(self):
        a = np.array(['2018-01-01']).astype(np.datetime64)
        
        c = MonotonicCoordinates1d('2018-01-01', name='time')
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(['2018-01-01', '2018-01-01']).astype(np.datetime64))
        assert c.size == 1
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending is None

    def test_ctype(self):
        # default
        assert MonotonicCoordinates1d(name='lat').ctype == 'midpoint'

        # valid
        MonotonicCoordinates1d(name='lat', ctype='point')
        MonotonicCoordinates1d(name='lat', ctype='left')
        MonotonicCoordinates1d(name='lat', ctype='right')
        MonotonicCoordinates1d(name='lat', ctype='midpoint')
        
        # invalid
        with pytest.raises(tl.TraitError):
            MonotonicCoordinates1d(name='lat', ctype='abc')

    def test_properties(self):
        # without extents
        c = MonotonicCoordinates1d(name='lat', ctype='point')
        assert isinstance(c.properties, dict)
        assert set(c.properties.keys()) == set(['units', 'ctype', 'coord_ref_sys'])

        # with extents
        c = MonotonicCoordinates1d(name='lat', extents=[0, 1])
        assert isinstance(c.properties, dict)
        assert set(c.properties.keys()) == set(['units', 'ctype', 'coord_ref_sys', 'extents'])

    def test_extents(self):
        c = MonotonicCoordinates1d(name='lat')
        assert c.extents is None

        # numerical
        c = MonotonicCoordinates1d([1, 2], name='lat', extents=[0.5, 2.5])
        assert_equal(c.extents, np.array([0.5, 2.5], dtype=float))

        # datetime
        c = MonotonicCoordinates1d(['2018-02-01', '2019-02-01'], name='time', extents=['2018-01-01', '2019-03-01'])
        assert_equal(c.extents, np.array(['2018-01-01', '2019-03-01']).astype(np.datetime64))

        # invalid (ctype=point)
        with pytest.raises(TypeError):
            MonotonicCoordinates1d([1, 2], name='lat', ctype='point', extents=[0.5, 2.5])

        # invalid (wrong dtype)
        with pytest.raises(ValueError):
            MonotonicCoordinates1d(['2018-02-01', '2019-02-01'], name='lat', extents=[0.5, 2.5])
        
        with pytest.raises(ValueError):
            MonotonicCoordinates1d([1, 2], name='time', extents=['2018-01-01', '2019-03-01'])

        # invalid (shape)
        with pytest.raises(ValueError):
            MonotonicCoordinates1d([1, 2], name='lat', extents=[0.5])

    def test_area_bounds_point(self):
        # numerical
        values = [0.0, 1.0, 4.0, 6.0]
        c = MonotonicCoordinates1d(values, name='lat', ctype='point')
        assert_equal(c.area_bounds, np.array([0.0, 6.0], dtype=float))
        c = MonotonicCoordinates1d(values[::-1], name='lat', ctype='point')
        assert_equal(c.area_bounds, np.array([0.0, 6.0], dtype=float))

        # datetime
        values = ['2017-01-01', '2017-01-02', '2018-01-01', '2019-01-01']
        c = MonotonicCoordinates1d(values, name='time', ctype='point')
        assert_equal(c.area_bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        c = MonotonicCoordinates1d(values[::-1], name='time', ctype='point')
        assert_equal(c.area_bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))

    def test_area_bounds_explicit_extents(self):
        # numerical
        values = [0.0, 1.0, 4.0, 6.0]
        c = MonotonicCoordinates1d(values, name='lat', extents=[-10, 10])
        assert_equal(c.area_bounds, np.array([-10.0, 10.0], dtype=float))
        c = MonotonicCoordinates1d(values[::-1], name='lat', extents=[-10, 10])
        assert_equal(c.area_bounds, np.array([-10.0, 10.0], dtype=float))

        # datetime
        values = ['2017-01-01', '2018-01-01', '2018-01-02', '2019-01-01']
        c = MonotonicCoordinates1d(values, name='time', extents=['2016-01-01', '2021-01-01'])
        assert_equal(c.area_bounds, np.array(['2016-01-01', '2021-01-01']).astype(np.datetime64))
        c = MonotonicCoordinates1d(values[::-1], name='time', extents=['2016-01-01', '2021-01-01'])
        assert_equal(c.area_bounds, np.array(['2016-01-01', '2021-01-01']).astype(np.datetime64))

    def test_area_bounds_left(self):
        # numerical
        values = [0.0, 1.0, 4.0, 6.0]
        c = MonotonicCoordinates1d(values, name='lat', ctype='left')
        assert_equal(c.area_bounds, np.array([0.0, 8.0], dtype=float))
        c = MonotonicCoordinates1d(values[::-1], name='lat', ctype='left')
        assert_equal(c.area_bounds, np.array([0.0, 8.0], dtype=float))

        # datetime
        values = ['2017-01-01', '2017-01-02', '2018-01-01', '2019-01-01']
        c = MonotonicCoordinates1d(values, name='time', ctype='left')
        assert_equal(c.area_bounds, np.array(['2017-01-01', '2020-01-01']).astype(np.datetime64))
        c = MonotonicCoordinates1d(values[::-1], name='time', ctype='left')
        assert_equal(c.area_bounds, np.array(['2017-01-01', '2020-01-01']).astype(np.datetime64))

    def test_area_bounds_right(self):
        # numerical
        values = [0.0, 1.0, 4.0, 6.0]
        c = MonotonicCoordinates1d(values, name='lat', ctype='right')
        assert_equal(c.area_bounds, np.array([-1.0, 6.0], dtype=float))
        c = MonotonicCoordinates1d(values[::-1], name='lat', ctype='right')
        assert_equal(c.area_bounds, np.array([-1.0, 6.0], dtype=float))

        # datetime
        values = ['2017-01-01', '2017-01-02', '2018-01-01', '2019-01-01']
        c = MonotonicCoordinates1d(values, name='time', ctype='right')
        assert_equal(c.area_bounds, np.array(['2016-12-31', '2019-01-01']).astype(np.datetime64))
        c = MonotonicCoordinates1d(values[::-1], name='time', ctype='right')
        assert_equal(c.area_bounds, np.array(['2016-12-31', '2019-01-01']).astype(np.datetime64))

    def test_area_bounds_midpoint_numerical(self):
        values = [0.0, 1.0, 4.0, 6.0]
        c = MonotonicCoordinates1d(values, name='lat', ctype='midpoint')
        assert_equal(c.area_bounds, np.array([-0.5, 7.0], dtype=float))
        c = MonotonicCoordinates1d(values[::-1], name='lat', ctype='midpoint')
        assert_equal(c.area_bounds, np.array([-0.5, 7.0], dtype=float))

    @pytest.mark.skip('TODO')
    def test_area_bounds_midpoint_datetime(self):
        values = ['2017-01-01', '2017-01-02', '2018-01-01', '2019-01-01']
        c = MonotonicCoordinates1d(values, name='time', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2016-12-31 12:00:00', '2019-06-01']).astype(np.datetime64))
        c = MonotonicCoordinates1d(values[::-1], name='time', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2016-12-31 12:00:00', '2019-07-02']).astype(np.datetime64))

    def test_area_bounds_segment_singleton(self):
        # numerical
        value = 10.0
        c = MonotonicCoordinates1d(value, name='lat', ctype='left')
        assert_equal(c.area_bounds, np.array([value, value], dtype=float))
        c = MonotonicCoordinates1d(value, name='lat', ctype='right')
        assert_equal(c.area_bounds, np.array([value, value], dtype=float))
        c = MonotonicCoordinates1d(value, name='lat', ctype='midpoint')
        assert_equal(c.area_bounds, np.array([value, value], dtype=float))

        # datetime
        value = '2018-01-01'
        c = MonotonicCoordinates1d(value, name='time', ctype='left')
        assert_equal(c.area_bounds, np.array([value, value]).astype(np.datetime64))
        c = MonotonicCoordinates1d(value, name='time', ctype='right')
        assert_equal(c.area_bounds, np.array([value, value]).astype(np.datetime64))
        c = MonotonicCoordinates1d(value, name='time', ctype='midpoint')
        assert_equal(c.area_bounds, np.array([value, value]).astype(np.datetime64))

    def test_index(self):
        c = MonotonicCoordinates1d([10, 20, 40, 50, 60, 90], name='lat')
        
        # int
        c2 = c[2]
        assert isinstance(c2, MonotonicCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, np.array([40], dtype=float))

        c2 = c[-2]
        assert isinstance(c2, MonotonicCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, np.array([60], dtype=float))

        # slice
        c2 = c[:2]
        assert isinstance(c2, MonotonicCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, np.array([10, 20], dtype=float))
        
        c2 = c[::2]
        assert isinstance(c2, MonotonicCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, np.array([10, 40, 60], dtype=float))
        
        c2 = c[1:-1]
        assert isinstance(c2, MonotonicCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, np.array([20, 40, 50, 60], dtype=float))
        
        c2 = c[::-1]
        assert isinstance(c2, MonotonicCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, np.array([90, 60, 50, 40, 20, 10], dtype=float))
        
        # ordered array
        c2 = c[[0, 1, 3]]
        assert isinstance(c2, MonotonicCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, np.array([10, 20, 50], dtype=float))

        c2 = c[[3, 1, 0]]
        assert isinstance(c2, MonotonicCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coords, np.array([50, 20, 10], dtype=float))

        # unordered array (tries to return ArrayCoordinates1d instead)
        with pytest.raises(tl.TraitError):
            c[[0, 3, 1]] # must be ctype=point

        cpoints = MonotonicCoordinates1d([10, 20, 40, 50, 60, 90], name='lat', ctype='point')
        c2 = cpoints[[0, 3, 1]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == cpoints.name
        assert c2.properties == cpoints.properties
        assert_equal(c2.coords, np.array([10, 50, 20], dtype=float))

        # boolean array
        c2 = c[[True, True, True, False, True, False]]
        assert isinstance(c2, MonotonicCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, np.array([10, 20, 40, 60], dtype=float))

        # invalid
        with pytest.raises(IndexError):
            c[0.3]

        with pytest.raises(IndexError):
            c[10]
    
#     def test_select_ascending(self):
#         c = MonotonicCoordinates1d([10., 20., 40., 50., 60., 90.])
        
#         # full and empty selection type
#         assert isinstance(c.select([0, 100]), MonotonicCoordinates1d)
#         assert isinstance(c.select([100, 200]), ArrayCoordinates1d)
#         assert isinstance(c.select([0, 5]), ArrayCoordinates1d)
        
#         # partial, above
#         s = c.select([50, 100], pad=0)
#         assert isinstance(s, MonotonicCoordinates1d)
#         assert_equal(s.coordinates, [50., 60., 90.])

#         # partial, below
#         s = c.select([0, 50], pad=0)
#         assert isinstance(s, MonotonicCoordinates1d)
#         assert_equal(s.coordinates, [10., 20., 40., 50.])

#         # partial, inner
#         s = c.select([30., 70.], pad=0)
#         assert isinstance(s, MonotonicCoordinates1d)
#         assert_equal(s.coordinates, [40., 50., 60.])

#         # partial, inner exact
#         s = c.select([40., 60.], pad=0)
#         assert isinstance(s, MonotonicCoordinates1d)
#         assert_equal(s.coordinates, [40., 50., 60.])

#         # partial, none
#         s = c.select([52, 55], pad=0)
#         assert isinstance(s, ArrayCoordinates1d)
#         assert_equal(s.coordinates, [])

#         # partial, backwards bounds
#         s = c.select([70, 30], pad=0)
#         assert isinstance(s, ArrayCoordinates1d)
#         assert_equal(s.coordinates, [])

#     def test_select_descending(self):
#         c = MonotonicCoordinates1d([90., 60., 50., 40., 20., 10.])
        
#         # full and empty selection type
#         assert isinstance(c.select([0, 100]), MonotonicCoordinates1d)
#         assert isinstance(c.select([100, 200]), ArrayCoordinates1d)
#         assert isinstance(c.select([0, 5]), ArrayCoordinates1d)
        
#         # partial, above
#         s = c.select([50, 100], pad=0)
#         assert isinstance(s, MonotonicCoordinates1d)
#         assert_equal(s.coordinates, [90., 60., 50.])

#         # partial, below
#         s = c.select([0, 50], pad=0)
#         assert isinstance(s, MonotonicCoordinates1d)
#         assert_equal(s.coordinates, [50., 40., 20., 10.])

#         # partial, inner
#         s = c.select([30., 70.], pad=0)
#         assert isinstance(s, MonotonicCoordinates1d)
#         assert_equal(s.coordinates, [60., 50., 40.])

#         # partial, inner exact
#         s = c.select([40., 60.], pad=0)
#         assert isinstance(s, MonotonicCoordinates1d)
#         assert_equal(s.coordinates, [60., 50., 40.])

#         # partial, none
#         s = c.select([52, 55], pad=0)
#         assert isinstance(s, ArrayCoordinates1d)
#         assert_equal(s.coordinates, [])

#         # partial, backwards bounds
#         s = c.select([70, 30], pad=0)
#         assert isinstance(s, ArrayCoordinates1d)
#         assert_equal(s.coordinates, [])

#     def test_select_ind(self):
#         c = MonotonicCoordinates1d([10., 20., 40., 50., 60., 90.])
        
#         # partial, above
#         s = c.select([50, 100], ind=True, pad=0)
#         assert_equal(c.coordinates[s], [50., 60., 90.])

#         # partial, below
#         s = c.select([0, 50], ind=True, pad=0)
#         assert_equal(c.coordinates[s], [10., 20., 40., 50.])

#         # partial, inner
#         s = c.select([30., 70.], ind=True, pad=0)
#         assert_equal(c.coordinates[s], [40., 50., 60.])

#         # partial, inner exact
#         s = c.select([40., 60.], ind=True, pad=0)
#         assert_equal(c.coordinates[s], [40., 50., 60.])

#         # partial, none
#         s = c.select([52, 55], ind=True, pad=0)
#         assert_equal(c.coordinates[s], [])

#         # partial, backwards bounds
#         s = c.select([70, 30], ind=True, pad=0)
#         assert_equal(c.coordinates[s], [])

#     def test_intersect(self):
#         # MonotonicCoordinates1d other
#         a = MonotonicCoordinates1d([10., 20., 50., 60.])
#         b = MonotonicCoordinates1d([45., 55., 65., 95.])
#         assert isinstance(a.intersect(b), MonotonicCoordinates1d)

#         # ArrayCoordinates1d other
#         c = ArrayCoordinates1d([20., 50., 60., 10.])
#         assert isinstance(a.intersect(c), MonotonicCoordinates1d)
        
#         # UniformCoordinates1d
#         u = UniformCoordinates1d(45, 95, 10)
#         assert isinstance(a.intersect(u), MonotonicCoordinates1d)

#     def test_concat(self):
#         a = MonotonicCoordinates1d([10., 20., 50., 60.])
#         b = MonotonicCoordinates1d([45., 55., 65., 95.])
#         c = MonotonicCoordinates1d([70., 80., 90.])
#         d = MonotonicCoordinates1d([35., 25., 15.])
#         e = MonotonicCoordinates1d()
#         f = MonotonicCoordinates1d([90., 80., 70.])
#         o = ArrayCoordinates1d([20., 50., 60., 10.])
#         u = UniformCoordinates1d(45, 95, 10)
#         v = UniformCoordinates1d(75, 95, 10)
#         t = MonotonicCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'])
        
#         # overlap, ascending
#         ab = a.concat(b)
#         assert isinstance(ab, ArrayCoordinates1d)
#         assert not isinstance(ab, MonotonicCoordinates1d)
#         assert_equal(ab.coordinates, [10., 20., 50., 60., 45., 55., 65., 95.])
        
#         ba = b.concat(a)
#         assert isinstance(ba, ArrayCoordinates1d)
#         assert not isinstance(ba, MonotonicCoordinates1d)
#         assert_equal(ba.coordinates, [45., 55., 65., 95., 10., 20., 50., 60.])

#         # overlap, descending
#         da = d.concat(a)
#         assert isinstance(da, ArrayCoordinates1d)
#         assert not isinstance(da, MonotonicCoordinates1d)
#         assert_equal(da.coordinates, [35., 25., 15., 10., 20., 50., 60.])

#         # ascending
#         ac = a.concat(c)
#         assert isinstance(ac, MonotonicCoordinates1d)
#         assert_equal(ac.coordinates, [10., 20., 50., 60., 70., 80., 90])

#         # ascending, reverse
#         cd = c.concat(d)
#         assert isinstance(cd, MonotonicCoordinates1d)
#         assert_equal(cd.coordinates, [15., 25., 35., 70., 80., 90])

#         # descending
#         fd = f.concat(d)
#         assert isinstance(fd, MonotonicCoordinates1d)
#         assert_equal(fd.coordinates, [90., 80., 70, 35, 25, 15])

#         # descending, reverse
#         dc = d.concat(c)
#         assert isinstance(cd, MonotonicCoordinates1d)
#         assert_equal(cd.coordinates, [15., 25., 35., 70., 80., 90])

#         # empty self
#         ea = e.concat(a)
#         assert isinstance(ea, MonotonicCoordinates1d)
#         assert_equal(ea.coordinates, a.coordinates)

#         et = e.concat(t)
#         assert isinstance(et, MonotonicCoordinates1d)
#         assert_equal(et.coordinates, t.coordinates)
        
#         eu = e.concat(u)
#         assert isinstance(eu, UniformCoordinates1d)
#         assert_equal(eu.coordinates, u.coordinates)
        
#         eo = e.concat(o)
#         assert isinstance(eo, ArrayCoordinates1d)
#         assert not isinstance(eo, MonotonicCoordinates1d)
#         assert_equal(eo.coordinates, o.coordinates)

#         # empty other
#         ae = a.concat(e)
#         assert isinstance(ae, ArrayCoordinates1d)
#         assert_equal(ae.coordinates, a.coordinates)

#         te = t.concat(e)
#         assert isinstance(te, ArrayCoordinates1d)
#         assert_equal(te.coordinates, t.coordinates)

#         # ArrayCoordinates1d other
#         co = c.concat(o)
#         assert isinstance(co, ArrayCoordinates1d)
#         assert not isinstance(co, MonotonicCoordinates1d)
#         assert_equal(co.coordinates, [70., 80., 90., 20., 50., 60., 10.])

#         # UniformCoordinates1d other, overlap
#         au = a.concat(u)
#         assert isinstance(au, ArrayCoordinates1d)
#         assert not isinstance(au, MonotonicCoordinates1d)
#         assert_equal(au.coordinates, [10., 20., 50., 60., 45., 55., 65., 75., 85., 95.])

#         # UniformCoordinates1d other, no overlap
#         av = a.concat(v)
#         assert isinstance(av, MonotonicCoordinates1d)
#         assert_equal(av.coordinates, [10., 20., 50., 60., 75., 85., 95.])

#     def test_concat_equal(self):        
#         # ascending
#         c = MonotonicCoordinates1d([10., 20., 50., 60.])
#         c.concat(MonotonicCoordinates1d([70., 80., 90.]), inplace=True)
#         assert_equal(c.coordinates, [10., 20., 50., 60., 70., 80., 90])

#         # ascending, reverse
#         c = MonotonicCoordinates1d([70., 80., 90.])
#         c.concat(MonotonicCoordinates1d([35., 25., 15.]), inplace=True)
#         assert_equal(c.coordinates, [15., 25., 35., 70., 80., 90])

#         # descending
#         c = MonotonicCoordinates1d([90., 80., 70.])
#         c.concat(MonotonicCoordinates1d([35., 25., 15.]), inplace=True)
#         assert_equal(c.coordinates, [90., 80., 70, 35, 25, 15])

#         # descending, reverse
#         c = MonotonicCoordinates1d([35., 25., 15.])
#         c.concat(MonotonicCoordinates1d([70., 80., 90.]), inplace=True)
#         assert_equal(c.coordinates, [90.,  80.,  70.,  35.,  25.,  15.])

#         # UniformCoordinates1d other, no overlap
#         c = MonotonicCoordinates1d([10., 20., 50., 60.])
#         c.concat(UniformCoordinates1d(75, 95, 10), inplace=True)
#         assert_equal(c.coordinates, [10., 20., 50., 60., 75., 85., 95.])

#         # empty self
#         c = MonotonicCoordinates1d()
#         c.concat(MonotonicCoordinates1d([10., 20., 50., 60.]), inplace=True)
#         assert_equal(c.coordinates, [10., 20., 50., 60.])

#         c = MonotonicCoordinates1d()
#         c.concat(MonotonicCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04']), inplace=True)
#         assert_equal(c.coordinates, np.array(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04']).astype(np.datetime64))
        
#         c = MonotonicCoordinates1d()
#         c.concat(UniformCoordinates1d(75, 95, 10), inplace=True)
#         assert_equal(c.coordinates, [75., 85., 95.])
        
#         # empty other
#         c = MonotonicCoordinates1d([10., 20., 50., 60.])
#         c.concat(MonotonicCoordinates1d(), inplace=True)
#         assert_equal(c.coordinates, [10., 20., 50., 60.])

#         c = MonotonicCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'])
#         c.concat(MonotonicCoordinates1d(), inplace=True)
#         assert_equal(c.coordinates, np.array(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04']).astype(np.datetime64))

#         # overlap, ascending
#         c = MonotonicCoordinates1d([10., 20., 50., 60.])
#         with pytest.raises(ValueError):
#             c.concat(MonotonicCoordinates1d([45., 55., 65., 95.]), inplace=True)
        
#         # overlap, descending
#         c = MonotonicCoordinates1d([35., 25., 15.])
#         with pytest.raises(ValueError):
#             c.concat(MonotonicCoordinates1d([10., 20., 50., 60.]), inplace=True)

#         # overlap, UniformCoordinates1d
#         c = MonotonicCoordinates1d([10., 20., 50., 60.])
#         with pytest.raises(ValueError):
#             c.concat(UniformCoordinates1d(45, 95, 10), inplace=True)

#         # ArrayCoordinates1d other
#         c = MonotonicCoordinates1d([45., 55., 65., 95.])
#         with pytest.raises(TypeError):
#             c.concat(ArrayCoordinates1d([20., 50., 60., 10.]), inplace=True)

#         c = MonotonicCoordinates1d() # should fail here even when empty
#         with pytest.raises(TypeError):
#             c.concat(ArrayCoordinates1d([20., 50., 60, 10.]), inplace=True)

#     def test_add(self):
#         a = MonotonicCoordinates1d([10., 20., 50., 60.])
#         assert isinstance(a + 1, MonotonicCoordinates1d)

class TestUniformCoordinates1d(object):
    def test_numerical(self):
        # ascending
        c = UniformCoordinates1d(0, 50, 10, name='lat')
        a = np.array([0, 10, 20, 30, 40, 50], dtype=float)
        
        assert c.start == 0
        assert c.stop == 50
        assert c.step == 10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([0, 50], dtype=float))
        assert c.size == 6
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == False

        # descending
        c = UniformCoordinates1d(50, 0, -10, name='lat')
        a = np.array([50, 40, 30, 20, 10, 0], dtype=float)

        assert c.start == 50
        assert c.stop == 0
        assert c.step == -10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([0, 50], dtype=float))
        assert c.size == 6
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == True

    def test_numerical_inexact(self):
        # ascending
        c = UniformCoordinates1d(0, 49, 10, name='lat')
        a = np.array([0, 10, 20, 30, 40], dtype=float)
        
        assert c.start == 0
        assert c.stop == 49
        assert c.step == 10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([0, 40], dtype=float))
        assert c.size == 5
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == False

        # descending
        c = UniformCoordinates1d(50, 1, -10, name='lat')
        a = np.array([50, 40, 30, 20, 10], dtype=float)
        
        assert c.start == 50
        assert c.stop == 1
        assert c.step == -10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([10, 50], dtype=float))
        assert c.dtype == float
        assert c.size == a.size
        assert c.is_monotonic == True
        assert c.is_descending == True
        
    def test_datetime(self):
        # ascending
        c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D', name='time')
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
        
        # descending
        c = UniformCoordinates1d('2018-01-04', '2018-01-01', '-1,D', name='time')
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

    def test_datetime_inexact(self):
        # ascending
        c = UniformCoordinates1d('2018-01-01', '2018-01-06', '2,D', name='time')
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

        # descending
        c = UniformCoordinates1d('2018-01-06', '2018-01-01', '-2,D', name='time')
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

    def test_datetime_month_step(self):
        # ascending
        c = UniformCoordinates1d('2018-01-01', '2018-04-01', '1,M', name='time')
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

        # descending
        c = UniformCoordinates1d('2018-04-01', '2018-01-01', '-1,M', name='time')
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

    def test_datetime_year_step(self):
        # ascending, exact
        c = UniformCoordinates1d('2018-01-01', '2021-01-01', '1,Y', name='time')
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

        # descending, exact
        c = UniformCoordinates1d('2021-01-01', '2018-01-01', '-1,Y', name='time')
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

        # ascending, inexact (two cases)
        c = UniformCoordinates1d('2018-01-01', '2021-04-01', '1,Y', name='time')
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

        c = UniformCoordinates1d('2018-04-01', '2021-01-01', '1,Y', name='time')
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

        # descending, inexact (two cases)
        c = UniformCoordinates1d('2021-01-01', '2018-04-01', '-1,Y', name='time')
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

        c = UniformCoordinates1d('2021-04-01', '2018-01-01', '-1,Y', name='time')
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

    def test_numerical_size(self):
        # ascending
        c = UniformCoordinates1d(0, 10, size=20, name='lat')
        assert c.start == 0
        assert c.stop == 10
        assert c.step == 10/19.
        assert_equal(c.coordinates, np.linspace(0, 10, 20))
        assert_equal(c.bounds, np.array([0, 10], dtype=float))
        assert c.size == 20
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == False

        # descending
        c = UniformCoordinates1d(10, 0, size=20, name='lat')
        assert c.start == 10
        assert c.stop == 0
        assert c.step == -10/19.
        assert_equal(c.coordinates, np.linspace(10, 0, 20))
        assert_equal(c.bounds, np.array([0, 10], dtype=float))
        assert c.size == 20
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == True

    def test_datetime_size(self):
        # ascending
        c = UniformCoordinates1d('2018-01-01', '2018-01-10', size=10, name='time')
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-01-10')
        assert_equal(c.bounds, [np.datetime64('2018-01-01'), np.datetime64('2018-01-10')])
        assert c.size == 10
        assert c.dtype == np.datetime64
        assert c.is_descending == False

        # descending
        c = UniformCoordinates1d('2018-01-10', '2018-01-01', size=10, name='time')
        assert c.start == np.datetime64('2018-01-10')
        assert c.stop == np.datetime64('2018-01-01')
        assert_equal(c.bounds, [np.datetime64('2018-01-01'), np.datetime64('2018-01-10')])
        assert c.size == 10
        assert c.dtype == np.datetime64
        assert c.is_descending == True

    @pytest.mark.skip("spec uncertain")
    def test_datetime_size_inexact(self):
        # ascending
        c = UniformCoordinates1d('2018-01-01', '2018-01-10', size=20, name='time')
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-01-10')
        assert_equal(c.bounds, [np.datetime64('2018-01-01'), np.datetime64('2018-01-10')])
        assert c.size == 20
        assert c.dtype == np.datetime64
        assert c.is_descending == False

    def test_size_floating_point_error(self):
        c = UniformCoordinates1d(50.619, 50.62795, size=30, name='lat')
        assert c.size == 30

    def test_numerical_singleton(self):
        # positive step
        c = UniformCoordinates1d(1, 1, 10, name='lat')
        a = np.array([1], dtype=float)
        
        assert c.start == 1
        assert c.stop == 1
        assert c.step == 10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([1, 1], dtype=float))
        assert c.size == 1
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == None

        # negative step
        c = UniformCoordinates1d(1, 1, -10, name='lat')
        a = np.array([1], dtype=float)
        
        assert c.start == 1
        assert c.stop == 1
        assert c.step == -10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([1, 1], dtype=float))
        assert c.size == 1
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == None

    def test_datetime_singleton(self):
        # positive step
        c = UniformCoordinates1d('2018-01-01', '2018-01-01', '1,D', name='time')
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
        
        # negative step
        c = UniformCoordinates1d('2018-01-01', '2018-01-01', '-1,D', name='time')
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

    def test_invalid_init(self):
        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 50, 0, name='lat')

        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 50, -10, name='lat')

        with pytest.raises(ValueError):
            UniformCoordinates1d(50, 0, 10, name='lat')
        
        with pytest.raises(TypeError):
            UniformCoordinates1d(0, '2018-01-01', 10, name='lat')

        with pytest.raises(TypeError):
            UniformCoordinates1d('2018-01-01', 50, 10, name='lat')

        with pytest.raises(TypeError):
            UniformCoordinates1d('2018-01-01', '2018-01-02', 10, name='lat')
        
        with pytest.raises(TypeError):
            UniformCoordinates1d(0., '2018-01-01', '1,D', name='lat')

        with pytest.raises(TypeError):
            UniformCoordinates1d('2018-01-01', 50, '1,D', name='lat')

        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 50, '1,D', name='lat')

        with pytest.raises(ValueError):
            UniformCoordinates1d('a', 50, 10, name='lat')

        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 'b', 10, name='lat')

        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 50, 'a', name='lat')

        with pytest.raises(TypeError):
            UniformCoordinates1d(name='lat')

        with pytest.raises(TypeError):
            UniformCoordinates1d(0, name='lat')

        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 50, name='lat')

        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 50, 10, size=6, name='lat')

        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 10, size=20., name='lat')
        
        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 10, size='string', name='lat')

        with pytest.raises(TypeError):
            UniformCoordinates1d('2018-01-10', '2018-01-01', size='1,D', name='lat')

    def test_ctype(self):
        # default
        assert UniformCoordinates1d(0, 50, 10, name='lat').ctype == 'midpoint'

        # valid
        UniformCoordinates1d(0, 50, 10, name='lat', ctype='point')
        UniformCoordinates1d(0, 50, 10, name='lat', ctype='left')
        UniformCoordinates1d(0, 50, 10, name='lat', ctype='right')
        UniformCoordinates1d(0, 50, 10, name='lat', ctype='midpoint')
        
        # invalid
        with pytest.raises(tl.TraitError):
            UniformCoordinates1d(0, 50, 10, name='lat', ctype='abc')

    def test_properties(self):
        # without extents
        c = UniformCoordinates1d(0, 50, 10, name='lat', ctype='point')
        assert isinstance(c.properties, dict)
        assert set(c.properties.keys()) == set(['units', 'ctype', 'coord_ref_sys'])

        # with extents
        c = UniformCoordinates1d(0, 50, 10, name='lat', extents=[0, 50])
        assert isinstance(c.properties, dict)
        assert set(c.properties.keys()) == set(['units', 'ctype', 'coord_ref_sys', 'extents'])

    def test_extents(self):
        # default None
        c = UniformCoordinates1d(0, 50, 10, name='lat')
        assert c.extents is None

        # numerical
        c = UniformCoordinates1d(0, 50, 10, name='lat', extents=[0, 55])
        assert_equal(c.extents, np.array([0, 55], dtype=float))

        # datetime
        c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D', name='time', extents=['2018-01-01', '2019-01-06'])
        assert_equal(c.extents, np.array(['2018-01-01', '2019-01-06']).astype(np.datetime64))

        # invalid (ctype=point)
        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 50, 10, name='lat', ctype='point', extents=[0, 55])

        # invalid (wrong dtype)
        with pytest.raises(ValueError):
            UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D', name='lat', extents=[0, 55])
        
        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 50, 10, name='time', extents=['2018-01-01', '2019-03-01'])

        # invalid (shape)
        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 50, 10, name='lat', extents=[0])

    def test_area_bounds_point(self):
        # numerical, ascending/descending and exact/inexact
        c = UniformCoordinates1d(0, 50, 10, name='lat', ctype='point')
        assert_equal(c.area_bounds, np.array([0, 50], dtype=float))
        c = UniformCoordinates1d(50, 0, -10, name='lat', ctype='point')
        assert_equal(c.area_bounds, np.array([0, 50], dtype=float))
        c = UniformCoordinates1d(0, 49, 10, name='lat', ctype='point')
        assert_equal(c.area_bounds, np.array([0, 40], dtype=float))
        c = UniformCoordinates1d(50, 9, -10, name='lat', ctype='point')
        assert_equal(c.area_bounds, np.array([10, 50], dtype=float))

        # datetime, ascending/descending and exact/inexact
        c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D', name='time', ctype='point')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-04', '2018-01-01', '-1,D', name='time', ctype='point')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-01', '2018-01-06', '2,D', name='time', ctype='point')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-05']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-06', '2018-01-01', '-2,D', name='time', ctype='point')
        assert_equal(c.area_bounds, np.array(['2018-01-02', '2018-01-06']).astype(np.datetime64))

    def test_area_bounds_explicit_extents(self):
        c = UniformCoordinates1d(0, 50, 10, name='lat', extents=[-10, 10])
        assert_equal(c.area_bounds, np.array([-10, 10], dtype=float))
        
        c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D', name='time', extents=['2016-01-01', '2021-01-01'])
        assert_equal(c.area_bounds, np.array(['2016-01-01', '2021-01-01']).astype(np.datetime64))
        
    def test_area_bounds_left(self):
        # numerical, ascending/descending and exact/inexact/singleton
        c = UniformCoordinates1d(0, 50, 10, name='lat', ctype='left')
        assert_equal(c.area_bounds, np.array([0, 60], dtype=float))
        c = UniformCoordinates1d(50, 0, -10, name='lat', ctype='left')
        assert_equal(c.area_bounds, np.array([0, 60], dtype=float))
        c = UniformCoordinates1d(0, 49, 10, name='lat', ctype='left')
        assert_equal(c.area_bounds, np.array([0, 50.0], dtype=float))
        c = UniformCoordinates1d(50, 9, -10, name='lat', ctype='left')
        assert_equal(c.area_bounds, np.array([10, 60.0], dtype=float))
        c = UniformCoordinates1d(0, 0, 10, name='lat', ctype='left')
        assert_equal(c.area_bounds, np.array([0, 10], dtype=float))
        c = UniformCoordinates1d(0, 0, -10, name='lat', ctype='left')
        assert_equal(c.area_bounds, np.array([0, 10], dtype=float))

        # # datetime, ascending/descending and exact/inexact/singleton
        # c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D', name='time', ctype='left')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-04', '2018-01-01', '-1,D', name='time', ctype='left')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-01', '2018-01-06', '2,D', name='time', ctype='left')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-05']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-06', '2018-01-01', '-2,D', name='time', ctype='left')
        # assert_equal(c.area_bounds, np.array(['2018-01-02', '2018-01-06']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-01', '2018-01-01', '1,D', name='time', ctype='left')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-01', '2018-01-01', '-1,D', name='time', ctype='left')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))

    def test_area_bounds_right(self):
        # numerical, ascending/descending and exact/inexact/singleton
        c = UniformCoordinates1d(0, 50, 10, name='lat', ctype='right')
        assert_equal(c.area_bounds, np.array([-10, 50], dtype=float))
        c = UniformCoordinates1d(50, 0, -10, name='lat', ctype='right')
        assert_equal(c.area_bounds, np.array([-10, 50], dtype=float))
        c = UniformCoordinates1d(0, 49, 10, name='lat', ctype='right')
        assert_equal(c.area_bounds, np.array([-10, 40], dtype=float))
        c = UniformCoordinates1d(50, 9, -10, name='lat', ctype='right')
        assert_equal(c.area_bounds, np.array([0, 50], dtype=float))
        c = UniformCoordinates1d(0, 0, 10, name='lat', ctype='right')
        assert_equal(c.area_bounds, np.array([-10, 0], dtype=float))
        c = UniformCoordinates1d(0, 0, -10, name='lat', ctype='right')
        assert_equal(c.area_bounds, np.array([-10, 0], dtype=float))

        # # datetime, ascending/descending and exact/inexact/singleton
        # c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D', name='time', ctype='right')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-04', '2018-01-01', '-1,D', name='time', ctype='right')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-01', '2018-01-06', '2,D', name='time', ctype='right')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-05']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-06', '2018-01-01', '-2,D', name='time', ctype='right')
        # assert_equal(c.area_bounds, np.array(['2018-01-02', '2018-01-06']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-01', '2018-01-01', '1,D', name='time', ctype='right')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-01', '2018-01-01', '-1,D', name='time', ctype='right')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))

    def test_area_bounds_midpoint_numerical(self):
        # numerical, ascending/descending and exact/inexact/singleton
        c = UniformCoordinates1d(0, 50, 10, name='lat', ctype='midpoint')
        assert_equal(c.area_bounds, np.array([-5, 55], dtype=float))
        c = UniformCoordinates1d(50, 0, -10, name='lat', ctype='midpoint')
        assert_equal(c.area_bounds, np.array([-5, 55], dtype=float))
        c = UniformCoordinates1d(0, 49, 10, name='lat', ctype='midpoint')
        assert_equal(c.area_bounds, np.array([-5, 45], dtype=float))
        c = UniformCoordinates1d(50, 9, -10, name='lat', ctype='midpoint')
        assert_equal(c.area_bounds, np.array([5, 55], dtype=float))
        c = UniformCoordinates1d(0, 0, 10, name='lat', ctype='midpoint')
        assert_equal(c.area_bounds, np.array([-5, 5], dtype=float))
        c = UniformCoordinates1d(0, 0, -10, name='lat', ctype='midpoint')
        assert_equal(c.area_bounds, np.array([-5, 5], dtype=float))

    @pytest.mark.skip('TODO')
    def test_area_bounds_midpoint_datetime(self):
        # datetime, ascending/descending and exact/inexact/singleton
        c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D', name='time', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-04', '2018-01-01', '-1,D', name='time', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-01', '2018-01-06', '2,D', name='time', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-05']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-06', '2018-01-01', '-2,D', name='time', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2018-01-02', '2018-01-06']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-01', '2018-01-01', '1,D', name='time', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-01', '2018-01-01', '-1,D', name='time', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))

    def test_index(self):
        c = UniformCoordinates1d(0, 50, 10, name='lat')
        
        # int
        c2 = c[2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 20
        assert c2.stop == 20
        assert c2.step == 10

        c2 = c[-2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 40
        assert c2.stop == 40
        assert c2.step == 10

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
        
        # ordered array
        c2 = c[[0, 1, 3]]
        assert isinstance(c2, MonotonicCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, np.array([0, 10, 30], dtype=float))

        c2 = c[[3, 1, 0]]
        assert isinstance(c2, MonotonicCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, np.array([30, 10, 0], dtype=float))

        # unordered array (tries to return ArrayCoordinates1d instead)
        with pytest.raises(tl.TraitError):
            c[[0, 3, 1]] # must be ctype=point

        cpoints = UniformCoordinates1d(0, 50, 10, name='lat', ctype='point')
        c2 = cpoints[[0, 3, 1]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == cpoints.name
        assert c2.properties == cpoints.properties
        assert_equal(c2.coordinates, np.array([0, 30, 10], dtype=float))

        # boolean array
        c2 = c[[True, True, True, False, True, False]]
        assert isinstance(c2, MonotonicCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, np.array([0, 10, 20, 40], dtype=float))

        # invalid
        with pytest.raises(IndexError):
            c[0.3]

        with pytest.raises(IndexError):
            c[10]

    def test_index_descending(self):
        c = UniformCoordinates1d(50, 0, -10, name='lat')
        
        # int
        c2 = c[2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 30
        assert c2.stop == 30
        assert c2.step == -10

        c2 = c[-2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 10
        assert c2.stop == 10
        assert c2.step == -10

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
        
        # ordered array
        c2 = c[[0, 1, 3]]
        assert isinstance(c2, MonotonicCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, np.array([50, 40, 20], dtype=float))

        c2 = c[[3, 1, 0]]
        assert isinstance(c2, MonotonicCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, np.array([20, 40, 50], dtype=float))

        # unordered array (tries to return ArrayCoordinates1d instead)
        with pytest.raises(tl.TraitError):
            c[[0, 3, 1]] # must be ctype=point

        cpoints = UniformCoordinates1d(50, 0, -10, name='lat', ctype='point')
        c2 = cpoints[[0, 3, 1]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == cpoints.name
        assert c2.properties == cpoints.properties
        assert_equal(c2.coordinates, np.array([50, 20, 40], dtype=float))

        # boolean array
        c2 = c[[True, True, True, False, True, False]]
        assert isinstance(c2, MonotonicCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, np.array([50, 40, 30, 10], dtype=float))

        # invalid
        with pytest.raises(IndexError):
            c[0.3]

        with pytest.raises(IndexError):
            c[10]
        
#     def test_select_ascending(self):
#         c = UniformCoordinates1d(20., 70., 10.)
        
#         # full and empty selection type
#         assert isinstance(c.select([0, 100]), UniformCoordinates1d)
#         assert isinstance(c.select([100, 200]), ArrayCoordinates1d)
#         assert isinstance(c.select([0, 5]), ArrayCoordinates1d)
        
#         # partial, above
#         s = c.select([45, 100], pad=0)
#         assert isinstance(s, UniformCoordinates1d)
#         assert s.start == 50.
#         assert s.stop == 70.
#         assert s.step == 10.
        
#         # partial, below
#         s = c.select([5, 55], pad=0)
#         assert isinstance(s, UniformCoordinates1d)
#         assert s.start == 20.
#         assert s.stop == 50.
#         assert s.step == 10.

#         # partial, inner
#         s = c.select([30., 60.], pad=0)
#         assert isinstance(s, UniformCoordinates1d)
#         assert s.start == 30.
#         assert s.stop == 60.
#         assert s.step == 10.

#         # partial, inner exact
#         s = c.select([35., 55.], pad=0)
#         assert isinstance(s, UniformCoordinates1d)
#         assert s.start == 40.
#         assert s.stop == 50.
#         assert s.step == 10.
        
#         # partial, none
#         s = c.select([52, 55], pad=0)
#         assert isinstance(s, ArrayCoordinates1d)
#         assert_equal(s.coordinates, [])

#         # partial, backwards bounds
#         s = c.select([70, 30], pad=0)
#         assert isinstance(s, ArrayCoordinates1d)
#         assert_equal(s.coordinates, [])

#     def test_select_descending(self):
#         c = UniformCoordinates1d(70., 20., -10.)
        
#         # full and empty selection type
#         assert isinstance(c.select([0, 100]), UniformCoordinates1d)
#         assert isinstance(c.select([100, 200]), ArrayCoordinates1d)
#         assert isinstance(c.select([0, 5]), ArrayCoordinates1d)
        
#         # partial, above
#         s = c.select([45, 100], pad=0)
#         assert isinstance(s, UniformCoordinates1d)
#         assert s.start == 70.
#         assert s.stop == 50.
#         assert s.step == -10.
        
#         # partial, below
#         s = c.select([5, 55], pad=0)
#         assert isinstance(s, UniformCoordinates1d)
#         assert s.start == 50.
#         assert s.stop == 20.
#         assert s.step == -10.

#         # partial, inner
#         s = c.select([30., 60.], pad=0)
#         assert isinstance(s, UniformCoordinates1d)
#         assert s.start == 60.
#         assert s.stop == 30.
#         assert s.step == -10.

#         # partial, inner exact
#         s = c.select([35., 55.], pad=0)
#         assert isinstance(s, UniformCoordinates1d)
#         assert s.start == 50.
#         assert s.stop == 40.
#         assert s.step == -10.
        
#         # partial, none
#         s = c.select([52, 55], pad=0)
#         assert isinstance(s, ArrayCoordinates1d)
#         assert_equal(s.coordinates, [])

#         # partial, backwards bounds
#         s = c.select([70, 30], pad=0)
#         assert isinstance(s, ArrayCoordinates1d)
#         assert_equal(s.coordinates, [])

#     def test_select_ind_ascending(self):
#         c = UniformCoordinates1d(20., 70., 10.)
        
#         # partial, above
#         s = c.select([45, 100], ind=True, pad=0)
#         assert_equal(c.coordinates[s], [50., 60., 70.])
        
#         # partial, below
#         s = c.select([5, 55], ind=True, pad=0)
#         assert_equal(c.coordinates[s], [20., 30., 40., 50])

#         # partial, inner
#         s = c.select([30., 60.], ind=True, pad=0)
#         assert_equal(c.coordinates[s], [30., 40., 50., 60])

#         # partial, inner exact
#         s = c.select([35., 55.], ind=True, pad=0)
#         assert_equal(c.coordinates[s], [40., 50.])
        
#         # partial, none
#         s = c.select([52, 55], ind=True, pad=0)
#         assert_equal(c.coordinates[s], [])

#         # partial, backwards bounds
#         s = c.select([70, 30], ind=True, pad=0)
#         assert_equal(c.coordinates[s], [])

#     def test_select_ind_descending(self):
#         c = UniformCoordinates1d(70., 20., -10.)
        
#         # partial, above
#         s = c.select([45, 100], ind=True, pad=0)
#         assert_equal(c.coordinates[s], [70., 60., 50.])
        
#         # partial, below
#         s = c.select([5, 55], ind=True, pad=0)
#         assert_equal(c.coordinates[s], [50., 40., 30., 20.])

#         # partial, inner
#         s = c.select([30., 60.], ind=True, pad=0)
#         assert_equal(c.coordinates[s], [60., 50., 40., 30.])

#         # partial, inner exact
#         s = c.select([35., 55.], ind=True, pad=0)
#         assert_equal(c.coordinates[s], [50., 40.])
        
#         # partial, none
#         s = c.select([52, 55], ind=True, pad=0)
#         assert_equal(c.coordinates[s], [])

#         # partial, backwards bounds
#         s = c.select([70, 30], ind=True, pad=0)
#         assert_equal(c.coordinates[s], [])

#     def test_intersect(self):
#         # MonotonicCoordinates1d other
#         a = UniformCoordinates1d(10., 60., 10.)
#         b = UniformCoordinates1d(45., 95., 5.)
        
#         ab = a.intersect(b, pad=0)
#         assert isinstance(ab, UniformCoordinates1d)
#         assert ab.start == 50.
#         assert ab.stop == 60.
#         assert ab.step == 10.

#         ba = b.intersect(a, pad=0)
#         assert isinstance(ba, UniformCoordinates1d)
#         assert ba.start == 45.
#         assert ba.stop == 60.
#         assert ba.step == 5.

#         # ArrayCoordinates1d other
#         c = ArrayCoordinates1d([40., 70., 50.,])
#         assert isinstance(a.intersect(c), UniformCoordinates1d)
        
#         # MonotonicCoordinates1d
#         m = MonotonicCoordinates1d([40., 50., 70.])
#         assert isinstance(a.intersect(m), UniformCoordinates1d)

#     def test_concat(self):
#         a = UniformCoordinates1d(30., 60., 10.)
#         b = UniformCoordinates1d(60., 30., -10.)

#         # empty other
#         r = a.concat(ArrayCoordinates1d())
#         assert isinstance(r, UniformCoordinates1d)
#         assert r.start == a.start
#         assert r.stop == a.stop
#         assert r.step == a.step

#         r = a._concat(ArrayCoordinates1d())
#         assert isinstance(r, UniformCoordinates1d)
#         assert r.start == a.start
#         assert r.stop == a.stop
#         assert r.step == a.step

#         # both ascending -> UniformCoordinates1d
#         r = a.concat(UniformCoordinates1d(70., 100, 10.))
#         assert isinstance(r, UniformCoordinates1d)
#         assert r.start == 30.
#         assert r.stop == 100.
#         assert r.step == 10.

#         r = a.concat(UniformCoordinates1d(0., 20, 10.))
#         assert isinstance(r, UniformCoordinates1d)
#         assert r.start == 0.
#         assert r.stop == 60.
#         assert r.step == 10.

#         # both descending -> UniformCoordinates1d
#         r = b.concat(UniformCoordinates1d(100., 70, -10.))
#         assert isinstance(r, UniformCoordinates1d)
#         assert r.start == 100.
#         assert r.stop == 30.
#         assert r.step == -10.

#         r = b.concat(UniformCoordinates1d(20., 0, -10.))
#         assert isinstance(r, UniformCoordinates1d)
#         assert r.start == 60.
#         assert r.stop == 0.
#         assert r.step == -10.

#         # mismatched -> UniformCoordinates1d
#         r = b.concat(UniformCoordinates1d(70., 100, 10.))
#         assert isinstance(r, UniformCoordinates1d)
#         assert r.start == 100.
#         assert r.stop == 30.
#         assert r.step == -10.

#         r = b.concat(UniformCoordinates1d(0., 20, 10.))
#         assert isinstance(r, UniformCoordinates1d)
#         assert r.start == 60.
#         assert r.stop == 0.
#         assert r.step == -10.

#         r = a.concat(UniformCoordinates1d(100., 70, -10.))
#         assert isinstance(r, UniformCoordinates1d)
#         assert r.start == 30.
#         assert r.stop == 100.
#         assert r.step == 10.

#         r = a.concat(UniformCoordinates1d(20., 0, -10.))
#         assert isinstance(r, UniformCoordinates1d)
#         assert r.start == 0.
#         assert r.stop == 60.
#         assert r.step == 10.

#         # separated, both ascending -> MonotonicCoordinates1d
#         r = a.concat(UniformCoordinates1d(80., 100, 10.))
#         assert isinstance(r, MonotonicCoordinates1d)
#         assert_equal(r.coordinates, [30., 40., 50., 60., 80., 90., 100.])

#         r = a.concat(UniformCoordinates1d(0., 10, 10.))
#         assert isinstance(r, MonotonicCoordinates1d)
#         assert_equal(r.coordinates, [0., 10., 30., 40., 50., 60.])

#         # separated, both descendeng -> MonotonicCoordinates1d
#         r = b.concat(UniformCoordinates1d(100., 80., -10.))
#         assert isinstance(r, MonotonicCoordinates1d)
#         assert_equal(r.coordinates, [100., 90., 80., 60., 50., 40., 30.])

#         r = b.concat(UniformCoordinates1d(10., 0, -10.))
#         assert isinstance(r, MonotonicCoordinates1d)
#         assert_equal(r.coordinates, [60., 50., 40., 30., 10., 0.])

#         # separated, mismatched -> MonotonicCoordinates1d
#         r = b.concat(UniformCoordinates1d(80., 100, 10.))
#         assert isinstance(r, MonotonicCoordinates1d)
#         assert_equal(r.coordinates, [100., 90., 80., 60., 50., 40., 30.])

#         r = b.concat(UniformCoordinates1d(0., 10, 10.))
#         assert isinstance(r, MonotonicCoordinates1d)
#         assert_equal(r.coordinates, [60., 50., 40., 30., 10., 0.])

#         # separated, both descendeng -> MonotonicCoordinates1d
#         r = a.concat(UniformCoordinates1d(100., 80., -10.))
#         assert isinstance(r, MonotonicCoordinates1d)
#         assert_equal(r.coordinates, [30., 40., 50., 60., 80., 90., 100.])

#         r = a.concat(UniformCoordinates1d(10., 0, -10.))
#         assert isinstance(r, MonotonicCoordinates1d)
#         assert_equal(r.coordinates, [0., 10., 30., 40., 50., 60.])

#         # mismatched step -> MonotonicCoordinates1d
#         r = a.concat(UniformCoordinates1d(70., 100, 5.))
#         assert isinstance(r, MonotonicCoordinates1d)
#         assert_equal(r.coordinates, [30., 40., 50., 60., 70., 75., 80., 85., 90., 95., 100.])

#         # not aligned -> MonotonicCoordinates1d
#         r = a.concat(UniformCoordinates1d(65., 100, 10.))
#         assert isinstance(r, MonotonicCoordinates1d)
#         assert_equal(r.coordinates, [30., 40., 50., 60., 65., 75., 85., 95.])

#         # overlap -> ArrayCoordinates1d
#         r = a.concat(UniformCoordinates1d(50., 100, 10))
#         assert isinstance(r, ArrayCoordinates1d)
#         assert_equal(r.coordinates, [30., 40., 50., 60., 50., 60., 70., 80., 90., 100.])

#         # MonotonicCoordinates1d other
#         r = a.concat(MonotonicCoordinates1d([75, 80, 90]))
#         assert isinstance(r, MonotonicCoordinates1d)
#         assert_equal(r.coordinates, [30., 40., 50., 60., 75, 80, 90])

#         r = a.concat(MonotonicCoordinates1d([55, 75, 80, 90]))
#         assert isinstance(r, ArrayCoordinates1d)
#         assert_equal(r.coordinates, [30., 40., 50., 60., 55, 75, 80, 90])
        
#         # ArrayCoordinates1d other
#         r = a.concat(ArrayCoordinates1d([75, 90, 80]))
#         assert_equal(r.coordinates, [30., 40., 50., 60., 75, 90, 80])
#         assert isinstance(r, ArrayCoordinates1d)

#     def test_concat_equal(self):
#         # empty other
#         a = UniformCoordinates1d(30., 60., 10.)
#         a.concat(ArrayCoordinates1d(), inplace=True)
#         assert a.start == 30.
#         assert a.stop == 60.
#         assert a.step == 10.

#         a = UniformCoordinates1d(30., 60., 10.)
#         a._concat_equal(ArrayCoordinates1d())
#         assert a.start == 30.
#         assert a.stop == 60.
#         assert a.step == 10.

#         # both ascending
#         a = UniformCoordinates1d(30., 60., 10.)
#         a.concat(UniformCoordinates1d(70., 100, 10.), inplace=True)
#         assert a.start == 30.
#         assert a.stop == 100.
#         assert a.step == 10.

#         a = UniformCoordinates1d(30., 60., 10.)
#         a.concat(UniformCoordinates1d(0., 20, 10.), inplace=True)
#         assert a.start == 0.
#         assert a.stop == 60.
#         assert a.step == 10.

#         # both descending
#         b = UniformCoordinates1d(60., 30., -10.)
#         b.concat(UniformCoordinates1d(100., 70, -10.), inplace=True)
#         assert b.start == 100.
#         assert b.stop == 30.
#         assert b.step == -10.

#         b = UniformCoordinates1d(60., 30., -10.)
#         b.concat(UniformCoordinates1d(20., 0, -10.), inplace=True)
#         assert b.start == 60.
#         assert b.stop == 0.
#         assert b.step == -10.

#         # mismatched
#         b = UniformCoordinates1d(60., 30., -10.)
#         b.concat(UniformCoordinates1d(70., 100, 10.), inplace=True)
#         assert b.start == 100.
#         assert b.stop == 30.
#         assert b.step == -10.

#         b = UniformCoordinates1d(60., 30., -10.)
#         b.concat(UniformCoordinates1d(0., 20, 10.), inplace=True)
#         assert b.start == 60.
#         assert b.stop == 0.
#         assert b.step == -10.

#         a = UniformCoordinates1d(30., 60., 10.)
#         a.concat(UniformCoordinates1d(100., 70, -10.), inplace=True)
#         assert a.start == 30.
#         assert a.stop == 100.
#         assert a.step == 10.

#         a = UniformCoordinates1d(30., 60., 10.)
#         a.concat(UniformCoordinates1d(20., 0, -10.), inplace=True)
#         assert a.start == 0.
#         assert a.stop == 60.
#         assert a.step == 10.

#         # separated -> ValueError
#         a = UniformCoordinates1d(30., 60., 10.)
#         with pytest.raises(ValueError):
#             a.concat(UniformCoordinates1d(80., 100, 10.), inplace=True)

#         a = UniformCoordinates1d(30., 60., 10.)
#         with pytest.raises(ValueError):
#             a.concat(UniformCoordinates1d(0., 10, 10.), inplace=True)

#         a = UniformCoordinates1d(30., 60., 10.)
#         with pytest.raises(ValueError):
#             b.concat(UniformCoordinates1d(100., 80., -10.), inplace=True)

#         a = UniformCoordinates1d(30., 60., 10.)
#         with pytest.raises(ValueError):
#             b.concat(UniformCoordinates1d(10., 0, -10.), inplace=True)

#         a = UniformCoordinates1d(30., 60., 10.)
#         with pytest.raises(ValueError):
#             b.concat(UniformCoordinates1d(80., 100, 10.), inplace=True)

#         a = UniformCoordinates1d(30., 60., 10.)
#         with pytest.raises(ValueError):
#             b.concat(UniformCoordinates1d(0., 10, 10.), inplace=True)

#         a = UniformCoordinates1d(30., 60., 10.)
#         with pytest.raises(ValueError):
#             a.concat(UniformCoordinates1d(100., 80., -10.), inplace=True)

#         a = UniformCoordinates1d(30., 60., 10.)
#         with pytest.raises(ValueError):
#             a.concat(UniformCoordinates1d(10., 0, -10.), inplace=True)

#         # mismatched step -> ValueError
#         a = UniformCoordinates1d(30., 60., 10.)
#         with pytest.raises(ValueError):
#             a.concat(UniformCoordinates1d(70., 100, 5.), inplace=True)

#         # not aligned -> ValueError
#         a = UniformCoordinates1d(30., 60., 10.)
#         with pytest.raises(ValueError):
#             a.concat(UniformCoordinates1d(65., 100, 10.), inplace=True)

#         # overlap -> ValueError
#         a = UniformCoordinates1d(30., 60., 10.)
#         with pytest.raises(ValueError):
#             a.concat(UniformCoordinates1d(50., 100, 10), inplace=True)

#         # non UniformCoordinates1d other -> TypeError
#         a = UniformCoordinates1d(30., 60., 10.)
#         with pytest.raises(TypeError):
#             a.concat(MonotonicCoordinates1d([75, 80, 90]), inplace=True)

#         # ArrayCoordinates1d other
#         a = UniformCoordinates1d(30., 60., 10.)
#         with pytest.raises(TypeError):
#             a.concat(ArrayCoordinates1d([75, 90, 80]), inplace=True)

#     def test_add(self):
#         # numerical
#         c = UniformCoordinates1d(20., 60., 10.)
#         c2 = c.add(1)
#         assert isinstance(c2, UniformCoordinates1d)
#         assert c2.start == 21.
#         assert c2.stop == 61.
#         assert c2.step == 10.

#         # simple datetime
#         t = UniformCoordinates1d('2018-01-01', '2018-01-10', '1,D')
#         t2d = t.add('2,D')
#         assert isinstance(t2d, UniformCoordinates1d)
#         assert t2d.start == np.datetime64('2018-01-03')
#         assert t2d.stop == np.datetime64('2018-01-12')
#         assert t2d.step == np.timedelta64(1, 'D')

#         t2d = t.add('-2,D')
#         assert isinstance(t2d, UniformCoordinates1d)
#         assert t2d.start == np.datetime64('2017-12-30')
#         assert t2d.stop == np.datetime64('2018-01-08')
#         assert t2d.step == np.timedelta64(1, 'D')

#         # nominal datetime
#         t2m = t.add('2,M')
#         assert isinstance(t2m, UniformCoordinates1d)
#         assert t2m.start == np.datetime64('2018-03-01')
#         assert t2m.stop == np.datetime64('2018-03-10')
#         assert t2m.step == np.timedelta64(1, 'D')

#         t2y = t.add('2,Y')
#         assert isinstance(t2y, UniformCoordinates1d)
#         assert t2y.start == np.datetime64('2020-01-01')
#         assert t2y.stop == np.datetime64('2020-01-10')
#         assert t2y.step == np.timedelta64(1, 'D')

#     def test_add_equal(self):
#         # numerical
#         c = UniformCoordinates1d(20., 60., 10.)
#         c.add(1, inplace=True)
#         assert c.start == 21.
#         assert c.stop == 61.
#         assert c.step == 10.

#         # simple datetime
#         t = UniformCoordinates1d('2018-01-01', '2018-01-10', '1,D')
#         t.add('2,D', inplace=True)
#         assert t.start == np.datetime64('2018-01-03')
#         assert t.stop == np.datetime64('2018-01-12')
#         assert t.step == np.timedelta64(1, 'D')

#         t = UniformCoordinates1d('2018-01-01', '2018-01-10', '1,D')
#         t.add('-2,D', inplace=True)
#         assert t.start == np.datetime64('2017-12-30')
#         assert t.stop == np.datetime64('2018-01-08')
#         assert t.step == np.timedelta64(1, 'D')

#         # nominal datetime
#         t = UniformCoordinates1d('2018-01-01', '2018-01-10', '1,D')
#         t.add('2,M', inplace=True)
#         assert t.start == np.datetime64('2018-03-01')
#         assert t.stop == np.datetime64('2018-03-10')
#         assert t.step == np.timedelta64(1, 'D')

#         t = UniformCoordinates1d('2018-01-01', '2018-01-10', '1,D')
#         t.add('2,Y', inplace=True)
#         assert t.start == np.datetime64('2020-01-01')
#         assert t.stop == np.datetime64('2020-01-10')
#         assert t.step == np.timedelta64(1, 'D')