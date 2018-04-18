
from datetime import datetime

import pytest
import traitlets as tl
import numpy as np
from numpy.testing import assert_equal

from podpac.core.units import Units
from podpac.core.coordinate.util import get_timedelta_unit
from podpac.core.coordinate import BaseCoord, Coord, MonotonicCoord, UniformCoord
from podpac.core.coordinate import coord_linspace

class TestBaseCoord(object):
    def test_abstract(self):
        c = BaseCoord()
        
        with pytest.raises(NotImplementedError):
            c.coordinates

        with pytest.raises(NotImplementedError):
            c.bounds

        with pytest.raises(NotImplementedError):
            c.is_datetime

        with pytest.raises(NotImplementedError):
            c.is_monotonic

        with pytest.raises(NotImplementedError):
            c.is_descending

        with pytest.raises(NotImplementedError):
            c.rasterio_regularity

        with pytest.raises(NotImplementedError):
            c.scipy_regularity

        with pytest.raises(NotImplementedError):
            c.select([0, 1])

class TestCoord(object):
    def test_empty(self):
        c = Coord()
        a = np.array([])

        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([np.nan, np.nan]))
        assert c.size == 0
        assert c.is_datetime == None
        assert c.is_monotonic == False
        assert c.is_descending is None
        assert c.rasterio_regularity == False
        assert c.scipy_regularity == True

        c = Coord([])
        assert_equal(c.coords, a)

        c = Coord(np.array([]))
        assert_equal(c.coords, a)

    def test_numerical_array(self):
        values = [0., 100., -3., 10.]
        a = np.array(values)
        
        # array input
        c = Coord(a, ctype='point')
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([-3., 100.]))
        assert np.issubdtype(c.coords.dtype, np.float)
        assert np.issubdtype(c.coordinates.dtype, np.float)
        assert np.issubdtype(c.bounds.dtype, np.float)
        assert c.size == 4
        assert c.is_datetime == False
        assert c.is_monotonic == False
        assert c.is_descending is None
        assert c.rasterio_regularity == False
        assert c.scipy_regularity == True

        # list input
        c = Coord(values)
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.float)

        # nested array input
        c = Coord(np.array([values]))
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.float)
        
        # int dtype
        c = Coord(a.astype(int))
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.float)

    def test_numerical_singleton(self):
        value = 10.0
        a = np.array([value])

        # basic input
        c = Coord(value, ctype='point')
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([value, value]))
        assert np.issubdtype(c.coords.dtype, np.float)
        assert np.issubdtype(c.coordinates.dtype, np.float)
        assert np.issubdtype(c.bounds.dtype, np.float)
        assert c.size == 1
        assert c.is_datetime == False
        assert c.is_monotonic == False
        assert c.is_descending is None
        assert c.rasterio_regularity == True
        assert c.scipy_regularity == True

        # np input
        c = Coord(np.array(value))
        assert_equal(c.coords, a)
        assert_equal(c.size, 1)
        assert np.issubdtype(c.coords.dtype, np.float)

        c = Coord(np.array([value]))
        assert_equal(c.coords, a)
        assert_equal(c.size, 1)
        assert np.issubdtype(c.coords.dtype, np.float)
        
        # int dtype
        c = Coord(int(value))
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.float)

    def test_datetime_array(self):
        values = ['2018-01-01', '2019-01-01', '2017-01-01', '2018-01-02']
        a = np.array(values).astype(np.datetime64)
        
        # array input
        c = Coord(a, ctype='point')
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        assert np.issubdtype(c.coords.dtype, np.datetime64)
        assert np.issubdtype(c.coordinates.dtype, np.datetime64)
        assert np.issubdtype(c.bounds.dtype, np.datetime64)
        assert c.size == 4
        assert c.is_datetime == True
        assert c.is_monotonic == False
        assert c.is_descending is None
        assert c.rasterio_regularity == False
        assert c.scipy_regularity == True

        # list of strings
        c = Coord(values)
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.datetime64)

        # nested array input
        c = Coord(np.array([values]))
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.datetime64)
        
        # datetime.datetime
        c = Coord([e.item() for e in a])
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.datetime64)

    def test_datetime_singleton(self):
        value_str = '2018-01-01'
        value_dt64 = np.datetime64(value_str)
        value_dt = value_dt64.item()
        a = np.array([value_dt64])

        # string input
        c = Coord(value_str, ctype='point')
        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([value_dt64, value_dt64]))
        assert np.issubdtype(c.coords.dtype, np.datetime64)
        assert np.issubdtype(c.coordinates.dtype, np.datetime64)
        assert np.issubdtype(c.bounds.dtype, np.datetime64)
        assert c.size == 1
        assert c.is_datetime == True
        assert c.is_monotonic == False
        assert c.is_descending is None
        assert c.rasterio_regularity == True
        assert c.scipy_regularity == True

        # ndim=0 array input
        c = Coord(np.array(value_str))
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.datetime64)

        # datetime64 input
        c = Coord(value_dt64)
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.datetime64)

        # datetime.datetime
        c = Coord(value_dt)
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.datetime64)

    def test_invalid_coords(self):
        with pytest.raises(TypeError):
            Coord(dict())

        with pytest.raises(ValueError):
            Coord([[1, 2], [3, 4]])

    def test_mixed_coords(self):
        with pytest.raises(TypeError):
            Coord([1, '2018-01-01'])
       
        with pytest.raises(TypeError):
            Coord(['2018-01-01', 1])

    def test_invalid_datetime(self):
        with pytest.raises(ValueError):
            Coord('a')

    def test_units(self):
        # default
        assert Coord(5.0).units is None
        
        # custom
        # TODO
        # c = Coord(5.0, units=Units())
        # assert isinstance(c.units, Units)

    def test_ctype(self):
        # default
        assert Coord(5.0).ctype == 'segment'
        
        # initialize
        assert Coord(5.0, ctype='segment').ctype == 'segment'
        assert Coord(5.0, ctype='point').ctype == 'point'
        
        # invalid
        with pytest.raises(tl.TraitError):
            Coord(5.0, ctype='abc')

    def test_segment_position(self):
        numerical = [1.0, 2.0]
        datetimes = ['2018-01-01', '2018-01-02']

        # default
        assert Coord(numerical).segment_position == 0.5
        assert Coord(datetimes).segment_position == 0.5
        
        # custom
        assert Coord(numerical, segment_position=0.8).segment_position == 0.8
        assert Coord(datetimes, segment_position=0.8).segment_position == 0.8
        # TODO datetime type

        with pytest.raises(ValueError):
            Coord(numerical, segment_position=1.5)

        with pytest.raises(ValueError):
            Coord(numerical, segment_position=-0.5)

    def test_extents(self):
        numerical = [1.0, 2.0]
        datetimes = ['2018-01-01', '2018-01-02']

        # default
        assert Coord(numerical).extents is None
        assert Coord(datetimes).extents is None

        # custom
        numerical_extents = [1.3, 2.7]
        datetimes_extents = ['2018-03-03', '2018-04-02']
        datetimes_extents_dt64 = np.array(['2018-03-03', '2018-04-02']).astype(np.datetime64)

        assert_equal(Coord(numerical, extents=numerical_extents).extents, numerical_extents)
        assert_equal(Coord(numerical, extents=datetimes_extents).extents, datetimes_extents_dt64)
        assert_equal(Coord(numerical, extents=datetimes_extents_dt64).extents, datetimes_extents_dt64)
        
        # invalid
        with pytest.raises(ValueError):
            Coord(numerical, extents=[1.0])

    def test_coord_ref_system(self):
        # default
        assert Coord(5.0).coord_ref_sys == u''

        # custom
        assert Coord(5.0, coord_ref_sys='test').coord_ref_sys == u'test'

        # invalid
        with pytest.raises(tl.TraitError):
            Coord(5.0, coord_ref_sys=1)

    def test_kwargs(self):
        # defaults
        c = Coord()
        assert isinstance(c.kwargs, dict)
        assert set(c.kwargs.keys()) == set(['units', 'ctype', 'segment_position', 'extents'])
        assert c.kwargs['units'] is None
        assert c.kwargs['ctype'] == 'segment'
        assert c.kwargs['segment_position'] == 0.5
        assert c.kwargs['extents'] is None

        # custom
        # TODO replace with the version with units...
        c = Coord(ctype="point", segment_position=0.8, extents=[0.0, 1.0])
        # units = Units()
        # c = Coord(units=units, ctype="segment", segment_position=0.8, extents=[0.0, 1.0])

        assert isinstance(c.kwargs, dict)
        assert set(c.kwargs.keys()) == set(['units', 'ctype', 'segment_position', 'extents'])
        assert c.kwargs['units'] is None
        # assert c.kwargs['units'] is units
        assert c.kwargs['ctype'] is 'point'
        assert c.kwargs['segment_position'] == 0.8
        assert_equal(c.kwargs['extents'], [0.0, 1.0])

    def test_area_bounds(self):
        numerical = [0., 100., -3., 10.]
        lo = -3.
        hi = 100.
        e = [lo-10, hi+10]

        datetimes = ['2018-01-01', '2019-01-01', '2017-01-01', '2018-01-02']
        dt_lo = np.datetime64('2017-01-01')
        dt_hi = np.datetime64('2019-01-01')
        dt_e = [dt_lo - np.timedelta64(10, 'D'), dt_hi + np.timedelta64(10, 'D')]
        
        # points: same as bounds
        assert_equal(Coord(numerical, ctype='point').area_bounds, [lo, hi])
        assert_equal(Coord(datetimes, ctype='point').area_bounds, [dt_lo, dt_hi])
        
        assert np.issubdtype(Coord(numerical, ctype='point').area_bounds.dtype, np.float)
        assert np.issubdtype(Coord(datetimes, ctype='point').area_bounds.dtype, np.datetime64)
        
        # points: ignore extents
        assert_equal(Coord(numerical, ctype='point', extents=e).area_bounds, [lo, hi])
        assert_equal(Coord(datetimes, ctype='point', extents=dt_e).area_bounds, [dt_lo, dt_hi])
        
        assert np.issubdtype(Coord(numerical, ctype='point', extents=e).area_bounds.dtype, np.float)
        assert np.issubdtype(Coord(datetimes, ctype='point', extents=dt_e).area_bounds.dtype, np.datetime64)

        # segments: explicit extents
        assert_equal(Coord(numerical, ctype='segment', extents=e).area_bounds, e)
        assert_equal(Coord(datetimes, ctype='segment', extents=dt_e).area_bounds, dt_e)

        assert np.issubdtype(Coord(numerical, ctype='segment', extents=e).area_bounds.dtype, np.float)
        assert np.issubdtype(Coord(datetimes, ctype='segment', extents=dt_e).area_bounds.dtype, np.datetime64)
        
        # segments: calculate from bounds, segment_position
        # TODO
        # assert_equal(Coord(numerical), ctype='segment').area_bounds, TODO)
        # assert_equal(Coord(numerical), ctype='segment', segment_position=0.8).area_bounds, TODO)
        # assert_equal(Coord(datetimes), ctype='segment').area_bounds, TODO)
        # assert_equal(Coord(datetimes), ctype='segment', segment_position=0.8).area_bounds, TODO)

        assert np.issubdtype(Coord(numerical, ctype='segment').area_bounds.dtype, np.float)
        assert np.issubdtype(Coord(datetimes, ctype='segment').area_bounds.dtype, np.datetime64)
        assert np.issubdtype(Coord(numerical, ctype='segment').area_bounds.dtype, np.float)
        assert np.issubdtype(Coord(datetimes, ctype='segment').area_bounds.dtype, np.datetime64)

    def test_select(self):
        c = Coord([20., 50., 60., 90., 40., 10.])
        
        # full selection
        s = c.select([0, 100])
        assert isinstance(s, Coord)
        assert_equal(s.coordinates, c.coordinates)

        # none, above
        s = c.select([100, 200])
        assert isinstance(s, Coord)
        assert_equal(s.coordinates, [])

        # none, below
        s = c.select([0, 5])
        assert isinstance(s, Coord)
        assert_equal(s.coordinates, [])

        # partial, above
        s = c.select([50, 100])
        assert isinstance(s, Coord)
        assert_equal(s.coordinates, [50., 60., 90.])

        # partial, below
        s = c.select([0, 50])
        assert isinstance(s, Coord)
        assert_equal(s.coordinates, [20., 50., 40., 10.])

        # partial, inner
        s = c.select([30., 70.])
        assert isinstance(s, Coord)
        assert_equal(s.coordinates, [50., 60., 40.])

        # partial, inner exact
        s = c.select([40., 60.])
        assert isinstance(s, Coord)
        assert_equal(s.coordinates, [50., 60., 40.])

        # partial, none
        s = c.select([52, 55])
        assert isinstance(s, Coord)
        assert_equal(s.coordinates, [])

        # partial, backwards bounds
        s = c.select([70, 30])
        assert isinstance(s, Coord)
        assert_equal(s.coordinates, [])

        # empty coords
        c = Coord()        
        s = c.select([0, 1])
        assert isinstance(s, Coord)
        assert_equal(s.coordinates, [])

    def test_select_ind(self):
        c = Coord([20., 50., 60., 90., 40., 10.])
        
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
        c = Coord()        
        I = c.select([0, 1], ind=True)
        assert_equal(c.coordinates[I], [])

    def test_intersect(self):
        a = Coord([20., 50., 60., 10.])
        b = Coord([55., 65., 95., 45.])
        c = Coord([80., 70., 90.])
        e = Coord()
        
        # Coord other, both directions
        ab = a.intersect(b)
        assert isinstance(ab, Coord)
        assert_equal(ab.coordinates, [50., 60.])
        
        ba = b.intersect(a)
        assert isinstance(ba, Coord)
        assert_equal(ba.coordinates, [55., 45.])

        # Coord other, no overlap
        ac = a.intersect(c)
        assert isinstance(ac, Coord)
        assert_equal(ac.coordinates, [])

        # empty self
        ea = e.intersect(a)
        assert isinstance(ea, Coord)
        assert_equal(ea.coordinates, [])

        # empty other
        ae = a.intersect(e)
        assert isinstance(ae, Coord)
        assert_equal(ae.coordinates, [])

        # MonotonicCoord other
        m = MonotonicCoord([45., 55., 65., 95.])
        am = a.intersect(m)
        assert isinstance(am, Coord)
        assert_equal(am.coordinates, [50., 60.])

        # UniformCoord other
        u = UniformCoord(45, 95, 10)
        au = a.intersect(u)
        assert isinstance(au, Coord)
        assert_equal(au.coordinates, [50., 60.])

    def test_intersect_ind(self):
        a = Coord([20., 50., 60., 10.])
        b = Coord([55., 65., 95., 45.])
        c = Coord([80., 70., 90.])
        e = Coord()
        
        # Coord other, both directions
        I = a.intersect(b, ind=True)
        assert_equal(a.coordinates[I], [50., 60.])
        
        I = b.intersect(a, ind=True)
        assert_equal(b.coordinates[I], [55., 45.])

        # Coord other, no overlap
        I = a.intersect(c, ind=True)
        assert_equal(a.coordinates[I], [])

        # empty self
        I = e.intersect(a, ind=True)
        assert_equal(e.coordinates[I], [])

        # empty other
        I = a.intersect(e, ind=True)
        assert_equal(a.coordinates[I], [])

        # MonotonicCoord other
        m = MonotonicCoord([45., 55., 65., 95.])
        I = a.intersect(m, ind=True)
        assert_equal(a.coordinates[I], [50., 60.])

        # UniformCoord other
        u = UniformCoord(45, 95, 10)
        I = a.intersect(u, ind=True)
        assert_equal(a.coordinates[I], [50., 60.])

    def test_concat(self):
        a = Coord([20., 50., 60., 10.])
        b = Coord([55., 65., 95., 45.])
        c = Coord([80., 70., 90.])
        e = Coord()

        t = Coord(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03'])
        
        # Coord other, both directions
        ab = a.concat(b)
        assert isinstance(ab, Coord)
        assert_equal(ab.coordinates, [20., 50., 60., 10., 55., 65., 95., 45.])
        
        ba = b.concat(a)
        assert isinstance(ba, Coord)
        assert_equal(ba.coordinates, [55., 65., 95., 45., 20., 50., 60., 10.])

        # empty self
        ea = e.concat(a)
        assert isinstance(ea, Coord)
        assert_equal(ea.coordinates, a.coordinates)

        et = e.concat(t)
        assert isinstance(et, Coord)
        assert_equal(et.coordinates, t.coordinates)

        # empty other
        ae = a.concat(e)
        assert isinstance(ae, Coord)
        assert_equal(ae.coordinates, a.coordinates)

        te = t.concat(e)
        assert isinstance(te, Coord)
        assert_equal(te.coordinates, t.coordinates)

        # MonotonicCoord other
        m = MonotonicCoord([45., 55., 65., 95.])
        am = a.concat(m)
        assert isinstance(am, Coord)
        assert_equal(am.coordinates, [20., 50., 60., 10., 45., 55., 65., 95.])

        # UniformCoord other
        u = UniformCoord(45, 95, 10)
        au = a.concat(u)
        assert isinstance(au, Coord)
        assert_equal(au.coordinates, [20., 50., 60., 10., 45., 55., 65., 75., 85., 95.])

        # type error
        with pytest.raises(TypeError):
            a.concat(5)

        with pytest.raises(TypeError):
            a.concat(t)

        with pytest.raises(TypeError):
            t.concat(a)

    def test_concat_equal(self):
        # Coord other
        c = Coord([20., 50., 60., 10.])
        c.concat(Coord([55., 65., 95., 45.]), inplace=True)
        assert_equal(c.coordinates, [20., 50., 60., 10., 55., 65., 95., 45.])
        
        # empty self
        c = Coord()
        c.concat(Coord([55., 65., 95., 45.]), inplace=True)
        assert_equal(c.coordinates, [55., 65., 95., 45.])

        c = Coord()
        c.concat(Coord(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03']), inplace=True)
        assert_equal(c.coordinates, np.array(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03']).astype(np.datetime64))

        # empty other
        c = Coord([20., 50., 60., 10.])
        c.concat(Coord(), inplace=True)
        assert_equal(c.coordinates, [20., 50., 60., 10.])

        c = Coord(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03'])
        c.concat(Coord(), inplace=True)
        assert_equal(c.coordinates, np.array(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03']).astype(np.datetime64))

        # MonotonicCoord other
        c = Coord([20., 50., 60., 10.])
        c.concat(MonotonicCoord([45., 55., 65., 95.]), inplace=True)
        assert_equal(c.coordinates, [20., 50., 60., 10., 45., 55., 65., 95.])

        # UniformCoord other
        c = Coord([20., 50., 60., 10.])
        c.concat(UniformCoord(45, 95, 10), inplace=True)
        assert_equal(c.coordinates, [20., 50., 60., 10., 45., 55., 65., 75., 85., 95.])

    def test_add(self):
        a = Coord([20., 50., 60., 10.])
        t = Coord(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03'])
        e = Coord()

        # empty
        r = e.add(1)
        assert isinstance(r, Coord)
        assert_equal(r.coordinates, [])

        r = e.add('10,D')
        assert isinstance(r, Coord)
        assert_equal(r.coordinates, [])
        
        # standard
        r = a.add(1)
        assert isinstance(r, Coord)
        assert_equal(r.coordinates, [21., 51., 61., 11.])

        r = t.add('10,D')
        assert isinstance(r, Coord)
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
        c = Coord()
        c.add(1, inplace=True)
        assert_equal(c.coordinates, [])

        c = Coord()
        c.add('10,D', inplace=True)
        assert_equal(c.coordinates, [])

        # standard
        c = Coord([20., 50., 60., 10.])
        c.add(1, inplace=True)
        assert_equal(c.coordinates, [21., 51., 61., 11.])

        c = Coord(['2018-01-02', '2018-01-01', '2018-01-04', '2018-01-03'])
        c.add('10,D', inplace=True)
        assert_equal(c.coordinates, np.array(['2018-01-12', '2018-01-11', '2018-01-14', '2018-01-13']).astype(np.datetime64))

    def test___add__(self):
        a = Coord([20., 50., 60., 10.])
        b = Coord([55., 65., 95., 45.])
        
        # concat
        r = a + b
        assert isinstance(r, Coord)
        assert_equal(r.coordinates, [20., 50., 60., 10., 55., 65., 95., 45.])
        
        r = b + a
        assert isinstance(r, Coord)
        assert_equal(r.coordinates, [55., 65., 95., 45., 20., 50., 60., 10.])

        # add
        r = a + 1
        assert isinstance(r, Coord)
        assert_equal(r.coordinates, [21., 51., 61., 11.])

    def test___iadd__(self):
        # concat
        c = Coord([20., 50., 60., 10.])
        c += Coord([55., 65., 95., 45.])
        assert_equal(c.coordinates, [20., 50., 60., 10., 55., 65., 95., 45.])
        
        # add
        c = Coord([20., 50., 60., 10.])
        c += 1
        assert_equal(c.coordinates, [21., 51., 61., 11.])

    def test___sub__(self):
        c = Coord([20., 50., 60., 10.])
        r = c - 1
        assert isinstance(r, Coord)
        assert_equal(r.coordinates, [19., 49., 59., 9.])

    def test___isub__(self):
        c = Coord([20., 50., 60., 10.])
        c -= 1
        assert_equal(c.coordinates, [19., 49., 59., 9.])

    def test___and__(self):
        a = Coord([20., 50., 60., 10.])
        b = Coord([55., 65., 95., 45.])
        
        r = a & b
        assert isinstance(r, Coord)
        assert_equal(r.coordinates, [50., 60.])
        
        r = b & a
        assert isinstance(r, Coord)
        assert_equal(r.coordinates, [55., 45.])

    def test___getitem__(self):
        c = Coord([20., 50., 60., 10.])
        
        assert c[0] == 20.
        assert_equal(c[1:3], [50., 60.])

    def test___len__(self):
        c = Coord([20., 50., 60., 10.])
        assert len(c) == 4

        e = Coord()
        assert len(e) == 0

    def test___in__(self):
        c = Coord([20., 50., 60., 10.])
        t = Coord(['2018-01-02', '2018-01-01', '2018-01-05', '2018-01-03'])
        e = Coord()
        
        # TODO area_bounds
        
        assert 10. in c
        assert 50. in c
        assert 55. in c
        assert 90 not in c
        
        assert np.datetime64('2018-01-02') in t
        assert np.datetime64('2018-01-04') in t
        assert np.datetime64('2017-01-04') not in t
        
        assert 50. not in e
        assert np.datetime64('2018-01-02') not in e

    def test___repr__(self):
        c = Coord([20., 50., 60., 10.])
        repr(c)

class TestMonotonicCoord(object):
    """
    MonotonicCoord extends Coord, so only some properties and methods are 
    tested here::
     - coords validation
     - bounds, is_datetime, is_datetime, and is_descending properties
     - intersect and select methods
    """

    def test_empty(self):
        c = MonotonicCoord()
        a = np.array([])

        assert_equal(c.coords, a)
        assert_equal(c.bounds, np.array([np.nan, np.nan]))
        assert c.is_datetime == None
        assert c.is_monotonic == True
        assert c.is_descending == False

        c = MonotonicCoord([])
        assert_equal(c.coords, a)

        c = MonotonicCoord(np.array([]))
        assert_equal(c.coords, a)

    def test_numerical_array(self):
        values = [0.0, 1.0, 50., 100.]
        a = np.array(values)
        
        # array input
        c = MonotonicCoord(a, ctype='point')
        assert_equal(c.coords, a)
        assert_equal(c.bounds, np.array([0., 100.]))
        assert np.issubdtype(c.coords.dtype, np.float)
        assert np.issubdtype(c.bounds.dtype, np.float)
        assert c.is_datetime == False
        assert c.is_monotonic == True
        assert c.is_descending == False
        
        # list input
        c = MonotonicCoord(values)
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.float)

        # nested array input
        c = MonotonicCoord(np.array([values]))
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.float)
        
        # int dtype
        c = MonotonicCoord(a.astype(int))
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.float)

        # descending
        c = MonotonicCoord(a[::-1])
        assert_equal(c.coords, a[::-1])
        assert_equal(c.bounds, np.array([0., 100.]))
        assert c.is_descending == True
        assert np.issubdtype(c.coords.dtype, np.float)

        # invalid (unsorted)
        with pytest.raises(ValueError):
            MonotonicCoord(a[[0, 2, 1, 3]])

    def test_numerical_singleton(self):
        value = 10.0
        a = np.array([value])

        # basic input
        c = MonotonicCoord(value, ctype='point')
        assert_equal(c.coords, a)
        assert_equal(c.bounds, np.array([value, value]))
        assert np.issubdtype(c.coords.dtype, np.float)
        assert np.issubdtype(c.bounds.dtype, np.float)
        assert c.is_datetime == False
        assert c.is_monotonic == True
        assert c.is_descending == False
        
        # np input
        c = MonotonicCoord(np.array(value))
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.float)

        c = MonotonicCoord(np.array([value]))
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.float)
        
        # int dtype
        c = MonotonicCoord(int(value))
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.float)

    def test_datetime_array(self):
        values = ['2017-01-01', '2018-01-01', '2018-01-02', '2019-01-01']
        a = np.array(values).astype(np.datetime64)
        
        # array input
        c = MonotonicCoord(a, ctype='point')
        assert_equal(c.coords, a)
        assert_equal(c.bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        assert np.issubdtype(c.coords.dtype, np.datetime64)
        assert np.issubdtype(c.bounds.dtype, np.datetime64)
        assert c.is_datetime == True
        assert c.is_monotonic == True
        assert c.is_descending == False
        
        # list of strings
        c = MonotonicCoord(values)
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.datetime64)

        # nested array input
        c = MonotonicCoord(np.array([values]))
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.datetime64)
        
        # datetime.datetime
        c = MonotonicCoord([e.item() for e in a])
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.datetime64)

        # descending
        c = MonotonicCoord(a[::-1])
        assert_equal(c.coords, a[::-1])
        assert_equal(c.bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        assert c.is_descending == True
        assert np.issubdtype(c.coords.dtype, np.datetime64)

        # invalid (unsorted)
        with pytest.raises(ValueError):
            MonotonicCoord(a[[0, 2, 1, 3]])

    def test_datetime_singleton(self):
        value_str = '2018-01-01'
        value_dt64 = np.datetime64(value_str)
        value_dt = value_dt64.item()
        a = np.array([value_dt64])

        # string input
        c = MonotonicCoord(value_str, ctype='point')
        assert_equal(c.coords, a)
        assert_equal(c.bounds, np.array([value_dt64, value_dt64]))
        assert np.issubdtype(c.coords.dtype, np.datetime64)
        assert np.issubdtype(c.bounds.dtype, np.datetime64)
        assert c.is_datetime == True
        assert c.is_monotonic == True
        assert c.is_descending == False
        
        # ndim=0 array input
        c = MonotonicCoord(np.array(value_str))
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.datetime64)

        # datetime64 input
        c = MonotonicCoord(value_dt64)
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.datetime64)

        # datetime.datetime
        c = MonotonicCoord(value_dt)
        assert_equal(c.coords, a)
        assert np.issubdtype(c.coords.dtype, np.datetime64)

    def test_select_ascending(self):
        c = MonotonicCoord([10., 20., 40., 50., 60., 90.])
        
        # full and empty selection type
        assert isinstance(c.select([0, 100]), MonotonicCoord)
        assert isinstance(c.select([100, 200]), Coord)
        assert isinstance(c.select([0, 5]), Coord)
        
        # partial, above
        s = c.select([50, 100], pad=0)
        assert isinstance(s, MonotonicCoord)
        assert_equal(s.coordinates, [50., 60., 90.])

        # partial, below
        s = c.select([0, 50], pad=0)
        assert isinstance(s, MonotonicCoord)
        assert_equal(s.coordinates, [10., 20., 40., 50.])

        # partial, inner
        s = c.select([30., 70.], pad=0)
        assert isinstance(s, MonotonicCoord)
        assert_equal(s.coordinates, [40., 50., 60.])

        # partial, inner exact
        s = c.select([40., 60.], pad=0)
        assert isinstance(s, MonotonicCoord)
        assert_equal(s.coordinates, [40., 50., 60.])

        # partial, none
        s = c.select([52, 55], pad=0)
        assert isinstance(s, Coord)
        assert_equal(s.coordinates, [])

        # partial, backwards bounds
        s = c.select([70, 30], pad=0)
        assert isinstance(s, Coord)
        assert_equal(s.coordinates, [])

    def test_select_descending(self):
        c = MonotonicCoord([90., 60., 50., 40., 20., 10.])
        
        # full and empty selection type
        assert isinstance(c.select([0, 100]), MonotonicCoord)
        assert isinstance(c.select([100, 200]), Coord)
        assert isinstance(c.select([0, 5]), Coord)
        
        # partial, above
        s = c.select([50, 100], pad=0)
        assert isinstance(s, MonotonicCoord)
        assert_equal(s.coordinates, [90., 60., 50.])

        # partial, below
        s = c.select([0, 50], pad=0)
        assert isinstance(s, MonotonicCoord)
        assert_equal(s.coordinates, [50., 40., 20., 10.])

        # partial, inner
        s = c.select([30., 70.], pad=0)
        assert isinstance(s, MonotonicCoord)
        assert_equal(s.coordinates, [60., 50., 40.])

        # partial, inner exact
        s = c.select([40., 60.], pad=0)
        assert isinstance(s, MonotonicCoord)
        assert_equal(s.coordinates, [60., 50., 40.])

        # partial, none
        s = c.select([52, 55], pad=0)
        assert isinstance(s, Coord)
        assert_equal(s.coordinates, [])

        # partial, backwards bounds
        s = c.select([70, 30], pad=0)
        assert isinstance(s, Coord)
        assert_equal(s.coordinates, [])

    def test_select_ind(self):
        c = MonotonicCoord([10., 20., 40., 50., 60., 90.])
        
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
        # MonotonicCoord other
        a = MonotonicCoord([10., 20., 50., 60.])
        b = MonotonicCoord([45., 55., 65., 95.])
        assert isinstance(a.intersect(b), MonotonicCoord)

        # Coord other
        c = Coord([20., 50., 60., 10.])
        assert isinstance(a.intersect(c), MonotonicCoord)
        
        # UniformCoord
        u = UniformCoord(45, 95, 10)
        assert isinstance(a.intersect(u), MonotonicCoord)

    def test_concat(self):
        a = MonotonicCoord([10., 20., 50., 60.])
        b = MonotonicCoord([45., 55., 65., 95.])
        c = MonotonicCoord([70., 80., 90.])
        d = MonotonicCoord([35., 25., 15.])
        e = MonotonicCoord()
        f = MonotonicCoord([90., 80., 70.])
        o = Coord([20., 50., 60., 10.])
        u = UniformCoord(45, 95, 10)
        v = UniformCoord(75, 95, 10)
        t = MonotonicCoord(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'])
        
        # overlap, ascending
        ab = a.concat(b)
        assert isinstance(ab, Coord)
        assert not isinstance(ab, MonotonicCoord)
        assert_equal(ab.coordinates, [10., 20., 50., 60., 45., 55., 65., 95.])
        
        ba = b.concat(a)
        assert isinstance(ba, Coord)
        assert not isinstance(ba, MonotonicCoord)
        assert_equal(ba.coordinates, [45., 55., 65., 95., 10., 20., 50., 60.])

        # overlap, descending
        da = d.concat(a)
        assert isinstance(da, Coord)
        assert not isinstance(da, MonotonicCoord)
        assert_equal(da.coordinates, [35., 25., 15., 10., 20., 50., 60.])

        # ascending
        ac = a.concat(c)
        assert isinstance(ac, MonotonicCoord)
        assert_equal(ac.coordinates, [10., 20., 50., 60., 70., 80., 90])

        # ascending, reverse
        cd = c.concat(d)
        assert isinstance(cd, MonotonicCoord)
        assert_equal(cd.coordinates, [15., 25., 35., 70., 80., 90])

        # descending
        fd = f.concat(d)
        assert isinstance(fd, MonotonicCoord)
        assert_equal(fd.coordinates, [90., 80., 70, 35, 25, 15])

        # descending, reverse
        dc = d.concat(c)
        assert isinstance(cd, MonotonicCoord)
        assert_equal(cd.coordinates, [15., 25., 35., 70., 80., 90])

        # empty self
        ea = e.concat(a)
        assert isinstance(ea, MonotonicCoord)
        assert_equal(ea.coordinates, a.coordinates)

        et = e.concat(t)
        assert isinstance(et, MonotonicCoord)
        assert_equal(et.coordinates, t.coordinates)
        
        eu = e.concat(u)
        assert isinstance(eu, UniformCoord)
        assert_equal(eu.coordinates, u.coordinates)
        
        eo = e.concat(o)
        assert isinstance(eo, Coord)
        assert not isinstance(eo, MonotonicCoord)
        assert_equal(eo.coordinates, o.coordinates)

        # empty other
        ae = a.concat(e)
        assert isinstance(ae, Coord)
        assert_equal(ae.coordinates, a.coordinates)

        te = t.concat(e)
        assert isinstance(te, Coord)
        assert_equal(te.coordinates, t.coordinates)

        # Coord other
        co = c.concat(o)
        assert isinstance(co, Coord)
        assert not isinstance(co, MonotonicCoord)
        assert_equal(co.coordinates, [70., 80., 90., 20., 50., 60., 10.])

        # UniformCoord other, overlap
        au = a.concat(u)
        assert isinstance(au, Coord)
        assert not isinstance(au, MonotonicCoord)
        assert_equal(au.coordinates, [10., 20., 50., 60., 45., 55., 65., 75., 85., 95.])

        # UniformCoord other, no overlap
        av = a.concat(v)
        assert isinstance(av, MonotonicCoord)
        assert_equal(av.coordinates, [10., 20., 50., 60., 75., 85., 95.])

    def test_concat_equal(self):        
        # ascending
        c = MonotonicCoord([10., 20., 50., 60.])
        c.concat(MonotonicCoord([70., 80., 90.]), inplace=True)
        assert_equal(c.coordinates, [10., 20., 50., 60., 70., 80., 90])

        # ascending, reverse
        c = MonotonicCoord([70., 80., 90.])
        c.concat(MonotonicCoord([35., 25., 15.]), inplace=True)
        assert_equal(c.coordinates, [15., 25., 35., 70., 80., 90])

        # descending
        c = MonotonicCoord([90., 80., 70.])
        c.concat(MonotonicCoord([35., 25., 15.]), inplace=True)
        assert_equal(c.coordinates, [90., 80., 70, 35, 25, 15])

        # descending, reverse
        c = MonotonicCoord([35., 25., 15.])
        c.concat(MonotonicCoord([70., 80., 90.]), inplace=True)
        assert_equal(c.coordinates, [90.,  80.,  70.,  35.,  25.,  15.])

        # UniformCoord other, no overlap
        c = MonotonicCoord([10., 20., 50., 60.])
        c.concat(UniformCoord(75, 95, 10), inplace=True)
        assert_equal(c.coordinates, [10., 20., 50., 60., 75., 85., 95.])

        # empty self
        c = MonotonicCoord()
        c.concat(MonotonicCoord([10., 20., 50., 60.]), inplace=True)
        assert_equal(c.coordinates, [10., 20., 50., 60.])

        c = MonotonicCoord()
        c.concat(MonotonicCoord(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04']), inplace=True)
        assert_equal(c.coordinates, np.array(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04']).astype(np.datetime64))
        
        c = MonotonicCoord()
        c.concat(UniformCoord(75, 95, 10), inplace=True)
        assert_equal(c.coordinates, [75., 85., 95.])
        
        # empty other
        c = MonotonicCoord([10., 20., 50., 60.])
        c.concat(MonotonicCoord(), inplace=True)
        assert_equal(c.coordinates, [10., 20., 50., 60.])

        c = MonotonicCoord(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'])
        c.concat(MonotonicCoord(), inplace=True)
        assert_equal(c.coordinates, np.array(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04']).astype(np.datetime64))

        # overlap, ascending
        c = MonotonicCoord([10., 20., 50., 60.])
        with pytest.raises(ValueError):
            c.concat(MonotonicCoord([45., 55., 65., 95.]), inplace=True)
        
        # overlap, descending
        c = MonotonicCoord([35., 25., 15.])
        with pytest.raises(ValueError):
            c.concat(MonotonicCoord([10., 20., 50., 60.]), inplace=True)

        # overlap, UniformCoord
        c = MonotonicCoord([10., 20., 50., 60.])
        with pytest.raises(ValueError):
            c.concat(UniformCoord(45, 95, 10), inplace=True)

        # Coord other
        c = MonotonicCoord([45., 55., 65., 95.])
        with pytest.raises(TypeError):
            c.concat(Coord([20., 50., 60., 10.]), inplace=True)

        c = MonotonicCoord() # should fail here even when empty
        with pytest.raises(TypeError):
            c.concat(Coord([20., 50., 60, 10.]), inplace=True)

    def test_add(self):
        a = MonotonicCoord([10., 20., 50., 60.])
        assert isinstance(a + 1, MonotonicCoord)

class TestUniformCoord(object):
    def test_numerical(self):
        # ascending
        c = UniformCoord(0., 50., 10.)
        a = np.array([0., 10., 20., 30., 40., 50])
        
        assert type(c.start) == float
        assert type(c.stop) == float
        assert type(c.delta) == float
        assert isinstance(c.coords, tuple)
        assert c.start == 0.
        assert c.stop == 50.
        assert c.delta == 10.
        assert c.coords == (0., 50.)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([0., 50.]))
        assert np.issubdtype(c.coordinates.dtype, np.float)
        assert np.issubdtype(c.bounds.dtype, np.float)
        assert c.size == 6
        assert c.is_datetime == False
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.rasterio_regularity == True
        assert c.scipy_regularity == True

        # descending
        c = UniformCoord(50., 0., -10.)
        a = np.array([50., 40., 30., 20., 10., 0])

        assert c.start == 50.
        assert c.stop == 0.
        assert c.delta == -10.
        assert c.coords == (50., 0.)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([0., 50.]))
        assert c.size == 6
        assert c.is_datetime == False
        assert c.is_monotonic == True
        assert c.is_descending == True

        # inexact step
        c = UniformCoord(0., 49., 10.)
        a = np.array([0., 10., 20., 30., 40.])
        
        assert type(c.start) == float
        assert type(c.stop) == float
        assert type(c.delta) == float
        assert isinstance(c.coords, tuple)
        assert c.start == 0.
        assert c.stop == 49.
        assert c.delta == 10.
        assert c.coords == (0., 49.)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([0., 40.]))
        assert c.size == 5
        
    def test_datetime(self):
        # ascending
        c = UniformCoord('2018-01-01', '2018-01-04', '1,D')
        a = np.array(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04']).astype(np.datetime64)
        
        assert isinstance(c.start, np.datetime64)
        assert isinstance(c.stop, np.datetime64)
        assert isinstance(c.delta, np.timedelta64)
        assert isinstance(c.coords, tuple)
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-01-04')
        assert c.delta == np.timedelta64(1, 'D')
        assert c.coords == (np.datetime64('2018-01-01'), np.datetime64('2018-01-04'))
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert np.issubdtype(c.coordinates.dtype, np.datetime64)
        assert np.issubdtype(c.bounds.dtype, np.datetime64)
        assert c.size == a.size
        assert c.is_datetime == True
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.rasterio_regularity == True
        assert c.scipy_regularity == True

        # descending
        c = UniformCoord('2018-01-04', '2018-01-01', '-1,D')
        a = np.array(['2018-01-04', '2018-01-03', '2018-01-02', '2018-01-01']).astype(np.datetime64)
        
        assert c.start == a[0]
        assert c.stop == a[-1]
        assert c.delta == np.timedelta64(-1, 'D')
        assert c.coords == (a[0], a[-1])
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.is_datetime == True
        assert c.is_monotonic == True
        assert c.is_descending == True

        # inexact step
        c = UniformCoord('2018-01-01', '2018-01-06', '2,D')
        a = np.array(['2018-01-01', '2018-01-03', '2018-01-05']).astype(np.datetime64)
        
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-01-06')
        assert c.delta == np.timedelta64(2, 'D')
        assert c.coords == (np.datetime64('2018-01-01'), np.datetime64('2018-01-06'))
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.is_datetime == True
        assert c.is_monotonic == True
        assert c.is_descending == False

        # month step
        c = UniformCoord('2018-01-01', '2018-04-01', '1,M')
        a = np.array(['2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01']).astype(np.datetime64)
        
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-04-01')
        assert c.delta == np.timedelta64(1, 'M')
        assert c.coords == (np.datetime64('2018-01-01'), np.datetime64('2018-04-01'))
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size

        # year step ascending, exact
        c = UniformCoord('2018-01-01', '2021-01-01', '1,Y')
        a = np.array(['2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01']).astype(np.datetime64)
        
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2021-01-01')
        assert c.delta == np.timedelta64(1, 'Y')
        assert c.coords == (np.datetime64('2018-01-01'), np.datetime64('2021-01-01'))
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size

        # year step descending, exact
        c = UniformCoord('2021-01-01', '2018-01-01', '-1,Y')
        a = np.array(['2021-01-01', '2020-01-01', '2019-01-01', '2018-01-01']).astype(np.datetime64)
        
        assert c.start == np.datetime64('2021-01-01')
        assert c.stop == np.datetime64('2018-01-01')
        assert c.delta == np.timedelta64(-1, 'Y')
        assert c.coords == (np.datetime64('2021-01-01'), np.datetime64('2018-01-01'))
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size

        # year step ascending, inexact (two cases)
        c = UniformCoord('2018-01-01', '2021-04-01', '1,Y')
        a = np.array(['2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01']).astype(np.datetime64)
        
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2021-04-01')
        assert c.delta == np.timedelta64(1, 'Y')
        assert c.coords == (np.datetime64('2018-01-01'), np.datetime64('2021-04-01'))
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size

        c = UniformCoord('2018-04-01', '2021-01-01', '1,Y')
        a = np.array(['2018-04-01', '2019-04-01', '2020-04-01']).astype(np.datetime64)
        
        assert c.start == np.datetime64('2018-04-01')
        assert c.stop == np.datetime64('2021-01-01')
        assert c.delta == np.timedelta64(1, 'Y')
        assert c.coords == (np.datetime64('2018-04-01'), np.datetime64('2021-01-01'))
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size

        # year step descending, inexact (two cases)
        c = UniformCoord('2021-01-01', '2018-04-01', '-1,Y')
        a = np.array(['2021-01-01', '2020-01-01', '2019-01-01', '2018-01-01']).astype(np.datetime64)
        
        assert c.start == np.datetime64('2021-01-01')
        assert c.stop == np.datetime64('2018-04-01')
        assert c.delta == np.timedelta64(-1, 'Y')
        assert c.coords == (np.datetime64('2021-01-01'), np.datetime64('2018-04-01'))
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size

        c = UniformCoord('2021-04-01', '2018-01-01', '-1,Y')
        a = np.array(['2021-04-01', '2020-04-01', '2019-04-01', '2018-04-01']).astype(np.datetime64)
        
        assert c.start == np.datetime64('2021-04-01')
        assert c.stop == np.datetime64('2018-01-01')
        assert c.delta == np.timedelta64(-1, 'Y')
        assert c.coords == (np.datetime64('2021-04-01'), np.datetime64('2018-01-01'))
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size

    def test_invalid_init(self):
        with pytest.raises(ValueError):
            UniformCoord(0., 50., 0)

        with pytest.raises(ValueError):
            UniformCoord(0., 50., -10)

        with pytest.raises(ValueError):
            UniformCoord(50., 0., 10)
        
        with pytest.raises(TypeError):
            UniformCoord(0., '2018-01-01', 10.)

        with pytest.raises(TypeError):
            UniformCoord('2018-01-01', 50., 10.)

        with pytest.raises(TypeError):
            UniformCoord('2018-01-01', '2018-01-02', 10.)
        
        with pytest.raises(TypeError):
            UniformCoord(0., '2018-01-01', '1,D')

        with pytest.raises(TypeError):
            UniformCoord('2018-01-01', 50., '1,D')

        with pytest.raises(TypeError):
            UniformCoord(0., 50., '1,D')

        with pytest.raises(ValueError):
            UniformCoord('a', 50., 10.)

        with pytest.raises(ValueError):
            UniformCoord(0., 'b', 10)

        with pytest.raises(ValueError):
            UniformCoord(0., 50., 'a')

        with pytest.raises(TypeError):
            UniformCoord()

        with pytest.raises(TypeError):
            UniformCoord(0., 50.)

    def test_select_ascending(self):
        c = UniformCoord(20., 70., 10.)
        
        # full and empty selection type
        assert isinstance(c.select([0, 100]), UniformCoord)
        assert isinstance(c.select([100, 200]), Coord)
        assert isinstance(c.select([0, 5]), Coord)
        
        # partial, above
        s = c.select([45, 100], pad=0)
        assert isinstance(s, UniformCoord)
        assert s.start == 50.
        assert s.stop == 70.
        assert s.delta == 10.
        
        # partial, below
        s = c.select([5, 55], pad=0)
        assert isinstance(s, UniformCoord)
        assert s.start == 20.
        assert s.stop == 50.
        assert s.delta == 10.

        # partial, inner
        s = c.select([30., 60.], pad=0)
        assert isinstance(s, UniformCoord)
        assert s.start == 30.
        assert s.stop == 60.
        assert s.delta == 10.

        # partial, inner exact
        s = c.select([35., 55.], pad=0)
        assert isinstance(s, UniformCoord)
        assert s.start == 40.
        assert s.stop == 50.
        assert s.delta == 10.
        
        # partial, none
        s = c.select([52, 55], pad=0)
        assert isinstance(s, Coord)
        assert_equal(s.coordinates, [])

        # partial, backwards bounds
        s = c.select([70, 30], pad=0)
        assert isinstance(s, Coord)
        assert_equal(s.coordinates, [])

    def test_select_descending(self):
        c = UniformCoord(70., 20., -10.)
        
        # full and empty selection type
        assert isinstance(c.select([0, 100]), UniformCoord)
        assert isinstance(c.select([100, 200]), Coord)
        assert isinstance(c.select([0, 5]), Coord)
        
        # partial, above
        s = c.select([45, 100], pad=0)
        assert isinstance(s, UniformCoord)
        assert s.start == 70.
        assert s.stop == 50.
        assert s.delta == -10.
        
        # partial, below
        s = c.select([5, 55], pad=0)
        assert isinstance(s, UniformCoord)
        assert s.start == 50.
        assert s.stop == 20.
        assert s.delta == -10.

        # partial, inner
        s = c.select([30., 60.], pad=0)
        assert isinstance(s, UniformCoord)
        assert s.start == 60.
        assert s.stop == 30.
        assert s.delta == -10.

        # partial, inner exact
        s = c.select([35., 55.], pad=0)
        assert isinstance(s, UniformCoord)
        assert s.start == 50.
        assert s.stop == 40.
        assert s.delta == -10.
        
        # partial, none
        s = c.select([52, 55], pad=0)
        assert isinstance(s, Coord)
        assert_equal(s.coordinates, [])

        # partial, backwards bounds
        s = c.select([70, 30], pad=0)
        assert isinstance(s, Coord)
        assert_equal(s.coordinates, [])

    def test_select_ind_ascending(self):
        c = UniformCoord(20., 70., 10.)
        
        # partial, above
        s = c.select([45, 100], ind=True, pad=0)
        assert_equal(c.coordinates[s], [50., 60., 70.])
        
        # partial, below
        s = c.select([5, 55], ind=True, pad=0)
        assert_equal(c.coordinates[s], [20., 30., 40., 50])

        # partial, inner
        s = c.select([30., 60.], ind=True, pad=0)
        assert_equal(c.coordinates[s], [30., 40., 50., 60])

        # partial, inner exact
        s = c.select([35., 55.], ind=True, pad=0)
        assert_equal(c.coordinates[s], [40., 50.])
        
        # partial, none
        s = c.select([52, 55], ind=True, pad=0)
        assert_equal(c.coordinates[s], [])

        # partial, backwards bounds
        s = c.select([70, 30], ind=True, pad=0)
        assert_equal(c.coordinates[s], [])

    def test_select_ind_descending(self):
        c = UniformCoord(70., 20., -10.)
        
        # partial, above
        s = c.select([45, 100], ind=True, pad=0)
        assert_equal(c.coordinates[s], [70., 60., 50.])
        
        # partial, below
        s = c.select([5, 55], ind=True, pad=0)
        assert_equal(c.coordinates[s], [50., 40., 30., 20.])

        # partial, inner
        s = c.select([30., 60.], ind=True, pad=0)
        assert_equal(c.coordinates[s], [60., 50., 40., 30.])

        # partial, inner exact
        s = c.select([35., 55.], ind=True, pad=0)
        assert_equal(c.coordinates[s], [50., 40.])
        
        # partial, none
        s = c.select([52, 55], ind=True, pad=0)
        assert_equal(c.coordinates[s], [])

        # partial, backwards bounds
        s = c.select([70, 30], ind=True, pad=0)
        assert_equal(c.coordinates[s], [])

    def test_intersect(self):
        # MonotonicCoord other
        a = UniformCoord(10., 60., 10.)
        b = UniformCoord(45., 95., 5.)
        
        ab = a.intersect(b, pad=0)
        assert isinstance(ab, UniformCoord)
        assert ab.start == 50.
        assert ab.stop == 60.
        assert ab.delta == 10.

        ba = b.intersect(a, pad=0)
        assert isinstance(ba, UniformCoord)
        assert ba.start == 45.
        assert ba.stop == 60.
        assert ba.delta == 5.

        # Coord other
        c = Coord([40., 70., 50.,])
        assert isinstance(a.intersect(c), UniformCoord)
        
        # MonotonicCoord
        m = MonotonicCoord([40., 50., 70.])
        assert isinstance(a.intersect(m), UniformCoord)

    def test_concat(self):
        a = UniformCoord(30., 60., 10.)
        b = UniformCoord(60., 30., -10.)

        # empty other
        r = a.concat(Coord())
        assert isinstance(r, UniformCoord)
        assert r.start == a.start
        assert r.stop == a.stop
        assert r.delta == a.delta

        r = a._concat(Coord())
        assert isinstance(r, UniformCoord)
        assert r.start == a.start
        assert r.stop == a.stop
        assert r.delta == a.delta

        # both ascending -> UniformCoord
        r = a.concat(UniformCoord(70., 100, 10.))
        assert isinstance(r, UniformCoord)
        assert r.start == 30.
        assert r.stop == 100.
        assert r.delta == 10.

        r = a.concat(UniformCoord(0., 20, 10.))
        assert isinstance(r, UniformCoord)
        assert r.start == 0.
        assert r.stop == 60.
        assert r.delta == 10.

        # both descending -> UniformCoord
        r = b.concat(UniformCoord(100., 70, -10.))
        assert isinstance(r, UniformCoord)
        assert r.start == 100.
        assert r.stop == 30.
        assert r.delta == -10.

        r = b.concat(UniformCoord(20., 0, -10.))
        assert isinstance(r, UniformCoord)
        assert r.start == 60.
        assert r.stop == 0.
        assert r.delta == -10.

        # mismatched -> UniformCoord
        r = b.concat(UniformCoord(70., 100, 10.))
        assert isinstance(r, UniformCoord)
        assert r.start == 100.
        assert r.stop == 30.
        assert r.delta == -10.

        r = b.concat(UniformCoord(0., 20, 10.))
        assert isinstance(r, UniformCoord)
        assert r.start == 60.
        assert r.stop == 0.
        assert r.delta == -10.

        r = a.concat(UniformCoord(100., 70, -10.))
        assert isinstance(r, UniformCoord)
        assert r.start == 30.
        assert r.stop == 100.
        assert r.delta == 10.

        r = a.concat(UniformCoord(20., 0, -10.))
        assert isinstance(r, UniformCoord)
        assert r.start == 0.
        assert r.stop == 60.
        assert r.delta == 10.

        # separated, both ascending -> MonotonicCoord
        r = a.concat(UniformCoord(80., 100, 10.))
        assert isinstance(r, MonotonicCoord)
        assert_equal(r.coordinates, [30., 40., 50., 60., 80., 90., 100.])

        r = a.concat(UniformCoord(0., 10, 10.))
        assert isinstance(r, MonotonicCoord)
        assert_equal(r.coordinates, [0., 10., 30., 40., 50., 60.])

        # separated, both descendeng -> MonotonicCoord
        r = b.concat(UniformCoord(100., 80., -10.))
        assert isinstance(r, MonotonicCoord)
        assert_equal(r.coordinates, [100., 90., 80., 60., 50., 40., 30.])

        r = b.concat(UniformCoord(10., 0, -10.))
        assert isinstance(r, MonotonicCoord)
        assert_equal(r.coordinates, [60., 50., 40., 30., 10., 0.])

        # separated, mismatched -> MonotonicCoord
        r = b.concat(UniformCoord(80., 100, 10.))
        assert isinstance(r, MonotonicCoord)
        assert_equal(r.coordinates, [100., 90., 80., 60., 50., 40., 30.])

        r = b.concat(UniformCoord(0., 10, 10.))
        assert isinstance(r, MonotonicCoord)
        assert_equal(r.coordinates, [60., 50., 40., 30., 10., 0.])

        # separated, both descendeng -> MonotonicCoord
        r = a.concat(UniformCoord(100., 80., -10.))
        assert isinstance(r, MonotonicCoord)
        assert_equal(r.coordinates, [30., 40., 50., 60., 80., 90., 100.])

        r = a.concat(UniformCoord(10., 0, -10.))
        assert isinstance(r, MonotonicCoord)
        assert_equal(r.coordinates, [0., 10., 30., 40., 50., 60.])

        # mismatched delta -> MonotonicCoord
        r = a.concat(UniformCoord(70., 100, 5.))
        assert isinstance(r, MonotonicCoord)
        assert_equal(r.coordinates, [30., 40., 50., 60., 70., 75., 80., 85., 90., 95., 100.])

        # not aligned -> MonotonicCoord
        r = a.concat(UniformCoord(65., 100, 10.))
        assert isinstance(r, MonotonicCoord)
        assert_equal(r.coordinates, [30., 40., 50., 60., 65., 75., 85., 95.])

        # overlap -> Coord
        r = a.concat(UniformCoord(50., 100, 10))
        assert isinstance(r, Coord)
        assert_equal(r.coordinates, [30., 40., 50., 60., 50., 60., 70., 80., 90., 100.])

        # MonotonicCoord other
        r = a.concat(MonotonicCoord([75, 80, 90]))
        assert isinstance(r, MonotonicCoord)
        assert_equal(r.coordinates, [30., 40., 50., 60., 75, 80, 90])

        r = a.concat(MonotonicCoord([55, 75, 80, 90]))
        assert isinstance(r, Coord)
        assert_equal(r.coordinates, [30., 40., 50., 60., 55, 75, 80, 90])
        
        # Coord other
        r = a.concat(Coord([75, 90, 80]))
        assert_equal(r.coordinates, [30., 40., 50., 60., 75, 90, 80])
        assert isinstance(r, Coord)

    def test_concat_equal(self):
        # empty other
        a = UniformCoord(30., 60., 10.)
        a.concat(Coord(), inplace=True)
        assert a.start == 30.
        assert a.stop == 60.
        assert a.delta == 10.

        a = UniformCoord(30., 60., 10.)
        a._concat_equal(Coord())
        assert a.start == 30.
        assert a.stop == 60.
        assert a.delta == 10.

        # both ascending
        a = UniformCoord(30., 60., 10.)
        a.concat(UniformCoord(70., 100, 10.), inplace=True)
        assert a.start == 30.
        assert a.stop == 100.
        assert a.delta == 10.

        a = UniformCoord(30., 60., 10.)
        a.concat(UniformCoord(0., 20, 10.), inplace=True)
        assert a.start == 0.
        assert a.stop == 60.
        assert a.delta == 10.

        # both descending
        b = UniformCoord(60., 30., -10.)
        b.concat(UniformCoord(100., 70, -10.), inplace=True)
        assert b.start == 100.
        assert b.stop == 30.
        assert b.delta == -10.

        b = UniformCoord(60., 30., -10.)
        b.concat(UniformCoord(20., 0, -10.), inplace=True)
        assert b.start == 60.
        assert b.stop == 0.
        assert b.delta == -10.

        # mismatched
        b = UniformCoord(60., 30., -10.)
        b.concat(UniformCoord(70., 100, 10.), inplace=True)
        assert b.start == 100.
        assert b.stop == 30.
        assert b.delta == -10.

        b = UniformCoord(60., 30., -10.)
        b.concat(UniformCoord(0., 20, 10.), inplace=True)
        assert b.start == 60.
        assert b.stop == 0.
        assert b.delta == -10.

        a = UniformCoord(30., 60., 10.)
        a.concat(UniformCoord(100., 70, -10.), inplace=True)
        assert a.start == 30.
        assert a.stop == 100.
        assert a.delta == 10.

        a = UniformCoord(30., 60., 10.)
        a.concat(UniformCoord(20., 0, -10.), inplace=True)
        assert a.start == 0.
        assert a.stop == 60.
        assert a.delta == 10.

        # separated -> ValueError
        a = UniformCoord(30., 60., 10.)
        with pytest.raises(ValueError):
            a.concat(UniformCoord(80., 100, 10.), inplace=True)

        a = UniformCoord(30., 60., 10.)
        with pytest.raises(ValueError):
            a.concat(UniformCoord(0., 10, 10.), inplace=True)

        a = UniformCoord(30., 60., 10.)
        with pytest.raises(ValueError):
            b.concat(UniformCoord(100., 80., -10.), inplace=True)

        a = UniformCoord(30., 60., 10.)
        with pytest.raises(ValueError):
            b.concat(UniformCoord(10., 0, -10.), inplace=True)

        a = UniformCoord(30., 60., 10.)
        with pytest.raises(ValueError):
            b.concat(UniformCoord(80., 100, 10.), inplace=True)

        a = UniformCoord(30., 60., 10.)
        with pytest.raises(ValueError):
            b.concat(UniformCoord(0., 10, 10.), inplace=True)

        a = UniformCoord(30., 60., 10.)
        with pytest.raises(ValueError):
            a.concat(UniformCoord(100., 80., -10.), inplace=True)

        a = UniformCoord(30., 60., 10.)
        with pytest.raises(ValueError):
            a.concat(UniformCoord(10., 0, -10.), inplace=True)

        # mismatched delta -> ValueError
        a = UniformCoord(30., 60., 10.)
        with pytest.raises(ValueError):
            a.concat(UniformCoord(70., 100, 5.), inplace=True)

        # not aligned -> ValueError
        a = UniformCoord(30., 60., 10.)
        with pytest.raises(ValueError):
            a.concat(UniformCoord(65., 100, 10.), inplace=True)

        # overlap -> ValueError
        a = UniformCoord(30., 60., 10.)
        with pytest.raises(ValueError):
            a.concat(UniformCoord(50., 100, 10), inplace=True)

        # non UniformCoord other -> TypeError
        a = UniformCoord(30., 60., 10.)
        with pytest.raises(TypeError):
            a.concat(MonotonicCoord([75, 80, 90]), inplace=True)

        # Coord other
        a = UniformCoord(30., 60., 10.)
        with pytest.raises(TypeError):
            a.concat(Coord([75, 90, 80]), inplace=True)

    def test_add(self):
        # numerical
        c = UniformCoord(20., 60., 10.)
        c2 = c.add(1)
        assert isinstance(c2, UniformCoord)
        assert c2.start == 21.
        assert c2.stop == 61.
        assert c2.delta == 10.

        # simple datetime
        t = UniformCoord('2018-01-01', '2018-01-10', '1,D')
        t2d = t.add('2,D')
        assert isinstance(t2d, UniformCoord)
        assert t2d.start == np.datetime64('2018-01-03')
        assert t2d.stop == np.datetime64('2018-01-12')
        assert t2d.delta == np.timedelta64(1, 'D')

        t2d = t.add('-2,D')
        assert isinstance(t2d, UniformCoord)
        assert t2d.start == np.datetime64('2017-12-30')
        assert t2d.stop == np.datetime64('2018-01-08')
        assert t2d.delta == np.timedelta64(1, 'D')

        # nominal datetime
        t2m = t.add('2,M')
        assert isinstance(t2m, UniformCoord)
        assert t2m.start == np.datetime64('2018-03-01')
        assert t2m.stop == np.datetime64('2018-03-10')
        assert t2m.delta == np.timedelta64(1, 'D')

        t2y = t.add('2,Y')
        assert isinstance(t2y, UniformCoord)
        assert t2y.start == np.datetime64('2020-01-01')
        assert t2y.stop == np.datetime64('2020-01-10')
        assert t2y.delta == np.timedelta64(1, 'D')

    def test_add_equal(self):
        # numerical
        c = UniformCoord(20., 60., 10.)
        c.add(1, inplace=True)
        assert c.start == 21.
        assert c.stop == 61.
        assert c.delta == 10.

        # simple datetime
        t = UniformCoord('2018-01-01', '2018-01-10', '1,D')
        t.add('2,D', inplace=True)
        assert t.start == np.datetime64('2018-01-03')
        assert t.stop == np.datetime64('2018-01-12')
        assert t.delta == np.timedelta64(1, 'D')

        t = UniformCoord('2018-01-01', '2018-01-10', '1,D')
        t.add('-2,D', inplace=True)
        assert t.start == np.datetime64('2017-12-30')
        assert t.stop == np.datetime64('2018-01-08')
        assert t.delta == np.timedelta64(1, 'D')

        # nominal datetime
        t = UniformCoord('2018-01-01', '2018-01-10', '1,D')
        t.add('2,M', inplace=True)
        assert t.start == np.datetime64('2018-03-01')
        assert t.stop == np.datetime64('2018-03-10')
        assert t.delta == np.timedelta64(1, 'D')

        t = UniformCoord('2018-01-01', '2018-01-10', '1,D')
        t.add('2,Y', inplace=True)
        assert t.start == np.datetime64('2020-01-01')
        assert t.stop == np.datetime64('2020-01-10')
        assert t.delta == np.timedelta64(1, 'D')


# testing standalone functions without a test class

def test_coord_linspace_numerical():
    # ascending
    c = coord_linspace(0., 10., 20)
    assert isinstance(c, UniformCoord)
    assert c.start == 0.
    assert c.stop == 10.
    assert c.size == 20
    assert_equal(c.bounds, [0., 10.])
    assert c.is_descending == False
    assert c.is_datetime == False

    # descending
    c = coord_linspace(10., 0., 20)
    assert isinstance(c, UniformCoord)
    assert c.start == 10.
    assert c.stop == 0.
    assert c.size == 20
    assert_equal(c.bounds, [0., 10.])
    assert c.is_descending == True
    assert c.is_datetime == False

@pytest.mark.skipif(pytest.config.getoption("ci"), reason="spec uncertain")
def test_coord_linspace_datetime():
    # ascending
    c = coord_linspace('2018-01-01', '2018-01-10', 10)
    assert isinstance(c, UniformCoord)
    assert c.start == np.datetime64('2018-01-01')
    assert c.stop == np.datetime64('2018-01-10')
    assert c.size == 20
    assert_equal(c.bounds, [np.datetime64('2018-01-01'), np.datetime64('2018-01-10')])
    assert c.is_descending == False
    assert c.is_datetime == True

    # descending
    c = coord_linspace('2018-01-10', '2018-01-01', 10)
    assert isinstance(c, UniformCoord)
    assert c.start == np.datetime64('2018-01-10')
    assert c.stop == np.datetime64('2018-01-01')
    assert c.size == 20
    assert_equal(c.bounds, [np.datetime64('2018-01-01'), np.datetime64('2018-01-10')])
    assert c.is_descending == True
    assert c.is_datetime == True

    # not exact
    c = coord_linspace('2018-01-01', '2018-01-10', 20)
    assert isinstance(c, UniformCoord)
    assert c.start == np.datetime64('2018-01-01')
    assert c.stop == np.datetime64('2018-01-10')
    assert c.size == 20
    assert_equal(c.bounds, [np.datetime64('2018-01-01'), np.datetime64('2018-01-10')])
    assert c.is_descending == False
    assert c.is_datetime == True

def test_coord_linspace_invalid():
    with pytest.raises(TypeError):
        coord_linspace(0., 10., 20.)
    
    with pytest.raises(TypeError):
        coord_linspace(0., 10., '')

    with pytest.raises(TypeError):
        coord_linspace('2018-01-10', '2018-01-01', '1,D')
    
def test_coord_linspace_floating_point_error():
    c = coord_linspace(50.619, 50.62795, 30)
    assert c.size == 30