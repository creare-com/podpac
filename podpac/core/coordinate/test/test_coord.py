
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
        assert c.is_datetime == False
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

    def test___sub__(self):
        pass

    def test___add__(self):
        pass

    def test___iadd__(self):
        pass

    def test___repr__(self):
        pass

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
        assert c.is_datetime == False
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
        assert s.size == 0

        # partial, backwards bounds
        s = c.select([70, 30], pad=0)
        assert isinstance(s, Coord)
        assert s.size == 0

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
        assert s.size == 0

        # partial, backwards bounds
        s = c.select([70, 30], pad=0)
        assert isinstance(s, Coord)
        assert s.size == 0

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

class TestUniformCoord(object):
    pass
 
class TestCoordLinspace(object):
    def test_floating_point_error(self):
        c = coord_linspace(50.619, 50.62795, 30)
        assert(c.size == 30)