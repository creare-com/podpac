
from datetime import datetime

import pytest
import traitlets as tl
import numpy as np
from numpy.testing import assert_equal

from podpac.core.units import Units
from podpac.core.coordinate.util import get_timedelta_unit
from podpac.core.coordinate import Coord, MonotonicCoord, UniformCoord
from podpac.core.coordinate import coord_linspace

class TestCoord(object):
    def test_empty(self):
        c = Coord()
        a = np.array([])

        assert_equal(c.coords, a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([np.nan, np.nan]))
        assert_equal(c.area_bounds, np.array([np.nan, np.nan]))
        assert c.size == 0
        assert c.is_datetime is False
        assert c.is_monotonic is False
        assert c.is_descending is None
        assert c.rasterio_regularity is False
        assert c.scipy_regularity is True

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
        assert_equal(c.area_bounds, np.array([-3.0, 100.0]))
        assert np.issubdtype(c.coords.dtype, np.float)
        assert np.issubdtype(c.coordinates.dtype, np.float)
        assert np.issubdtype(c.bounds.dtype, np.float)
        assert np.issubdtype(c.area_bounds.dtype, np.float)
        assert c.size == 4
        assert c.is_datetime is False
        assert c.is_monotonic is False
        assert c.is_descending is None
        assert c.rasterio_regularity is False
        assert c.scipy_regularity is True

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
        assert_equal(c.area_bounds, np.array([value, value]))
        assert np.issubdtype(c.coords.dtype, np.float)
        assert np.issubdtype(c.coordinates.dtype, np.float)
        assert np.issubdtype(c.bounds.dtype, np.float)
        assert np.issubdtype(c.area_bounds.dtype, np.float)
        assert c.size == 1
        assert c.is_datetime is False
        assert c.is_monotonic is False
        assert c.is_descending is None
        assert c.rasterio_regularity is True
        assert c.scipy_regularity is True

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
        assert_equal(c.area_bounds, np.array(['2017-01-01', '2019-01-01']).astype(np.datetime64))
        assert np.issubdtype(c.coords.dtype, np.datetime64)
        assert np.issubdtype(c.coordinates.dtype, np.datetime64)
        assert np.issubdtype(c.bounds.dtype, np.datetime64)
        assert np.issubdtype(c.area_bounds.dtype, np.datetime64)
        assert c.size == 4
        assert c.is_datetime is True
        assert c.is_monotonic is False
        assert c.is_descending is None
        assert c.rasterio_regularity is False
        assert c.scipy_regularity is True

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
        assert_equal(c.area_bounds, np.array([value_dt64, value_dt64]))
        assert np.issubdtype(c.coords.dtype, np.datetime64)
        assert np.issubdtype(c.coordinates.dtype, np.datetime64)
        assert np.issubdtype(c.bounds.dtype, np.datetime64)
        assert np.issubdtype(c.area_bounds.dtype, np.datetime64)
        assert c.size == 1
        assert c.is_datetime is True
        assert c.is_monotonic is False
        assert c.is_descending is None
        assert c.rasterio_regularity is True
        assert c.scipy_regularity is True

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

    def test_delta(self):
        # TODO

        # empty
        assert np.isnan(Coord().delta)

        # single numerical values
        # assert Coord(5.0).delta == # TODO

        # single datetimes
        assert type(Coord(['2018-01-01']).delta) == np.timedelta64
        assert get_timedelta_unit(Coord(['2018-01-01']).delta) == 'D'

        # multiple values
        assert Coord([1.0, 2.0, 3.0]).delta == 1.0
        assert 1.0 < Coord([1.0, 2.0, 4.0]).delta < 2.0

        assert Coord(['2018-01-01', '2018-01-02', '2018-01-03']).delta == np.timedelta64(1, 'D')

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
        
        # points: ignore extents
        assert_equal(Coord(numerical, ctype='point', extents=e).area_bounds, [lo, hi])
        assert_equal(Coord(datetimes, ctype='point', extents=dt_e).area_bounds, [dt_lo, dt_hi])

        # segments: explicit extents
        assert_equal(Coord(numerical, ctype='segment', extents=e).area_bounds, e)
        assert_equal(Coord(datetimes, ctype='segment', extents=dt_e).area_bounds, dt_e)
        
        # segments: calculate from bounds, segment_position, and delta
        # TODO
        # assert_equal(Coord(numerical), ctype='segment').area_bounds, TODO)
        # assert_equal(Coord(numerical), ctype='segment', segment_position=0.8).area_bounds, TODO)
        # assert_equal(Coord(datetimes), ctype='segment').area_bounds, TODO)
        # assert_equal(Coord(datetimes), ctype='segment', segment_position=0.8).area_bounds, TODO)

    def test_intersect(self):
        pass

    def test_select(self):
        pass

    def test___sub__(self):
        pass

    def test___add__(self):
        pass

    def test___iadd__(self):
        pass

    def test___repr__(self):
        pass

class TestMonotonicCoord(object):
    pass

class TestUniformCoord(object):
    pass
 
class TestCoordLinspace(object):
    def test_floating_point_error(self):
        c = coord_linspace(50.619, 50.62795, 30)
        assert(c.size == 30)