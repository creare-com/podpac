
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
    def test_coords(self):
        dt64 = np.datetime64

        # empty
        assert_equal(Coord().coords, np.array([]))
        assert_equal(Coord([]), np.array([]))
        assert_equal(Coord(np.array([])), np.array([]))

        # single numerical values
        assert_equal(Coord(5).coords, np.array([5.0]))
        assert_equal(Coord(5L).coords, np.array([5.0]))
        assert_equal(Coord(5.0).coords, np.array([5.0]))
        assert_equal(Coord([5.0]).coords, np.array([5.0]))
        assert_equal(Coord(np.array(5.0)).coords, np.array([5.0]))
        assert_equal(Coord(np.array([5.0])).coords, np.array([5.0]))
        assert_equal(Coord(np.array([[5.0]])).coords, np.array([5.0]))
        
        # single datetimes
        assert_equal(Coord('2018-01-01').coords, np.array([dt64('2018-01-01')]))
        assert_equal(Coord(['2018-01-01']).coords, np.array([dt64('2018-01-01')]))
        assert_equal(Coord(np.array('2018-01-01')).coords, np.array([dt64('2018-01-01')]))
        assert_equal(Coord(np.array(['2018-01-01'])).coords, np.array([dt64('2018-01-01')]))
        assert_equal(Coord(np.array([['2018-01-01']])).coords, np.array([dt64('2018-01-01')]))

        assert_equal(Coord(dt64('2018-01-01')).coords, np.array([dt64('2018-01-01')]))
        assert_equal(Coord([dt64('2018-01-01')]).coords, np.array([dt64('2018-01-01')]))
        assert_equal(Coord(np.array(dt64('2018-01-01'))).coords, np.array([dt64('2018-01-01')]))
        assert_equal(Coord(np.array([dt64('2018-01-01')])).coords, np.array([dt64('2018-01-01')]))
        assert_equal(Coord(np.array([[dt64('2018-01-01')]])).coords, np.array([dt64('2018-01-01')]))
        
        assert_equal(Coord(datetime.now()).coords, np.array([datetime.now()]).astype(dt64))
        assert_equal(Coord([datetime.now()]).coords, np.array([datetime.now()]).astype(dt64))
        assert_equal(Coord(np.array(datetime.now())).coords, np.array([datetime.now()]).astype(dt64))
        assert_equal(Coord(np.array([datetime.now()])).coords, np.array([datetime.now()]).astype(dt64))
        assert_equal(Coord(np.array([[datetime.now()]])).coords, np.array([datetime.now()]).astype(dt64))

        # multiple values
        assert_equal(Coord([1, 2, 3]).coords, np.array([1, 2, 3]))
        assert_equal(Coord([[1, 2, 3]]).coords, np.array([1, 2, 3]))
        assert_equal(Coord(np.array([1, 2, 3])).coords, np.array([1, 2, 3]))
        assert_equal(Coord(np.array([[1, 2, 3]])).coords, np.array([1, 2, 3]))
        
        assert_equal(Coord(['2018-01-01', '2018-02-01']).coords, np.array(['2018-01-01', '2018-02-01']).astype(dt64))
        assert_equal(Coord([['2018-01-01', '2018-02-01']]).coords, np.array(['2018-01-01', '2018-02-01']).astype(dt64))
        assert_equal(Coord(np.array(['2018-01-01', '2018-02-01'])).coords, np.array(['2018-01-01', '2018-02-01']).astype(dt64))
        assert_equal(Coord(np.array([['2018-01-01', '2018-02-01']])).coords, np.array(['2018-01-01', '2018-02-01']).astype(dt64))
        
        assert_equal(Coord([dt64('2018-01-01'), dt64('2018-02-01')]).coords, np.array(['2018-01-01', '2018-02-01']).astype(dt64))
        assert_equal(Coord([[dt64('2018-01-01'), dt64('2018-02-01')]]).coords, np.array(['2018-01-01', '2018-02-01']).astype(dt64))
        assert_equal(Coord(np.array(['2018-01-01', '2018-02-01']).astype(dt64)).coords, np.array(['2018-01-01', '2018-02-01']).astype(dt64))
        assert_equal(Coord(np.array([['2018-01-01', '2018-02-01']]).astype(dt64)).coords, np.array(['2018-01-01', '2018-02-01']).astype(dt64))
        
        dt1 = datetime.now()
        dt2 = datetime.now()
        assert_equal(Coord([dt1, dt2]).coords, np.array([dt1, dt2]).astype(dt64))
        assert_equal(Coord([[dt1, dt2]]).coords, np.array([dt1, dt2]).astype(dt64))
        assert_equal(Coord(np.array([dt1, dt2])).coords, np.array([dt1, dt2]).astype(dt64))
        assert_equal(Coord(np.array([[dt1, dt2]])).coords, np.array([dt1, dt2]).astype(dt64))

        # invalid
        with pytest.raises(TypeError):
            Coord(dict())
        
        with pytest.raises(TypeError):
            Coord([1, '2018-01-01'])

        with pytest.raises(ValueError):
            Coord('a')

        with pytest.raises(ValueError):
            Coord([[1, 2], [3, 4]])

    def test_units(self):
        # default
        assert Coord(5.0).units is None
        
        # initialize
        # TODO
        # c = Coord(5.0, units=Units())
        # assert isinstance(c.units, Units)

    def test_ctype(self):
        # default
        assert Coord(5.0).ctype == 'segment'
        
        # initialize
        assert Coord(5.0, ctype='segment').ctype == 'segment'
        assert Coord(5.0, ctype='point').ctype == 'point'
        assert Coord(5.0, ctype='fence').ctype == 'fence'
        assert Coord(5.0, ctype='post').ctype == 'post'
        
        # invalid
        with pytest.raises(tl.TraitError):
            Coord(5.0, ctype='abc')

    def test_segment_position(self):
        # default
        assert Coord(5.0).segment_position == 0.5
        
        # initialize
        assert Coord(5.0, segment_position=0.8).segment_position == 0.8

    def test_extents(self):
        # default
        assert len(Coord(5.0).extents) == 0

        # initialize
        c = Coord([1.0, 1.8, 2.0], extents=[1.0, 2.0])
        assert_equal(c.extents, [1.0, 2.0])

    def test_coord_ref_system(self):
        # default
        assert Coord(5.0).coord_ref_sys == u''

        # initialize
        assert Coord(5.0, coord_ref_sys='test').coord_ref_sys == u'test'

        # invalid
        with pytest.raises(tl.TraitError):
            Coord(5.0, coord_ref_sys=1)

    def test_delta(self):
        # empty
        assert np.isnan(Coord().delta)

        # single numerical values
        # assert Coord(5.0).delta == # TODO

        # single datetimes
        assert get_timedelta_unit(Coord(['2018-01-01']).delta) == 'D'
        assert get_timedelta_unit(Coord(['2018-01-01 12:00']).delta) == 'm'

        # multiple values
        assert Coord([1.0, 2.0, 3.0]).delta == 1.0
        assert 1.0 < Coord([1.0, 2.0, 4.0]).delta < 2.0

        assert Coord(['2018-01-01', '2018-01-02', '2018-01-03']).delta == np.timedelta64(1, 'D')
        assert Coord(['2018-01-01 12:00', '2018-01-01 13:00', '2018-01-01 14:00']).delta == np.timedelta64(1, 'h')


    def test_kwargs(self):
        pass

    def test_coordinates(self):
        pass

    def test_area_bounds(self):
        pass

    def test_bounds(self):
        pass

    def test_size(self):
        pass

    def test_is_datetime(self):
        pass

    def test_is_monotonic(self):
        pass

    def test_is_descending(self):
        pass

    def test_rasterio_regularity(self):
        pass

    def test_scipy_regularity(self):
        pass

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
    def test_floating_point_error(self):
        c = coord_linspace(50.619, 50.62795, 30)
        assert(c.size == 30)

class TestUniformCoord(object):
    pass
 
class TestCoordLinspace(object):
    pass