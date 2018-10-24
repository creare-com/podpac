
from __future__ import division, unicode_literals, print_function, absolute_import

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from podpac.core.coordinates.utils import get_timedelta, get_timedelta_unit, make_timedelta_string
from podpac.core.coordinates.utils import make_coord_value, make_coord_delta, make_coord_array
from podpac.core.coordinates.utils import add_coord

def test_get_timedelta():
    td64 = np.timedelta64
    assert get_timedelta('2,ms') == td64(2, 'ms')
    assert get_timedelta('2,s') == td64(2, 's')
    assert get_timedelta('2,Y') == td64(2, 'Y')
    assert get_timedelta('-1,s') == td64(-1, 's')

    with pytest.raises(ValueError):
        get_timedelta('1.5,s')

    with pytest.raises(ValueError):
        get_timedelta('1')

    with pytest.raises(TypeError):
        get_timedelta('1,x')

def test_get_timedelta_unit():
    td64 = np.timedelta64
    assert get_timedelta_unit(td64(2, 'ms')) == 'ms'
    assert get_timedelta_unit(td64(2, 's')) == 's'
    assert get_timedelta_unit(td64(2, 'Y')) == 'Y'
    
    with pytest.raises(TypeError):
        get_timedelta_unit('a string')

    with pytest.raises(TypeError):
        get_timedelta_unit(np.array([1, 2]))

def test_make_timedelta_string():
    td64 = np.timedelta64
    assert make_timedelta_string(td64(2, 'ms')) == '2,ms'
    assert make_timedelta_string(td64(2, 's')) == '2,s'
    assert make_timedelta_string(td64(2, 'Y')) == '2,Y'
    assert make_timedelta_string(td64(-1, 's')) == '-1,s'
    
    with pytest.raises(TypeError):
        assert make_timedelta_string(1)

def test_make_coord_value():
    # numbers
    assert make_coord_value(10.5) == 10.5
    assert make_coord_value(10) == 10.0
    assert make_coord_value(np.array(10.5)) == 10.5
    assert make_coord_value(np.array([10.5])) == 10.5

    assert type(make_coord_value(10.5)) is float
    assert type(make_coord_value(10)) is float
    assert type(make_coord_value(np.array(10.5))) is float
    assert type(make_coord_value(np.array([10.5]))) is float

    # datetimes
    dt = np.datetime64('2018-01-01')
    assert make_coord_value(dt) == dt
    assert make_coord_value(dt.item()) == dt
    assert make_coord_value('2018-01-01') == dt
    assert make_coord_value(u'2018-01-01') == dt
    assert make_coord_value(np.array(dt)) == dt
    assert make_coord_value(np.array([dt])) == dt
    assert make_coord_value(np.array('2018-01-01')) == dt
    assert make_coord_value(np.array(['2018-01-01'])) == dt

    # arrays and lists 
    with pytest.raises(ValueError):
        make_coord_value(np.arange(5))
    
    with pytest.raises(ValueError):
        make_coord_value(range(5))

    # invalid strings
    with pytest.raises(ValueError):
        make_coord_value('not a valid datetime')

def test_make_coord_delta():
    # numbers
    assert make_coord_delta(10.5) == 10.5
    assert make_coord_delta(10) == 10.0
    assert make_coord_delta(np.array(10.5)) == 10.5
    assert make_coord_delta(np.array([10.5])) == 10.5

    assert type(make_coord_delta(10.5)) is float
    assert type(make_coord_delta(10)) is float
    assert type(make_coord_delta(np.array(10.5))) is float
    assert type(make_coord_delta(np.array([10.5]))) is float

    # timedelta
    td = np.timedelta64(2, 'D')
    assert make_coord_delta(td) == td
    assert make_coord_delta(td.item()) == td
    assert make_coord_delta('2,D') == td
    assert make_coord_delta(u'2,D') == td
    assert make_coord_delta(np.array(td)) == td
    assert make_coord_delta(np.array([td])) == td
    assert make_coord_delta(np.array('2,D')) == td
    assert make_coord_delta(np.array(['2,D'])) == td

    # arrays and lists 
    with pytest.raises(TypeError):
        make_coord_delta(np.arange(5))
    
    with pytest.raises(TypeError):
        make_coord_delta(range(5))

    # invalid strings
    with pytest.raises(ValueError):
        make_coord_delta('not a valid timedelta')

class TestMakeCoordArray(object):
    def test_numerical_singleton(self):
        a = np.array([5.0])
        f = 5.0
        i = 5

        # float
        np.testing.assert_array_equal(make_coord_array(f), a)
        np.testing.assert_array_equal(make_coord_array([f]), a)
        np.testing.assert_array_equal(make_coord_array([[f]]), a)

        # float array
        np.testing.assert_array_equal(make_coord_array(np.array(f)), a)
        np.testing.assert_array_equal(make_coord_array(np.array([f])), a)
        np.testing.assert_array_equal(make_coord_array(np.array([[f]])), a)
        
        # int
        np.testing.assert_array_equal(make_coord_array(i), a)
        np.testing.assert_array_equal(make_coord_array([i]), a)
        np.testing.assert_array_equal(make_coord_array([[i]]), a)

        # int array
        np.testing.assert_array_equal(make_coord_array(np.array(i)), a)
        np.testing.assert_array_equal(make_coord_array(np.array([i])), a)
        np.testing.assert_array_equal(make_coord_array(np.array([[i]])), a)
    
    def test_numerical_array(self):
        a = np.array([5.0, 5.5])
        l = [5, 5.5]

        np.testing.assert_array_equal(make_coord_array(l), a)
        np.testing.assert_array_equal(make_coord_array([l]), a)
        np.testing.assert_array_equal(make_coord_array(np.array(l)), a)
        np.testing.assert_array_equal(make_coord_array(np.array([l])), a)

    def test_date_singleton(self):
        a = np.array(['2018-01-01']).astype(np.datetime64)
        s = '2018-01-01'
        u = u'2018-01-01'
        dt64 = np.datetime64('2018-01-01')
        dt = np.datetime64('2018-01-01').item()

        # str
        np.testing.assert_array_equal(make_coord_array(s), a)
        np.testing.assert_array_equal(make_coord_array([s]), a)
        np.testing.assert_array_equal(make_coord_array([[s]]), a)

        # unicode
        np.testing.assert_array_equal(make_coord_array(u), a)
        np.testing.assert_array_equal(make_coord_array([u]), a)
        np.testing.assert_array_equal(make_coord_array([[u]]), a)
        
        # datetime64
        np.testing.assert_array_equal(make_coord_array(dt64), a)
        np.testing.assert_array_equal(make_coord_array([dt64]), a)
        np.testing.assert_array_equal(make_coord_array([[dt64]]), a)
        
        # python Datetime
        np.testing.assert_array_equal(make_coord_array(dt), a)
        np.testing.assert_array_equal(make_coord_array([dt]), a)
        np.testing.assert_array_equal(make_coord_array([[dt]]), a)

        # pandas Timestamp
        # not tested here because these always have h:m:s

    def test_datetime_singleton(self):
        a = np.array(['2018-01-01T01:01:01']).astype(np.datetime64)
        s = '2018-01-01T01:01:01'
        u = u'2018-01-01T01:01:01'
        dt64 = np.datetime64('2018-01-01T01:01:01')
        dt = np.datetime64('2018-01-01T01:01:01').item()
        ts = pd.Timestamp('2018-01-01T01:01:01')

        # str
        np.testing.assert_array_equal(make_coord_array(s), a)
        np.testing.assert_array_equal(make_coord_array([s]), a)
        np.testing.assert_array_equal(make_coord_array([[s]]), a)

        # unicode
        np.testing.assert_array_equal(make_coord_array(u), a)
        np.testing.assert_array_equal(make_coord_array([u]), a)
        np.testing.assert_array_equal(make_coord_array([[u]]), a)
        
        # datetime64
        np.testing.assert_array_equal(make_coord_array(dt64), a)
        np.testing.assert_array_equal(make_coord_array([dt64]), a)
        np.testing.assert_array_equal(make_coord_array([[dt64]]), a)
        
        # python Datetime
        np.testing.assert_array_equal(make_coord_array(dt), a)
        np.testing.assert_array_equal(make_coord_array([dt]), a)
        np.testing.assert_array_equal(make_coord_array([[dt]]), a)

        # pandas Timestamp
        np.testing.assert_array_equal(make_coord_array(ts), a)
        np.testing.assert_array_equal(make_coord_array([ts]), a)
        np.testing.assert_array_equal(make_coord_array([[ts]]), a)

    def test_date_array(self):
        a = np.array(['2018-01-01', '2018-01-02']).astype(np.datetime64)
        s = ['2018-01-01', '2018-01-02']
        u = [u'2018-01-01', u'2018-01-02']
        dt64 = [np.datetime64('2018-01-01'), np.datetime64('2018-01-02')]
        dt = [np.datetime64('2018-01-01').item(), np.datetime64('2018-01-02').item()]

        # str
        np.testing.assert_array_equal(make_coord_array(s), a)
        np.testing.assert_array_equal(make_coord_array([s]), a)
        np.testing.assert_array_equal(make_coord_array(np.array(s)), a)
        np.testing.assert_array_equal(make_coord_array(np.array([s])), a)

        # unicode
        np.testing.assert_array_equal(make_coord_array(u), a)
        np.testing.assert_array_equal(make_coord_array([u]), a)
        np.testing.assert_array_equal(make_coord_array(np.array(u)), a)
        np.testing.assert_array_equal(make_coord_array(np.array([u])), a)

        # datetime64
        np.testing.assert_array_equal(make_coord_array(dt64), a)
        np.testing.assert_array_equal(make_coord_array([dt64]), a)
        np.testing.assert_array_equal(make_coord_array(np.array(dt64)), a)
        np.testing.assert_array_equal(make_coord_array(np.array([dt64])), a)

        # python datetime
        np.testing.assert_array_equal(make_coord_array(dt), a)
        np.testing.assert_array_equal(make_coord_array([dt]), a)
        np.testing.assert_array_equal(make_coord_array(np.array(dt)), a)
        np.testing.assert_array_equal(make_coord_array(np.array([dt])), a)
        
        # pandas Timestamp
        # not tested here because these always have h:m:s

    def test_datetime_array(self):
        a = np.array(['2018-01-01T01:01:01', '2018-01-01T01:01:02']).astype(np.datetime64)
        s = ['2018-01-01T01:01:01', '2018-01-01T01:01:02']
        u = [u'2018-01-01T01:01:01', u'2018-01-01T01:01:02']
        dt64 = [np.datetime64('2018-01-01T01:01:01'), np.datetime64('2018-01-01T01:01:02')]
        dt = [np.datetime64('2018-01-01T01:01:01').item(), np.datetime64('2018-01-01T01:01:02').item()]
        ts = [pd.Timestamp('2018-01-01T01:01:01'), pd.Timestamp('2018-01-01T01:01:02')]

        # str
        np.testing.assert_array_equal(make_coord_array(s), a)
        np.testing.assert_array_equal(make_coord_array([s]), a)
        np.testing.assert_array_equal(make_coord_array(np.array(s)), a)
        np.testing.assert_array_equal(make_coord_array(np.array([s])), a)

        # unicode
        np.testing.assert_array_equal(make_coord_array(u), a)
        np.testing.assert_array_equal(make_coord_array([u]), a)
        np.testing.assert_array_equal(make_coord_array(np.array(u)), a)
        np.testing.assert_array_equal(make_coord_array(np.array([u])), a)

        # datetime64
        np.testing.assert_array_equal(make_coord_array(dt64), a)
        np.testing.assert_array_equal(make_coord_array([dt64]), a)
        np.testing.assert_array_equal(make_coord_array(np.array(dt64)), a)
        np.testing.assert_array_equal(make_coord_array(np.array([dt64])), a)

        # python datetime
        np.testing.assert_array_equal(make_coord_array(dt), a)
        np.testing.assert_array_equal(make_coord_array([dt]), a)
        np.testing.assert_array_equal(make_coord_array(np.array(dt)), a)
        np.testing.assert_array_equal(make_coord_array(np.array([dt])), a)
        
        # pandas Timestamp
        np.testing.assert_array_equal(make_coord_array(ts), a)
        np.testing.assert_array_equal(make_coord_array([ts]), a)
        np.testing.assert_array_equal(make_coord_array([[ts]]), a)

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            make_coord_array([{}])
    
    def test_mixed_type(self):
        with pytest.raises(TypeError):
            make_coord_array([5.0, '2018-01-01'])

        with pytest.raises(TypeError):
            make_coord_array(['2018-01-01', 5.0])

        with pytest.raises(TypeError):
            make_coord_array([5.0, np.datetime64('2018-01-01')])

        with pytest.raises(TypeError):
            make_coord_array([np.datetime64('2018-01-01'), 5.0])
        
    def test_invalid_time_string(self):
        with pytest.raises(TypeError):
            make_coord_array(['invalid'])

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            make_coord_array([[0, 1], [5, 6]])

        with pytest.raises(ValueError):
            make_coord_array(np.array([[0, 1], [5, 6]]))

def test_add_coord():
    # numbers
    assert add_coord(5, 1) == 6
    assert add_coord(5, -1) == 4
    assert np.allclose(add_coord(5, np.array([-1, 1])), [4, 6])

    # simple timedeltas
    td64 = np.timedelta64
    dt64 = np.datetime64
    assert add_coord(dt64('2018-01-30'), td64( 1, 'D')) == dt64('2018-01-31')
    assert add_coord(dt64('2018-01-30'), td64( 2, 'D')) == dt64('2018-02-01')
    assert add_coord(dt64('2018-01-30'), td64(-1, 'D')) == dt64('2018-01-29')
    assert add_coord(dt64('2018-01-01'), td64(-1, 'D')) == dt64('2017-12-31')
    assert np.all(
        add_coord(dt64('2018-01-30'), np.array([td64(1, 'D'), td64(2, 'D')])) ==
        np.array([dt64('2018-01-31'), dt64('2018-02-01')]))

    # year timedeltas
    assert add_coord(dt64('2018-01-01'), td64( 1, 'Y')) == dt64('2019-01-01')
    assert add_coord(dt64('2018-01-01'), td64(-1, 'Y')) == dt64('2017-01-01')
    assert add_coord(dt64('2020-02-29'), td64( 1, 'Y')) == dt64('2021-02-28')

    # month timedeltas
    assert add_coord(dt64('2018-01-01'), td64( 1, 'M')) == dt64('2018-02-01')
    assert add_coord(dt64('2018-01-01'), td64(-1, 'M')) == dt64('2017-12-01')
    assert add_coord(dt64('2018-01-01'), td64(24, 'M')) == dt64('2020-01-01')
    assert add_coord(dt64('2018-01-31'), td64( 1, 'M')) == dt64('2018-02-28')
    assert add_coord(dt64('2018-01-31'), td64( 2, 'M')) == dt64('2018-03-31')
    assert add_coord(dt64('2018-01-31'), td64( 3, 'M')) == dt64('2018-04-30')
    assert add_coord(dt64('2020-01-31'), td64( 1, 'M')) == dt64('2020-02-29')

    # type error
    with pytest.raises(TypeError):
        add_coord(25.0, dt64('2020-01-31'))

    # this base case is generally not encountered
    from podpac.core.coordinates.utils import _add_nominal_timedelta
    assert _add_nominal_timedelta(dt64('2018-01-30'), td64( 1, 'D')) == dt64('2018-01-31')
