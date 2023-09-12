from __future__ import division, unicode_literals, print_function, absolute_import

from datetime import datetime

import pytest
import numpy as np
import pandas as pd
import pyproj
import traitlets as tl

import podpac

from podpac.core.coordinates.utils import get_timedelta, get_timedelta_unit, make_timedelta_string
from podpac.core.coordinates.utils import make_coord_value, make_coord_delta, make_coord_array, make_coord_delta_array
from podpac.core.coordinates.utils import add_coord, divide_delta, divide_timedelta, timedelta_divisible
from podpac.core.coordinates.utils import has_alt_units, lower_precision_time_bounds, higher_precision_time_bounds
from podpac.core.coordinates.utils import add_valid_dimension

def test_get_timedelta():
    td64 = np.timedelta64
    assert get_timedelta("2,ms") == td64(2, "ms")
    assert get_timedelta("2,s") == td64(2, "s")
    assert get_timedelta("2,Y") == td64(2, "Y")
    assert get_timedelta("-1,s") == td64(-1, "s")

    with pytest.raises(ValueError):
        get_timedelta("1.5,s")

    with pytest.raises(ValueError):
        get_timedelta("1")

    with pytest.raises(TypeError, match="Invalid datetime unit"):
        get_timedelta("1,x")


def test_get_timedelta_unit():
    td64 = np.timedelta64
    assert get_timedelta_unit(td64(2, "ms")) == "ms"
    assert get_timedelta_unit(td64(2, "s")) == "s"
    assert get_timedelta_unit(td64(2, "Y")) == "Y"

    with pytest.raises(TypeError):
        get_timedelta_unit("a string")

    with pytest.raises(TypeError):
        get_timedelta_unit(np.array([1, 2]))


def test_make_timedelta_string():
    td64 = np.timedelta64
    assert make_timedelta_string(td64(2, "ms")) == "2,ms"
    assert make_timedelta_string(td64(2, "s")) == "2,s"
    assert make_timedelta_string(td64(2, "Y")) == "2,Y"
    assert make_timedelta_string(td64(-1, "s")) == "-1,s"

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
    dt = np.datetime64("2018-01-01")
    assert make_coord_value(dt) == dt
    assert make_coord_value(dt.item()) == dt
    assert make_coord_value("2018-01-01") == dt
    assert make_coord_value("2018-01-01") == dt
    assert make_coord_value(np.array(dt)) == dt
    assert make_coord_value(np.array([dt])) == dt
    assert make_coord_value(np.array("2018-01-01")) == dt
    assert make_coord_value(np.array(["2018-01-01"])) == dt

    # arrays and lists
    with pytest.raises(TypeError, match="Invalid coordinate value"):
        make_coord_value(np.arange(5))

    with pytest.raises(TypeError, match="Invalid coordinate value"):
        make_coord_value(range(5))

    # invalid strings
    with pytest.raises(ValueError, match="Error parsing datetime string"):
        make_coord_value("not a valid datetime")


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
    td = np.timedelta64(2, "D")
    assert make_coord_delta(td) == td
    assert make_coord_delta(td.item()) == td
    assert make_coord_delta("2,D") == td
    assert make_coord_delta("2,D") == td
    assert make_coord_delta(np.array(td)) == td
    assert make_coord_delta(np.array([td])) == td
    assert make_coord_delta(np.array("2,D")) == td
    assert make_coord_delta(np.array(["2,D"])) == td

    # arrays and lists
    with pytest.raises(TypeError, match="Invalid coordinate delta"):
        make_coord_delta(np.arange(5))

    with pytest.raises(TypeError, match="Invalid coordinate delta"):
        make_coord_delta(range(5))

    # invalid strings
    with pytest.raises(ValueError):
        make_coord_delta("not a valid timedelta")


class TestMakeCoordArray(object):
    def test_numerical_singleton(self):
        a = np.array([5.0])
        f = 5.0
        i = 5

        # float
        np.testing.assert_array_equal(make_coord_array(f), a)
        np.testing.assert_array_equal(make_coord_array([f]), a)

        # float array
        np.testing.assert_array_equal(make_coord_array(np.array(f)), a)
        np.testing.assert_array_equal(make_coord_array(np.array([f])), a)

        # int
        np.testing.assert_array_equal(make_coord_array(i), a)
        np.testing.assert_array_equal(make_coord_array([i]), a)

        # int array
        np.testing.assert_array_equal(make_coord_array(np.array(i)), a)
        np.testing.assert_array_equal(make_coord_array(np.array([i])), a)

    def test_numerical_array(self):
        a = np.array([5.0, 5.5])
        l = [5, 5.5]

        np.testing.assert_array_equal(make_coord_array(l), a)
        np.testing.assert_array_equal(make_coord_array(np.array(l)), a)

    def test_numerical_ndarray(self):
        a = [[0, 1], [5, 6]]
        np.testing.assert_array_equal(make_coord_array(a), a)
        np.testing.assert_array_equal(make_coord_array(np.array(a)), a)

    def test_date_singleton(self):
        a = np.array(["2018-01-01"]).astype(np.datetime64)
        s = "2018-01-01"
        u = "2018-01-01"
        dt64 = np.datetime64("2018-01-01")
        dt = np.datetime64("2018-01-01").item()

        # str
        np.testing.assert_array_equal(make_coord_array(s), a)
        np.testing.assert_array_equal(make_coord_array([s]), a)

        # unicode
        np.testing.assert_array_equal(make_coord_array(u), a)
        np.testing.assert_array_equal(make_coord_array([u]), a)

        # datetime64
        np.testing.assert_array_equal(make_coord_array(dt64), a)
        np.testing.assert_array_equal(make_coord_array([dt64]), a)

        # python Datetime
        np.testing.assert_array_equal(make_coord_array(dt), a)
        np.testing.assert_array_equal(make_coord_array([dt]), a)

        # pandas Timestamp
        # not tested here because these always have h:m:s

    def test_datetime_singleton(self):
        a = np.array(["2018-01-01T01:01:01"]).astype(np.datetime64)
        s = "2018-01-01T01:01:01"
        u = "2018-01-01T01:01:01"
        dt64 = np.datetime64("2018-01-01T01:01:01")
        dt = np.datetime64("2018-01-01T01:01:01").item()
        ts = pd.Timestamp("2018-01-01T01:01:01")

        # str
        np.testing.assert_array_equal(make_coord_array(s), a)
        np.testing.assert_array_equal(make_coord_array([s]), a)

        # unicode
        np.testing.assert_array_equal(make_coord_array(u), a)
        np.testing.assert_array_equal(make_coord_array([u]), a)

        # datetime64
        np.testing.assert_array_equal(make_coord_array(dt64), a)
        np.testing.assert_array_equal(make_coord_array([dt64]), a)

        # python Datetime
        np.testing.assert_array_equal(make_coord_array(dt), a)
        np.testing.assert_array_equal(make_coord_array([dt]), a)

        # pandas Timestamp
        np.testing.assert_array_equal(make_coord_array(ts), a)
        np.testing.assert_array_equal(make_coord_array([ts]), a)

    def test_date_array(self):
        a = np.array(["2018-01-01", "2018-01-02"]).astype(np.datetime64)
        s = ["2018-01-01", "2018-01-02"]
        u = ["2018-01-01", "2018-01-02"]
        dt64 = [np.datetime64("2018-01-01"), np.datetime64("2018-01-02")]
        dt = [np.datetime64("2018-01-01").item(), np.datetime64("2018-01-02").item()]

        # str
        np.testing.assert_array_equal(make_coord_array(s), a)
        np.testing.assert_array_equal(make_coord_array(np.array(s)), a)

        # unicode
        np.testing.assert_array_equal(make_coord_array(u), a)
        np.testing.assert_array_equal(make_coord_array(np.array(u)), a)

        # datetime64
        np.testing.assert_array_equal(make_coord_array(dt64), a)
        np.testing.assert_array_equal(make_coord_array(np.array(dt64)), a)

        # python datetime
        np.testing.assert_array_equal(make_coord_array(dt), a)
        np.testing.assert_array_equal(make_coord_array(np.array(dt)), a)

        # pandas Timestamp
        # not tested here because these always have h:m:s

    def test_datetime_array(self):
        a = np.array(["2018-01-01T01:01:01", "2018-01-01T01:01:02"]).astype(np.datetime64)
        s = ["2018-01-01T01:01:01", "2018-01-01T01:01:02"]
        u = ["2018-01-01T01:01:01", "2018-01-01T01:01:02"]
        dt64 = [np.datetime64("2018-01-01T01:01:01"), np.datetime64("2018-01-01T01:01:02")]
        dt = [np.datetime64("2018-01-01T01:01:01").item(), np.datetime64("2018-01-01T01:01:02").item()]
        ts = [pd.Timestamp("2018-01-01T01:01:01"), pd.Timestamp("2018-01-01T01:01:02")]

        # str
        np.testing.assert_array_equal(make_coord_array(s), a)
        np.testing.assert_array_equal(make_coord_array(np.array(s)), a)

        # unicode
        np.testing.assert_array_equal(make_coord_array(u), a)
        np.testing.assert_array_equal(make_coord_array(np.array(u)), a)

        # datetime64
        np.testing.assert_array_equal(make_coord_array(dt64), a)
        np.testing.assert_array_equal(make_coord_array(np.array(dt64)), a)

        # python datetime
        np.testing.assert_array_equal(make_coord_array(dt), a)
        np.testing.assert_array_equal(make_coord_array(np.array(dt)), a)

        # pandas Timestamp
        np.testing.assert_array_equal(make_coord_array(ts), a)

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            make_coord_array([{}])

    def test_mixed_type(self):
        with pytest.raises(ValueError, match="Invalid coordinate values"):
            make_coord_array([5.0, "2018-01-01"])

        with pytest.raises(ValueError, match="Invalid coordinate values"):
            make_coord_array(["2018-01-01", 5.0])

        with pytest.raises(ValueError, match="Invalid coordinate values"):
            make_coord_array([5.0, np.datetime64("2018-01-01")])

        with pytest.raises(ValueError, match="Invalid coordinate values"):
            make_coord_array([np.datetime64("2018-01-01"), 5.0])

    def test_invalid_time_string(self):
        with pytest.raises(ValueError, match="Error parsing datetime string"):
            make_coord_array(["invalid"])


class TestMakeCoordDeltaArray(object):
    def test_numerical_singleton(self):
        a = np.array([5.0])
        f = 5.0
        i = 5

        # float
        np.testing.assert_array_equal(make_coord_delta_array(f), a)
        np.testing.assert_array_equal(make_coord_delta_array([f]), a)

        # float array
        np.testing.assert_array_equal(make_coord_delta_array(np.array(f)), a)
        np.testing.assert_array_equal(make_coord_delta_array(np.array([f])), a)

        # int
        np.testing.assert_array_equal(make_coord_delta_array(i), a)
        np.testing.assert_array_equal(make_coord_delta_array([i]), a)

        # int array
        np.testing.assert_array_equal(make_coord_delta_array(np.array(i)), a)
        np.testing.assert_array_equal(make_coord_delta_array(np.array([i])), a)

    def test_numerical_array(self):
        a = np.array([5.0, 5.5])
        l = [5, 5.5]

        np.testing.assert_array_equal(make_coord_delta_array(l), a)
        np.testing.assert_array_equal(make_coord_delta_array(np.array(l)), a)

    def test_timedelta_singleton(self):
        a = np.array([np.timedelta64(1, "D")])
        s = "1,D"
        u = "1,D"
        td64 = np.timedelta64(1, "D")
        td = np.timedelta64(1, "D").item()

        # str
        np.testing.assert_array_equal(make_coord_delta_array(s), a)
        np.testing.assert_array_equal(make_coord_delta_array([s]), a)

        # unicode
        np.testing.assert_array_equal(make_coord_delta_array(u), a)
        np.testing.assert_array_equal(make_coord_delta_array([u]), a)

        # timedelta64
        np.testing.assert_array_equal(make_coord_delta_array(td64), a)
        np.testing.assert_array_equal(make_coord_delta_array([td64]), a)

        # python timedelta
        np.testing.assert_array_equal(make_coord_delta_array(td), a)
        np.testing.assert_array_equal(make_coord_delta_array([td]), a)

    def test_date_array(self):
        a = np.array([np.timedelta64(1, "D"), np.timedelta64(2, "D")])
        s = ["1,D", "2,D"]
        u = ["1,D", "2,D"]
        td64 = [np.timedelta64(1, "D"), np.timedelta64(2, "D")]
        td = [np.timedelta64(1, "D").item(), np.timedelta64(2, "D").item()]

        # str
        np.testing.assert_array_equal(make_coord_delta_array(s), a)
        np.testing.assert_array_equal(make_coord_delta_array(np.array(s)), a)

        # unicode
        np.testing.assert_array_equal(make_coord_delta_array(u), a)
        np.testing.assert_array_equal(make_coord_delta_array(np.array(u)), a)

        # timedelta64
        np.testing.assert_array_equal(make_coord_delta_array(td64), a)
        np.testing.assert_array_equal(make_coord_delta_array(np.array(td64)), a)

        # python timedelta
        np.testing.assert_array_equal(make_coord_delta_array(td), a)
        np.testing.assert_array_equal(make_coord_delta_array(np.array(td)), a)

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            make_coord_delta_array([{}])

    def test_mixed_type(self):
        with pytest.raises(ValueError):
            make_coord_delta_array([5.0, "1,D"])

        with pytest.raises(ValueError):
            make_coord_delta_array(["1,D", 5.0])

        with pytest.raises(ValueError):
            make_coord_delta_array([5.0, np.timedelta64(1, "D")])

        with pytest.raises(ValueError):
            make_coord_delta_array([np.timedelta64(1, "D"), 5.0])

    def test_invalid_time_string(self):
        with pytest.raises(ValueError):
            make_coord_delta_array(["invalid"])

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            make_coord_delta_array([[0, 1], [5, 6]])

        with pytest.raises(ValueError):
            make_coord_delta_array(np.array([[0, 1], [5, 6]]))


def test_add_coord():
    # numbers
    assert add_coord(5, 1) == 6
    assert add_coord(5, -1) == 4
    assert np.allclose(add_coord(5, np.array([-1, 1])), [4, 6])

    # simple timedeltas
    td64 = np.timedelta64
    dt64 = np.datetime64
    assert add_coord(dt64("2018-01-30"), td64(1, "D")) == dt64("2018-01-31")
    assert add_coord(dt64("2018-01-30"), td64(2, "D")) == dt64("2018-02-01")
    assert add_coord(dt64("2018-01-30"), td64(-1, "D")) == dt64("2018-01-29")
    assert add_coord(dt64("2018-01-01"), td64(-1, "D")) == dt64("2017-12-31")
    assert np.all(
        add_coord(dt64("2018-01-30"), np.array([td64(1, "D"), td64(2, "D")]))
        == np.array([dt64("2018-01-31"), dt64("2018-02-01")])
    )

    # year timedeltas
    assert add_coord(dt64("2018-01-01"), td64(1, "Y")) == dt64("2019-01-01")
    assert add_coord(dt64("2018-01-01T00:00:00.0000000"), td64(1, "Y")) == dt64("2019-01-01")
    assert add_coord(dt64("2018-01-01"), td64(-1, "Y")) == dt64("2017-01-01")
    assert add_coord(dt64("2020-02-29"), td64(1, "Y")) == dt64("2021-02-28")

    # month timedeltas
    assert add_coord(dt64("2018-01-01"), td64(1, "M")) == dt64("2018-02-01")
    assert add_coord(dt64("2018-01-01"), td64(-1, "M")) == dt64("2017-12-01")
    assert add_coord(dt64("2018-01-01"), td64(24, "M")) == dt64("2020-01-01")
    assert add_coord(dt64("2018-01-31"), td64(1, "M")) == dt64("2018-02-28")
    assert add_coord(dt64("2018-01-31"), td64(2, "M")) == dt64("2018-03-31")
    assert add_coord(dt64("2018-01-31"), td64(3, "M")) == dt64("2018-04-30")
    assert add_coord(dt64("2020-01-31"), td64(1, "M")) == dt64("2020-02-29")

    # type error
    with pytest.raises(TypeError):
        add_coord(25.0, dt64("2020-01-31"))

    # this base case is generally not encountered
    from podpac.core.coordinates.utils import _add_nominal_timedelta

    assert _add_nominal_timedelta(dt64("2018-01-30"), td64(1, "D")) == dt64("2018-01-31")


def test_divide_timedelta():
    # simple
    assert divide_timedelta(np.timedelta64(2, "D"), 2) == np.timedelta64(1, "D")
    assert divide_timedelta(np.timedelta64(5, "D"), 2.5) == np.timedelta64(2, "D")

    # increase resolution, if necessary
    assert divide_timedelta(np.timedelta64(1, "Y"), 365) == np.timedelta64(1, "D")
    assert divide_timedelta(np.timedelta64(1, "D"), 2) == np.timedelta64(12, "h")
    assert divide_timedelta(np.timedelta64(1, "h"), 2) == np.timedelta64(30, "m")
    assert divide_timedelta(np.timedelta64(1, "m"), 2) == np.timedelta64(30, "s")
    assert divide_timedelta(np.timedelta64(1, "s"), 2) == np.timedelta64(500, "ms")

    # increase resolution several times, if necessary
    assert divide_timedelta(np.timedelta64(1, "D"), 40) == np.timedelta64(36, "m")

    # sometimes, a time is not divisible since we need integer time units
    with pytest.raises(ValueError, match="Cannot divide timedelta .* evenly"):
        divide_timedelta(np.timedelta64(1, "ms"), 3)

    with pytest.raises(ValueError, match="Cannot divide timedelta .* evenly"):
        divide_timedelta(np.timedelta64(1, "D"), 17)


def test_divide_delta():
    # numerical
    assert divide_delta(5.0, 2.0) == 2.5

    # timedelta
    assert divide_delta(np.timedelta64(2, "D"), 2) == np.timedelta64(1, "D")
    assert divide_delta(np.timedelta64(1, "D"), 2) == np.timedelta64(12, "h")
    with pytest.raises(ValueError, match="Cannot divide timedelta .* evenly"):
        divide_delta(np.timedelta64(1, "D"), 17)


def test_timedelta_divisible():
    assert timedelta_divisible(np.timedelta64(1, "D"), np.timedelta64(1, "D"))
    assert timedelta_divisible(np.timedelta64(4, "D"), np.timedelta64(2, "D"))
    assert timedelta_divisible(np.timedelta64(1, "D"), np.timedelta64(6, "h"))
    assert timedelta_divisible(np.timedelta64(1, "D"), np.timedelta64(6, "m"))
    assert timedelta_divisible(np.timedelta64(1, "D"), np.timedelta64(6, "s"))
    assert timedelta_divisible(np.timedelta64(1, "Y"), np.timedelta64(2, "M"))

    assert not timedelta_divisible(np.timedelta64(4, "D"), np.timedelta64(3, "D"))
    assert not timedelta_divisible(np.timedelta64(1, "D"), np.timedelta64(2, "D"))
    assert not timedelta_divisible(np.timedelta64(6, "h"), np.timedelta64(1, "D"))
    assert not timedelta_divisible(np.timedelta64(1, "D"), np.timedelta64(5, "h"))

    assert not timedelta_divisible(np.timedelta64(1, "M"), np.timedelta64(1, "D"))


def test_has_alt_units():
    assert has_alt_units(pyproj.CRS("+proj=merc")) is False
    assert has_alt_units(pyproj.CRS("+proj=merc +vunits=m")) is True


def test_lower_precision_time_bounds():
    a = [np.datetime64("2020-01-01"), np.datetime64("2020-01-02")]
    b = [np.datetime64("2020-01-01T12:00"), np.datetime64("2020-01-01T14:00")]

    with pytest.raises(TypeError, match="Input bounds should be of type np.datetime64"):
        lower_precision_time_bounds(a, [10, 20], False)

    with pytest.raises(TypeError, match="Native bounds should be of type np.datetime64"):
        lower_precision_time_bounds([10, 20], b, False)

    # outer True
    a1, b1 = lower_precision_time_bounds(a, b, True)
    assert a1 == [np.datetime64("2020-01-01"), np.datetime64("2020-01-02")]
    assert a1[0].dtype == "<M8[D]"
    assert a1[1].dtype == "<M8[D]"
    assert b1 == [np.datetime64("2020-01-01"), np.datetime64("2020-01-01")]
    assert b1[0].dtype == "<M8[D]"
    assert b1[1].dtype == "<M8[D]"

    b1, a1 = lower_precision_time_bounds(b, a, True)
    assert b1 == [np.datetime64("2020-01-01"), np.datetime64("2020-01-01")]
    assert b1[0].dtype == "<M8[D]"
    assert b1[1].dtype == "<M8[D]"
    assert a1 == [np.datetime64("2020-01-01"), np.datetime64("2020-01-02")]
    assert a1[0].dtype == "<M8[D]"
    assert a1[1].dtype == "<M8[D]"

    # outer False
    a1, b1 = lower_precision_time_bounds(a, b, False)
    assert a1 == [np.datetime64("2020-01-01T00:00"), np.datetime64("2020-01-02T00:00")]
    assert a1[0].dtype == "<M8[m]"
    assert a1[1].dtype == "<M8[m]"
    assert b1 == [np.datetime64("2020-01-01T12:00"), np.datetime64("2020-01-01T14:00")]
    assert b1[0].dtype == "<M8[m]"
    assert b1[1].dtype == "<M8[m]"

    b1, a1 = lower_precision_time_bounds(b, a, False)
    assert b1 == [np.datetime64("2020-01-01"), np.datetime64("2020-01-01")]
    assert b1[0].dtype == "<M8[D]"
    assert b1[1].dtype == "<M8[D]"
    assert a1 == [np.datetime64("2020-01-01"), np.datetime64("2020-01-02")]
    assert a1[0].dtype == "<M8[D]"
    assert a1[1].dtype == "<M8[D]"


def test_higher_precision_time_bounds():
    a = [np.datetime64("2020-01-01"), np.datetime64("2020-01-02")]
    b = [np.datetime64("2020-01-01T12:00"), np.datetime64("2020-01-01T14:00")]

    with pytest.raises(TypeError, match="Input bounds should be of type np.datetime64"):
        higher_precision_time_bounds(a, [10, 20], False)

    with pytest.raises(TypeError, match="Native bounds should be of type np.datetime64"):
        higher_precision_time_bounds([10, 20], b, False)

    # outer True
    a1, b1 = higher_precision_time_bounds(a, b, True)
    assert a1 == [np.datetime64("2020-01-01T00:00"), np.datetime64("2020-01-02T00:00")]
    assert a1[0].dtype == "<M8[m]"
    assert a1[1].dtype == "<M8[m]"
    assert b1 == [np.datetime64("2020-01-01T12:00"), np.datetime64("2020-01-01T14:00")]
    assert b1[0].dtype == "<M8[m]"
    assert b1[1].dtype == "<M8[m]"

    b1, a1 = higher_precision_time_bounds(b, a, True)
    assert b1 == [np.datetime64("2020-01-01T12:00"), np.datetime64("2020-01-01T14:00")]
    assert b1[0].dtype == "<M8[m]"
    assert b1[1].dtype == "<M8[m]"
    assert a1 == [np.datetime64("2020-01-01T00:00"), np.datetime64("2020-01-02T23:59")]
    assert a1[0].dtype == "<M8[m]"
    assert a1[1].dtype == "<M8[m]"

    # outer False
    a1, b1 = higher_precision_time_bounds(a, b, False)
    assert a1 == [np.datetime64("2020-01-01T00:00"), np.datetime64("2020-01-02T00:00")]
    assert a1[0].dtype == "<M8[m]"
    assert a1[1].dtype == "<M8[m]"
    assert b1 == [np.datetime64("2020-01-01T12:00"), np.datetime64("2020-01-01T14:00")]
    assert b1[0].dtype == "<M8[m]"
    assert b1[1].dtype == "<M8[m]"

    b1, a1 = higher_precision_time_bounds(b, a, False)
    assert b1 == [np.datetime64("2020-01-01"), np.datetime64("2020-01-01")]
    assert b1[0].dtype == "<M8[D]"
    assert b1[1].dtype == "<M8[D]"
    assert a1 == [np.datetime64("2020-01-01"), np.datetime64("2020-01-02")]
    assert a1[0].dtype == "<M8[D]"
    assert a1[1].dtype == "<M8[D]"

def test_add_custom_dimension():

    # Make sure dimensions can't be duplicated
    with pytest.raises(ValueError):
        add_valid_dimension("lat")
    
    # Make sure I can't input non-strings
    with pytest.raises(TypeError):
        add_valid_dimension(["dim1", "dim2"])
    
    # Assert add_valid_dimension must be called
    with pytest.raises(tl.TraitError):
        c1 =  podpac.Coordinates([[1, 2, 3]], ['mydim'])
    
    # Add a valid dimension
    add_valid_dimension("mydim")
    
    ### *Unstacked Coords ###
    
    # Unstacked Coords, One Dimension
    c1 = podpac.Coordinates([[1, 2, 3]], ['mydim'])
    n1 = podpac.data.Array(source=[1, 2, 3], coordinates=c1)
    data1 = n1.eval(c1[1:])
    assert 'mydim' in data1.dims
    
    # Unstacked Coords, Multiple Dimensions
    c2 = podpac.Coordinates([[1, 2], [1, 2, 3]], ['mydim', 'lat'])
    n2 = podpac.data.Array(source=[[1, 2, 3], [4, 5, 6]], coordinates=c2)
    data2 = n2.eval(c2)
    assert ('mydim' in data2.dims) and ('lat' in data2.dims)
    
    ### Stacked Coords ###
    c3 = podpac.Coordinates([[[1,2,3], [4,5,6]]], dims=['mydim_lat'])
    n3 = podpac.data.Array(source=[1,2,3], coordinates=c3)
    data3 = n3.eval(c3)
    assert 'mydim_lat' in data3.dims