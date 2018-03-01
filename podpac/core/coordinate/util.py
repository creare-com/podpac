from __future__ import division, unicode_literals, print_function, absolute_import

import datetime
import calendar
import numbers
import numpy as np
from six import string_types

def get_timedelta(s):
    """
    Make a numpy timedelta from a podpac timedelta string.

    The time delta string must be in the form '<n>,<unit>', where <n> is an
    integer and <unit> is the character for the timedelta unit.

    Parameters
    ----------
    s : str
        podpac timedelta string, in the form '<n>,<unit>'

    Returns
    -------
    np.timedelta64
        numpy timedelta

    Examples
    --------

    >>> get_timedelta('2,D')
    numpy.timedelta64(2,'D')

    >>> get_timedelta('-3,h')
    numpy.timedelta64(-3,'h')

    """

    a, b = s.split(',')
    return np.timedelta64(int(a), b)

def get_timedelta_unit(delta):
    """
    Get the unit character from a numpy timedelta.

    Parameters
    ----------
    delta : np.timedelta64
        numpy timedelta

    Returns
    -------
    str
        the character for the timedelta unit

    Examples
    --------

    >>> get_timedelta_unit(np.timedelta64(1, 'D'))
    'D'

    """

    dname = delta.dtype.name
    if not dname.startswith('timedelta'):
        raise TypeError("Cannot get timedelta unit from type '%s'" % dname)
    return dname[12]

def get_timedelta_string(delta):
    """
    Get the podpac timedelta string from a numpy timedelta.

    Parameters
    ----------
    delta : np.timedelta64
        numpy timedelta

    Returns
    -------
    str
        podpac timedelta string, in the form '<n>,<units>'

    Examples
    --------

    >>> get_timedelta_string(np.timedelta64(2, 'D'))
    '2,D'

    """

    a = delta.item()
    b = get_timedelta_unit(delta)
    return a, b

def make_coord_value(val):
    """
    Make a podpac coordinate value by casting to the correct type.

    Parameters
    ----------
    val : str, number, datetime.datetime, np.array
        Input coordinate value.

    Returns
    -------
    val : float, np.datetime64
        Cast coordinate value.

    Notes
    -----
     * the value is extracted from singleton and 0-dimensional arrays
     * strings interpreted as inputs to numpy.datetime64
     * datetime datetimes are converted to numpy datetimes
     * numbers are converted to floats

    """

    # extract value from singleton and 0-dimensional arrays
    if isinstance(val, np.ndarray):
        if val.dtype.ndim == 0:
            val = val[()]
        elif val.size == 1:
            val = val[0]

    # type checking and conversion
    if isinstance(val, (string_types, datetime.datetime)):
        val = np.datetime64(val)
    elif isinstance(val, np.datetime64):
        pass
    elif isinstance(val, numbers.Number):
        val = float(val)
    else:
        raise TypeError("Invalid coordinate value '%s'" % type(val))

    return val

def make_coord_delta(val):
    """
    Make a podpac coordinate delta by casting to the correct type.

    Parameters
    ----------
    val : str, number, datetime.timedelta, np.array
        Input coordinate delta.

    Returns
    -------
    val : float, np.timedelta64
        Cast coordinate delta.

    Notes
    -----
     * the value is extracted from singleton and 0-dimensional arrays
     * strings are interpreted as inputs to get_timedelta
     * datetime timedeltas are converted to numpy timedeltas
     * numbers are converted to floats

    """

    # extract value from singleton and 0-dimensional arrays
    if isinstance(val, np.ndarray):
        if val.dtype.ndim == 0:
            val = val[()]
        elif val.size == 1:
            val = val[0]

    # type checking and conversion
    if isinstance(val, string_types):
        val = get_timedelta(val)
    elif isinstance(val, datetime.timedelta):
        val = np.timedelta64(val)
    elif isinstance(val, np.timedelta64):
        pass
    elif isinstance(val, numbers.Number):
        val = float(val)
    else:
        raise TypeError("Invalid coordinate delta '%s'" % type(val))

    return val

def add_coord(base, delta):
    """
    Add a coordinate delta to a coordinate value.

    Parameters
    ----------
    base : float, np.datetime64
        The base coordinate value.
    delta : float, np.timedelta64
        The coordinate delta. This can also be a numpy array.

    Returns
    -------
    result : float, np.datetime64
        The sum, with month and year timedeltas handled. If dalta is an array,
        the result will be an array.

    Notes
    -----
    Month and year deltas are added nominally, which differs from how numpy
    adds timedeltas to datetimes. When adding months or years, if the new date
    exceeds the number of days in the month, it is set to the last day of the
    month.

    Examples
    --------

    >>> add_coord(1.5, 1.0)
    2.5

    >>> add_coord(1.5, np.array([1.0, 2.0]))
    array([ 2.5,  3.5])

    >>> add_coord(np.datetime64('2018-01-01'), np.timedelta64(1, 'D'))
    numpy.datetime64('2018-01-02')

    >>> np.datetime64('2018-01-01') + np.timedelta64(1, 'M')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: Cannot get a common metadata divisor for NumPy datetime metadata [D] and [M] because they have incompatible nonlinear base time units

    >>> add_coord(np.datetime64('2018-01-01'), np.timedelta64(1, 'M'))
    numpy.datetime64('2018-02-01')

    """

    try:
        return base + delta
    except TypeError as e:
        if (isinstance(base, np.datetime64) and
            np.issubdtype(delta.dtype, np.timedelta64)):
            return _add_nominal_timedelta(base, delta)
        else:
            raise e

def _add_nominal_timedelta(base, delta):
    dunit = get_timedelta_unit(delta)
    if dunit not in ['Y', 'M']:
        return base + delta

    shape = delta.shape
    base = base.item()
    tds = np.array(delta).astype(int).flatten()

    dates = []
    for td in tds:
        if dunit == 'Y':
            date = _replace_safe(base, year=base.year+td)
        elif dunit == 'M':
            date = _replace_safe(base, month=base.month+td)
        dates.append(date)

    dates = np.array(dates).astype(np.datetime64).reshape(shape)
    if shape == ():
        dates = dates[()]
    return dates

def _replace_safe(dt, year=None, month=None):
    if year is None:
        year = dt.year
    if month is None:
        month = dt.month
    
    year = dt.year + (month-1) // 12
    month = (month-1) % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)