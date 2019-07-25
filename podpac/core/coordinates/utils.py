"""
Utilities functions for handling podpac coordinates.

.. testsetup:: podpac.core.coordinates.utils
    
    import numpy as np
    from podpac.core.coordinates.utils import *
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import datetime
import re
import calendar
import numbers
import numpy as np
import traitlets as tl
from six import string_types
import pyproj


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

    .. doctest:: podpac.core.coordinates.utils

        >>> get_timedelta('2,D')
        numpy.timedelta64(2,'D')
        
        >>> get_timedelta('-3,h')
        numpy.timedelta64(-3,'h')
    
    """

    a, b = s.split(",")
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
    
    .. doctest:: podpac.core.coordinates.utils

        >>> get_timedelta_unit(np.timedelta64(1, 'D'))
        'D'
    
    Raises
    ------
    TypeError
        Description
    
    """

    try:
        dname = delta.dtype.name
    except AttributeError:
        raise TypeError("Cannot get timedelta unit from type '%s'" % type(delta))
    if not dname.startswith("timedelta"):
        raise TypeError("Cannot get timedelta unit from dtype '%s'" % dname)
    return dname[12:-1]


def make_timedelta_string(delta):
    """
    Make a podpac timedelta string from a numpy timedelta.
    
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
    
    .. doctest:: podpac.core.coordinates.utils

        >>> make_timedelta_string(np.timedelta64(2, 'D'))
        '2,D'
    
    Raises
    ------
    TypeError
        Description
    
    """

    if not isinstance(delta, np.timedelta64):
        raise TypeError("Cannot make timedelta string from type '%s'" % type(delta))

    mag = delta.astype(int)
    unit = get_timedelta_unit(delta)
    return "%d,%s" % (mag, unit)


def make_coord_value(val):
    """
    Make a podpac coordinate value by casting to the correct type.
    
    Parameters
    ----------
    val : str, number, datetime.date, np.ndarray
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
    
    Raises
    ------
    Value
        val is an unsupported type
    
    """

    # extract value from singleton and 0-dimensional arrays
    if isinstance(val, np.ndarray):
        try:
            val = val.item()
        except ValueError:
            raise TypeError("Invalid coordinate value, unsupported type '%s'" % type(val))

    # type checking and conversion
    if isinstance(val, (string_types, datetime.date)):
        val = np.datetime64(val)
    elif isinstance(val, np.datetime64):
        pass
    elif isinstance(val, numbers.Number):
        val = float(val)
    else:
        raise TypeError("Invalid coordinate value, unsupported type '%s'" % type(val))

    return val


def make_coord_delta(val):
    """
    Make a podpac coordinate delta by casting to the correct type.
    
    Parameters
    ----------
    val : str, number, datetime.timedelta, np.ndarray
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
    
    Raises
    ------
    TypeError
        Description
    
    """

    # extract value from singleton and 0-dimensional arrays
    if isinstance(val, np.ndarray):
        try:
            val = val.item()
        except ValueError:
            raise TypeError("Invalid coordinate delta, unsuported type '%s'" % type(val))

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
        raise TypeError("Invalid coordinate delta, unsuported type '%s'" % type(val))

    return val


def make_coord_array(values):
    """
    Make an array of podpac coordinate values by casting to the correct type.

    Parameters
    ----------
    values : array-like
        Input coordinates.
    
    Returns
    -------
    a : np.ndarray
        Cast coordinate values.
    
    Notes
    -----
     * all of the values must be of the same type
     * strings and datetimes are converted to numpy datetime64
     * numbers are converted to floats
    """

    a = np.atleast_1d(values)

    if a.ndim != 1:
        raise ValueError("Invalid coordinate values (ndim=%d, must be ndim=1)" % a.ndim)

    if a.dtype == float or np.issubdtype(a.dtype, np.datetime64):
        pass

    elif np.issubdtype(a.dtype, np.number):
        a = a.astype(float)

    else:
        a = np.array([make_coord_value(e) for e in np.atleast_1d(np.array(values, dtype=object))])

        if not np.issubdtype(a.dtype, np.datetime64):
            raise ValueError("Invalid coordinate values (must be all numbers or all datetimes)")

    return a


def make_coord_delta_array(values):
    """
    Make an array of podpac coordinate deltas by casting to the correct type.

    Parameters
    ----------
    values : array-like
        Input coordinate deltas.
    
    Returns
    -------
    a : np.ndarray
        Cast coordinate deltas.
    
    Notes
    -----
     * all of the deltas must be of the same type
     * strings and timedeltas are converted to numpy timdelta64
     * numbers are converted to floats
    """

    a = np.atleast_1d(values)

    if a.ndim != 1:
        raise ValueError("Invalid coordinate deltas (ndim=%d, must be ndim=1)" % a.ndim)

    if a.dtype == float or np.issubdtype(a.dtype, np.timedelta64):
        pass

    elif np.issubdtype(a.dtype, np.number):
        a = a.astype(float)

    else:
        a = np.array([make_coord_delta(e) for e in np.atleast_1d(np.array(values, dtype=object))])

        if not np.issubdtype(a.dtype, np.timedelta64):
            raise ValueError("Invalid coordinate deltas (must be all numbers or all compatible timedeltas)")

    return a


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
        The sum, with month and year timedeltas handled. If delta is an array,
        the result will be an array.
    
    Notes
    -----
    Month and year deltas are added nominally, which differs from how numpy
    adds timedeltas to datetimes. When adding months or years, if the new date
    exceeds the number of days in the month, it is set to the last day of the
    month.
    
    Examples
    --------
    
    .. doctest:: podpac.core.coordinates.utils

        >>> add_coord(1.5, 1.0)
        2.5
        
        >>> add_coord(1.5, np.array([1.0, 2.0]))
        array([2.5, 3.5])
        
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
        if isinstance(base, np.datetime64) and np.issubdtype(delta.dtype, np.timedelta64):
            return _add_nominal_timedelta(base, delta)
        else:
            raise e


def _add_nominal_timedelta(base, delta):
    dunit = get_timedelta_unit(delta)
    if dunit not in ["Y", "M"]:
        return base + delta

    shape = delta.shape
    if not isinstance(base.item(), datetime.datetime):
        base = base.astype("datetime64[ms]")
    base = base.item()
    tds = np.array(delta).astype(int).flatten()

    dates = []
    for td in tds:
        if dunit == "Y":
            date = _replace_safe(base, year=base.year + td)
        elif dunit == "M":
            date = _replace_safe(base, month=base.month + td)
        dates.append(date)

    dates = np.array([np.datetime64(date) for date in dates]).reshape(shape)
    if shape == ():
        dates = dates[()]
    return dates


def _replace_safe(dt, year=None, month=None):
    if year is None:
        year = dt.year
    if month is None:
        month = dt.month

    year = year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def divide_delta(delta, divisor):
    """
    Divide a coordinate delta by a numerical divisor.
    
    Parameters
    ----------
    delta : float, np.timedelta64
        The base delta.
    divisor : number
        The divisor
    
    Returns
    -------
    result : float, np.timedelta64
        The result, with timedeltas converted to higher resolution if necessary.

    """

    if isinstance(delta, np.timedelta64):
        try:
            return divide_timedelta(delta, divisor)
        except ValueError:
            raise ValueError("Cannot divide timedelta '%s' evenly by %d" % (make_timedelta_string(delta), divisor))
    else:
        return delta / divisor


def divide_timedelta(delta, divisor):
    result = delta / divisor
    if divisor * result.astype(int) == delta.astype(int):
        return result

    if delta.dtype.str in _TIMEDELTA_ZOOM:
        return divide_timedelta(delta.astype(_TIMEDELTA_ZOOM[delta.dtype.str]), divisor)

    # months, for example
    raise ValueError("Cannot divide timedelta '%s' evenly by %d" % (make_timedelta_string(delta), divisor))


_TIMEDELTA_ZOOM = {
    "<m8[Y]": "<m8[D]",
    "<m8[D]": "<m8[h]",
    "<m8[h]": "<m8[m]",
    "<m8[m]": "<m8[s]",
    "<m8[s]": "<m8[ms]",  # already probably farther then necessary...
    "<m8[ms]": "<m8[us]",
    "<m8[us]": "<m8[ns]",
}


class Dimension(tl.Enum):
    def __init__(self, *args, **kwargs):
        super(Dimension, self).__init__(["lat", "lon", "alt", "time"], *args, **kwargs)


class CoordinateType(tl.Enum):
    def __init__(self, *args, **kwargs):
        super(CoordinateType, self).__init__(["point", "left", "right", "midpoint"], *args, **kwargs)


def get_vunits(crs):
    """
    Get vunits from a coordinate reference system string.

    Arguments
    ---------
    crs : str
        PROJ4 coordinate reference system.

    Returns
    -------
    vunits : str
        PROJ4 distance units for altitude, or None if no vunits present.
    """

    if "+vunits" not in crs:
        return None

    return re.search(r"(?<=\+vunits=)[a-z\-]+", crs).group(0)


def set_vunits(crs, vunits):
    """
    Set the vunits of a coordinate reference system string. The vunits will be replaced or added, as needed.

    Arguments
    ---------
    crs : str
        PROJ4 coordinate reference system.
    vunits : str
        desired altitude units in PROJ4 distance units.

    Returns
    -------
    crs : str
        PROJ4 coordinate reference system string with the desired vunits.
    """

    if "+vunits" in crs:
        crs = re.sub(r"(?<=\+vunits=)[a-z\-]+", vunits, crs)
    else:
        crs = pyproj.CRS(crs).to_proj4()  # convert EPSG-style strings
        crs += " +vunits={}".format(vunits)

    crs = pyproj.CRS(crs).to_proj4()  # standardize, this is optional

    return crs


def rem_vunits(crs):
    """
    Remove the vunits of a coordinate reference system string, if present.

    Arguments
    ---------
    crs : str
        PROJ4 coordinate reference system.

    Returns
    crs : str
        PROJ4 coordinate referenc system without vunits.
    """

    if "+vunits" in crs:
        crs = re.sub(r"\+vunits=[a-z\-]+", "", crs)
    return crs
