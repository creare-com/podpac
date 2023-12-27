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
import warnings

import numpy as np
import traitlets as tl
from six import string_types
import pyproj

from lazy_import import lazy_function

geodesic = lazy_function("geopy.distance.geodesic")

import podpac


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
        try:
            val = np.datetime64(val)
        except ValueError as e:
            if "," in val:
                val = get_timedelta(val)
            else:
                raise e

    elif isinstance(val, np.datetime64) | isinstance(val, np.timedelta64):
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
            raise TypeError("Invalid coordinate delta, unsupported type '%s'" % type(val))

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
        raise TypeError("Invalid coordinate delta, unsupported type '%s'" % type(val))

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

    if a.dtype == float or np.issubdtype(a.dtype, np.datetime64) or np.issubdtype(a.dtype, np.timedelta64):
        pass

    elif np.issubdtype(a.dtype, np.number):
        a = a.astype(float)

    else:
        a = np.array([make_coord_value(e) for e in np.atleast_1d(np.array(values, dtype=object)).flatten()]).reshape(
            a.shape
        )

        if not (np.issubdtype(a.dtype, np.datetime64) or np.issubdtype(a.dtype, np.timedelta64)):
            raise ValueError("Invalid coordinate values (must be all numbers, all datetimes, or all timedeltas)")

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
    # The following is needed when the time resolution is smaller than a ms -- cannot create datetime in those cases

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
    """
    Divide a timedelta by a numerical divisor. This is a helper function for divide_delta.

    Parameters
    ----------
    delta : np.timedelta64
        The base delta.
    divisor : number
        The divisor

    Returns
    -------
    result : np.timedelta64
        The result, converted to higher resolution if necessary.

    """

    result = delta / divisor
    if divisor * result.astype(int) == delta.astype(int):
        return result

    if delta.dtype.str in _TIMEDELTA_ZOOM:
        return divide_timedelta(delta.astype(_TIMEDELTA_ZOOM[delta.dtype.str]), divisor)

    # months, for example
    raise ValueError("Cannot divide timedelta '%s' evenly by %d" % (make_timedelta_string(delta), divisor))


def timedelta_divisible(numerator, divisor):
    """Check if a numpy timedelta64 is evenly divisible by another.

    Arguments
    ---------
    numerator : numpy.timedelta64
        numerator
    divisor : numpy.timedelta64
        divisor

    Returns
    -------
    divisible : bool
        if the numerator is evenly divisible by the divisor
    """

    try:
        # NOTE: numerator % divisor works in some versions of numpy, but not all
        r = numerator / divisor
        return float(r) == int(r)
    except TypeError:
        # e.g. months and days are not comparible
        return False


_TIMEDELTA_ZOOM = {
    "<m8[Y]": "<m8[D]",
    "<m8[D]": "<m8[h]",
    "<m8[h]": "<m8[m]",
    "<m8[m]": "<m8[s]",
    "<m8[s]": "<m8[ms]",  # already probably farther then necessary...
    "<m8[ms]": "<m8[us]",
    "<m8[us]": "<m8[ns]",
}

VALID_DIMENSION_NAMES = ["lat", "lon", "alt", "time"]


class Dimension(tl.Enum):
    def __init__(self, *args, **kwargs):
        super(Dimension, self).__init__(VALID_DIMENSION_NAMES, *args, **kwargs)


def lower_precision_time_bounds(my_bounds, other_bounds, outer):
    """
    When given two bounds of np.datetime64, this function will convert both bounds to the lower-precision (in terms of
    time unit) numpy datetime4 object if outer==True, otherwise only my_bounds will be converted.

    Parameters
    -----------
    my_bounds : List(np.datetime64)
        The bounds of the coordinates of the dataset
    other_bounds : List(np.datetime64)
        The bounds used for the selection
    outer : bool
        When the other_bounds are higher precision than the input_bounds, only convert these IF outer=True

    Returns
    --------
    my_bounds : List(np.datetime64)
        The bounds of the coordinates of the dataset at the new precision
    other_bounds : List(np.datetime64)
        The bounds used for the selection at the new precision, if outer == True, otherwise return original coordinates
    """
    if not isinstance(other_bounds[0], np.datetime64) or not isinstance(other_bounds[1], np.datetime64):
        raise TypeError("Input bounds should be of type np.datetime64 when selecting data from:", str(my_bounds))

    if not isinstance(my_bounds[0], np.datetime64) or not isinstance(my_bounds[1], np.datetime64):
        raise TypeError("Native bounds should be of type np.datetime64 when selecting data using:", str(other_bounds))

    if my_bounds[0].dtype < other_bounds[0].dtype and outer:
        other_bounds = [b.astype(my_bounds[0].dtype) for b in other_bounds]
    else:
        my_bounds = [b.astype(other_bounds[0].dtype) for b in my_bounds]

    return my_bounds, other_bounds


def higher_precision_time_bounds(my_bounds, other_bounds, outer):
    """
    When given two bounds of np.datetime64, this function will convert both bounds to the higher-precision (in terms of
    time unit) numpy datetime4 object if outer==True, otherwise only my_bounds will be converted.

    Parameters
    -----------
    my_bounds : List(np.datetime64)
        The bounds of the coordinates of the dataset
    other_bounds : List(np.datetime64)
        The bounds used for the selection
    outer : bool
        When the other_bounds are higher precision than the input_bounds, only convert these IF outer=True

    Returns
    --------
    my_bounds : List(np.datetime64)
        The bounds of the coordinates of the dataset at the new precision
    other_bounds : List(np.datetime64)
        The bounds used for the selection at the new precision, if outer == True, otherwise return original coordinates

    Notes
    ------
    When converting the upper bound with outer=True, the whole lower-precision time unit is valid. E.g. when converting
    YYYY-MM-DD to YYYY-MM-DD HH, the largest value for HH is used, since the whole day is valid.
    """
    if not isinstance(other_bounds[0], np.datetime64) or not isinstance(other_bounds[1], np.datetime64):
        raise TypeError("Input bounds should be of type np.datetime64 when selecting data from:", str(my_bounds))

    if not isinstance(my_bounds[0], np.datetime64) or not isinstance(my_bounds[1], np.datetime64):
        raise TypeError("Native bounds should be of type np.datetime64 when selecting data using:", str(other_bounds))

    if my_bounds[0].dtype > other_bounds[0].dtype and outer:
        # for the upper bound, the whole lower-precision time unit is valid (see note)
        # select the largest value for the higher-precision time unit by adding one lower-precision time unit and
        # subtracting one higher-precision time unit.
        other_bounds = [
            other_bounds[0].astype(my_bounds[0].dtype),
            (other_bounds[1] + 1).astype(my_bounds[0].dtype) - 1,
        ]
    else:
        my_bounds = [b.astype(other_bounds[0].dtype) for b in my_bounds]

    return my_bounds, other_bounds


def has_alt_units(crs):
    """
    Check if the CRS has vertical units.

    Arguments
    ---------
    crs : pyproj.CRS
        CRS to check

    Returns
    -------
    has_alt_units : bool
        True if the CRS has vunits or other altitude units.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return crs.is_vertical or "vunits" in crs.to_dict() or any(axis.direction == "up" for axis in crs.axis_info)


def calculate_distance(point1, point2, ellipsoid_tuple, coordinate_name, units="meter"):
    """Return distance of 2 points in desired unit measurement

    Parameters
    ----------
    point1 : tuple
    point2 : tuple

    Returns
    -------
    float
        The distance between point1 and point2, according to the current coordinate system's distance metric, using the desired units
    """
    if coordinate_name == "cartesian":
        return np.linalg.norm(point1 - point2, axis=-1, units="meter") * podpac.units(units)
    else:
        if not isinstance(point1, tuple) and point1.size > 2:
            distances = np.empty(len(point1))
            for i in range(len(point1)):
                distances[i] = geodesic(point1[i], point2[i], ellipsoid=ellipsoid_tuple).m
            return distances * podpac.units("metre").to(podpac.units(units))
        if not isinstance(point2, tuple) and point2.size > 2:
            distances = np.empty(len(point2))
            for i in range(len(point2)):
                distances[i] = geodesic(point1, point2[i], ellipsoid=ellipsoid_tuple).m
            return distances * podpac.units("metre").to(podpac.units(units))
        else:
            return (geodesic(point1, point2, ellipsoid=ellipsoid_tuple).m) * podpac.units("metre").to(
                podpac.units(units)
            )


def add_valid_dimension(dimension_name):
    """
    Add a new dimension to VALID_DIMENSION_NAMES.

    Parameters
    ----------
    dimension_name : string
        Name of dimension to make a valid dimension
    """

    # Assert inputted value is a string
    if not isinstance(dimension_name, str):
        raise TypeError(f"Expected arg to be a string, but got {type(dimension_name).__name__}")

    if dimension_name in VALID_DIMENSION_NAMES:
        raise ValueError(f"Dim `{dimension_name}` already a valid dimension.")

    if "-" in dimension_name or "_" in dimension_name:
        raise ValueError(f"Dim `{dimension_name}` may note contain `-` or `_`.")

    VALID_DIMENSION_NAMES.append(dimension_name)
