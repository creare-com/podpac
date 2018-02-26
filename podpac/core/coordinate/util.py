from __future__ import division, unicode_literals, print_function, absolute_import

import numbers
import numpy as np
from six import string_types

def get_timedelta(s):
    a, b = s.split(',')
    return np.timedelta64(int(a), b)

def get_timedelta_unit(delta):
    dname = delta.dtype.name
    if not dname.startswith('timedelta'):
        raise TypeError("Cannot get timedelta unit from type '%s'" % dname)
    return dname[12]

def get_timedelta_string(delta):
    a = delta.item()
    b = get_timedelta_unit(delta)
    return a, b

def make_coord_value(val):
    # type checking and conversion
    if isinstance(val, string_types):
        val = np.datetime64(val)
    elif isinstance(val, np.datetime64):
        pass
    elif isinstance(val, numbers.Number):
        val = float(val)
    else:
        raise TypeError("Invalid coordinate value '%s'" % type(val))

    return val

def make_coord_delta(val):
    if isinstance(val, string_types):
        val = get_timedelta(val)
    elif isinstance(val, np.timedelta64):
        pass
    elif isinstance(val, numbers.Number):
        val = float(val)
    else:
        raise TypeError("Invalid coordinate delta '%s'" % type(val))

    return val

def add_coord(base, offset):
    try:
        return base + offset
    except TypeError as e:
        if (isinstance(base, np.datetime64) and
            np.issubdtype(offset.dtype, np.timedelta64)):
            return _add_nominal_timedelta(base, offset)
        else:
            raise e

def divide_coord(base, divisor):
    try:
        return base / divisor
    except TypeError as e:
        if (isinstance(base, np.datetime64) and
            np.issubdtype(divisor.dtype, np.timedelta64)):
            return _divide_nominal_timedelta(base, divisor)
        else:
            raise e

def _add_nominal_timedelta(base, delta):
    dunit = get_timedelta_unit(delta)
    if dunit not in ['Y', 'M']:
        return base + delta

    shape = delta.shape
    dt_base = base.item()
    tds = np.array(delta).astype(int).flatten()

    dates = []
    for td in tds:
        if dunit == 'Y':
            year = dt_base.year + td
            month = dt_base.month
        elif dunit == 'M':
            months = dt_base.month + td - 1
            year = dt_base.year + months // 12
            month = months % 12 + 1

        try:
            dt_new = dt_base.replace(year=year, month=month)
        except ValueError as e:
            # non leap year?
            dt_new = dt_base.replace(year=year, month=month, day=28)
        
        dates.append(dt_new)

    return np.array(dates).astype(np.datetime64).reshape(shape)

def _divide_nominal_timedelta(base, delta):
    # TODO
    unit = get_time_unit(delta)
    date = base.astype(object)
    
    try: 
        date = date.replace(**{unit: getattr(date, unit) / delta.astype(object)})
    except ValueError as e:
        date = date.replace(**{unit: getattr(date, unit) / delta.astype(object)
                , 'day': 28})
    return np.datetime64(date)