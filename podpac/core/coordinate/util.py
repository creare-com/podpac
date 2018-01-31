from __future__ import division, unicode_literals, print_function, absolute_import

import numbers
import numpy as np
from six import string_types

def get_timedelta(s):
    a, b = s.split(',')
    return np.timedelta64(int(a), b)

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