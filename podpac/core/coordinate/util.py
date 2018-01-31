from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np

def get_timedelta(s):
    a, b = s.split(',')
    return np.timedelta64(int(a), b)