
from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np

from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates

def crange(start, stop, step):
    return UniformCoordinates1d(start, stop, step)

def clinspace(start, stop, size):
    try:
        a = np.array([start, stop])
    except ValueError:
        raise ValueError("start, stop, and step must have the same shape")

    if a.ndim == 2:
        return StackedCoordinates([UniformCoordinates1d(start[i], stop[i], size=size) for i in range(a[0].size)])
    else:
        return UniformCoordinates1d(start, stop, size=size)