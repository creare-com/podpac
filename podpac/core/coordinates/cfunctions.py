
from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np

from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates

def crange(start, stop, step, name=None):
    return UniformCoordinates1d(start, stop, step, name=name)

def clinspace(start, stop, size, name=None):
    try:
        a = np.array([start, stop])
    except ValueError:
        raise ValueError("start and stop must have the same shape")

    if a.ndim == 2:
        c = StackedCoordinates([UniformCoordinates1d(start[i], stop[i], size=size) for i in range(a[0].size)])
    else:
        c = UniformCoordinates1d(start, stop, size=size)

    if name is not None:
        c.name = name
        
    return c