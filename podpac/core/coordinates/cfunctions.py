
from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np

from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates

def crange(start, stop, step):
    return UniformCoordinates1d(start, stop, step)

def clinspace(start, stop, size):
    try:
        a = np.stack([start, stop])
    except ValueError:
        raise ValueError("start, stop, and step must have the same shape")

    if a.ndim == 2:
        return StackedCoordinates([UniformCoordinates1d(start[i], stop[i], size=size) for i in range(a[0].size)])
    else:
        return UniformCoordinates1d(start, stop, size=size)

def cstack(a):
    if len(a) == 0:
        return ArrayCoordinates()

    if len(a) == 1:
        return a[0].copy() if isinstance(a[0], Coordinates1d) else ArrayCoordinates1d(a[0])
    
    cs = [e.copy() if isinstance(e, Coordinates1d) else ArrayCoordinates1d(e) for e in a]
    return StackedCoordinates(cs)