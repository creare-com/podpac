
from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np

from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates

def crange(start, stop, step, name=None):
    """
    Create uniformly-spaced 1d coordinates with a start, stop, and step.

    For numerical coordinates, the start, stop, and step are converted to ``float``. For time
    coordinates, the start and stop are converted to numpy ``datetime64``, and the step is converted to numpy
    ``timedelta64``. For convenience, podpac automatically converts datetime strings such as ``'2018-01-01'`` to
    ``datetime64`` and timedelta strings such as ``'1,D'`` to ``timedelta64``.

    Arguments
    ---------
    start : float, datetime64, datetime, str
        Start coordinate.
    stop : float, datetime64, datetime, str
        Stop coordinate.
    step : float, timedelta64, timedelta, str
        Signed, non-zero step between coordinates.
    name : str, optional
        Dimension name.

    Returns
    -------
    :class:`UniformCoordinates1d`
        Uniformly-spaced 1d coordinates.
    """

    return UniformCoordinates1d(start, stop, step=step, name=name)

def clinspace(start, stop, size, name=None):
    """
    Create uniformly-spaced 1d coordinates with a start, stop, and size.

    For numerical coordinates, the start and stop are converted to ``float``. For time coordinates, the start and stop
    are converted to numpy ``datetime64``. For convenience, podpac automatically converts datetime strings such as
    ``'2018-01-01'`` to ``datetime64``.

    Arguments
    ---------
    start : float, datetime64, datetime, str
        Start coordinate for 1d coordinates
    stop : float, datetime64, datetime, str
        Stop coordinate for 1d coordinates
    size : int
        Number of coordinates.
    name : str, optional
        Dimension name.

    Returns
    -------
    :class:`UniformCoordinates1d`
        Uniformly-spaced 1d coordinates.

    Raises
    ------
    ValueError
        If the start and stop are not the same size.
    """
    if np.array(start).size != np.array(stop).size:
        raise ValueError("Size mismatch, 'start' and 'stop' must have the same size (%s != %s)" % (
            np.array(start).size, np.array(stop).size))
    
    c = UniformCoordinates1d(start, stop, size=size)

    if name is not None:
        c.name = name
    
    return c
