"""
Utils Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import os
from collections import OrderedDict

import numpy as np

from podpac.core.coordinates import Coordinates, StackedCoordinates, ArrayCoordinates1d


def get_dims_list():
    return [
        ("lat",),
        ("lon",),
        ("alt",),
        ("tim",),
        ("lat", "lon"),
        ("lat", "alt"),
        ("lat", "tim"),
        ("lon", "lat"),
        ("lon", "alt"),
        ("lon", "tim"),
        ("alt", "lat"),
        ("alt", "lon"),
        ("alt", "tim"),
        ("tim", "lat"),
        ("tim", "lon"),
        ("tim", "alt"),
        ("lat", "lon", "alt"),
        ("lat", "lon", "tim"),
        ("lat", "alt", "tim"),
        ("lat", "tim", "alt"),
        ("lon", "lat", "alt"),
        ("lon", "lat", "tim"),
        ("lon", "alt", "tim"),
        ("lon", "tim", "alt"),
        ("alt", "lat", "lon"),
        ("alt", "lat", "tim"),
        ("alt", "lon", "lat"),
        ("alt", "lon", "tim"),
        ("alt", "tim", "lat"),
        ("alt", "tim", "lon"),
        ("tim", "lat", "lon"),
        ("tim", "lat", "alt"),
        ("tim", "lon", "lat"),
        ("tim", "lon", "alt"),
        ("tim", "alt", "lat"),
        ("tim", "alt", "lon"),
        ("lat", "lon", "alt", "tim"),
        ("lat", "lon", "tim", "alt"),
        ("lon", "lat", "alt", "tim"),
        ("lon", "lat", "tim", "alt"),
        ("alt", "lat", "lon", "tim"),
        ("alt", "lon", "lat", "tim"),
        ("alt", "tim", "lat", "lon"),
        ("alt", "tim", "lon", "lat"),
        ("tim", "lat", "lon", "alt"),
        ("tim", "lon", "lat", "alt"),
        ("tim", "alt", "lat", "lon"),
        ("tim", "alt", "lon", "lat"),
        ("lat_lon",),
        ("lat_alt",),
        ("lat_tim",),
        ("lon_lat",),
        ("lon_alt",),
        ("lon_tim",),
        ("alt_lat",),
        ("alt_lon",),
        ("alt_tim",),
        ("tim_lat",),
        ("tim_lon",),
        ("tim_alt",),
        ("lat_lon", "alt"),
        ("lat_lon", "tim"),
        ("lat_alt", "tim"),
        ("lat_tim", "alt"),
        ("lon_lat", "tim"),
        ("lon_lat", "alt"),
        ("lon_alt", "tim"),
        ("lon_tim", "alt"),
        ("alt_lat", "tim"),
        ("alt_lon", "tim"),
        ("alt_tim", "lat"),
        ("alt_tim", "lon"),
        ("tim_lat", "alt"),
        ("tim_lon", "alt"),
        ("tim_alt", "lat"),
        ("tim_alt", "lon"),
        ("lat", "alt_tim"),
        ("lat", "tim_alt"),
        ("lon", "alt_tim"),
        ("lon", "tim_alt"),
        ("alt", "lat_lon"),
        ("alt", "lat_tim"),
        ("alt", "lon_lat"),
        ("alt", "lon_tim"),
        ("alt", "tim_lat"),
        ("alt", "tim_lon"),
        ("tim", "lat_lon"),
        ("tim", "lat_alt"),
        ("tim", "lon_lat"),
        ("tim", "lon_alt"),
        ("tim", "alt_lat"),
        ("tim", "alt_lon"),
        ("lat_lon", "alt_tim"),
        ("lat_lon", "tim_alt"),
        ("lon_lat", "tim_alt"),
        ("lon_lat", "alt_tim"),
        ("alt_tim", "lat_lon"),
        ("alt_tim", "lon_lat"),
        ("tim_alt", "lat_lon"),
        ("tim_alt", "lon_lat"),
        ("lat_lon_alt", "tim"),
        ("lat_lon_tim", "alt"),
        ("lon_lat_tim", "alt"),
        ("lon_lat_alt", "tim"),
        ("alt_lat_lon", "tim"),
        ("alt_lon_lat", "tim"),
        ("tim_lat_lon", "alt"),
        ("tim_lon_lat", "alt"),
        ("alt", "lat_lon_tim"),
        ("alt", "lon_lat_tim"),
        ("alt", "tim_lat_lon"),
        ("alt", "tim_lon_lat"),
        ("tim", "lat_lon_alt"),
        ("tim", "lon_lat_alt"),
        ("tim", "alt_lat_lon"),
        ("tim", "alt_lon_lat"),
        ("lat", "lon", "alt_tim"),
        ("lat", "lon", "tim_alt"),
        ("lon", "lat", "alt_tim"),
        ("lon", "lat", "tim_alt"),
        ("alt", "tim", "lat_lon"),
        ("alt", "tim", "lon_lat"),
        ("tim", "alt", "lat_lon"),
        ("tim", "alt", "lon_lat"),
        ("alt", "lat_lon", "tim"),
        ("alt", "lon_lat", "tim"),
        ("tim", "lat_lon", "alt"),
        ("tim", "lon_lat", "alt"),
        ("lat_lon", "alt", "tim"),
        ("lat_lon", "tim", "alt"),
        ("lon_lat", "alt", "tim"),
        ("lon_lat", "tim", "alt"),
        ("alt_tim", "lat", "lon"),
        ("alt_tim", "lon", "lat"),
        ("tim_alt", "lat", "lon"),
        ("tim_alt", "lon", "lat"),
        ("lat_lon_alt",),
        ("lat_lon_tim",),
        ("lat_alt_tim",),
        ("lat_tim_alt",),
        ("lon_lat_alt",),
        ("lon_lat_tim",),
        ("lon_alt_tim",),
        ("lon_tim_alt",),
        ("alt_lat_lon",),
        ("alt_lat_tim",),
        ("alt_lon_lat",),
        ("alt_lon_tim",),
        ("alt_tim_lat",),
        ("alt_tim_lon",),
        ("tim_lat_lon",),
        ("tim_lat_alt",),
        ("tim_lon_lat",),
        ("tim_lon_alt",),
        ("tim_alt_lat",),
        ("tim_alt_lon",),
        ("lat_lon_alt_tim",),
        ("lat_lon_tim_alt",),
        ("lon_lat_alt_tim",),
        ("lon_lat_tim_alt",),
        ("alt_lat_lon_tim",),
        ("alt_lon_lat_tim",),
        ("alt_tim_lat_lon",),
        ("alt_tim_lon_lat",),
        ("tim_lat_lon_alt",),
        ("tim_lon_lat_alt",),
        ("tim_alt_lat_lon",),
        ("tim_alt_lon_lat",),
    ]


def make_coordinate_combinations(lat=None, lon=None, alt=None, time=None):
    """Generates every combination of stacked and unstacked coordinates podpac expects to handle

    Parameters
    -----------
    lat: podpac.core.coordinates.Coordinates1d, optional
        1D coordinate object used to create the Coordinate objects that contain the latitude dimension. By default uses:
        UniformCoord(start=0, stop=2, size=3)
    lon: podpac.core.coordinates.Coordinates1d, optional
        Same as above but for longitude. By default uses:
        UniformCoord(start=2, stop=6, size=3)
    alt: podpac.core.coordinates.Coordinates1d, optional
        Same as above but for longitude. By default uses:
        UniformCoord(start=6, stop=12, size=3)
    time: podpac.core.coordinates.Coordinates1d, optional
        Same as above but for longitude. By default uses:
        UniformCoord(start='2018-01-01T00:00:00', stop='2018-03-01T00:00:00', size=3)

    Returns
    -------
    OrderedDict:
        Dictionary of all the podpac.Core.Coordinate objects podpac expects to handle. The dictionary keys is a tuple of
        coordinate dimensions, and the values are the actual Coordinate objects.

    Notes
    ------
    When custom lat, lon, alt, and time 1D coordinates are given, only those with the same number of coordinates are
    stacked together. For example, if lat, lon, alt, and time have sizes 3, 4, 5, and 6, respectively, no stacked
    coordinates are created. Also, no exception or warning is thrown for this case.
    """

    # make the 1D coordinates
    if lat is None:
        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
    if lon is None:
        lon = ArrayCoordinates1d([2, 4, 6], name="lon")
    if alt is None:
        alt = ArrayCoordinates1d([6, 9, 12], name="alt")
    if time is None:
        time = ArrayCoordinates1d(["2018-01-01", "2018-02-01", "2018-03-01"], name="time")

    d = dict([("lat", lat), ("lon", lon), ("alt", alt), ("tim", time)])

    dims_list = get_dims_list()

    # make the stacked coordinates
    for dim in [dim for dims in dims_list for dim in dims if "_" in dim]:
        cs = [d[k] for k in dim.split("_")]
        if any(c.size != cs[0].size for c in cs):
            continue  # can't stack these
        d[dim] = StackedCoordinates(cs)

    # make the ND coordinates
    coord_collection = OrderedDict()
    for dims in dims_list:
        if any(dim not in d for dim in dims):
            continue
        coord_collection[dims] = Coordinates([d[dim] for dim in dims])
    return coord_collection
