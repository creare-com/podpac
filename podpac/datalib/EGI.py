"""
PODPAC node to access the NASA EGI Programmatic Interface
https://developer.earthdata.nasa.gov/sdps/programmatic-access-docs#overview
"""



import logging
import copy
import zipfile

import requests
from six import string_types
import numpy as np
import xarray as xr
import traitlets as tl

# Set up logging
_logger = logging.getLogger(__name__)

# Internal dependencies
from podpac import Coordinates, Node
from podpac.compositor import OrderedCompositor
from podpac.data import DataSource

# Base URLs
# https://developer.earthdata.nasa.gov/sdps/programmatic-access-docs#egiparameters
BASE_URL = "https://n5eil02u.ecs.nsidc.org/egi/request"


class EGISource(DataSource):
    pass

class EGI(OrderedCompositor):
    """
    PODPAC node to access the NASA EGI Programmatic Interface
    https://developer.earthdata.nasa.gov/sdps/programmatic-access-docs#cmrparameters

    Design:
        - only allow one set of "data layers" (aka coverage)
        - always download geotif since we don't need to know lat/lon keys

    """

    @tl.default('sources')
    def _default_sources(self):
        return np.array([])

    base_url = tl.Unicode().tag(attr=True)
    @tl.default('base_url')
    def _base_url_default(self):
        return BASE_URL

    # required
    short_name = tl.Unicode(allow_none=False).tag(attr=True)

    # optional
    
    # full list of supported formats ["GeoTIFF", "HDF-EOS5", "NetCDF4-CF", "NetCDF-3", "ASCII", "HDF-EOS", "KML"]
    # we still need to know how to interpret results for each format
    response_format = tl.Enum(["GeoTIFF", "HDF-EOS5"], default_value="GeoTIFF", allow_none=True)
    source_type = tl.Instance(DataSource, allow_none=True)

    version = tl.Union([tl.Unicode(default_value=None, allow_none=True), \
                        tl.Int(default_value=None, allow_none=True)]).tag(attr=True)
    coverage = tl.Unicode(default_value=None, allow_none=True)
    updated_since = tl.Unicode(default_value=None, allow_none=True)
    
    # access
    username = tl.Unicode(default_value=None, allow_none=True)
    password = tl.Unicode(default_value=None, allow_none=True)
    token = tl.Unicode(default_value=None, allow_none=True)

    @property
    def source(self):
        """
        URL Endpoint built from input parameters

        Returns
        -------
        str
        """
        url = copy.copy(self.base_url)

        def _append(key, val):
            url += "?{key}={val}".format(key=key, val=val)

        _append("short_name", self.short_name)
        _append("format", self.response_format)

        if self.version:
            _append("version", self.version)

        if self.data_layers:
            _append("Subset_Data_Layers", self.data_layers)
        
        if self.updated_since:
            _append("Updated_since", self.updated_since)
        
        # other parameters are included at eval time

        return url

    def select_sources(self, coordinates):
        """

        """
        pass
