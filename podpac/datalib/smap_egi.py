"""
PODPAC Nodes to access SMAP data via EGI Interface
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import os
import copy
import logging

import requests
from six import string_types
import numpy as np
import xarray as xr
import traitlets as tl

# Set up logging
_log = logging.getLogger(__name__)

# Helper utility for optional imports
from lazy_import import lazy_module
h5py = lazy_module('h5py')

# fixing problem with older versions of numpy
if not hasattr(np, 'isnat'):
    def isnat(a):
        return a.astype(str) == 'None'
    np.isnat = isnat

# Internal dependencies
import podpac
from podpac.core.coordinates import Coordinates
from podpac.datalib import EGI
from podpac.core.units import UnitsDataArray, create_data_array

SMAP_PRODUCT_DICT = {
    #'shortname': ['lat_key', 'lon_key', 'data_key', 'default_verison']
    'SPL4SMAU':   ['cell_lat', 'cell_lon', '/Analysis_Data/sm_surface_analysis',  4],
    'SPL4SMGP':   ['cell_lat', 'cell_lon', '/Geophysical_Data/sm_surface',        4],
    'SPL4SMLM':   ['cell_lat', 'cell_lon', '/Land_Model_Constants_Data',          4],
    'SPL3SMAP':   ['/Soil_Moisture_Retrieval_Data/latitude',    '/Soil_Moisture_Retrieval_Data/longitude',    '/Soil_Moisture_Retrieval_Data/soil_moisture', 3],
    'SPL3SMA':    ['/Soil_Moisture_Retrieval_Data/latitude',    '/Soil_Moisture_Retrieval_Data/longitude',    '/Soil_Moisture_Retrieval_Data/soil_moisture', 3],
    'SPL3SMP':    ['/Soil_Moisture_Retrieval_Data/AM_latitude', '/Soil_Moisture_Retrieval_Data/AM_longitude', '/Soil_Moisture_Retrieval_Data/soil_moisture', 5],
}
SMAP_PRODUCTS = list(SMAP_PRODUCT_DICT.keys())


class SMAP(EGI):
    """
    """

    product = tl.Enum(SMAP_PRODUCTS).tag(attr=True)
    nan_vals = [-9999.0]

    # set default short_name, data_key, lat_key, lon_key, version
    @tl.default('short_name')
    def _short_name_default(self):
        return self.product

    @tl.default('lat_key')
    def _lat_key_default(self):
        return SMAP_PRODUCT_DICT[self.product][0]

    @tl.default('lon_key')
    def _lon_key_default(self):
        return SMAP_PRODUCT_DICT[self.product][1]

    @tl.default('data_key')
    def _data_key_default(self):
        return SMAP_PRODUCT_DICT[self.product][2]

    @tl.default('version')
    def _version_default(self):
        return SMAP_PRODUCT_DICT[self.product][3]

    def read_file(self, filelike):
        """Interpret individual SMAP file from  EGI zip archive.
        
        Parameters
        ----------
        filelike : filelike
            Reference to file inside EGI zip archive
        
        Returns
        -------
        podpac.UnitsDataArray
        
        Raises
        ------
        ValueError
        """
        hdf5_file = h5py.File(filelike)

        # handle data
        data = hdf5_file[self.data_key][()]
        data = np.array([data])  # add extra dimension for time slice

        # handle time
        if self.product in ['SPL3SMA', 'SPL3SMAP']:
            # handle time (Not py2.7 compatible)
            # take the midpoint between the range identified in the file
            t_start = np.datetime64(hdf5_file['Metadata/Extent'].attrs['rangeBeginningDateTime'].replace(b'Z', b''))
            t_end = np.datetime64(hdf5_file['Metadata/Extent'].attrs['rangeEndingDateTime'].replace(b'Z', b''))
            time = np.array([t_start + (t_end - t_start)/2])

        # handle coordinates
        # take nan mean along each axis 
        lons = hdf5_file[self.lon_key][()]
        lats = hdf5_file[self.lat_key][()]
        lons[lons == self.nan_vals[0]] = np.nan
        lats[lats == self.nan_vals[0]] = np.nan
        lon = np.nanmean(lons, axis=0)
        lat = np.nanmean(lats, axis=1)

        # if all dims are returned None, we can't just along the dims
        # so we need to skip this data (?)
        if np.all(np.isnan(lat)) and np.all(np.isnan(lon)):
            return None

        return (data, lat, lon, time)
        # # make podpac coordinates
        # c = Coordinates([time, lat, lon], dims=['time', 'lat', 'lon'])

        # # make units data array with coordinates and data
        # return create_data_array(c, data=data)
