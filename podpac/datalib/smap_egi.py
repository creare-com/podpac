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
    'SPL3SMA':    ['/Soil_Moisture_Retrieval_Data/latitude',    '/Soil_Moisture_Retrieval_Data/longitude',    '/Soil_Moisture_Retrieval_Data/soil_moisture', 3],
    'SPL3SMP':    ['/Soil_Moisture_Retrieval_Data/AM_latitude', '/Soil_Moisture_Retrieval_Data/AM_longitude', '/Soil_Moisture_Retrieval_Data/soil_moisture', 5],
}
SMAP_PRODUCTS = list(SMAP_PRODUCT_DICT.keys())


class SMAP(EGI):
    """
    """

    product = tl.Enum(SMAP_PRODUCTS).tag(attr=True)
    nan_vals = [-9999.0]

    # set default data_key, lat_key, lon_key, version
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

    def _read_file(self):

    def merge_files(self, files):
        """Interpret SMAP files from EGI zip archive.

        Parameters
        ----------
        files : list of h5py.Group
            list of h5py files read from zip archive to merge
        """

        for f in files:

            # handle data
            data = hdf5_file[self.data_key]
            data = np.array([data])  # add extra dimension for time slice

        # handle time
        if self.product in ['SPL3SMA']:
            # handle time (Not py2.7 compatible)
            # take the midpoint between the range identified in the file
            t_start = np.datetime64(hdf5_file['Metadata/Extent'].attrs['rangeBeginningDateTime'].replace(b'Z', b''))
            t_end = np.datetime64(hdf5_file['Metadata/Extent'].attrs['rangeEndingDateTime'].replace(b'Z', b''))
            time = np.array([t_start + (t_end - t_start)/2])


        # handle coordinates
        # take nan mean along each axis 
        lons = np.array(hdf5_file[self.lat_key][:, :])
        lats = np.array(hdf5_file[self.lon_key][:, :])
        lons[lons == self.nan_vals[0]] = np.nan
        lats[lats == self.nan_vals[0]] = np.nan
        lon = np.nanmean(lons, axis=0)
        lat = np.nanmean(lats, axis=1)


        f_data, f_lat, f_lon, f_time = self.read_file(bio)   # ND, 1D, 1D, 1D

        # initialize
        if data is None:
            data = f_data
            lat = f_lat
            lon = f_lon
            time = f_time

            continue

        # TODO: We are assuming the lat/lon will not change between files of the same type
        # this is likely a bad assumption, but will handle in the future
        # lat/lon may either be gridded or stacked
        if not np.all(lat == f_lat):
            lat = np.nanmean([lat, f_lat], axis=0)

        if not np.all(lon == f_lon):
            lon = np.nanmean([lat, f_lat], axis=0)

        # if not np.all(lat == f_lat) or not np.all(lon == f_lon):
        #     raise ValueError('Coordinates vary between individual data files in EGI zip archive')

        # concatenate all data with new data @ time slice
        data = np.concatenate([data, f_data])
        time = np.concatenate([time, f_time])

        # stacked coords
        if data.ndim == 2:
            c = Coordinates([(lat, lon), time], dims=['lat_lon', 'time'])

        # gridded coords
        elif data.ndim == 3:
            c = Coordinates([lat, lon, time], dims=['lat', 'lon', 'time'])
        else:
            raise ValueError('Data must have either 2 or 3 dimensions')
        
        self.data = create_data_array(c, data=data)
