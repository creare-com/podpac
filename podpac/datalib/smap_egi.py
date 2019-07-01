"""
PODPAC Nodes to access SMAP data via EGI Interface
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import os
import copy
import logging
from datetime import datetime

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
import podpac.datalib
from podpac.core.coordinates import Coordinates
from podpac.datalib import EGI
from podpac.core.units import create_data_array

SMAP_PRODUCT_DICT = {
    #'shortname': ['lat_key', 'lon_key', 'data_key', 'default_verison']
    'SPL4SMAU':   ['/x', '/y', '/Analysis_Data/sm_surface_analysis',  4],
    'SPL4SMGP':   ['/x', '/y', '/Geophysical_Data/sm_surface',        4],
    'SPL4SMLM':   ['/x', '/y', '/Land_Model_Constants_Data',          4],
    'SPL3SMAP':   ['/Soil_Moisture_Retrieval_Data/latitude',    '/Soil_Moisture_Retrieval_Data/longitude',    '/Soil_Moisture_Retrieval_Data/soil_moisture', 3],
    'SPL3SMA':    ['/Soil_Moisture_Retrieval_Data/latitude',    '/Soil_Moisture_Retrieval_Data/longitude',    '/Soil_Moisture_Retrieval_Data/soil_moisture', 3],
    'SPL3SMP_AM':    ['/Soil_Moisture_Retrieval_Data_AM/latitude', '/Soil_Moisture_Retrieval_Data_AM/longitude', '/Soil_Moisture_Retrieval_Data_AM/soil_moisture', 5],
    'SPL3SMP_PM':    ['/Soil_Moisture_Retrieval_Data_PM/latitude', '/Soil_Moisture_Retrieval_Data_PM/longitude', '/Soil_Moisture_Retrieval_Data_PM/soil_moisture', 5],
    'SPL3SMP_E_AM':  ['/Soil_Moisture_Retrieval_Data_AM/latitude', '/Soil_Moisture_Retrieval_Data_AM/longitude', '/Soil_Moisture_Retrieval_Data_AM/soil_moisture', 2],
    'SPL3SMP_E_PM':  ['/Soil_Moisture_Retrieval_Data_PM/latitude', '/Soil_Moisture_Retrieval_Data_PM/longitude', '/Soil_Moisture_Retrieval_Data_PM/soil_moisture', 2],
}
SMAP_PRODUCTS = list(SMAP_PRODUCT_DICT.keys())


class SMAP(EGI):
    """
    SMAP interface using the EGI Data Portal
    https://developer.earthdata.nasa.gov/sdps/programmatic-access-docs

    Parameters
    ----------
    product : str
        One of the :list:`SMAP_PRODUCTS` strings

    Attributes
    ----------
    nan_vals : list
        Nan values in SMAP data
    """

    product = tl.Enum(SMAP_PRODUCTS).tag(attr=True)
    nan_vals = [-9999.0]

    # set default short_name, data_key, lat_key, lon_key, version
    @tl.default('short_name')
    def _short_name_default(self):
        if 'SPL3SMP' in self.product:
            return self.product.replace('_AM', '').replace('_PM', '')
        else:
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
        ds = h5py.File(filelike)

        # handle data
        data = ds[self.data_key][()]
        data = np.array([data])  # add extra dimension for time slice

        # handle time
        if 'SPL3' in self.product:
            # TODO: make this py2.7 compatible
            # take the midpoint between the range identified in the file
            t_start = np.datetime64(ds['Metadata/Extent'].attrs['rangeBeginningDateTime'].replace(b'Z', b''))
            t_end = np.datetime64(ds['Metadata/Extent'].attrs['rangeEndingDateTime'].replace(b'Z', b''))
            time = np.array([t_start + (t_end - t_start)/2])
        
        elif 'SPL4' in self.product:
            t_offset = datetime(2000, 1, 1).timestamp()  # all time relative to 2000-01-01
            t_obs = ds['time'][()][0]                    # time give as seconds since 2000-01-01
            time = np.datetime64(datetime.fromtimestamp(t_offset + t_obs))


        # handle spatial coordinates
        if 'SPL3' in self.product:
            # take nan mean along each axis 
            lons = ds[self.lon_key][()]
            lats = ds[self.lat_key][()]
            lons[lons == self.nan_vals[0]] = np.nan
            lats[lats == self.nan_vals[0]] = np.nan
            lon = np.nanmean(lons, axis=0)
            lat = np.nanmean(lats, axis=1)

            # make podpac coordinates
            c = Coordinates([time, lat, lon], dims=['time', 'lat', 'lon'])
        
        elif 'SPL4' in self.product:
            # lat lon coordinates in EPSG:6933 (https://epsg.io/6933)
            lon = ds['x'][()]
            lat = ds['y'][()]

            c = Coordinates([time, lat, lon], dims=['time', 'lat', 'lon'], crs='epsg:6933')

        # Short-circuit
        # if all dims are returned None, we can't just concat along the dims
        # so we need to skip this data (?)
        if np.all(np.isnan(lat)) and np.all(np.isnan(lon)):
            return None

        # make units data array with coordinates and data
        return create_data_array(c, data=data)

    def append_file(self, all_data, data):
        """Append data
        
        Parameters
        ----------
        all_data : podpac.UnitsDataArray
            aggregated data
        data : podpac.UnitsDataArray
            new data to append
        
        Raises
        ------
        NotImplementedError
        """

        all_data = xr.concat([all_data, data], dim='time')

        return all_data
