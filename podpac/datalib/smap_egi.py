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

from podpac.datalib import nasaCMR

# Set up logging
_log = logging.getLogger(__name__)

# Helper utility for optional imports
from lazy_import import lazy_module, lazy_class

h5py = lazy_module("h5py")
lazy_class("h5py.File")

# fixing problem with older versions of numpy
if not hasattr(np, "isnat"):

    def isnat(a):
        return a.astype(str) == "None"

    np.isnat = isnat

# Internal dependencies
from podpac import Coordinates, UnitsDataArray, cached_property
from podpac.datalib import EGI

BASE_URL = "https://n5eil01u.ecs.nsidc.org/egi/request"

SMAP_PRODUCT_DICT = {
    #'shortname':    ['lat_key', 'lon_key', '_data_key', 'quality_flag', 'default_verison']
    "SPL4SMAU": ["/x", "/y", "/Analysis_Data/sm_surface_analysis", None, None],
    "SPL4SMGP": ["/x", "/y", "/Geophysical_Data/sm_surface", None, 4],
    "SPL4SMLM": ["/x", "/y", "/Land_Model_Constants_Data", None, 4],
    "SPL3SMAP": [
        "/Soil_Moisture_Retrieval_Data/latitude",
        "/Soil_Moisture_Retrieval_Data/longitude",
        "/Soil_Moisture_Retrieval_Data/soil_moisture",
        "/Soil_Moisture_Retrieval_Data/retrieval_qual_flag",
        "003",
    ],
    "SPL3SMA": [
        "/Soil_Moisture_Retrieval_Data/latitude",
        "/Soil_Moisture_Retrieval_Data/longitude",
        "/Soil_Moisture_Retrieval_Data/soil_moisture",
        "/Soil_Moisture_Retrieval_Data/retrieval_qual_flag",
        "003",
    ],
    "SPL3SMP_AM": [
        "/Soil_Moisture_Retrieval_Data_AM/latitude",
        "/Soil_Moisture_Retrieval_Data_AM/longitude",
        "/Soil_Moisture_Retrieval_Data_AM/soil_moisture",
        "/Soil_Moisture_Retrieval_Data_AM/retrieval_qual_flag",
        "005",
    ],
    "SPL3SMP_PM": [
        "/Soil_Moisture_Retrieval_Data_PM/latitude",
        "/Soil_Moisture_Retrieval_Data_PM/longitude",
        "/Soil_Moisture_Retrieval_Data_PM/soil_moisture_pm",
        "/Soil_Moisture_Retrieval_Data_PM/retrieval_qual_flag_pm",
        "005",
    ],
    "SPL3SMP_E_AM": [
        "/Soil_Moisture_Retrieval_Data_AM/latitude",
        "/Soil_Moisture_Retrieval_Data_AM/longitude",
        "/Soil_Moisture_Retrieval_Data_AM/soil_moisture",
        "/Soil_Moisture_Retrieval_Data_AM/retrieval_qual_flag",
        "004",
    ],
    "SPL3SMP_E_PM": [
        "/Soil_Moisture_Retrieval_Data_PM/latitude_pm",
        "/Soil_Moisture_Retrieval_Data_PM/longitude_pm",
        "/Soil_Moisture_Retrieval_Data_PM/soil_moisture_pm",
        "/Soil_Moisture_Retrieval_Data_PM/retrieval_qual_flag_pm",
        "004",
    ],
}

SMAP_PRODUCTS = list(SMAP_PRODUCT_DICT.keys())


class SMAP(EGI):
    """
    SMAP Node. For more information about SMAP, see https://nsidc.org/data/smap

    SMAP interface using the EGI Data Portal
    https://developer.earthdata.nasa.gov/sdps/programmatic-access-docs
    with the base URL: https://n5eil01u.ecs.nsidc.org/egi/request

    To access data from this node, an Earthdata login is required. This can either be specified when
    creating the node:
    ```python
    smap = SMAP(username="your_user_name", password="your_password")
    ```
    OR you can set the following PODPAC settings:
    ```python
    podpac.settings["username@urs.earthdata.nasa.gov"] = "your_user_name"
    podpac.settings["password@urs.earthdata.nasa.gov"] = "your_password"
    podpac.settings.save()  # To have this information persist
    smap = SMAP()
    ```

    Parameters
    ----------
    product : str
        One of the :list:`SMAP_PRODUCTS` strings
    check_quality_flags : bool, optional
        Default is True. If True, data will be filtered based on the SMAP data quality flag, and only
        high quality data is returned.
    data_key : str, optional
        Default will return soil moisture and is set automatically based on the product selected. Other
        possible data keys can be found

    Attributes
    ----------
    nan_vals : list
        Nan values in SMAP data
    username : str, optional
        Earthdata username (https://urs.earthdata.nasa.gov/)
        If undefined, node will look for a username under setting key "username@urs.earthdata.nasa.gov"
    password : str, optional
        Earthdata password (https://urs.earthdata.nasa.gov/)
        If undefined, node will look for a password under setting key "password@urs.earthdata.nasa.gov"
    """

    product = tl.Enum(SMAP_PRODUCTS, default_value="SPL4SMAU").tag(attr=True)
    nan_vals = [-9999.0]
    min_bounds_span = tl.Dict(default_value={"lon": 0.3, "lat": 0.3, "time": "3,h"}).tag(attr=True)
    check_quality_flags = tl.Bool(True).tag(attr=True, default=True)
    quality_flag_key = tl.Unicode(allow_none=True).tag(attr=True)
    data_key = tl.Unicode(allow_none=True, default_value=None).tag(attr=True)
    base_url = tl.Unicode(default_value=BASE_URL).tag(attr=True)

    @property
    def short_name(self):
        if "SPL3SMP" in self.product:
            return self.product.replace("_AM", "").replace("_PM", "")
        else:
            return self.product

    # pull _data_key, lat_key, lon_key, and version from product dict
    @cached_property
    def _product_data(self):
        return SMAP_PRODUCT_DICT[self.product]

    @property
    def udims(self):
        return ["lat", "lon", "time"]

    @property
    def lat_key(self):
        return self._product_data[0]

    @property
    def lon_key(self):
        return self._product_data[1]

    @property
    def _data_key(self):
        if self.data_key is None:
            return self._product_data[2]
        else:
            return self.data_key

    @property
    def quality_flag_key(self):
        return self._product_data[3]

    @property
    def version(self):
        try:
            return nasaCMR.get_collection_entries(short_name=self.product)[-1]["version_id"]
        except:
            _log.warning("Could not automatically retrieve newest product version id from NASA CMR.")
            return self._product_data[4]

    @property
    def coverage(self):
        if self.quality_flag_key:
            return (self._data_key, self.quality_flag_key, self.lat_key, self.lon_key)
        else:
            return (self._data_key, self.lat_key, self.lon_key)

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
        ds = h5py.File(filelike, "r")

        # handle data
        data = ds[self._data_key][()]

        if self.check_quality_flags and self.quality_flag_key:
            flag = ds[self.quality_flag_key][()]
            flag = flag > 0
            [flag] == np.nan

        data = np.array([data])  # add extra dimension for time slice

        # handle time
        if "SPL3" in self.product:
            # TODO: make this py2.7 compatible
            # take the midpoint between the range identified in the file
            t_start = np.datetime64(ds["Metadata/Extent"].attrs["rangeBeginningDateTime"].replace("Z", ""))
            t_end = np.datetime64(ds["Metadata/Extent"].attrs["rangeEndingDateTime"].replace("Z", ""))
            time = np.array([t_start + (t_end - t_start) / 2])
            time = time.astype("datetime64[D]")

        elif "SPL4" in self.product:
            time_unit = ds["time"].attrs["units"].decode()
            time = xr.coding.times.decode_cf_datetime(ds["time"][()][0], units=time_unit)
            time = time.astype("datetime64[h]")

        # handle spatial coordinates
        if "SPL3" in self.product:

            # take nan mean along each axis
            lons = ds[self.lon_key][()]
            lats = ds[self.lat_key][()]
            lons[lons == self.nan_vals[0]] = np.nan
            lats[lats == self.nan_vals[0]] = np.nan

            # short-circuit if all lat/lon are non
            if np.all(np.isnan(lats)) and np.all(np.isnan(lons)):
                return None

            # make podpac coordinates
            lon = np.nanmean(lons, axis=0)
            lat = np.nanmean(lats, axis=1)
            c = Coordinates([time, lat, lon], dims=["time", "lat", "lon"])

        elif "SPL4" in self.product:
            # lat/lon coordinates in EPSG:6933 (https://epsg.io/6933)
            lon = ds["x"][()]
            lat = ds["y"][()]

            # short-circuit if all lat/lon are nan
            if np.all(np.isnan(lat)) and np.all(np.isnan(lon)):
                return None

            c = Coordinates([time, lat, lon], dims=["time", "lat", "lon"], crs="epsg:6933")

        # make units data array with coordinates and data
        return UnitsDataArray.create(c, data=data)

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
        if all_data.shape[1:] == data.shape[1:]:
            data.lat.data[:] = all_data.lat.data
            data.lon.data[:] = all_data.lon.data
        else:
            # select only data with finite coordinates
            data = data.isel(lon=np.isfinite(data.lon), lat=np.isfinite(data.lat))

            # select lat based on the old data
            lat = all_data.lat.sel(lat=data.lat, method="nearest")

            # When the difference between old and new coordintaes are large, it means there are new coordinates
            Ilat = (np.abs(lat.data - data.lat) > 1e-3).data
            # Use the new data's coordinates for the new coordinates
            lat.data[Ilat] = data.lat.data[Ilat]

            # Repeat for lon
            lon = all_data.lon.sel(lon=data.lon, method="nearest")
            Ilon = (np.abs(lon.data - data.lon) > 1e-3).data
            lon.data[Ilon] = data.lon.data[Ilon]

            # Assign to data
            data.lon.data[:] = lon.data
            data.lat.data[:] = lat.data

        return all_data.combine_first(data)
