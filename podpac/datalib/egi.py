"""
PODPAC node to access the NASA EGI Programmatic Interface
https://developer.earthdata.nasa.gov/sdps/programmatic-access-docs#overview
"""


import os
from io import BytesIO
import socket
import logging
import copy
import zipfile
import xml.etree.ElementTree
from xml.etree.ElementTree import ParseError

import requests
from six import string_types
import numpy as np
import xarray as xr
import traitlets as tl
from lazy_import import lazy_module

# optional imports
h5py = lazy_module('h5py')

# Internal dependencies
from podpac import Coordinates, Node
from podpac.compositor import OrderedCompositor
from podpac.data import DataSource
from podpac import authentication
from podpac import settings
from podpac.core.units import UnitsDataArray, create_data_array

# Set up logging
_log = logging.getLogger(__name__)


# Base URLs
# https://developer.earthdata.nasa.gov/sdps/programmatic-access-docs#egiparameters
BASE_URL = "https://n5eil01u.ecs.nsidc.org/egi/request"


class EGI(DataSource):
    """
    PODPAC DataSource node to access the NASA EGI Programmatic Interface
    https://developer.earthdata.nasa.gov/sdps/programmatic-access-docs#cmrparameters
    
    Parameters
    ----------
    short_name : str
        Specifies the short name of the collection used to find granules for the coverage requested. Required.
        See https://developer.earthdata.nasa.gov/sdps/programmatic-access-docs#cmrparameters
    data_key : str
        Path to the subset data layer or group for Parameter Subsetting. Required.
        Equivalent to "Coverage" paramter described in
        https://developer.earthdata.nasa.gov/sdps/programmatic-access-docs#cmrparameters
    lat_key : str
        Key for latitude data in endpoint HDF-5 file. Required.
    lon_key : str
        Key for longitude data in endpoint HDF-5 file. Required.
    base_url : str
        URL for EGI data endpoint. Optional.
        Defaults to :str:`BASE_URL`
        See https://developer.earthdata.nasa.gov/sdps/programmatic-access-docs#egiparameters
    updated_since : str
        Can be used to find granules recently updated in CMR. Optional.
        See https://developer.earthdata.nasa.gov/sdps/programmatic-access-docs#cmrparameters
    version : str, int
        Data product version. Optional.
        Number input will be cast into a 3 character string NNN, i.e. 3 -> "003"
    download_files : bool
        NOT YET IMPLEMENTED
        If True, Download response data as files to cache.
        If False, PODPAC will attempt to handle all response data in memory.
        Set to True for large time/spatial queries
    token : str
        EGI Token from authentication process.
        See https://wiki.earthdata.nasa.gov/display/CMR/Creating+a+Token+Common
        If undefined, the node will look for a token under the setting key "token@EGI".
        If this setting is not defined, the node will attempt to generate a token using
        :attr:`self.username` and :attr:`self.password`
    username : str
        EarthData username (https://urs.earthdata.nasa.gov/)
        If undefined, node will look for a username under setting key "username@urs.earthdata.nasa.gov"
    password : str
        EarthData password (https://urs.earthdata.nasa.gov/)
        If undefined, node will look for a password under setting key "password@urs.earthdata.nasa.gov"

    Attributes
    ----------
    data : :class:`podpac.UnitsDataArray`
        The data array compiled from downloaded EGI data
    """

    base_url = tl.Unicode().tag(attr=True)
    @tl.default('base_url')
    def _base_url_default(self):
        return BASE_URL

    # required
    short_name = tl.Unicode().tag(attr=True)
    data_key = tl.Unicode().tag(attr=True)
    lat_key = tl.Unicode(allow_none=True).tag(attr=True)
    lon_key = tl.Unicode(allow_none=True).tag(attr=True)
    time_key = tl.Unicode(allow_none=True).tag(attr=True)

    # optional
    
    # full list of supported formats ["GeoTIFF", "HDF-EOS5", "NetCDF4-CF", "NetCDF-3", "ASCII", "HDF-EOS", "KML"]
    # response_format = tl.Enum(["HDF-EOS5"], default_value="HDF-EOS5", allow_none=True)
    # download_files = tl.Bool(default_value=False)
    version = tl.Union([tl.Unicode(default_value=None, allow_none=True), \
                        tl.Int(default_value=None, allow_none=True)]).tag(attr=True)
    @tl.validate('version')
    def _version_to_str(self, proposal):
        v = proposal['value']
        if isinstance(v, int):
            return '{:03d}'.format(v)
        
        if isinstance(v, string_types):
            return v.zfill(3)

        return None

    updated_since = tl.Unicode(default_value=None, allow_none=True)
    
    # auth
    username = tl.Unicode(allow_none=True)
    @tl.default('username')
    def _username_default(self):
        if 'username@EGI' in settings:
            return settings['username@EGI']

        return None

    password = tl.Unicode(allow_none=True)
    @tl.default('password')
    def _password_default(self):
        if 'password@EGI' in settings:
            return settings['password@EGI']

        return None

    token = tl.Unicode(allow_none=True)
    @tl.default('token')
    def _token_default(self):
        if 'token@EGI' in settings:
            return settings['token@EGI']

        return None

    # attributes
    data = tl.Any(allow_none=True)
    _url = tl.Unicode(allow_none=True)

    @property
    def source(self):
        """
        URL Endpoint built from input parameters

        Returns
        -------
        str
        """
        url = copy.copy(self.base_url)
        url += "?short_name={}".format(self.short_name)

        def _append(u, key, val):
            u += "&{key}={val}".format(key=key, val=val)
            return u

        url = _append(url, "Coverage", "{},{},{}".format(self.data_key, self.lat_key, self.lon_key))

        # Format could be customized - see response_format above
        # For now we set to HDF5
        # url = _append(url, "format", self.response_format)
        url = _append(url, "format", "HDF-EOS")

        if self.version:
            url = _append(url, "version", self.version)
        
        if self.updated_since:
            url = _append(url, "Updated_since", self.updated_since)
        
        # other parameters are included at eval time
        return url

    def get_native_coordinates(self):
        if self.data is not None:
            return Coordinates.from_xarray(self.data.coords, crs=self.data.attrs['crs'])
        else:
            _log.warning('No coordinates found in EGI source')
            return Coordinates([], dims=[])

    def get_data(self, coordinates, coordinates_index):
        if self.data is not None:
            da = self.data[coordinates_index]
            return da
        else:
            _log.warning('No data found in EGI source')
            return np.array([])

    def eval(self, coordinates, output=None):
        # download data for coordinate bounds, then handle that data as an H5PY node
        zip_file = self._download_zip(coordinates)
        self._read_zip(zip_file)  # reads each file in zip archive and creates single dataarray

        # run normal eval once self.data is prepared
        return super(EGI, self).eval(coordinates, output)

    ##########
    # Data I/O
    ##########
    def read_file(self, filelike):
        """Interpret individual file from  EGI zip archive.
        
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

        raise NotImplementedError('read_file must be implemented for EGI DataSource')
        
        ## TODO: implement generic handler based on keys and dimensions

        # # load file
        # hdf5_file = h5py.File(filelike)

        # # handle data
        # data = hdf5_file[self.data_key]
        # lat = hdf5_file[self.lat_key] if self.lat_key in hdf5_file else None
        # lon = hdf5_file[self.lon_key] if self.lon_key in hdf5_file else None
        # time = hdf5_file[self.time_key] if self.time_key in hdf5_file else None

        # # stacked coords
        # if data.ndim == 2:
        #     c = Coordinates([(lat, lon), time], dims=['lat_lon', 'time'])

        # # gridded coords
        # elif data.ndim == 3:
        #     c = Coordinates([lat, lon, time], dims=['lat', 'lon', 'time'])
        # else:
        #     raise ValueError('Data must have either 2 or 3 dimensions')

    def append_file(self, all_data, data):
        """Append new data
        
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
        raise NotImplementedError()

    def _download_zip(self, coordinates):
        """
        Download data from EGI Interface within PODPAC coordinates
        
        Parameters
        ----------
        coordinates : :class:`podpac.Coordinates`
            PODPAC coordinates specifying spatial and temporal bounds
        
        Raises
        ------
        ValueError
            Error raised when no spatial or temporal bounds are provided
        
        Returns
        -------
        zipfile.ZipFile
            Returns zip file byte-str to downloaded data
        """
        self._authenticate()

        time_bounds = None
        bbox = None

        if 'time' in coordinates.udims:
            time_bounds = [str(np.datetime64(bound, 's')) for bound in \
                           coordinates['time'].bounds if isinstance(bound, np.datetime64)]
            if len(time_bounds) < 2:
                raise ValueError("Time coordinates must be of type np.datetime64")

        if 'lat' in coordinates.udims or 'lon' in coordinates.udims:
            lat = coordinates['lat'].bounds
            lon = coordinates['lon'].bounds
            bbox = "{},{},{},{}".format(lon[0], lat[0], lon[1], lat[1])

        # TODO: do we actually want to limit an open query?
        if time_bounds is None and bbox is None:
            raise ValueError("No time or spatial coordinates requested")

        url = self.source

        if time_bounds is not None:
            url += "&time={start_time},{end_time}".format(start_time=time_bounds[0], end_time=time_bounds[1])

        if bbox is not None:
            url += "&Bbox={bbox}".format(bbox=bbox)

        url += "&token={token}".format(token=self.token)

        _log.debug("Querying EGI url: {}".format(url))
        self._url = url         # for debugging
        r = requests.get(url)

        if r.status_code != 200:
            raise ValueError("Failed to download data from EGI Interface. EGI Reponse: {}".format(r.content))

        # load content into file-like object and then read into zip file
        f = BytesIO(r.content)
        zip_file = zipfile.ZipFile(f)

        return zip_file

    def _read_zip(self, zip_file):

        all_data = None
        _log.debug("Processing {} EGI files from zip archive".format(len(zip_file.namelist())))

        for name in zip_file.namelist():
            _log.debug("Reading file: {}".format(name))
            
            # BytesIO
            bio = BytesIO(zip_file.read(name))

            # read file
            uda = self.read_file(bio)

            # TODO: this can likely be simpler and automated
            if uda is not None:
                if all_data is None:
                    all_data = uda
                else:
                    # self.data = xr.combine_by_coords([self.data, uda], data_vars='minimal', coords='all')
                    all_data = self.append_file(all_data, uda)

            else:
                _log.warning('No data returned from file: {}'.format(name))

        self.data = all_data

    ################
    # Token Handling
    ################
    def _authenticate(self):
        if self.token is None:
            self.get_token()

        # if token's not valid, try getting a new token
        if not self.token_valid():
            self.get_token()

        # if token is still not valid, throw error
        if not self.token_valid():
            raise ValueError("Failed to get a valid token from EGI Interface. " + \
                             "Try requesting a token manually using `self.get_token()`")

        _log.debug('EGI Token valid')

    def token_valid(self):
        """
        Validate EGI token set in :attr:`token` attribute of EGI Node
        
        Returns
        -------
        Bool
            True if token is valid, False if token is invalid
        """
        r = requests.get("{base_url}?token={token}".format(base_url=self.base_url, token=self.token))

        return r.status_code != 401
        
    def get_token(self):
        """
        Get token for EGI interface using EarthData credentials
        
        Returns
        -------
        str
            Token for access to EGI interface
        
        Raises
        ------
        ValueError
            Raised if EarthData username or password is unavailable
        """
        # token access URL
        url = 'https://cmr.earthdata.nasa.gov/legacy-services/rest/tokens'

        if self.username is not None:
            settings['username@EGI'] = self.username
        else:
            raise ValueError('No EarthData username available to request EGI token')

        if self.password is not None:
            settings['password@EGI'] = self.password
        else:
            raise ValueError('No EarthData password available to request EGI token')

        _ip = self._get_ip()
        request = """
        <token>
            <username>{username}</username>
            <password>{password}</password>
            <client_id>podpac</client_id>
            <user_ip_address>{ip}</user_ip_address>
        </token>
        """.format(username=self.username, password=self.password, ip=_ip)
        headers = {"Content-Type": "application/xml"}
        r = requests.post(url, data=request, headers=headers)

        try:
            tree = xml.etree.ElementTree.fromstring(r.text)
        except ParseError:
            _log.error('Failed to parse returned text from EGI interface: {}'.format(r.text))
            return

        try:
            token = [element.text for element in tree.findall('id')][0]
        except IndexError:
            _log.error('No token found in XML response from EGI: {}'.format(r.text))
            return

        settings['token@EGI'] = token
        self.token = token

    def _get_ip(self):
        """
        Utility to return a best guess at the IP address of the local machine.
        Required by EGI authentication to get EGI token.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()

        return ip

