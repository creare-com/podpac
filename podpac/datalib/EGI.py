"""
PODPAC node to access the NASA EGI Programmatic Interface
https://developer.earthdata.nasa.gov/sdps/programmatic-access-docs#overview
"""


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

# Set up logging
log = logging.getLogger(__name__)

# Internal dependencies
from podpac import Coordinates, Node
from podpac.compositor import OrderedCompositor
from podpac.data import DataSource
from podpac import authentication
from podpac import settings

# Base URLs
# https://developer.earthdata.nasa.gov/sdps/programmatic-access-docs#egiparameters
BASE_URL = "https://n5eil02u.ecs.nsidc.org/egi/request"



class EGI(DataSource):
    """
    PODPAC DataSource node to access the NASA EGI Programmatic Interface
    https://developer.earthdata.nasa.gov/sdps/programmatic-access-docs#cmrparameters
    
    Design
    ------
    - only allow one set of "data layers" (aka coverage)
    - always download geotif since we don't need to know lat/lon keys
    
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

    """

    base_url = tl.Unicode().tag(attr=True)
    @tl.default('base_url')
    def _base_url_default(self):
        return BASE_URL

    # required
    short_name = tl.Unicode().tag(attr=True)
    data_key = tl.Unicode().tag(attr=True)
    lat_key = tl.Unicode().tag(attr=True)
    lon_key = tl.Unicode().tag(attr=True)

    # optional
    
    # full list of supported formats ["GeoTIFF", "HDF-EOS5", "NetCDF4-CF", "NetCDF-3", "ASCII", "HDF-EOS", "KML"]
    # response_format = tl.Enum(["HDF-EOS5"], default_value="HDF-EOS5", allow_none=True)

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
    username = tl.Unicode(default_value=None, allow_none=True)
    password = tl.Unicode(default_value=None, allow_none=True)
    token = tl.Unicode(allow_none=True)
    @tl.default('token')
    def _token_default(self):
        if 'token@EGI' in settings:
            return settings['token@EGI']

        return None

    @property
    def source(self):
        """
        URL Endpoint built from input parameters

        Returns
        -------
        str
        """
        url = copy.copy(self.base_url)

        def _append(u, key, val):
            u += "?{key}={val}".format(key=key, val=val)
            return u

        url = _append(url, "short_name", self.short_name)

        # Format could be customized - see response_format above
        # For now we set to HDF5
        # url = _append(url, "format", self.response_format)
        url = _append(url, "format", "HDF-EOS5")

        if self.version:
            url = _append(url, "version", self.version)

        if self.data_key:
            url = _append(url, "Subset_Data_Layers", self.data_key)
        
        if self.updated_since:
            url = _append(url, "Updated_since", self.updated_since)
        
        # other parameters are included at eval time
        return url


    def get_native_coordinates(self):
        pass


    def get_data(self, coordinates, coordinates_index):
        self._authenticate()


    def _authenticate(self):
        if self.token is None:
            self.get_token()

        if not self.token_valid():
            raise ValueError("EGI Token Invalid")
        

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

        # use EathDataSession auth class to handle auth parameters and settings
        _session = authentication.EarthDataSession(username=self.username, password=self.password)
        
        if _session.username is None:
            raise ValueError('No EarthData username available to request EGI token')

        if _session.password is None:
            raise ValueError('No EarthData password available to request EGI token')

        _ip = self._get_ip()
        request = """
        <token>
            <username>{username}</username>
            <password>{password}</password>
            <client_id>podpac</client_id>
            <user_ip_address>{ip}</user_ip_address>
        </token>
        """.format(username=_session.username, password=_session.password, ip=_ip)
        headers = {"Content-Type": "application/xml"}
        r = requests.post(url, data=request, headers=headers)

        try:
            tree = xml.etree.ElementTree.fromstring(r.text)
        except ParseError:
            log.error('Failed to parse returned text from EGI interface: {}'.format(r.text))
            return

        try:
            token = [element.text for element in tree.findall('id')][0]
        except IndexError:
            log.error('No token found in XML response from EGI: {}'.format(r.text))
            return

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
