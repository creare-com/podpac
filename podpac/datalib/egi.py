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
h5py = lazy_module("h5py")

# Internal dependencies
from podpac import Coordinates, Node
from podpac.compositor import OrderedCompositor
from podpac.data import DataSource
from podpac import authentication
from podpac import settings
from podpac.core.units import UnitsDataArray, create_data_array
from podpac.core.node import node_eval

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
    base_url : str, optional
        URL for EGI data endpoint.
        Defaults to :str:`BASE_URL`
        See https://developer.earthdata.nasa.gov/sdps/programmatic-access-docs#egiparameters
    page_size : int, optional
        Number of granules returned from CMR per HTTP call. Defaults to 20.
        See https://developer.earthdata.nasa.gov/sdps/programmatic-access-docs#cmrparameters
    updated_since : str, optional
        Can be used to find granules recently updated in CMR. Optional.
        See https://developer.earthdata.nasa.gov/sdps/programmatic-access-docs#cmrparameters
    version : str, int, optional
        Data product version. Optional.
        Number input will be cast into a 3 character string NNN, i.e. 3 -> "003"
    token : str, optional
        EGI Token from authentication process.
        See https://wiki.earthdata.nasa.gov/display/CMR/Creating+a+Token+Common
        If undefined, the node will look for a token under the setting key "token@EGI".
        If this setting is not defined, the node will attempt to generate a token using
        :attr:`self.username` and :attr:`self.password`
    username : str, optional
        EarthData username (https://urs.earthdata.nasa.gov/)
        If undefined, node will look for a username under setting key "username@urs.earthdata.nasa.gov"
    password : str, optional
        EarthData password (https://urs.earthdata.nasa.gov/)
        If undefined, node will look for a password under setting key "password@urs.earthdata.nasa.gov"

    Attributes
    ----------
    data : :class:`podpac.UnitsDataArray`
        The data array compiled from downloaded EGI data
    """

    base_url = tl.Unicode().tag(attr=True)

    @tl.default("base_url")
    def _base_url_default(self):
        return BASE_URL

    # required
    short_name = tl.Unicode().tag(attr=True)
    data_key = tl.Unicode().tag(attr=True)
    lat_key = tl.Unicode(allow_none=True).tag(attr=True)
    lon_key = tl.Unicode(allow_none=True).tag(attr=True)
    time_key = tl.Unicode(allow_none=True).tag(attr=True)

    min_bounds_span = tl.Dict(allow_none=True).tag(attr=True)

    # optional

    # full list of supported formats ["GeoTIFF", "HDF-EOS5", "NetCDF4-CF", "NetCDF-3", "ASCII", "HDF-EOS", "KML"]
    # response_format = tl.Enum(["HDF-EOS5"], default_value="HDF-EOS5", allow_none=True)
    page_size = tl.Int(default_value=20)
    version = tl.Union(
        [tl.Unicode(default_value=None, allow_none=True), tl.Int(default_value=None, allow_none=True)]
    ).tag(attr=True)

    @tl.validate("version")
    def _version_to_str(self, proposal):
        v = proposal["value"]
        if isinstance(v, int):
            return "{:03d}".format(v)

        if isinstance(v, string_types):
            return v.zfill(3)

        return None

    updated_since = tl.Unicode(default_value=None, allow_none=True)

    # auth
    username = tl.Unicode(allow_none=True)

    @tl.default("username")
    def _username_default(self):
        if "username@EGI" in settings:
            return settings["username@EGI"]

        return None

    password = tl.Unicode(allow_none=True)

    @tl.default("password")
    def _password_default(self):
        if "password@EGI" in settings:
            return settings["password@EGI"]

        return None

    token = tl.Unicode(allow_none=True)

    @tl.default("token")
    def _token_default(self):
        if "token@EGI" in settings:
            return settings["token@EGI"]

        return None

    @property
    def coverage(self):
        return (self.data_key, self.lat_key, self.lon_key)

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

        url = _append(url, "Coverage", ",".join(self.coverage))

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
            return Coordinates.from_xarray(self.data.coords, crs=self.data.attrs["crs"])
        else:
            _log.warning("No coordinates found in EGI source")
            return Coordinates([], dims=[])

    def get_data(self, coordinates, coordinates_index):
        if self.data is not None:
            da = self.data[coordinates_index]
            return da
        else:
            _log.warning("No data found in EGI source")
            return np.array([])

    @node_eval
    def eval(self, coordinates, output=None):
        # download data for coordinate bounds, then handle that data as an H5PY node
        zip_files = self._download(coordinates)
        try:
            self.data = self._read_zips(zip_files)  # reads each file in zip archive and creates single dataarray
        except KeyError as e:
            print("This following error may occur if data_key, lat_key, or lon_key is not correct.")
            print(
                "This error may also occur if the specified area bounds are smaller than the dataset pixel size, in"
                " which case EGI is returning no data."
            )
            raise e
        # Force update on native_coordinates (in case of multiple evals)
        self.native_coordinates = self.get_native_coordinates()

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

        raise NotImplementedError("read_file must be implemented for EGI DataSource")

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

    def _download(self, coordinates):
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

        if "time" in coordinates.udims:
            time_bounds = [
                str(np.datetime64(bound, "s"))
                for bound in coordinates["time"].bounds
                if isinstance(bound, np.datetime64)
            ]
            if self.min_bounds_span != None and "time" in self.min_bounds_span:
                raise NotImplementedError  # TODO

            if len(time_bounds) < 2:
                raise ValueError("Time coordinates must be of type np.datetime64")

        if "lat" in coordinates.udims or "lon" in coordinates.udims:
            lat = coordinates["lat"].bounds
            lon = coordinates["lon"].bounds
            if (self.min_bounds_span != None) and ("lat" in self.min_bounds_span) and ("lon" in self.min_bounds_span):
                latdiff = np.diff(lat)
                londiff = np.diff(lon)
                if latdiff < self.min_bounds_span["lat"]:
                    pad = ((self.min_bounds_span["lat"] - latdiff) / 2)[0]
                    lat = [lat[0] - pad, lat[1] + pad]

                if londiff < self.min_bounds_span["lon"]:
                    pad = ((self.min_bounds_span["lon"] - londiff) / 2)[0]
                    lon = [lon[0] - pad, lon[1] + pad]

            bbox = "{},{},{},{}".format(lon[0], lat[0], lon[1], lat[1])

        # TODO: do we actually want to limit an open query?
        if time_bounds is None and bbox is None:
            raise ValueError("No time or spatial coordinates requested")

        url = self.source

        if time_bounds is not None:
            url += "&time={start_time},{end_time}".format(start_time=time_bounds[0], end_time=time_bounds[1])

        if bbox is not None:
            url += "&Bbox={bbox}".format(bbox=bbox)

        # admin parameters
        url += "&token={token}&page_size={page_size}".format(token=self.token, page_size=self.page_size)
        self._url = url  # for debugging

        # iterate through pages to build up zipfiles containg data
        return list(self._query_egi(url))

    def _query_egi(self, url, page_num=1):
        """Generator for getting zip files from EGI interface
        
        Parameters
        ----------
        url : str
            base url without page_num attached
        page_num : int, optional
            page_num to query
        
        Yields
        ------
        zipfile.ZipFile
            ZipFile of results from page
        
        Raises
        ------
        ValueError
            Raises value error if no granules available from EGI
        """

        # create the full url
        page_url = "{}&page_num={}".format(url, page_num)
        _log.debug("Querying EGI url: {}".format(page_url))
        r = requests.get(page_url)

        if r.status_code != 200:

            # raise exception if the status is not 200 on the first page
            if page_num == 1:
                raise ValueError("Failed to download data from EGI Interface. EGI Reponse: {}".format(r.text))

            # end iteration
            elif r.status_code == 501 and "No granules returned by CMR" in r.text:
                _log.debug("Last page returned from EGI Interface: {}".format(page_num - 1))

            # not sure of response, so end iteration
            else:
                _log.warning("Page returned from EGI Interface with unknown response: {}".format(r.text))

        else:

            # most of the time, EGI returns a zip file
            if ".zip" in r.headers["Content-Disposition"]:
                # load content into file-like object and then read into zip file
                f = BytesIO(r.content)
                zip_file = zipfile.ZipFile(f)

            # if only one file exists, it will return the single file. This puts the single file in a zip archive
            else:
                filename = r.headers["Content-Disposition"].split('filename="')[1].replace('"', "")
                zip_file = zipfile.ZipFile("{}.zip".format(filename), "w")
                zip_file.writestr(filename, r.content)

            # yield the current zip file
            yield zip_file

            try:
                while True:  # broken by StopIteration
                    page_num += 1  # increase page_num
                    yield next(self._query_egi(url, page_num=page_num))
            except StopIteration:
                pass

    def _read_zips(self, zip_files):

        all_data = None
        _log.debug("Processing {} zip files from EGI response".format(len(zip_files)))

        for zip_file in zip_files:
            for name in zip_file.namelist():
                _log.debug("Reading file: {}".format(name))

                # BytesIO
                try:
                    bio = BytesIO(zip_file.read(name))
                except (zipfile.BadZipfile, EOFError) as e:
                    _log.warning(str(e))
                    continue

                # read file
                uda = self.read_file(bio)

                # TODO: this can likely be simpler and automated
                if uda is not None:
                    if all_data is None:
                        all_data = uda.isel(lon=np.isfinite(uda.lon), lat=np.isfinite(uda.lat))
                    else:
                        all_data = self.append_file(all_data, uda)
                else:
                    _log.warning("No data returned from file: {}".format(name))

        return all_data

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
            raise ValueError(
                "Failed to get a valid token from EGI Interface. "
                + "Try requesting a token manually using `self.get_token()`"
            )

        _log.debug("EGI Token valid")

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
        url = "https://cmr.earthdata.nasa.gov/legacy-services/rest/tokens"

        if self.username is not None:
            settings["username@EGI"] = self.username
        else:
            raise ValueError("No EarthData username available to request EGI token")

        if self.password is not None:
            settings["password@EGI"] = self.password
        else:
            raise ValueError("No EarthData password available to request EGI token")

        _ip = self._get_ip()
        request = """
        <token>
            <username>{username}</username>
            <password>{password}</password>
            <client_id>podpac</client_id>
            <user_ip_address>{ip}</user_ip_address>
        </token>
        """.format(
            username=self.username, password=self.password, ip=_ip
        )
        headers = {"Content-Type": "application/xml"}
        r = requests.post(url, data=request, headers=headers)

        try:
            tree = xml.etree.ElementTree.fromstring(r.text)
        except ParseError:
            _log.error("Failed to parse returned text from EGI interface: {}".format(r.text))
            return

        try:
            token = [element.text for element in tree.findall("id")][0]
        except IndexError:
            _log.error("No token found in XML response from EGI: {}".format(r.text))
            return

        settings["token@EGI"] = token
        self.token = token

    def _get_ip(self):
        """
        Utility to return a best guess at the IP address of the local machine.
        Required by EGI authentication to get EGI token.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()

        return ip
