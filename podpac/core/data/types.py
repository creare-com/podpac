"""
Type Summary

Attributes
----------
WCS_DEFAULT_CRS : str
    Description
WCS_DEFAULT_VERSION : str
    Description
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import warnings
import os
import re
from io import BytesIO
from collections import OrderedDict, defaultdict
from six import string_types
import logging

import numpy as np
import traitlets as tl
import pandas as pd  # Core dependency of xarray
import xarray as xr

# Helper utility for optional imports
from lazy_import import lazy_module, lazy_class

# Internal dependencies
from podpac.core import authentication
from podpac.core.node import Node
from podpac.core.settings import settings
from podpac.core.utils import common_doc, trait_is_defined, ArrayTrait, NodeTrait
from podpac.core.data.datasource import COMMON_DATA_DOC, DataSource
from podpac.core.coordinates import Coordinates, UniformCoordinates1d, ArrayCoordinates1d, StackedCoordinates
from podpac.core.coordinates.utils import Dimension
from podpac.core.algorithm.algorithm import Algorithm
from podpac.core.data.interpolation import interpolation_trait

# Optional dependencies
bs4 = lazy_module("bs4")
# Not used directly, but used indirectly by bs4 so want to check if it's available
lxml = lazy_module("lxml")
pydap = lazy_module("pydap")
lazy_module("pydap.client")
lazy_module("pydap.model")
rasterio = lazy_module("rasterio")
h5py = lazy_module("h5py")
boto3 = lazy_module("boto3")
requests = lazy_module("requests")
zarr = lazy_module("zarr")
zarrGroup = lazy_class("zarr.Group")
s3fs = lazy_module("s3fs")
# esri
RasterToNumPyArray = lazy_module("arcpy.RasterToNumPyArray")
urllib3 = lazy_module("urllib3")
certifi = lazy_module("certifi")

# Set up logging
_logger = logging.getLogger(__name__)


class Array(DataSource):
    """Create a DataSource from an array
    
    Attributes
    ----------
    source : np.ndarray
        Numpy array containing the source data
        
    Notes
    ------
    `native_coordinates` need to supplied by the user when instantiating this node.
    """

    source = ArrayTrait().tag(readonly=True)

    @tl.validate("source")
    def _validate_source(self, d):
        a = d["value"]
        try:
            a.astype(float)
        except:
            raise ValueError("Array source must be numerical")
        return a

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        s = coordinates_index
        d = self.create_output_array(coordinates, data=self.source[s])
        return d


@common_doc(COMMON_DATA_DOC)
class PyDAP(DataSource):
    """Create a DataSource from an OpenDAP server feed.
    
    Attributes
    ----------
    auth_class : :class:`podpac.authentication.Session`
        :class:`requests.Session` derived class providing authentication credentials.
        When username and password are provided, an auth_session is created using this class.
    auth_session : :class:`podpac.authentication.Session`
        Instance of the auth_class. This is created if username and password is supplied, but this object can also be
        supplied directly
    datakey : str
        Pydap 'key' for the data to be retrieved from the server. Datasource may have multiple keys, so this key
        determines which variable is returned from the source.
    dataset : pydap.model.DatasetType
        The open pydap dataset. This is provided for troubleshooting.
    native_coordinates : Coordinates
        {native_coordinates}
    password : str, optional
        Password used for authenticating against OpenDAP server. WARNING: this is stored as plain-text, provide
        auth_session instead if you have security concerns.
    source : str
        URL of the OpenDAP server.
    username : str, optional
        Username used for authenticating against OpenDAP server. WARNING: this is stored as plain-text, provide
        auth_session instead if you have security concerns.
    """

    source = tl.Unicode().tag(readonly=True)
    dataset = tl.Instance("pydap.model.DatasetType").tag(readonly=True)

    # node attrs
    datakey = tl.Unicode().tag(attr=True)

    # optional inputs
    auth_class = tl.Type(authentication.Session)
    auth_session = tl.Instance(authentication.Session, allow_none=True)
    username = tl.Unicode(default_value=None, allow_none=True)
    password = tl.Unicode(default_value=None, allow_none=True)

    @tl.default("auth_session")
    def _auth_session_default(self):

        # requires username and password
        if not self.username or not self.password:
            return None

        # requires auth_class
        # TODO: default auth_class?
        if not self.auth_class:
            return None

        # instantiate and check utl
        try:
            session = self.auth_class(username=self.username, password=self.password)
            session.get(self.source + ".dds")
        except:
            # TODO: catch a 403 error
            return None

        return session

    @tl.default("dataset")
    def _open_dataset(self):
        """Summary
        
        Parameters
        ----------
        source : str, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        """

        # auth session
        # if self.auth_session:
        try:
            dataset = self._open_url()
        except Exception:
            # TODO handle a 403 error
            # TODO: Check Url (probably inefficient...)
            try:
                self.auth_session.get(self.source + ".dds")
                dataset = self._open_url()
            except Exception:
                # TODO: handle 403 error
                print ("Warning, dataset could not be opened. Check login credentials.")
                dataset = None

        return dataset

    def _open_url(self):
        return pydap.client.open_url(self.source, session=self.auth_session)

    @common_doc(COMMON_DATA_DOC)
    def get_native_coordinates(self):
        """{get_native_coordinates}
        
        Raises
        ------
        NotImplementedError
            DAP has no mechanism for creating coordinates automatically, so this is left up to child classes.
        """
        raise NotImplementedError(
            "DAP has no mechanism for creating coordinates"
            + ", so this is left up to child class "
            + "implementations."
        )

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        data = self.dataset[self.datakey][tuple(coordinates_index)]
        # PyDAP 3.2.1 gives a numpy array for the above, whereas 3.2.2 needs the .data attribute to get a numpy array
        if not isinstance(data, np.ndarray) and hasattr(data, "data"):
            data = data.data
        d = self.create_output_array(coordinates, data=data.reshape(coordinates.shape))
        return d

    @property
    def keys(self):
        """The list of available keys from the OpenDAP dataset.
        
        Returns
        -------
        List
            The list of available keys from the OpenDAP dataset. Any of these keys can be set as self.datakey
        """
        return self.dataset.keys()


@common_doc(COMMON_DATA_DOC)
class CSV(DataSource):
    """Create a DataSource from a .csv file. This class assumes that the data has a storage format such as:
    header 1,   header 2,   header 3, ...
    row1_data1, row1_data2, row1_data3, ...
    row2_data1, row2_data2, row2_data3, ...
    
    Attributes
    ----------
    native_coordinates : Coordinates
        {native_coordinates}
    source : str
        Path to the data source
    alt_col : str or int
        Column number or column title containing altitude data
    lat_col : str or int
        Column number or column title containing latitude data
    lon_col : str or int
        Column number or column title containing longitude data
    time_col : str or int
        Column number or column title containing time data
    data_col : str or int
        Column number or column title containing output data
    dims : list[str]
        Default is ['alt', 'lat', 'lon', 'time']. List of dimensions tested. This list determined the order of the
        stacked dimensions.
    dataset : pd.DataFrame
        Raw Pandas DataFrame used to read the data
    """

    source = tl.Unicode().tag(readonly=True)
    dataset = tl.Instance(pd.DataFrame).tag(readonly=True)

    # node attrs
    dims = tl.List(default_value=["alt", "lat", "lon", "time"]).tag(attr=True)
    alt_col = tl.Union([tl.Unicode(), tl.Int()]).tag(attr=True)
    lat_col = tl.Union([tl.Unicode(), tl.Int()]).tag(attr=True)
    lon_col = tl.Union([tl.Unicode(), tl.Int()]).tag(attr=True)
    time_col = tl.Union([tl.Unicode(), tl.Int()]).tag(attr=True)
    data_col = tl.Union([tl.Unicode(), tl.Int()]).tag(attr=True)

    def _first_init(self, **kwargs):
        # First part of if tests to make sure this is the CSV parent class
        # It's assumed that derived classes will define alt_col etc for specialized readers
        if type(self) == CSV and not (
            ("alt_col" in kwargs) or ("time_col" in kwargs) or ("lon_col" in kwargs) or ("lat_col" in kwargs)
        ):
            raise TypeError("CSV requires at least one of time_col, alt_col, lat_col, or lon_col.")

        return super(CSV, self)._first_init(**kwargs)

    @property
    def _alt_col(self):
        if isinstance(self.alt_col, int):
            return self.alt_col
        return self.dataset.columns.get_loc(self.alt_col)

    @property
    def _lat_col(self):
        if isinstance(self.lat_col, int):
            return self.lat_col
        return self.dataset.columns.get_loc(self.lat_col)

    @property
    def _lon_col(self):
        if isinstance(self.lon_col, int):
            return self.lon_col
        return self.dataset.columns.get_loc(self.lon_col)

    @property
    def _time_col(self):
        if isinstance(self.time_col, int):
            return self.time_col
        return self.dataset.columns.get_loc(self.time_col)

    @property
    def _data_col(self):
        if isinstance(self.data_col, int):
            return self.data_col
        return self.dataset.columns.get_loc(self.data_col)

    @tl.default("dataset")
    def _open_dataset(self):
        """Opens the data source
        
        Returns
        -------
        pd.DataFrame
            pd.read_csv(source)
        """
        return pd.read_csv(self.source, parse_dates=True, infer_datetime_format=True)

    @common_doc(COMMON_DATA_DOC)
    def get_native_coordinates(self):
        """{get_native_coordinates}
        
        The default implementation tries to find the lat/lon coordinates based on dataset.affine or dataset.transform
        (depending on the version of rasterio). It cannot determine the alt or time dimensions, so child classes may
        have to overload this method.
        """
        coords = []
        for d in self.dims:
            if trait_is_defined(self, d + "_col") or (
                d + "_col" not in self.trait_names() and hasattr(self, d + "_col")
            ):
                i = getattr(self, "_{}_col".format(d))
                if d is "time":
                    c = np.array(self.dataset.iloc[:, i], np.datetime64)
                else:
                    c = np.array(self.dataset.iloc[:, i])
                coords.append(ArrayCoordinates1d(c, name=d))
        if len(coords) > 1:
            coords = [StackedCoordinates(coords)]
        return Coordinates(coords)

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        d = self.dataset.iloc[coordinates_index[0], self._data_col]
        return self.create_output_array(coordinates, data=d)


@common_doc(COMMON_DATA_DOC)
class Rasterio(DataSource):
    """Create a DataSource using Rasterio.
 
    Parameters
    ----------
    source : str, :class:`io.BytesIO`
        Path to the data source
    band : int
        The 'band' or index for the variable being accessed in files such as GeoTIFFs

    Attributes
    ----------
    dataset : :class:`rasterio._io.RasterReader`
        A reference to the datasource opened by rasterio
    native_coordinates : :class:`podpac.Coordinates`
        {native_coordinates}


    Notes
    ------
    The source could be a path to an s3 bucket file, e.g.: s3://landsat-pds/L8/139/045/LC81390452014295LGN00/LC81390452014295LGN00_B1.TIF  
    In that case, make sure to set the environmental variable: 
    * Windows: set CURL_CA_BUNDLE=<path_to_conda_env>\Library\ssl\cacert.pem
    * Linux: export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
    """

    source = tl.Union([tl.Unicode(), tl.Instance(BytesIO)]).tag(readonly=True)
    dataset = tl.Any().tag(readonly=True)

    # node attrs
    band = tl.CInt(1).tag(attr=True)

    @tl.default("dataset")
    def _open_dataset(self):
        """Opens the data source
        
        Returns
        -------
        :class:`rasterio.io.DatasetReader`
            Rasterio dataset
        """

        # TODO: dataset should not open by default
        # prefer with as: syntax

        if isinstance(self.source, BytesIO):
            # https://rasterio.readthedocs.io/en/latest/topics/memory-files.html
            # TODO: this is still not working quite right - likely need to work
            # out the BytesIO format or how we are going to read/write in memory
            with rasterio.MemoryFile(self.source) as memfile:
                return memfile.open(driver="GTiff")

        # local file
        else:
            return rasterio.open(self.source)

    def close_dataset(self):
        """Closes the file for the datasource
        """
        self.dataset.close()

    @common_doc(COMMON_DATA_DOC)
    def get_native_coordinates(self):
        """{get_native_coordinates}
        
        The default implementation tries to find the lat/lon coordinates based on dataset.affine.
        It cannot determine the alt or time dimensions, so child classes may
        have to overload this method.
        """

        # check to see if the coordinates are rotated used affine
        affine = self.dataset.transform
        if affine[1] != 0.0 or affine[3] != 0.0:
            raise NotImplementedError("Rotated coordinates are not yet supported")

        try:
            crs = self.dataset.crs["init"].upper()
        except:
            crs = None

        # get bounds
        left, bottom, right, top = self.dataset.bounds

        # rasterio reads data upside-down from coordinate conventions, so lat goes from top to bottom
        lat = UniformCoordinates1d(top, bottom, size=self.dataset.height, name="lat")
        lon = UniformCoordinates1d(left, right, size=self.dataset.width, name="lon")
        return Coordinates([lat, lon], dims=["lat", "lon"], crs=crs)

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        data = self.create_output_array(coordinates)
        slc = coordinates_index

        # read data within coordinates_index window
        window = ((slc[0].start, slc[0].stop), (slc[1].start, slc[1].stop))
        raster_data = self.dataset.read(self.band, out_shape=tuple(coordinates.shape), window=window)

        # set raster data to output array
        data.data.ravel()[:] = raster_data.ravel()
        return data

    @property
    def band_count(self):
        """The number of bands

        Returns
        -------
        int
            The number of bands in the dataset
        """

        if not hasattr(self, "_band_count"):
            self._band_count = self.dataset.count

        return self._band_count

    @property
    def band_descriptions(self):
        """A description of each band contained in dataset.tags
        
        Returns
        -------
        OrderedDict
            Dictionary of band_number: band_description pairs. The band_description values are a dictionary, each 
            containing a number of keys -- depending on the metadata
        """

        if not hasattr(self, "_band_descriptions"):
            self._band_descriptions = OrderedDict((i, self.dataset.tags(i + 1)) for i in range(self.band_count))

        return self._band_descriptions

    @property
    def band_keys(self):
        """An alternative view of band_descriptions based on the keys present in the metadata
        
        Returns
        -------
        dict
            Dictionary of metadata keys, where the values are the value of the key for each band. 
            For example, band_keys['TIME'] = ['2015', '2016', '2017'] for a dataset with three bands.
        """

        if not hasattr(self, "_band_keys"):
            keys = {k for i in range(self.band_count) for k in self.band_descriptions[i]}  # set
            self._band_keys = {k: [self.band_descriptions[i].get(k) for i in range(self.band_count)] for k in keys}

        return self._band_keys

    def get_band_numbers(self, key, value):
        """Return the bands that have a key equal to a specified value.
        
        Parameters
        ----------
        key : str / list
            Key present in the metadata of the band. Can be a single key, or a list of keys.
        value : str / list
            Value of the key that should be returned. Can be a single value, or a list of values
        
        Returns
        -------
        np.ndarray
            An array of band numbers that match the criteria
        """
        if (not hasattr(key, "__iter__") or isinstance(key, string_types)) and (
            not hasattr(value, "__iter__") or isinstance(value, string_types)
        ):
            key = [key]
            value = [value]

        match = np.ones(self.band_count, bool)
        for k, v in zip(key, value):
            match = match & (np.array(self.band_keys[k]) == v)
        matches = np.where(match)[0] + 1

        return matches


class DatasetCoordinatedMixin(tl.HasTraits):
    latkey = tl.Unicode(allow_none=True, default_value="lat").tag(attr=True)
    lonkey = tl.Unicode(allow_none=True, default_value="lon").tag(attr=True)
    timekey = tl.Unicode(allow_none=True, default_value="time").tag(attr=True)
    altkey = tl.Unicode(allow_none=True, default_value="alt").tag(attr=True)
    dims = tl.List(trait=Dimension()).tag(attr=True)
    crs = tl.Unicode(allow_none=True, default_value=None).tag(attr=True)
    cf_time = tl.Bool(False).tag(attr=True)
    cf_units = tl.Unicode(allow_none=True).tag(attr=True)
    cf_calendar = tl.Unicode(allow_none=True).tag(attr=True)

    @common_doc(COMMON_DATA_DOC)
    def get_native_coordinates(self):
        """{get_native_coordinates}
        """

        cs = []
        for dim in self.dims:
            if dim == "lat":
                cs.append(self.get_lat())
            elif dim == "lon":
                cs.append(self.get_lon())
            elif dim == "time":
                cs.append(self.get_time())
            elif dim == "alt":
                cs.append(self.get_alt())

        return Coordinates(cs, dims=self.dims, crs=self.crs)

    def get_lat(self):
        """
        Get the native lat coordinates. Subclasses should customize as needed.

        Returns
        -------
        lat : array-like or Coordinates1d
            native latitude coordinates.
        """

        return self.dataset[self.latkey]

    def get_lon(self):
        """
        Get the native lon coordinates. Subclasses should customize as needed.

        Returns
        -------
        lon : array-like or Coordinates1d
            native latitude coordinates.
        """

        return self.dataset[self.lonkey]

    def get_time(self):
        """
        Get the native time coordinates. Subclasses should customize as needed.

        Returns
        -------
        time : array-like or Coordinates1d
            native latitude coordinates.
        """

        values = self.dataset[self.timekey]
        if self.cf_time:
            values = xr.coding.times.decode_cf_datetime(values, self.cf_units, self.cf_calendar)
        return values

    def get_alt(self):
        """
        Get the native alt coordinates. Subclasses should customize as needed.

        Returns
        -------
        alt : array-like or Coordinates1d
            native latitude coordinates.
        """

        return self.dataset[self.altkey]


@common_doc(COMMON_DATA_DOC)
class H5PY(DatasetCoordinatedMixin, DataSource):
    """Create a DataSource node using h5py.
    
    Attributes
    ----------
    datakey : str
        The 'key' for the data to be retrieved from the file. Datasource may have multiple keys, so this key
        determines which variable is returned from the source.
    dataset : h5py.File
        The h5py File object used for opening the file
    native_coordinates : Coordinates
        {native_coordinates}
    source : str
        Path to the data source
For example,
        if self.datasets[datakey] has shape (1, 2, 3) and the (time, lon, lat) dimensions have sizes (1, 2, 3)
        then dims should be ['time', 'lon', 'lat']
    file_mode : str, optional
        Default is 'r'. The mode used to open the HDF5 file. Options are r, r+, w, w- or x, a (see h5py.File).
    """

    source = tl.Unicode().tag(readonly=True)
    dataset = tl.Any().tag(readonly=True)
    file_mode = tl.Unicode(default_value="r").tag(readonly=True)

    # node attrs
    datakey = tl.Unicode().tag(attr=True)

    @tl.default("dataset")
    def _open_dataset(self):
        """Opens the data source
        
        Parameters
        ----------
        source : str, optional
            Uses self.source by default. Path to the data source.
        
        Returns
        -------
        Any
            raster.open(source)
        """

        # TODO: dataset should not open by default
        # prefer with as: syntax
        return h5py.File(self.source, self.file_mode)

    def close_dataset(self):
        """Closes the file for the datasource
        """
        self.dataset.close()

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        data = self.create_output_array(coordinates)
        slc = coordinates_index
        a = self.dataset[self.datakey][slc]
        data.data.ravel()[:] = a.ravel()
        return data

    @property
    def keys(self):
        return H5PY._find_h5py_keys(self.dataset)

    def attrs(self, key="/"):
        """
        Dataset or group key for which attributes will be summarized.
        """
        return dict(self.dataset[key].attrs)

    @staticmethod
    def _find_h5py_keys(obj, keys=[]):
        if isinstance(obj, (h5py.Group, h5py.File)):
            for k in obj.keys():
                keys = H5PY._find_h5py_keys(obj[k], keys)
        else:
            keys.append(obj.name)
            return keys
        keys = list(set(keys))
        keys.sort()
        return keys


class Zarr(DatasetCoordinatedMixin, DataSource):
    source = tl.Unicode(default_value=None, allow_none=True).tag(readonly=True)
    dataset = tl.Any().tag(readonly=True)

    # node attrs
    datakey = tl.Unicode().tag(attr=True)

    # optional inputs
    access_key_id = tl.Unicode()
    secret_access_key = tl.Unicode()
    region_name = tl.Unicode()

    @tl.default("access_key_id")
    def _get_access_key_id(self):
        return settings["AWS_ACCESS_KEY_ID"]

    @tl.default("secret_access_key")
    def _get_secret_access_key(self):
        return settings["AWS_SECRET_ACCESS_KEY"]

    @tl.default("region_name")
    def _get_region_name(self):
        return settings["AWS_REGION_NAME"]

    def init(self):
        # check that source or dataset is provided
        if self.source is None:
            self.dataset

        # check dim keys
        for dim in self.dims:
            if dim == "lat":
                if self.latkey is None:
                    raise TypeError("Zarr node 'latkey' is required for dims %s" % self.dims)
                if self.latkey not in self.dataset:
                    raise ValueError("Zarr lat key '%s' not found" % self.latkey)
            elif dim == "lon":
                if self.lonkey is None:
                    raise TypeError("Zarr node 'lonkey' is required for dims %s" % self.dims)
                if self.lonkey not in self.dataset:
                    raise ValueError("Zarr lon key '%s' not found" % self.lonkey)
            elif dim == "time":
                if self.timekey is None:
                    raise TypeError("Zarr node 'timekey' is required for dims %s" % self.dims)
                if self.timekey not in self.dataset:
                    raise ValueError("Zarr time key '%s' not found" % self.timekey)
            elif dim == "alt":
                if self.altkey is None:
                    raise TypeError("Zarr node 'altkey' is required for dims %s" % self.dims)
                if self.altkey not in self.dataset:
                    raise ValueError("Zarr alt key '%s' not found" % self.altkey)

        # check data key
        if self.datakey not in self.dataset:
            raise ValueError("Zarr data key '%s' not found" % self.datakey)

    @tl.default("dataset")
    def _open_dataset(self):
        if self.source is None:
            raise TypeError("Zarr node requires 'source' or 'dataset'")

        if self.source.startswith("s3://"):
            root = self.source.strip("s3://")
            kwargs = {"region_name": self.region_name}
            s3 = s3fs.S3FileSystem(key=self.access_key_id, secret=self.secret_access_key, client_kwargs=kwargs)
            s3map = s3fs.S3Map(root=root, s3=s3, check=False)
            store = s3map
        else:
            store = str(self.source)  # has to be a string in Python2.7 for local files

        try:
            return zarr.open(store, mode="r")
        except ValueError:
            raise ValueError("No Zarr store found at path '%s'" % self.source)

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        data = self.create_output_array(coordinates)
        a = self.dataset[self.datakey][coordinates_index]
        data.data.ravel()[:] = a.ravel()
        return data


WCS_DEFAULT_VERSION = "1.0.0"
WCS_DEFAULT_CRS = "EPSG:4326"


class WCS(DataSource):
    """Create a DataSource from an OGC-complient WCS service
    
    Attributes
    ----------
    crs : 'str'
        Default is EPSG:4326 (WGS84 Geodic) EPSG number for the coordinate reference system that the data should
        be returned in.
    layer_name : str
        Name of the WCS layer that should be fetched from the server
    source : str
        URL of the WCS server endpoint
    version : str
        Default is 1.0.0. WCS version string.
    wcs_coordinates : Coordinates
        The coordinates of the WCS source
    """

    source = tl.Unicode().tag(readonly=True)
    wcs_coordinates = tl.Instance(Coordinates).tag(readonly=True)  # default below

    # node attrs
    layer_name = tl.Unicode().tag(attr=True)
    version = tl.Unicode(WCS_DEFAULT_VERSION).tag(attr=True)
    crs = tl.Unicode(WCS_DEFAULT_CRS).tag(attr=True)

    _get_capabilities_qs = tl.Unicode("SERVICE=WCS&REQUEST=DescribeCoverage&" "VERSION={version}&COVERAGE={layer}")
    _get_data_qs = tl.Unicode(
        "SERVICE=WCS&VERSION={version}&REQUEST=GetCoverage&"
        "FORMAT=GeoTIFF&COVERAGE={layer}&"
        "BBOX={w},{s},{e},{n}&CRS={crs}&RESPONSE_CRS={crs}&"
        "WIDTH={width}&HEIGHT={height}&TIME={time}"
    )

    # TODO: This should be capabilities_url, not get_
    @property
    def get_capabilities_url(self):
        """Constructs the url that requests the WCS capabilities
        
        Returns
        -------
        str
            The url that requests the WCS capabilities
        """
        return self.source + "?" + self._get_capabilities_qs.format(version=self.version, layer=self.layer_name)

    @tl.default("wcs_coordinates")
    def get_wcs_coordinates(self):
        """Retrieves the native coordinates reported by the WCS service.
        
        Returns
        -------
        Coordinates
            The native coordinates reported by the WCS service.
        
        Notes
        -------
        This assumes a `time`, `lat`, `lon` order for the coordinates, and currently doesn't handle `alt` coordinates
        
        Raises
        ------
        Exception
            Raises this if the required dependencies are not installed.
        """
        if requests is not None:
            capabilities = requests.get(self.get_capabilities_url)
            if capabilities.status_code != 200:
                raise Exception("Could not get capabilities from WCS server")
            capabilities = capabilities.text

        # TODO: remove support urllib3 - requests is sufficient
        elif urllib3 is not None:
            if certifi is not None:
                http = urllib3.PoolManager(ca_certs=certifi.where())
            else:
                http = urllib3.PoolManager()

            r = http.request("GET", self.get_capabilities_url)
            capabilities = r.data
            if r.status != 200:
                raise Exception("Could not get capabilities from WCS server:" + self.get_capabilities_url)
        else:
            raise Exception("Do not have a URL request library to get WCS data.")

        if (
            lxml is not None
        ):  # could skip using lxml and always use html.parser instead, which seems to work but lxml might be faster
            capabilities = bs4.BeautifulSoup(capabilities, "lxml")
        else:
            capabilities = bs4.BeautifulSoup(capabilities, "html.parser")

        domain = capabilities.find("wcs:spatialdomain")
        pos = domain.find("gml:envelope").get_text().split()
        lonlat = np.array(pos, float).reshape(2, 2)
        grid_env = domain.find("gml:gridenvelope")
        low = np.array(grid_env.find("gml:low").text.split(), int)
        high = np.array(grid_env.find("gml:high").text.split(), int)
        size = high - low
        dlondlat = (lonlat[1, :] - lonlat[0, :]) / size
        bottom = lonlat[:, 1].min() + dlondlat[1] / 2
        top = lonlat[:, 1].max() - dlondlat[1] / 2
        left = lonlat[:, 0].min() + dlondlat[0] / 2
        right = lonlat[:, 0].max() - dlondlat[0] / 2

        timedomain = capabilities.find("wcs:temporaldomain")
        if timedomain is None:
            return Coordinates(
                [
                    UniformCoordinates1d(top, bottom, size=size[1], name="lat"),
                    UniformCoordinates1d(left, right, size=size[0], name="lon"),
                ]
            )

        date_re = re.compile("[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}")
        times = str(timedomain).replace("<gml:timeposition>", "").replace("</gml:timeposition>", "").split("\n")
        times = np.array([t for t in times if date_re.match(t)], np.datetime64)

        if len(times) == 0:
            return Coordinates(
                [
                    UniformCoordinates1d(top, bottom, size=size[1], name="lat"),
                    UniformCoordinates1d(left, right, size=size[0], name="lon"),
                ]
            )

        return Coordinates(
            [
                ArrayCoordinates1d(times, name="time"),
                UniformCoordinates1d(top, bottom, size=size[1], name="lat"),
                UniformCoordinates1d(left, right, size=size[0], name="lon"),
            ]
        )

    @property
    @common_doc(COMMON_DATA_DOC)
    def native_coordinates(self):
        """{native_coordinates}
        
        Returns
        -------
        Coordinates
            {native_coordinates}
            
        Notes
        ------
        This is a little tricky and doesn't fit into the usual PODPAC method, as the service is actually doing the 
        data wrangling for us...
        """

        # TODO update so that we don't rely on _requested_coordinates if possible
        if not self._requested_coordinates:
            return self.wcs_coordinates

        cs = []
        for dim in self.wcs_coordinates.dims:
            if dim in self._requested_coordinates.dims:
                c = self._requested_coordinates[dim]
                if c.size == 1:
                    cs.append(ArrayCoordinates1d(c.coordinates[0], name=dim))
                elif isinstance(c, UniformCoordinates1d):
                    cs.append(UniformCoordinates1d(c.bounds[0], c.bounds[1], abs(c.step), name=dim))
                else:
                    # TODO: generalize/fix this
                    # WCS calls require a regular grid, could (otherwise we have to do multiple WCS calls)
                    cs.append(UniformCoordinates1d(c.bounds[0], c.bounds[1], size=c.size, name=dim))
            else:
                cs.append(self.wcs_coordinates[dim])
        c = Coordinates(cs)
        return c

    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        
        Raises
        ------
        Exception
            Raises this if there is a network error or required dependencies are not installed.
        """
        output = self.create_output_array(coordinates)
        dotime = "time" in self.wcs_coordinates.dims

        if "time" in coordinates.dims and dotime:
            sd = np.timedelta64(0, "s")
            times = [str(t + sd) for t in coordinates["time"].coordinates]
        else:
            times = [""]

        if len(times) > 1:
            for i, time in enumerate(times):
                url = (
                    self.source
                    + "?"
                    + self._get_data_qs.format(
                        version=self.version,
                        layer=self.layer_name,
                        w=min(coordinates["lon"].area_bounds),
                        e=max(coordinates["lon"].area_bounds),
                        s=min(coordinates["lat"].area_bounds),
                        n=max(coordinates["lat"].area_bounds),
                        width=coordinates["lon"].size,
                        height=coordinates["lat"].size,
                        time=time,
                        crs=self.crs,
                    )
                )

                if not dotime:
                    url = url.replace("&TIME=", "")

                if requests is not None:
                    data = requests.get(url)
                    if data.status_code != 200:
                        raise Exception("Could not get data from WCS server:" + url)
                    io = BytesIO(bytearray(data.content))
                    content = data.content

                # TODO: remove support urllib3 - requests is sufficient
                elif urllib3 is not None:
                    if certifi is not None:
                        http = urllib3.PoolManager(ca_certs=certifi.where())
                    else:
                        http = urllib3.PoolManager()
                    r = http.request("GET", url)
                    if r.status != 200:
                        raise Exception("Could not get capabilities from WCS server:" + url)
                    content = r.data
                    io = BytesIO(bytearray(r.data))
                else:
                    raise Exception("Do not have a URL request library to get WCS data.")

                try:
                    try:  # This works with rasterio v1.0a8 or greater, but not on python 2
                        with rasterio.open(io) as dataset:
                            output.data[i, ...] = dataset.read()
                    except Exception as e:  # Probably python 2
                        print (e)
                        tmppath = os.path.join(settings["DISK_CACHE_DIR"], "wcs_temp.tiff")

                        if not os.path.exists(os.path.split(tmppath)[0]):
                            os.makedirs(os.path.split(tmppath)[0])

                        # TODO: close tmppath? os does this on remove?
                        open(tmppath, "wb").write(content)

                        with rasterio.open(tmppath) as dataset:
                            output.data[i, ...] = dataset.read()

                        os.remove(tmppath)  # Clean up

                except ImportError:
                    # Writing the data to a temporary tiff and reading it from there is hacky
                    # However reading directly from r.data or io doesn't work
                    # Should improve in the future
                    open("temp.tiff", "wb").write(r.data)
                    output.data[i, ...] = RasterToNumPyArray("temp.tiff")
        else:
            time = times[0]

            url = (
                self.source
                + "?"
                + self._get_data_qs.format(
                    version=self.version,
                    layer=self.layer_name,
                    w=min(coordinates["lon"].area_bounds),
                    e=max(coordinates["lon"].area_bounds),
                    s=min(coordinates["lat"].area_bounds),
                    n=max(coordinates["lat"].area_bounds),
                    width=coordinates["lon"].size,
                    height=coordinates["lat"].size,
                    time=time,
                    crs=self.crs,
                )
            )
            if not dotime:
                url = url.replace("&TIME=", "")
            if requests is not None:
                data = requests.get(url)
                if data.status_code != 200:
                    raise Exception("Could not get data from WCS server:" + url)
                io = BytesIO(bytearray(data.content))
                content = data.content

            # TODO: remove support urllib3 - requests is sufficient
            elif urllib3 is not None:
                if certifi is not None:
                    http = urllib3.PoolManager(ca_certs=certifi.where())
                else:
                    http = urllib3.PoolManager()
                r = http.request("GET", url)
                if r.status != 200:
                    raise Exception("Could not get capabilities from WCS server:" + url)
                content = r.data
                io = BytesIO(bytearray(r.data))
            else:
                raise Exception("Do not have a URL request library to get WCS data.")

            try:
                try:  # This works with rasterio v1.0a8 or greater, but not on python 2
                    with rasterio.open(io) as dataset:
                        if dotime:
                            output.data[0, ...] = dataset.read()
                        else:
                            output.data[:] = dataset.read()
                except Exception as e:  # Probably python 2
                    print (e)
                    tmppath = os.path.join(settings["DISK_CACHE_DIR"], "wcs_temp.tiff")
                    if not os.path.exists(os.path.split(tmppath)[0]):
                        os.makedirs(os.path.split(tmppath)[0])
                    open(tmppath, "wb").write(content)
                    with rasterio.open(tmppath) as dataset:
                        output.data[:] = dataset.read()
                    os.remove(tmppath)  # Clean up
            except ImportError:
                # Writing the data to a temporary tiff and reading it from there is hacky
                # However reading directly from r.data or io doesn't work
                # Should improve in the future
                open("temp.tiff", "wb").write(r.data)
                try:
                    output.data[:] = RasterToNumPyArray("temp.tiff")
                except:
                    raise Exception("Rasterio or Arcpy not available to read WCS feed.")
        if not coordinates["lat"].is_descending:
            if dotime:
                output.data[:] = output.data[:, ::-1, :]
            else:
                output.data[:] = output.data[::-1, :]

        return output

    @property
    def base_ref(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        return self.layer_name.rsplit(".", 1)[1]


class ReprojectedSource(DataSource):
    """Create a DataSource with a different resolution from another Node. This can be used to bilinearly interpolated a
    dataset after averaging over a larger area.
    
    Attributes
    ----------
    source : Node
        The source node
    source_interpolation : str
        Type of interpolation method to use for the source node
    reprojected_coordinates : Coordinates
        Coordinates where the source node should be evaluated. 
    """

    source = NodeTrait().tag(readonly=True)

    # node attrs
    source_interpolation = interpolation_trait().tag(attr=True)
    reprojected_coordinates = tl.Instance(Coordinates).tag(attr=True)

    def _first_init(self, **kwargs):
        if "reprojected_coordinates" in kwargs:
            if isinstance(kwargs["reprojected_coordinates"], dict):
                kwargs["reprojected_coordinates"] = Coordinates.from_definition(kwargs["reprojected_coordinates"])
            elif isinstance(kwargs["reprojected_coordinates"], string_types):
                kwargs["reprojected_coordinates"] = Coordinates.from_json(kwargs["reprojected_coordinates"])

        return super(ReprojectedSource, self)._first_init(**kwargs)

    @common_doc(COMMON_DATA_DOC)
    def get_native_coordinates(self):
        """{get_native_coordinates}
        """
        if isinstance(self.source, DataSource):
            sc = self.source.native_coordinates
        else:  # Otherwise we cannot guarantee that native_coordinates exist
            sc = self.reprojected_coordinates
        rc = self.reprojected_coordinates
        coords = [rc[dim] if dim in rc.dims else sc[dim] for dim in sc.dims]
        return Coordinates(coords)

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        if hasattr(self.source, "interpolation") and self.source_interpolation is not None:
            si = self.source.interpolation
            self.source.interpolation = self.source_interpolation
        elif self.source_interpolation is not None:
            _logger.warn(
                "ReprojectedSource cannot set the 'source_interpolation'"
                " since self.source does not have an 'interpolation' "
                " attribute. \n type(self.source): %s\nself.source: " % (str(type(self.source)), str(self.source))
            )
        data = self.source.eval(coordinates)
        if hasattr(self.source, "interpolation") and self.source_interpolation is not None:
            self.source.interpolation = si
        # The following is needed in case the source is an algorithm
        # or compositor node that doesn't have all the dimensions of
        # the reprojected coordinates
        # TODO: What if data has coordinates that reprojected_coordinates doesn't have
        keep_dims = list(data.coords.keys())
        drop_dims = [d for d in coordinates.dims if d not in keep_dims]
        coordinates.drop(drop_dims)
        return data

    @property
    def base_ref(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        return "{}_reprojected".format(self.source.base_ref)


@common_doc(COMMON_DATA_DOC)
class Dataset(DataSource):
    """Create a DataSource node using xarray.open_dataset.
    
    Attributes
    ----------
    datakey : str
        The 'key' for the data to be retrieved from the file. Datasource may have multiple keys, so this key
        determines which variable is returned from the source.
    dataset : xarray.Dataset, optional
        The xarray dataset from which to retrieve data. If not specified, will be automatically created from the 'source'
    native_coordinates : Coordinates
        {native_coordinates}
    source : str
        Path to the data source
    extra_dim : dict
        In cases where the data contain dimensions other than ['lat', 'lon', 'time', 'alt'], these dimensions need to be selected. 
        For example, if the data contains ['lat', 'lon', 'channel'], the second channel can be selected using `extra_dim=dict(channel=1)`
    """

    source = tl.Any(allow_none=True).tag(readonly=True)
    dataset = tl.Instance(xr.Dataset).tag(readonly=True)

    # node attrs
    extra_dim = tl.Dict({}).tag(attr=True)
    datakey = tl.Unicode().tag(attr=True)

    @tl.default("dataset")
    def _dataset_default(self):
        return xr.open_dataset(self.source)

    @property
    @common_doc(COMMON_DATA_DOC)
    def native_coordinates(self):
        """{native_coordinates}
        """
        # we have to remove any dimensions not in 'lat', 'lon', 'time', 'alt' for the 'get_data' machinery to work properly
        coords = self.dataset[self.datakey].coords
        crds = []
        dims = []
        for d in coords.dims:
            if d not in ["lat", "lon", "time", "alt"]:
                continue
            crds.append(coords[d].data)
            dims.append(d)
        return Coordinates(crds, dims)

    def get_data(self, coordinates, coordinates_index):
        return self.create_output_array(coordinates, self.dataset[self.datakey][self.extra_dim].data[coordinates_index])

    @property
    def keys(self):
        """The list of available keys from the xarray dataset.
        
        Returns
        -------
        List
            The list of available keys from the xarray dataset. Any of these keys can be set as self.datakey.
        """
        return list(self.dataset.keys())
