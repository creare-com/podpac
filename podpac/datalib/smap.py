"""Specialized PODPAC nodes use to access SMAP data via OpenDAP from nsidc.

Attributes
----------
SMAP_BASE_URL() : str
    Url to nsidc openDAP server
SMAP_INCOMPLETE_SOURCE_COORDINATES : list
    List of products whose source coordinates are incomplete. This means any shared coordinates cannot be extracted
SMAP_PRODUCT_DICT: dict
    Mapping of important keys into the openDAP dataset that deals with inconsistencies across SMAP products. Used to add
    new SMAP products.
SMAP_PRODUCT_MAP : xr.DataArray
    Same as SMAP_PRODUCT_DICT, but stored as a more convenient DataArray object
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import os
import re
import copy
import logging

import requests
from six import string_types
import numpy as np
import xarray as xr
import traitlets as tl

# Set up logging
_logger = logging.getLogger(__name__)

# Helper utility for optional imports
from lazy_import import lazy_module

# Optional dependencies
bs4 = lazy_module("bs4")
boto3 = lazy_module("boto3")

# fixing problem with older versions of numpy
if not hasattr(np, "isnat"):

    def isnat(a):
        return a.astype(str) == "None"

    np.isnat = isnat

# Internal dependencies
import podpac
from podpac.core.coordinates import Coordinates, union, merge_dims, concat
from podpac.core.data import pydap_source
from podpac.core import authentication
from podpac.core.utils import common_doc
from podpac.core.data.datasource import COMMON_DATA_DOC
from podpac.core.node import cache_func
from podpac.core.node import NodeException
from podpac.core import cache

from . import nasaCMR

COMMON_DOC = COMMON_DATA_DOC.copy()
COMMON_DOC.update(
    {
        "smap_date": "str\n        SMAP date string",
        "np_date": "np.datetime64\n        Numpy date object",
        "auth_class": (
            "EarthDataSession (Class object)\n        Class used to make an authenticated session from a"
            " username and password (both are defined in base class)"
        ),
        "auth_session": (
            "Instance(EarthDataSession)\n        Authenticated session used to make http requests using"
            "NASA Earth Data Login credentials"
        ),
        "base_url": "str\n        Url to nsidc openDAP server",
        "layerkey": (
            "str\n        Key used to retrieve data from OpenDAP dataset. This specifies the key used to retrieve "
            "the data"
        ),
        "password": "User's EarthData password",
        "username": "User's EarthData username",
        "product": "SMAP product name",
        "version": "Version number for the SMAP product",
        "source_coordinates": """Returns the coordinates that uniquely describe each source

        Returns
        -------
        :class:`podpac.Coordinates`
            Coordinates that uniquely describe each source""",
        "keys": """Available layers that are in the OpenDAP dataset

        Returns
        -------
        List
            The list of available keys from the OpenDAP dataset. Any of these keys can be set as self.datakey.

        Notes
        -----
        This function assumes that all of the keys in the available dataset are the same for every file.
        """,
    }
)


@common_doc(COMMON_DOC)
def smap2np_date(date):
    """Convert dates using the format in SMAP to numpy datetime64

    Parameters
    ----------
    date : {smap_date}

    Returns
    -------
    {np_date}
    """
    if isinstance(date, string_types):
        ymd = "-".join([date[:4], date[4:6], date[6:8]])
        if len(date) == 15:
            HMS = " " + ":".join([date[9:11], date[11:13], date[13:15]])
        else:
            HMS = ""
        date = np.array([ymd + HMS], dtype="datetime64")
    return date


@common_doc(COMMON_DOC)
def np2smap_date(date):
    """Convert dates using the numpy format to SMAP strings

    Parameters
    ----------
    date : {np_date}

    Returns
    -------
    {smap_date}
    """
    if isinstance(date, np.datetime64):
        date = str(date).replace("-", ".")
    return date


def _get_from_url(url, auth_session):
    """Helper function to get data from an NSIDC url with error checking. 
    
    Parameters
    -----------
    url: str
        URL to website
    auth_session: podpac.core.authentication.EarthDataSession
        Authenticated EDS session
    """
    try:
        r = auth_session.get(url)
        if r.status_code != 200:
            _logger.warning("Could not connect to {}, status code {}".format(url, r.status_code))
            _logger.info("Trying to connect to {}".format(url.replace("opendap/", "")))
            r = auth_session.get(url.replace("opendap/", ""))
            if r.status_code != 200:
                _logger.error("Could not connect to {} to retrieve data, status code {}".format(url, r.status_code))
                raise RuntimeError("HTTP error: <%d>\n" % (r.status_code) + r.text[:4096])
    except requests.ConnectionError as e:
        _logger.warning("Cannot connect to {}:".format(url) + str(e))
        r = None
    except RuntimeError as e:
        _logger.warning("Cannot authenticate to {}. Check credentials. Error was as follows:".format(url) + str(e))
    return r


def _infer_SMAP_product_version(product, base_url, auth_session):
    """Helper function to automatically infer the version number of SMAP 
    products in case user did not specify a version, or the version changed
    
    Parameters
    ------------
    product: str
        Name of the SMAP product (e.g. one of SMAP_PRODUCT_DICT.keys())
    base_url: str
        URL to base SMAP product page
    auth_session: podpac.core.authentication.EarthDataSession
        Authenticated EDS session
    """

    r = _get_from_url(base_url, auth_session)
    if r:
        m = re.search(product, r.text)
        return int(r.text[m.end() + 1 : m.end() + 4])
    return int(SMAP_PRODUCT_MAP.sel(product=product, attr="default_version").item())


# NOTE: {rdk} will be substituted for the entry's 'rootdatakey'
SMAP_PRODUCT_DICT = {
    #'<Product>.ver': ['latkey',               'lonkey',                     'rootdatakey',                       'layerkey'              'default_verison'
    "SPL4SMAU": ["cell_lat", "cell_lon", "Analysis_Data_", "{rdk}sm_surface_analysis", 4],
    "SPL4SMGP": ["cell_lat", "cell_lon", "Geophysical_Data_", "{rdk}sm_surface", 4],
    "SPL3SMA": ["{rdk}latitude", "{rdk}longitude", "Soil_Moisture_Retrieval_Data_", "{rdk}soil_moisture", 3],
    "SPL3SMAP": ["{rdk}latitude", "{rdk}longitude", "Soil_Moisture_Retrieval_Data_", "{rdk}soil_moisture", 3],
    "SPL3SMP": ["{rdk}AM_latitude", "{rdk}AM_longitude", "Soil_Moisture_Retrieval_Data_", "{rdk}_soil_moisture", 5],
    "SPL3SMP_E": ["{rdk}AM_latitude", "{rdk}AM_longitude", "Soil_Moisture_Retrieval_Data_", "{rdk}_soil_moisture", 5],
    "SPL4SMLM": ["cell_lat", "cell_lon", "Land_Model_Constants_Data_", "", 4],
    "SPL2SMAP_S": [
        "{rdk}latitude_1km",
        "{rdk}longitude_1km",
        "Soil_Moisture_Retrieval_Data_1km_",
        "{rdk}soil_moisture_1km",
        2,
    ],
}

SMAP_PRODUCT_MAP = xr.DataArray(
    list(SMAP_PRODUCT_DICT.values()),
    dims=["product", "attr"],
    coords={
        "product": list(SMAP_PRODUCT_DICT.keys()),
        "attr": ["latkey", "lonkey", "rootdatakey", "layerkey", "default_version"],
    },
)

SMAP_INCOMPLETE_SOURCE_COORDINATES = ["SPL2SMAP_S"]
SMAP_IRREGULAR_COORDINATES = ["SPL2SMAP_S"]

# Discover SMAP OpenDAP url from podpac s3 server
SMAP_BASE_URL_FILE = os.path.join(os.path.dirname(__file__), "nsidc_smap_opendap_url.txt")
_SMAP_BASE_URL = None


def SMAP_BASE_URL():
    global _SMAP_BASE_URL
    if _SMAP_BASE_URL is not None:
        return _SMAP_BASE_URL
    BASE_URL = "https://n5eil01u.ecs.nsidc.org/opendap/SMAP"
    try:
        with open(SMAP_BASE_URL_FILE, "r") as fid:
            rf = fid.read()
        if "https://" in rf and "nsidc.org" in rf:
            BASE_URL = rf
    except Exception as e:
        _logger.warning("Could not retrieve SMAP url from %s: " % (SMAP_BASE_URL_FILE) + str(e))
        rf = None
    try:
        r = requests.get("https://s3.amazonaws.com/podpac-s3/settings/nsidc_smap_opendap_url.txt").text
        if "https://" in r and "nsidc.org" in r:
            if rf != r:
                _logger.warning("Updating SMAP url from PODPAC S3 Server.")
                BASE_URL = r
                try:
                    with open(SMAP_BASE_URL_FILE, "w") as fid:
                        fid.write(r)
                except Exception as e:
                    _logger.warning("Could not overwrite SMAP url update on disk:" + str(e))
    except Exception as e:
        _logger.warning("Could not retrieve SMAP url from PODPAC S3 Server. Using default." + str(e))
    _SMAP_BASE_URL = BASE_URL
    return BASE_URL


@common_doc(COMMON_DOC)
class SMAPSource(pydap_source.PyDAP):
    """Accesses SMAP data given a specific openDAP URL. This is the base class giving access to SMAP data, and knows how
    to extract the correct coordinates and data keys for the soil moisture data.

    Attributes
    ----------
    auth_class : {auth_class}
    auth_session : {auth_session}
    date_file_url_re : SRE_Pattern
        Regular expression used to retrieve date from self.source (OpenDAP Url)
    date_time_file_url_re : SRE_Pattern
        Regular expression used to retrieve date and time from self.source (OpenDAP Url)
    layerkey : str
        Key used to retrieve data from OpenDAP dataset. This specifies the key used to retrieve the data
    nan_vals : list
        List of values that should be treated as no-data (these are replaced by np.nan)
    rootdatakey : str
        String the prepends every or most keys for data in the OpenDAP dataset
    """

    auth_session = tl.Instance(authentication.EarthDataSession)
    auth_class = tl.Type(authentication.EarthDataSession)
    # Need to overwrite parent because of recursive definition
    outputs = None

    @tl.default("auth_session")
    def _auth_session_default(self):
        session = self.auth_class(username=self.username, password=self.password, product_url=SMAP_BASE_URL())

        # check url
        try:
            session.get(SMAP_BASE_URL())
        except Exception as e:
            _logger.warning("Unknown exception: ", e)
        return session

    # date_url_re = re.compile('[0-9]{4}\.[0-9]{2}\.[0-9]{2}')
    date_time_file_url_re = re.compile("[0-9]{8}T[0-9]{6}")
    date_file_url_re = re.compile("[0-9]{8}")

    rootdatakey = tl.Unicode()

    @tl.default("rootdatakey")
    def _rootdatakey_default(self):
        return SMAP_PRODUCT_MAP.sel(product=self.product, attr="rootdatakey").item()

    layerkey = tl.Unicode()

    @tl.default("layerkey")
    def _layerkey_default(self):
        return SMAP_PRODUCT_MAP.sel(product=self.product, attr="layerkey").item()

    nan_vals = [-9999.0]

    @property
    def product(self):
        """Returns the SMAP product from the OpenDAP Url

        Returns
        -------
        str
            {product}
        """
        src = self.source.split("/")
        return src[src.index("SMAP") + 1].split(".")[0]

    @property
    def version(self):
        """Returns the SMAP product version from the OpenDAP Url

        Returns
        -------
        int
            {version}
        """
        src = self.source.split("/")
        return int(src[src.index("SMAP") + 1].split(".")[1])

    @tl.default("datakey")
    def _datakey_default(self):
        return self.layerkey.format(rdk=self.rootdatakey)

    @property
    def latkey(self):
        """The key used to retrieve the latitude

        Returns
        -------
        str
            OpenDap dataset key for latitude
        """
        return SMAP_PRODUCT_MAP.sel(product=self.product, attr="latkey").item().format(rdk=self.rootdatakey)

    @property
    def lonkey(self):
        """The key used to retrieve the latitude

        Returns
        -------
        str
            OpenDap dataset key for longitude
        """
        return SMAP_PRODUCT_MAP.sel(product=self.product, attr="lonkey").item().format(rdk=self.rootdatakey)

    @common_doc(COMMON_DOC)
    @cache_func("native.coordinates")
    def get_native_coordinates(self):
        """{get_native_coordinates}
        """
        times = self.get_available_times()
        ds = self.dataset
        lons = np.array(ds[self.lonkey][:, :])
        lats = np.array(ds[self.latkey][:, :])
        lons[lons == self.nan_vals[0]] = np.nan
        lats[lats == self.nan_vals[0]] = np.nan
        lons = np.nanmean(lons, axis=0)
        lats = np.nanmean(lats, axis=1)
        coords = podpac.Coordinates([times, lats, lons], dims=["time", "lat", "lon"])
        return coords

    def get_available_times(self):
        """Retrieve the available times from the SMAP file. This is primarily based on the filename, but some products 
        have multiple times stored in a single file.

        Returns
        -------
        np.ndarray(dtype=np.datetime64)
            Available times in the SMAP source
        """
        m = self.date_time_file_url_re.search(self.source)
        if not m:
            m = self.date_file_url_re.search(self.source)
        times = m.group()
        times = smap2np_date(times)
        if "SM_P_" in self.source:
            times = times + np.array([6, 18], "timedelta64[h]")
        return times

    @common_doc(COMMON_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        # We actually ignore the time slice
        s = tuple([slc for d, slc in zip(coordinates.dims, coordinates_index) if "time" not in d])
        if "SM_P_" in self.source:
            d = self.create_output_array(coordinates)
            am_key = self.layerkey.format(rdk=self.rootdatakey + "AM")
            pm_key = self.layerkey.format(rdk=self.rootdatakey + "PM") + "_pm"

            try:
                t = self.native_coordinates.coords["time"][0]
                d.loc[dict(time=t)] = np.array(self.dataset[am_key][s])
            except:
                pass

            try:
                t = self.native_coordinates.coords["time"][1]
                d.loc[dict(time=t)] = np.array(self.dataset[pm_key][s])
            except:
                pass

        else:
            data = np.array(self.dataset[self.datakey][s])
            d = self.create_output_array(coordinates, data=data.reshape(coordinates.shape))

        return d


class SMAPProperties(SMAPSource):
    """Accesses properties related to the generation of SMAP products. 

    Attributes
    ----------
    property : str
        A SMAP property, which includes: 
                        'clsm_dzsf', 'mwrtm_bh', 'clsm_cdcr2', 'mwrtm_poros',
                        'clsm_dzgt3', 'clsm_dzgt2', 'mwrtm_rghhmax',
                        'mwrtm_rghpolmix', 'clsm_dzgt1', 'clsm_wp', 'mwrtm_lewt',
                        'clsm_dzgt4', 'clsm_cdcr1', 'cell_elevation',
                        'mwrtm_rghwmin', 'clsm_dzrz', 'mwrtm_vegcls', 'mwrtm_bv',
                        'mwrtm_rghwmax', 'mwrtm_rghnrh', 'clsm_dztsurf',
                        'mwrtm_rghhmin', 'mwrtm_wangwp', 'mwrtm_wangwt',
                        'clsm_dzgt5', 'clsm_dzpr', 'clsm_poros',
                        'cell_land_fraction', 'mwrtm_omega', 'mwrtm_soilcls',
                        'clsm_dzgt6', 'mwrtm_rghnrv', 'mwrtm_clay', 'mwrtm_sand'
    source : str, optional
         Source OpenDAP url for SMAP properties. Default is (SMAP_BASE_URL() + 
                                                             'SPL4SMLM{latest_version}/2015.03.31/'
                                                             'SMAP_L4_SM_lmc_00000000T000000_Vv{latest_version}.h5')
    """

    file_url_re = re.compile(r"SMAP.*_[0-9]{8}T[0-9]{6}_.*\.h5")

    source = tl.Unicode().tag(readonly=True)

    @tl.default("source")
    def _property_source_default(self):
        v = _infer_SMAP_product_version("SPL4SMLM", SMAP_BASE_URL(), self.auth_session)
        url = SMAP_BASE_URL() + "/SPL4SMLM.%03d/2015.03.31/" % (v)
        r = _get_from_url(url, self.auth_session)
        if not r:
            return "None"
        n = self.file_url_re.search(r.text).group()
        return url + n

    property = tl.Enum(
        [
            "clsm_dzsf",
            "mwrtm_bh",
            "clsm_cdcr2",
            "mwrtm_poros",
            "clsm_dzgt3",
            "clsm_dzgt2",
            "mwrtm_rghhmax",
            "mwrtm_rghpolmix",
            "clsm_dzgt1",
            "clsm_wp",
            "mwrtm_lewt",
            "clsm_dzgt4",
            "clsm_cdcr1",
            "cell_elevation",
            "mwrtm_rghwmin",
            "clsm_dzrz",
            "mwrtm_vegcls",
            "mwrtm_bv",
            "mwrtm_rghwmax",
            "mwrtm_rghnrh",
            "clsm_dztsurf",
            "mwrtm_rghhmin",
            "mwrtm_wangwp",
            "mwrtm_wangwt",
            "clsm_dzgt5",
            "clsm_dzpr",
            "clsm_poros",
            "cell_land_fraction",
            "mwrtm_omega",
            "mwrtm_soilcls",
            "clsm_dzgt6",
            "mwrtm_rghnrv",
            "mwrtm_clay",
            "mwrtm_sand",
        ]
    ).tag(attr=True)

    @tl.default("layerkey")
    def _layerkey_default(self):
        return "{rdk}" + self.property

    @common_doc(COMMON_DOC)
    @cache_func("native.coordinates")
    def get_native_coordinates(self):
        """{get_native_coordinates}
        """
        ds = self.dataset
        lons = np.array(ds[self.lonkey][:, :])
        lats = np.array(ds[self.latkey][:, :])
        lons[lons == self.nan_vals[0]] = np.nan
        lats[lats == self.nan_vals[0]] = np.nan
        lons = np.nanmean(lons, axis=0)
        lats = np.nanmean(lats, axis=1)
        coords = podpac.Coordinates([lats, lons], dims=["lat", "lon"])
        return coords


class SMAPPorosity(SMAPProperties):
    """Retrieve the specific SMAP property: Porosity

    Attributes
    ----------
    property : str, Optional
        Uses 'clsm_poros'
    """

    property = tl.Unicode("clsm_poros")


class SMAPWilt(SMAPProperties):
    """Retrieve the specific SMAP property: Wilting Point

    Attributes
    ----------
    property : str, Optional
        Uses 'clsm_wp'
    """

    property = tl.Unicode("clsm_wp")


@common_doc(COMMON_DOC)
class SMAPDateFolder(podpac.compositor.OrderedCompositor):
    """Compositor of all the SMAP source urls present in a particular folder which is defined for a particular date

    Attributes
    ----------
    auth_class : {auth_class}
    auth_session : {auth_session}
    base_url : {base_url}
    cache_native_coordinates : bool, optional
        Default is False. If True, the native_coordinates will be cached to disk after being computed the first time
    date_time_url_re : SRE_Pattern
        Regular expression used to retrieve the date and time from the filename if file_url_re matches
    date_url_re : SRE_Pattern
        Regular expression used to retrieve the date from the filename if file_url_re2 matches
    file_url_re : SRE_Pattern
        Regular expression used to find files in a folder that match the expected format of a SMAP source file
    file_url_re2 : SRE_Pattern
        Same as file_url_re, but for variation of SMAP files that do not contain time in the filename
    folder_date : str
        The name of the folder. This is used to construct the OpenDAP URL from the base_url
    latlon_delta : float, optional
        Default is 1.5 degrees. For SMAP files that contain LAT-LON data (i.e. SMAP-Sentinel), how many degrees does the
        tile cover?
    latlon_url_re : SRE_Pattern
        Regular expression used to find the lat-lon coordinates associated with the file from the file name
    layerkey : {layerkey}
    password : {password}
    product : str
        {product}
    version : int
        {version}
    username : {username}
    """

    auth_session = tl.Instance(authentication.EarthDataSession)
    auth_class = tl.Type(authentication.EarthDataSession)
    username = tl.Unicode(None, allow_none=True)
    password = tl.Unicode(None, allow_none=True)
    # Need to overwrite parent because of recursive definition
    outputs = None

    @tl.validate("source_coordinates")
    def _validate_source_coordinates(self, d):
        # Need to overwrite parent because of recursive definition
        return d["value"]

    @tl.default("cache_ctrl")
    def _cache_ctrl_default(self):
        # append disk store to default cache_ctrl if not present
        default_ctrl = cache.get_default_cache_ctrl()
        stores = default_ctrl._cache_stores
        if not any(isinstance(store, cache.DiskCacheStore) for store in default_ctrl._cache_stores):
            stores.append(cache.DiskCacheStore())
        return cache.CacheCtrl(stores)

    @tl.default("auth_session")
    def _auth_session_default(self):
        return self.auth_class(username=self.username, password=self.password, product_url=SMAP_BASE_URL())

    base_url = tl.Unicode().tag(attr=True)

    @tl.default("base_url")
    def _base_url_default(self):
        return SMAP_BASE_URL()

    product = tl.Enum(SMAP_PRODUCT_MAP.coords["product"].data.tolist()).tag(attr=True)
    version = tl.Int(allow_none=True).tag(attr=True)

    @tl.default("version")
    def _detect_product_version(self):
        return _infer_SMAP_product_version(self.product, self.base_url, self.auth_session)

    folder_date = tl.Unicode("").tag(attr=True)

    file_url_re = re.compile(r".*_[0-9]{8}T[0-9]{6}_.*\.h5")
    file_url_re2 = re.compile(r".*_[0-9]{8}_.*\.h5")
    date_time_url_re = re.compile(r"[0-9]{8}T[0-9]{6}")
    date_url_re = re.compile(r"[0-9]{8}")
    latlon_url_re = re.compile(r"[0-9]{3}[E,W][0-9]{2}[N,S]")

    latlon_delta = tl.Float(default_value=1.5).tag(attr=True)

    cache_native_coordinates = tl.Bool(False)

    layerkey = tl.Unicode()

    @tl.default("layerkey")
    def _layerkey_default(self):
        return SMAP_PRODUCT_MAP.sel(product=self.product, attr="layerkey").item()

    @tl.observe("layerkey")
    def _layerkey_change(self, change):
        if change["old"] != change["new"] and change["old"] != "":
            for s in self.sources:
                s.layerkey = change["new"]

    def __repr__(self):
        rep = "{}".format("SMAP")
        rep += "\n\tproduct: {}".format(self.product)

        return rep

    @property
    def source(self):
        """URL to OpenDAP dataset folder

        Returns
        -------
        str
            URL to OpenDAP dataset folder
        """
        return "/".join([self.base_url, "%s.%03d" % (self.product, self.version), self.folder_date])

    @tl.default("sources")
    def sources_default(self):
        """SMAPSource objects pointing to URLs of specific SMAP files in the folder

        Returns
        -------
        np.ndarray(dtype=object(SMAPSource))
            Array of SMAPSource instances tied to specific SMAP files
        """
        # Swapped the try and except blocks. SMAP filenames may change version numbers, which causes cached source to
        # break. Hence, try to get the new source everytime, unless data is offline, in which case rely on the cache.
        try:
            _, _, sources = self.get_available_coords_sources()
            self.put_cache(sources, "sources", overwrite=True)
        except:  # No internet or authentication error
            try:
                sources = self.get_cache("sources")
            except NodeException as e:
                raise NodeException(
                    "Connection or Authentication error, and no disk cache to fall back on for determining sources."
                )

        b = self.source + "/"
        time_crds = self.source_coordinates["time"]
        if time_crds.is_monotonic and time_crds.is_uniform and time_crds.size > 1:
            tol = time_crds.coordinates[1] - time_crds.coordinates[0]
        else:
            tol = self.source_coordinates["time"].coordinates[0]
            tol = tol - tol
            tol = np.timedelta64(1, dtype=(tol.dtype))

        src_objs = [
            SMAPSource(
                source=b + s,
                auth_session=self.auth_session,
                layerkey=self.layerkey,
                interpolation={"method": "nearest", "time_tolerance": tol},
            )
            for s in sources
        ]
        return np.array(src_objs)

    @tl.default("is_source_coordinates_complete")
    def src_crds_complete_default(self):
        """Flag use to optimize creation of native_coordinates. If the source_coordinates are complete,
        native_coordinates can easily be reconstructed, and same with shared coordinates. 

        Returns
        -------
        bool
            Flag indicating whether the source coordinates completely describe the source's coordinates for that dimension
        """
        return self.product not in SMAP_INCOMPLETE_SOURCE_COORDINATES

    def get_source_coordinates(self):
        """{source_coordinates}
        """
        try:
            times, latlon, _ = self.get_available_coords_sources()
        except:
            try:
                return self.get_cache("source.coordinates")
            except NodeException as e:
                raise NodeException(
                    "Connection or Authentication error, and no disk cache to fall back on for determining sources."
                )

        if latlon is not None and latlon.size > 0:
            crds = podpac.Coordinates([[times, latlon[:, 0], latlon[:, 1]]], dims=["time_lat_lon"])
        else:
            crds = podpac.Coordinates([times], dims=["time"])
        self.put_cache(crds, "source.coordinates", overwrite=True)
        return crds

    @cache_func("shared.coordinates")
    def get_shared_coordinates(self):
        """Coordinates that are shared by all files in the folder.

        Returns
        -------
        podpac.Coordinates
            Coordinates shared by all files in the folder
        """
        if self.product in SMAP_INCOMPLETE_SOURCE_COORDINATES:
            return None

        coords = copy.deepcopy(self.sources[0].native_coordinates)
        return coords.drop("time")

    def get_available_coords_sources(self):
        """Read NSIDC site for available coordinate sources

        Returns
        -------
        np.ndarray
            Available times of sources in the folder
        np.ndarray
            Available lat lon coordinates of sources in the folder, None if empty
        np.ndarray
            The url's of the sources

        Raises
        ------
        RuntimeError
            If the NSIDC website cannot be accessed 
        """
        url = self.source
        r = _get_from_url(url, self.auth_session)
        if r is None:
            _logger.warning("Could not contact {} to retrieve source coordinates".format(url))
            return np.array([]), None, np.array([])
        soup = bs4.BeautifulSoup(r.text, "lxml")
        a = soup.find_all("a")
        file_regex = self.file_url_re
        file_regex2 = self.file_url_re2
        date_time_regex = self.date_time_url_re
        date_regex = self.date_url_re
        latlon_regex = self.latlon_url_re
        times = []
        latlons = []
        sources = []
        for aa in a:
            t = aa.get_text().strip("\n")
            if "h5.iso.xml" in t:
                continue
            m = file_regex.match(t)
            m2 = file_regex2.match(t)

            lonlat = None
            if m:
                date_time = date_time_regex.search(m.group()).group()
                times.append(smap2np_date(date_time))

            elif m2:
                m = m2
                date = date_regex.search(m.group()).group()
                times.append(smap2np_date(date))
            if m:
                sources.append(m.group())
                lonlat = latlon_regex.search(m.group())
            if lonlat:
                lonlat = lonlat.group()
                latlons.append(
                    (
                        float(lonlat[4:6]) * (1 - 2 * (lonlat[6] == "S")),
                        float(lonlat[:3]) * (1 - 2 * (lonlat[3] == "W")),
                    )
                )

        times = np.atleast_1d(np.array(times).squeeze())
        latlons = np.array(latlons)
        sources = np.array(sources)
        I = np.argsort(times)
        if latlons.shape[0] == times.size:
            return times[I], latlons[I], sources[I]
        return times[I], None, sources[I]

    @property
    @common_doc(COMMON_DOC)
    def keys(self):
        """{keys}
        """
        return self.sources[0].keys

    @property
    def base_definition(self):
        """ Definition for SMAP node. Sources not required as these are computed.
        """
        d = super(podpac.compositor.Compositor, self).base_definition
        d["interpolation"] = self.interpolation
        return d


@common_doc(COMMON_DOC)
class SMAP(podpac.compositor.OrderedCompositor):
    """Compositor of all the SMAPDateFolder's for every available SMAP date. Essentially a compositor of all SMAP data
    for a particular product.

    Attributes
    ----------
    auth_class : {auth_class}
    auth_session : {auth_session}
    base_url : {base_url}
    date_url_re : SRE_Pattern
        Regular expression used to extract all folder dates (or folder names) for the particular SMAP product.
    layerkey : {layerkey}
    password : {password}
    product : str
        {product}
    username : {username}
    """

    # Need to overwrite parent because of recursive definition
    outputs = None
    base_url = tl.Unicode().tag(attr=True)

    @tl.validate("source_coordinates")
    def _validate_source_coordinates(self, d):
        # Need to overwrite parent because of recursive definition
        return d["value"]

    @tl.default("base_url")
    def _base_url_default(self):
        return SMAP_BASE_URL()

    product = tl.Enum(SMAP_PRODUCT_MAP.coords["product"].data.tolist(), default_value="SPL4SMAU").tag(attr=True)
    version = tl.Int(allow_none=True).tag(attr=True)

    @tl.default("version")
    def _detect_product_version(self):
        return _infer_SMAP_product_version(self.product, self.base_url, self.auth_session)

    date_url_re = re.compile(r"[0-9]{4}\.[0-9]{2}\.[0-9]{2}")

    auth_session = tl.Instance(authentication.EarthDataSession)
    auth_class = tl.Type(authentication.EarthDataSession)
    username = tl.Unicode(None, allow_none=True).tag(attr=True)
    password = tl.Unicode(None, allow_none=True).tag(attr=True)

    @tl.default("auth_session")
    def _auth_session_default(self):
        return self.auth_class(username=self.username, password=self.password, product_url=SMAP_BASE_URL())

    layerkey = tl.Unicode()

    @tl.default("layerkey")
    def _layerkey_default(self):
        return SMAP_PRODUCT_MAP.sel(product=self.product, attr="layerkey").item()

    @tl.observe("layerkey")
    def _layerkey_change(self, change):
        if change["old"] != change["new"] and change["old"] != "":
            for s in self.sources:
                s.layerkey = change["new"]

    def __repr__(self):
        rep = "{}".format("SMAP")
        rep += "\n\tproduct: {}".format(self.product)
        rep += "\n\tinterpolation: {}".format(self.interpolation)

        return rep

    @property
    def source(self):
        """The source is used for a unique name to cache SMAP products. 

        Returns
        -------
        str
            The SMAP product name.
        """
        return "%s.%03d" % (self.product, self.version)

    @tl.default("sources")
    def sources_default(self):
        """SMAPDateFolder objects pointing to specific SMAP folders

        Returns
        -------
        np.ndarray(dtype=object(SMAPDateFolder))
            Array of SMAPDateFolder instances tied to specific SMAP folders
        """
        dates = self.get_available_times_dates()[1]
        src_objs = np.array(
            [
                SMAPDateFolder(
                    product=self.product,
                    version=self.version,
                    folder_date=date,
                    shared_coordinates=self.shared_coordinates,
                    auth_session=self.auth_session,
                    layerkey=self.layerkey,
                )
                for date in dates
            ]
        )
        return src_objs

    @common_doc(COMMON_DOC)
    def find_coordinates(self):
        """
        {native_coordinates}
        
        Notes
        -----
        These coordinates are computed, assuming dataset is regular.
        """
        if self.product in SMAP_IRREGULAR_COORDINATES:
            raise Exception("Native coordinates too large. Try using get_filename_coordinates_sources().")

        shared = self.get_shared_coordinates()
        partial_sources = self.get_source_coordinates()["time"].coordinates
        complete_source_0 = self.sources[0].get_source_coordinates()["time"].coordinates
        offset = complete_source_0 - partial_sources[0]
        full_times = (partial_sources[:, None] + offset[None, :]).ravel()
        return [merge_dims([podpac.Coordinates([full_times], ["time"]), shared])]

    @common_doc(COMMON_DOC)
    def get_source_coordinates(self):
        """{source_coordinates}
        """
        return podpac.Coordinates([self.get_available_times_dates()[0]], dims=["time"])

    def get_available_times_dates(self):
        """Returns the available folder dates in the SMAP product

        Returns
        -------
        np.ndarray
            Array of dates in numpy datetime64 format
        list
            list of dates in SMAP date format

        Raises
        ------
        RuntimeError
            If the http resource could not be accessed (check Earthdata login credentials)
        """
        url = "/".join([self.base_url, "%s.%03d" % (self.product, self.version)])
        r = _get_from_url(url, self.auth_session)
        if r is None:
            _logger.warning("Could not contact {} to retrieve source coordinates".format(url))
            return np.array([]), []
        soup = bs4.BeautifulSoup(r.text, "lxml")
        a = soup.find_all("a")
        regex = self.date_url_re
        times = []
        dates = []
        for aa in a:
            m = regex.match(aa.get_text())
            if m:
                times.append(np.datetime64(m.group().replace(".", "-")))
                dates.append(m.group())
        times.sort()
        dates.sort()
        return np.array(times), dates

    @cache_func("shared.coordinates")
    def get_shared_coordinates(self):
        """Coordinates that are shared by all files in the SMAP product family. 

        Returns
        -------
        podpac.Coordinates
            Coordinates shared by all files in the SMAP product. 

        Notes
        ------
        For example, the gridded SMAP data have the same lat-lon coordinates in every file (global at some resolution), 
        and the only difference between files is the time coordinate. 
        This is not true for the SMAP-Sentinel product, in which case this function returns None
        """
        if self.product in SMAP_INCOMPLETE_SOURCE_COORDINATES:
            return None

        coords = SMAPDateFolder(
            product=self.product,
            version=self.version,
            folder_date=self.get_available_times_dates()[1][0],
            auth_session=self.auth_session,
        ).shared_coordinates
        return coords

    def get_filename_coordinates_sources(self, bounds=None, update_cache=False):
        """Returns coordinates solely based on the filenames of the sources. This function was motivated by the 
        SMAP-Sentinel product, which does not have regularly stored tiles (in space and time). 

        Parameters
        -----------
        bounds: podpac.Coordinates, Optional
            Default is None. Return the coordinates based on filenames of the source only within the specified bounds. 
            When not None, the result is not cached.
            
        update_cache: bool, optional
            Default is False. The results of this call are automatically cached to disk. This function will try to 
            update the cache if new data arrives. Only set this flag to True to rebuild the entire index locally (which
            may be needed when version numbers in the filenames change).

        Returns
        -------
        podpac.Coordinates
            Coordinates of all the sources in the product family
        Container
            Container that will generate an array of the SMAPSources pointing to unique OpenDAP urls corresponding to
            the returned coordinates
        

        Notes
        ------
        The outputs of this function can be used to find source that overlap spatially or temporally with a subset 
        region specified by the user.

        If 'bounds' is not specified, the result is cached for faster future access after the first invocation.
        
        This call uses NASA's Common Metadata Repository (CMR) and requires an internet connection.
        """

        def cmr_query(kwargs=None, bounds=None):
            """ Helper function for making and parsing cmr queries. This is used for building the initial index
            and for updating the cached index with new data.
            """
            if not kwargs:
                kwargs = {}

            # Set up regular expressions and maps to convert filenames to coordinates
            date_re = self.sources[0].date_url_re
            date_time_re = self.sources[0].date_time_url_re
            latlon_re = self.sources[0].latlon_url_re

            def datemap(x):
                m = date_time_re.search(x)
                if not m:
                    m = date_re.search(x)
                return smap2np_date(m.group())

            def latlonmap(x):
                m = latlon_re.search(x)
                if not m:
                    return ()
                lonlat = m.group()
                return (
                    float(lonlat[4:6]) * (1 - 2 * (lonlat[6] == "S")),
                    float(lonlat[:3]) * (1 - 2 * (lonlat[3] == "W")),
                )

            # Restrict the query to any specified bounds
            if bounds:
                kwargs["temporal"] = ",".join([str(b.astype("datetime64[s]")) for b in bounds["time"].area_bounds])

            # Get CMR data
            filenames = nasaCMR.search_granule_json(
                auth_session=self.auth_session,
                entry_map=lambda x: x["producer_granule_id"],
                short_name=self.product,
                **kwargs,
            )
            if not filenames:
                return Coordinates([]), [], []

            # Extract coordinate information from filenames
            # filenames.sort()  # Assume it comes sorted...
            dims = ["time"]
            dates = [d for d in np.array(list(map(datemap, filenames))).squeeze()]
            coords = [dates]
            if latlonmap(filenames[0]):
                latlons = list(map(latlonmap, filenames))
                lats = np.array([l[0] for l in latlons])
                lons = np.array([l[1] for l in latlons])
                dims = ["time_lat_lon"]
                coords = [[dates, lats, lons]]

            # Create PODPAC Coordinates object, and return relevant data structures
            crds = Coordinates(coords, dims)
            return crds, filenames, dates

        # Create kwargs for making a SMAP source
        create_kwargs = {"auth_session": self.auth_session, "layer_key": self.layerkey}
        if self.interpolation:
            create_kwargs["interpolation"] = self.interpolation

        try:  # Try retrieving index from cache
            if update_cache:
                raise NodeException
            crds, sources = (self.get_cache("filename.coordinates"), self.get_cache("filename.sources"))
            try:  # update the cache
                # Specify the bounds based on the last entry in the cached coordinates
                # Add a minute to the bounds to make sure we get unique coordinates
                kwargs = {
                    "temporal": str(crds["time"].area_bounds[-1].astype("datetime64[s]") + np.timedelta64(5, "m")) + "/"
                }
                crds_new, filenames_new, dates_new = cmr_query(kwargs)

                # Update the cached coordinates
                if len(filenames_new) > 1:
                    # Append the new coordinates to the relevant data structures
                    crdsfull = podpac.coordinates.concat([crds, crds_new])
                    sources.filenames.extend(filenames_new)
                    sources.dates.extend(dates_new)

                    # Make sure the coordinates are unique
                    # (we actually know SMAP-Sentinel is NOT unique, so we can't do this)
                    # crdsunique, inds = crdsfull.unique(return_indices=True)
                    # sources.filenames = np.array(sources.filenames)[inds[0]].tolist()
                    # sources.dates = np.array(sources.dates)[inds[0]].tolist()

                    # Update the cache
                    if filenames_new:
                        self.put_cache(crdsfull, "filename.coordinates", overwrite=True)
                        self.put_cache(sources, "filename.sources", overwrite=True)

            except Exception as e:  # likely a connection or authentication error
                _logger.warning("Failed to update cached filenames: ", str(e))

            if bounds:  # Restrict results to user-specified bounds
                crds, I = crds.intersect(bounds, outer=True, return_indices=True)
                sources = sources.intersect(I[0])

        except NodeException:  # Not in cache or forced update
            crds, filenames, dates = cmr_query(bounds=bounds)
            sources = GetSMAPSources(self.product, filenames, dates, create_kwargs)

            if bounds is None:
                self.put_cache(crds, "filename.coordinates", overwrite=update_cache)
                self.put_cache(sources, "filename.sources", overwrite=update_cache)

        # Update the auth_session and/or interpolation and/or other keyword arguments in the sources class
        sources.create_kwargs = create_kwargs
        return crds, sources

    @property
    def base_ref(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return "{0}_{1}".format(self.__class__.__name__, self.product)

    @property
    def base_definition(self):
        """ Definition for SMAP node. Sources not required as these are computed.
        """
        d = super(podpac.compositor.Compositor, self).base_definition
        d["interpolation"] = self.interpolation
        return d

    @property
    @common_doc(COMMON_DOC)
    def keys(self):
        """{keys}
        """
        return self.sources[0].keys


class SMAPBestAvailable(podpac.compositor.OrderedCompositor):
    """Compositor of SMAP-Sentinel and the Level 4 SMAP Analysis Update soil moisture
    """

    @tl.default("sources")
    def sources_default(self):
        """Orders the compositor of SPL2SMAP_S in front of SPL4SMAU

        Returns
        -------
        np.ndarray(dtype=object(SMAP))
            Array of SMAP product sources
        """
        src_objs = np.array(
            [
                SMAP(interpolation=self.interpolation, product="SPL2SMAP_S"),
                SMAP(interpolation=self.interpolation, product="SPL4SMAU"),
            ]
        )
        return src_objs

    def __repr__(self):
        rep = "{}".format("SMAP (Best Available)")
        return rep

    def get_shared_coordinates(self):
        return None  # NO shared coordiantes


class GetSMAPSources(object):
    def __init__(self, product, filenames, dates, create_kwargs):
        self.product = product
        self.filenames = filenames
        self.dates = dates
        self.create_kwargs = create_kwargs
        self._base_url = None

    def __getitem__(self, slc):
        return_slice = slice(None)
        if not isinstance(slc, slice):
            if isinstance(slc, (np.integer, int)):
                slc = slice(slc, slc + 1)
                return_slice = 0
            else:
                raise ValueError("Invalid slice")
        base_url = self.base_url
        source_urls = [base_url + np2smap_date(d)[:10] + "/" + f for d, f in zip(self.dates[slc], self.filenames[slc])]
        return np.array([SMAPSource(source=s, **self.create_kwargs) for s in source_urls], object)[return_slice]

    @property
    def base_url(self):
        if not self._base_url:
            self._base_url = SMAPDateFolder(
                product=self.product, folder_date="00001122", auth_session=self.create_kwargs["auth_session"]
            ).source[:-8]
        return self._base_url

    def __len__(self):
        return len(self.filenames)

    def intersect(self, I):
        return GetSMAPSources(
            product=self.product,
            filenames=[self.filenames[i] for i in I],
            dates=[self.dates[i] for i in I],
            create_kwargs=self.create_kwargs,
        )
