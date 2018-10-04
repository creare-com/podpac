"""Specialized PODPAC nodes use to access SMAP data via OpenDAP from nsidc.

Attributes
----------
SMAP_BASE_URL : str
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

import re
import copy

from bs4 import BeautifulSoup
from six import string_types
import numpy as np
import xarray as xr
import traitlets as tl

# fixing problem with older versions of numpy
if not hasattr(np, 'isnat'):
    def isnat(a):
        return a.astype(str) == 'None'
    np.isnat = isnat

# Internal dependencies
import podpac
from podpac.core.data import types as datatype
from podpac.core import authentication
from podpac.core.utils import common_doc
from podpac.core.data.datasource import COMMON_DATA_DOC

COMMON_DOC = COMMON_DATA_DOC.copy()
COMMON_DOC.update(
    {'smap_date': 'str\n        SMAP date string',
     'np_date':   'np.datetime64\n        Numpy date object',
     'auth_class': ('EarthDataSession (Class object)\n        Class used to make an authenticated session from a'
                   ' username and password (both are defined in base class)'),
     'auth_session' : ('Instance(EarthDataSession)\n        Authenticated session used to make http requests using'
                      'NASA Earth Data Login credentials'),
     'base_url' : 'str\n        Url to nsidc openDAP server', 
     'layerkey': ('str\n        Key used to retrieve data from OpenDAP dataset. This specifies the key used to retrieve '
                 'the data'),
     'password': 'User\'s EarthData password',
     'username': 'User\'s EarthData username',
     'product': 'SMAP Product name',
     'source_coordinates': '''Returns the coordinates that uniquely describe each source

        Returns
        -------
        podpac.Coordinates
            Coordinates that uniquely describe each source''',
     'keys': '''Available layers that are in the OpenDAP dataset

        Returns
        -------
        List
            The list of available keys from the OpenDAP dataset. Any of these keys can be set as self.datakey.

        Notes
        -----
        This function assumes that all of the keys in the available dataset are the same for every file.
        ''',
       })


# Optional Dependencies
try:
    import boto3
except:
    boto3 = None

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
        ymd = '-'.join([date[:4], date[4:6], date[6:8]])
        if len(date) == 15:
            HMS = ' ' + ':'.join([date[9:11], date[11:13], date[13:15]])
        else:
            HMS = ''
        date = np.array([ymd + HMS], dtype='datetime64')
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
        date = str(date).replace('-', '.')
    return date

# NOTE: {rdk} will be substituted for the entry's 'rootdatakey'
SMAP_PRODUCT_DICT = {
    #'<Product>.ver': ['latkey',               'lonkey',                     'rootdatakey',                       'layerkey'
    'SPL4SMAU.003':   ['cell_lat',             'cell_lon',                   'Analysis_Data_',                    '{rdk}sm_surface_analysis'],
    'SPL4SMGP.003':   ['cell_lat',             'cell_lon',                   'Geophysical_Data_',                 '{rdk}sm_surface'],
    'SPL3SMA.003':    ['{rdk}latitude',        '{rdk}longitude',             'Soil_Moisture_Retrieval_Data_',     '{rdk}soil_moisture'],
    'SPL3SMAP.003':   ['{rdk}latitude',        '{rdk}longitude',             'Soil_Moisture_Retrieval_Data_',     '{rdk}soil_moisture'],
    'SPL3SMP.004':    ['{rdk}AM_latitude',     '{rdk}AM_longitude',          'Soil_Moisture_Retrieval_Data_',     '{rdk}AM_soil_moisture'],
    'SPL4SMLM.003':   ['cell_lat',             'cell_lon',                   'Land_Model_Constants_Data_',        ''],
    'SPL2SMAP_S.001': ['{rdk}latitude_1km',    '{rdk}longitude_1km',         'Soil_Moisture_Retrieval_Data_1km_', '{rdk}soil_moisture_1km'],
}

SMAP_PRODUCT_MAP = xr.DataArray(list(SMAP_PRODUCT_DICT.values()),
                                dims=['product', 'attr'],
                                coords={'product': list(SMAP_PRODUCT_DICT.keys()),
                                        'attr':['latkey', 'lonkey', 'rootdatakey', 'layerkey']
              }
)

SMAP_INCOMPLETE_SOURCE_COORDINATES = ['SPL2SMAP_S.001']
SMAP_BASE_URL = 'https://n5eil01u.ecs.nsidc.org/opendap/hyrax/SMAP'

@common_doc(COMMON_DOC)
class SMAPSource(datatype.PyDAP):
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

    @tl.default('auth_session')
    def _auth_session_default(self):
        session = self.auth_class(
            username=self.username, password=self.password)
        # check url
        try:
            session.get(self.source + '.dds')
        except:
            return None
        return session

    #date_url_re = re.compile('[0-9]{4}\.[0-9]{2}\.[0-9]{2}')
    date_time_file_url_re = re.compile('[0-9]{8}T[0-9]{6}')
    date_file_url_re = re.compile('[0-9]{8}')

    rootdatakey = tl.Unicode()
    @tl.default('rootdatakey')
    def _rootdatakey_default(self):
        return SMAP_PRODUCT_MAP.sel(product=self.product,
                                    attr='rootdatakey').item()

    layerkey = tl.Unicode()
    @tl.default('layerkey')
    def _layerkey_default(self):
        return SMAP_PRODUCT_MAP.sel(
            product=self.product,
            attr='layerkey').item()

    nan_vals = [-9999.0]

    @property
    def product(self):
        """Returns the SMAP product from the OpenDAP Url

        Returns
        -------
        str
            {product}
        """
        src = self.source.split('/')
        return src[src.index('SMAP')+1]

    @tl.default('datakey')
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
        return SMAP_PRODUCT_MAP.sel(product=self.product, attr='latkey') \
               .item().format(rdk=self.rootdatakey)

    @property
    def lonkey(self):
        """The key used to retrieve the latitude

        Returns
        -------
        str
            OpenDap dataset key for longitude
        """
        return SMAP_PRODUCT_MAP.sel(product=self.product, attr='lonkey').item().format(rdk=self.rootdatakey)

    @common_doc(COMMON_DOC)
    def get_native_coordinates(self):
        """{get_native_coordinates}
        """
        try:
            return self.load_cached_obj('native.coordinates')
        except:
            pass
        times = self.get_available_times()
        ds = self.dataset
        lons = np.array(ds[self.lonkey][:, :])
        lats = np.array(ds[self.latkey][:, :])
        lons[lons == self.nan_vals[0]] = np.nan
        lats[lats == self.nan_vals[0]] = np.nan
        lons = np.nanmean(lons, axis=0)
        lats = np.nanmean(lats, axis=1)
        coords = podpac.Coordinates([lat, lon, time], dims=['time', 'lat', 'lon'])
        self.cache_obj(coords, 'native.coordinates')
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
        if 'SM_P_' in self.source:
            times = times + np.array([6, 18], 'timedelta64[h]')
        return times

    @common_doc(COMMON_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        # We actually ignore the time slice
        s = tuple([slc for d, slc in zip(coordinates.dims, coordinates_index)
                   if 'time' not in d])
        if 'SM_P_' in self.source:
            d = self.initialize_coord_array(coordinates, 'nan')
            am_key = self.layerkey.format(rdk=self.rootdatakey + 'AM_')
            pm_key = self.layerkey.format(rdk=self.rootdatakey + 'PM_') + '_pm'

            try:
                t = self.native_coordinates.coords['time'][0]
                d.loc[dict(time=t)] = np.array(self.dataset[am_key][s])
            except: 
                pass

            try: 
                t = self.native_coordinates.coords['time'][1]
                d.loc[dict(time=t)] = np.array(self.dataset[pm_key][s])
            except: 
                pass

        else:
            data = np.array(self.dataset[self.datakey][s])
            d = self.initialize_coord_array(coordinates, 'data',
                                            fillval=data.reshape(coordinates.shape))
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
         Source OpenDAP url for SMAP properties. Default is ('https://n5eil01u.ecs.nsidc.org/opendap/SMAP/'
                                                             'SPL4SMLM.003/2015.03.31/'
                                                             'SMAP_L4_SM_lmc_00000000T000000_Vv3030_001.h5')
    """

    source = tl.Unicode('https://n5eil01u.ecs.nsidc.org/opendap/SMAP/'
                        'SPL4SMLM.003/2015.03.31/'
                        'SMAP_L4_SM_lmc_00000000T000000_Vv3030_001.h5').tag(attr=True)

    property = tl.Enum(['clsm_dzsf', 'mwrtm_bh', 'clsm_cdcr2', 'mwrtm_poros',
                        'clsm_dzgt3', 'clsm_dzgt2', 'mwrtm_rghhmax',
                        'mwrtm_rghpolmix', 'clsm_dzgt1', 'clsm_wp', 'mwrtm_lewt',
                        'clsm_dzgt4', 'clsm_cdcr1', 'cell_elevation',
                        'mwrtm_rghwmin', 'clsm_dzrz', 'mwrtm_vegcls', 'mwrtm_bv',
                        'mwrtm_rghwmax', 'mwrtm_rghnrh', 'clsm_dztsurf',
                        'mwrtm_rghhmin', 'mwrtm_wangwp', 'mwrtm_wangwt',
                        'clsm_dzgt5', 'clsm_dzpr', 'clsm_poros',
                        'cell_land_fraction', 'mwrtm_omega', 'mwrtm_soilcls',
                        'clsm_dzgt6', 'mwrtm_rghnrv', 'mwrtm_clay', 'mwrtm_sand']).tag(attr=True)

    @tl.default('layerkey')
    def _layerkey_default(self):
        return self.property

    @common_doc(COMMON_DOC)
    def get_native_coordinates(self):
        """{get_native_coordinates}
        """
        try:
            coords = self.load_cached_obj('native.coordinates')
        except:
            ds = self.dataset
            lons = np.array(ds[self.lonkey][:, :])
            lats = np.array(ds[self.latkey][:, :])
            lons[lons == self.nan_vals[0]] = np.nan
            lats[lats == self.nan_vals[0]] = np.nan
            lons = np.nanmean(lons, axis=0)
            lats = np.nanmean(lats, axis=1)
            coords = podpac.Coordinates([lats, lons], dims=['lat', 'lon'])
            self.cache_obj(coords, 'native.coordinates')
        
        return coords

class SMAPPorosity(SMAPProperties):
    """Retrieve the specific SMAP property: Porosity

    Attributes
    ----------
    property : str, Optional
        Uses 'clsm_poros'
    """
    property = tl.Unicode('clsm_poros')

class SMAPWilt(SMAPProperties):
    """Retrieve the specific SMAP property: Wilting Point

    Attributes
    ----------
    property : str, Optional
        Uses 'clsm_wp'
    """
    property = tl.Unicode('clsm_wp')

@common_doc(COMMON_DOC)
class SMAPDateFolder(podpac.OrderedCompositor):
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
    username : {username}
    """

    auth_session = tl.Instance(authentication.EarthDataSession)
    auth_class = tl.Type(authentication.EarthDataSession)
    username = tl.Unicode(None, allow_none=True).tag(attr=True)
    password = tl.Unicode(None, allow_none=True).tag(attr=True)

    @tl.default('auth_session')
    def _auth_session_default(self):
        session = self.auth_class(username=self.username, password=self.password)
        return session

    base_url = tl.Unicode(SMAP_BASE_URL).tag(attr=True)
    product = tl.Enum(SMAP_PRODUCT_MAP.coords['product'].data.tolist()).tag(attr=True)
    folder_date = tl.Unicode(u'').tag(attr=True)

    file_url_re = re.compile(r'.*_[0-9]{8}T[0-9]{6}_.*\.h5')
    file_url_re2 = re.compile(r'.*_[0-9]{8}_.*\.h5')
    date_time_url_re = re.compile(r'[0-9]{8}T[0-9]{6}')
    date_url_re = re.compile(r'[0-9]{8}')
    latlon_url_re = re.compile(r'[0-9]{3}[E,W][0-9]{2}[N,S]')

    latlon_delta = tl.Float(default_value=1.5).tag(attr=True)

    cache_native_coordinates = tl.Bool(False)

    layerkey = tl.Unicode()
    @tl.default('layerkey')
    def _layerkey_default(self):
        return SMAP_PRODUCT_MAP.sel(product=self.product, attr='layerkey').item()

    @tl.observe('layerkey')
    def _layerkey_change(self, change):
        if change['old'] != change['new'] and change['old'] != '':
            for s in self.sources:
                s.layerkey = change['new']

    @property
    def source(self):
        """URL to OpenDAP dataset folder

        Returns
        -------
        str
            URL to OpenDAP dataset folder
        """
        return '/'.join([self.base_url, self.product, self.folder_date])

    @tl.default('sources')
    def sources_default(self):
        """SMAPSource objects pointing to URLs of specific SMAP files in the folder

        Returns
        -------
        np.ndarray(dtype=object(SMAPSource))
            Array of SMAPSource instances tied to specific SMAP files
        """
        try:
            sources = self.load_cached_obj('sources')
        except:
            _, _, sources = self.get_available_coords_sources()
            self.cache_obj(sources, 'sources')

        b = self.source + '/'
        tol = self.source_coordinates['time'].delta / 2
        if np.isnat(tol):
            tol = self.source_coordinates['time'].coordinates[0]
            tol = tol - tol
            tol = np.timedelta64(1, dtype=(tol.dtype))

        src_objs = [
            SMAPSource(source=b+s, interpolation_tolerance=tol, auth_session=self.auth_session, layerkey=self.layerkey)
            for s in sources]
        return np.array(src_objs)

    @tl.default('is_source_coordinates_complete')
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
            return self.load_cached_obj('source.coordinates')
        except:
            pass
        times, latlon, _ = self.get_available_coords_sources()

        if latlon is not None and latlon.size > 0:
            lat = podpac.Coord(latlon[:, 0], delta=self.latlon_delta) # TODO
            lon = podpac.Coord(latlon[:, 1], delta=self.latlon_delta) # TODO
            crds = podpac.Coordinates([[times, lat, lon]], dims=['lat_lon_time'])
        else:
            crds = podpac.Coordinates([times], dims=['times'])
        self.cache_obj(crds, 'source.coordinates')
        return crds

    def get_shared_coordinates(self):
        """Coordinates that are shared by all files in the folder.

        Returns
        -------
        podpac.Coordinates
            Coordinates shared by all files in the folder
        """
        if self.product in SMAP_INCOMPLETE_SOURCE_COORDINATES:
            return None

        try:
            return self.load_cached_obj('shared.coordinates')
        except:
            pass

        coords = copy.deepcopy(self.sources[0].native_coordinates)
        del coords._coords['time']
        self.cache_obj(coords, 'shared.coordinates')
        return coords

    def get_available_coords_sources(self):
        """Summary

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
        r = self.auth_session.get(url)
        if r.status_code != 200:
            r = self.auth_session.get(url.replace('opendap/hyrax/', ''))
            if r.status_code != 200:
                raise RuntimeError('HTTP error: <%d>\n' % (r.status_code) + r.text[:256])
        soup = BeautifulSoup(r.text, 'lxml')
        a = soup.find_all('a')
        file_regex = self.file_url_re
        file_regex2 = self.file_url_re2
        date_time_regex = self.date_time_url_re
        date_regex = self.date_url_re
        latlon_regex = self.latlon_url_re
        times = []
        latlons = []
        sources = []
        for aa in a:
            t = aa.get_text().strip('\n')
            if 'h5.iso.xml' in t:
                continue
            m = file_regex.match(t)
            m2 = file_regex2.match(t)

            lonlat = None
            if m:
                date_time = date_time_regex.search(m.group()).group()
                times.append(np.datetime64(
                    '%s-%s-%sT%s:%s:%s' % (date_time[:4], date_time[4:6], date_time[6:8], date_time[9:11],
                                           date_time[11:13], date_time[13:15])
                ))

            elif m2:
                m = m2
                date = date_regex.search(m.group()).group()
                times.append(np.datetime64('%s-%s-%s' % (date[:4], date[4:6], date[6:8])))
            if m:
                sources.append(m.group())
                lonlat = latlon_regex.search(m.group())
            if lonlat:
                lonlat = lonlat.group()
                latlons.append((float(lonlat[4:6]) * (1 - 2 * (lonlat[6] == 'S')),
                                float(lonlat[:3]) * (1 - 2 * (lonlat[3] == 'W'))
                                ))

        times = np.array(times)
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

@common_doc(COMMON_DOC)
class SMAP(podpac.OrderedCompositor):
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

    base_url = tl.Unicode(SMAP_BASE_URL).tag(attr=True)
    product = tl.Enum(SMAP_PRODUCT_MAP.coords['product'].data.tolist(), 
                      default_value='SPL4SMAU.003').tag(attr=True)
    date_url_re = re.compile(r'[0-9]{4}\.[0-9]{2}\.[0-9]{2}')

    auth_session = tl.Instance(authentication.EarthDataSession)
    auth_class = tl.Type(authentication.EarthDataSession)
    username = tl.Unicode(None, allow_none=True).tag(attr=True)
    password = tl.Unicode(None, allow_none=True).tag(attr=True)

    @tl.default('auth_session')
    def _auth_session_default(self):
        session = self.auth_class(username=self.username, password=self.password)
        return session

    layerkey = tl.Unicode()
    @tl.default('layerkey')
    def _layerkey_default(self):
        return SMAP_PRODUCT_MAP.sel(
            product=self.product,
            attr='layerkey').item()

    @tl.observe('layerkey')
    def _layerkey_change(self, change):
        if change['old'] != change['new'] and change['old'] != '':
            for s in self.sources:
                s.layerkey = change['new']

    @property
    def source(self):
        """The source is used for a unique name to cache SMAP products. 

        Returns
        -------
        str
            The SMAP product name.
        """
        return self.product

    @tl.default('sources')
    def sources_default(self):
        """SMAPDateFolder objects pointing to specific SMAP folders

        Returns
        -------
        np.ndarray(dtype=object(SMAPDateFolder))
            Array of SMAPDateFolder instances tied to specific SMAP folders
        """
        dates = self.get_available_times_dates()[1]
        src_objs = np.array([
            SMAPDateFolder(product=self.product, folder_date=date,
                           shared_coordinates=self.shared_coordinates,
                           auth_session=self.auth_session,
                           layerkey=self.layerkey)
            for date in dates])
        return src_objs

    def get_source_coordinates(self):
        """{source_coordinates}
        """
        return podpac.Coordinates([self.get_available_times_dates()[0]], dims=['time'])

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
        url = '/'.join([self.base_url, self.product])
        r = self.auth_session.get(url)
        if r.status_code != 200:
            r = self.auth_session.get(url.replace('opendap/hyrax/', ''))
            if r.status_code != 200:
                raise RuntimeError('HTTP error: <%d>\n' % (r.status_code)
                                   + r.text[:256])
        soup = BeautifulSoup(r.text, 'lxml')
        a = soup.find_all('a')
        regex = self.date_url_re
        times = []
        dates = []
        for aa in a:
            m = regex.match(aa.get_text())
            if m:
                times.append(np.datetime64(m.group().replace('.', '-')))
                dates.append(m.group())
        times.sort()
        dates.sort()
        return np.array(times), dates

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

        try:
            return self.load_cached_obj('shared.coordinates')
        except:
            pass

        coords = SMAPDateFolder(product=self.product,
                                folder_date=self.get_available_times_dates()[1][0],
                                auth_session=self.auth_session,
                               ).shared_coordinates
        self.cache_obj(coords, 'shared.coordinates')
        return coords

    def get_partial_native_coordinates_sources(self):
        """Returns coordinates solely based on the filenames of the sources. This function was motivated by the 
        SMAP-Sentinel product, which does not have regularly stored tiles (in space and time). 

        Returns
        -------
        podpac.Coordinates
            Coordinates of all the sources in the product family
        np.ndarray(dtype=object(SMAPSource))
            Array of all the SMAPSources pointing to unique OpenDAP urls corresponding to the partial native coordinates

        Notes
        ------
        The outputs of this function can be used to find source that overlap spatially or temporally with a subset 
        region specified by the user.
        """
        try:
            return (self.load_cached_obj('partial_native.coordinates'),
                    self.load_cached_obj('partial_native.sources'))
        except:
            pass

        crds = self.sources[0].source_coordinates
        sources = [self.sources[0].sources]
        for s in self.sources[1:]:
            if np.prod(s.source_coordinates.shape) > 0:
                crds = crds + s.source_coordinates
                sources.append(s.sources)
        #if self.shared_coordinates is not None:
            #crds = crds + self.shared_coordinates
        sources = np.concatenate(sources)
        self.cache_obj(crds, 'partial_native.coordinates')
        self.cache_obj(sources, 'partial_native.sources')
        return crds, sources

    @property
    def base_ref(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return '{0}_{1}'.format(self.__class__.__name__, self.product)

    @property
    def definition(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        d = self.base_definition()
        if self.interpolation:
            d['attrs']['interpolation'] = self.interpolation
        return d

    @property
    @common_doc(COMMON_DOC)
    def keys(self):
        """{keys}
        """
        return self.sources[0].keys


class SMAPBestAvailable(podpac.OrderedCompositor):
    """Compositor of SMAP-Sentinel and the Level 4 SMAP Analysis Update soil moisture
    """

    @tl.default('sources')
    def sources_default(self):
        """Orders the compositor of SPL2SMAP_S.001 in front of SPL4SMAU.003

        Returns
        -------
        np.ndarray(dtype=object(SMAP))
            Array of SMAP product sources
        """
        src_objs = np.array([
            SMAP(interpolation=self.interpolation, product='SPL2SMAP_S.001'),
            SMAP(interpolation=self.interpolation, product='SPL4SMAU.003')
        ])
        return src_objs

    def get_shared_coordinates(self):
        return None # NO shared coordiantes


