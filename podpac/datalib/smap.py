"""Summary

Attributes
----------
SMAP_BASE_URL : str
    Description
SMAP_INCOMPLETE_SOURCE_COORDINATES : list
    Description
SMAP_PRODUCT_MAP : TYPE
    Description
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import re
import copy
from collections import OrderedDict

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
from podpac.core.data import type as datatype
from podpac.core import authentication

# Optional Dependencies
try:
    import boto3
except:
    boto3 = None


def smap2np_date(date):
    """Summary

    Parameters
    ----------
    date : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    if isinstance(date, string_types):
        ymd = '-'.join([date[:4], date[4:6], date[6:8]])
        if len(date) == 15:
            HMS = ' ' + ':'.join([date[9:11], date[11:13], date[13:15]])
        else:
            HMS = ''
        date = np.array([ymd + HMS], dtype='datetime64')
    return date


def np2smap_date(date):
    """Summary

    Parameters
    ----------
    date : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    if isinstance(date, np.datetime64):
        date = str(date).replace('-', '.')
    return date

SMAP_PRODUCT_MAP = xr.DataArray([
    ['cell_lat', 'cell_lon', 'Analysis_Data_', '{rdk}sm_surface_analysis'],
    ['cell_lat', 'cell_lon', 'Geophysical_Data_', '{rdk}sm_surface'],
    ['{rdk}latitude', '{rdk}longitude', 'Soil_Moisture_Retrieval_Data_', '{rdk}soil_moisture'],
    ['{rdk}latitude', '{rdk}longitude', 'Soil_Moisture_Retrieval_Data_', '{rdk}soil_moisture'],
    ['{rdk}AM_latitude', '{rdk}AM_longitude', 'Soil_Moisture_Retrieval_Data_', '{rdk}AM_soil_moisture'],
    ['cell_lat', 'cell_lon', 'Land_Model_Constants_Data_', ''],
    ['{rdk}latitude_1km', '{rdk}longitude_1km', 'Soil_Moisture_Retrieval_Data_1km_', '{rdk}soil_moisture_1km']],
    dims = ['product', 'attr'],
    coords = {'product': ['SPL4SMAU.003', 'SPL4SMGP.003', 'SPL3SMA.003', 'SPL3SMAP.003', 'SPL3SMP.004',
                          'SPL4SMLM.003', 'SPL2SMAP_S.001'],
              'attr':['latkey', 'lonkey', 'rootdatakey', 'layerkey']
             }
)

SMAP_INCOMPLETE_SOURCE_COORDINATES = ['SPL2SMAP_S.001']
SMAP_BASE_URL = 'https://n5eil01u.ecs.nsidc.org/opendap/hyrax/SMAP'


class SMAPSource(datatype.PyDAP):
    """Summary

    Attributes
    ----------
    auth_class : TYPE
        Description
    auth_session : TYPE
        Description
    date_file_url_re : TYPE
        Description
    date_time_file_url_re : TYPE
        Description
    date_url_re : TYPE
        Description
    layerkey : TYPE
        Description
    no_data_vals : list
        Description
    rootdatakey : TYPE
        Description
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

    date_url_re = re.compile('[0-9]{4}\.[0-9]{2}\.[0-9]{2}')
    date_time_file_url_re = re.compile('[0-9]{8}T[0-9]{6}')
    date_file_url_re = re.compile('[0-9]{8}')

    rootdatakey = tl.Unicode(u'Soil_Moisture_Retrieval_Data_')
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

    no_data_vals = [-9999.0]

    @property
    def product(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        src = self.source.split('/')
        return src[src.index('SMAP')+1]

    @tl.default('datakey')
    def _datakey_default(self):
        return self.layerkey.format(rdk=self.rootdatakey)

    @property
    def latkey(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return SMAP_PRODUCT_MAP.sel(product=self.product, attr='latkey') \
                               .item().format(rdk=self.rootdatakey)

    @property
    def lonkey(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return SMAP_PRODUCT_MAP.sel(product=self.product, attr='lonkey').item().format(rdk=self.rootdatakey)

    def get_native_coordinates(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        try:
            return self.load_cached_obj('native.coordinates')
        except:
            pass
        times = self.get_available_times()
        ds = self.dataset
        lons = np.array(ds[self.lonkey][:, :])
        lats = np.array(ds[self.latkey][:, :])
        lons[lons == self.no_data_vals[0]] = np.nan
        lats[lats == self.no_data_vals[0]] = np.nan
        lons = np.nanmean(lons, axis=0)
        lats = np.nanmean(lats, axis=1)
        coords = podpac.Coordinate(lat=lats, lon=lons, time=np.array(times),
                                   order=['time', 'lat', 'lon'])
        self.cache_obj(coords, 'native.coordinates')
        return coords

    def get_available_times(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        m = self.date_time_file_url_re.search(self.source)
        if not m:
            m = self.date_file_url_re.search(self.source)
        times = m.group()
        times = smap2np_date(times)
        if 'SM_P_' in self.source:
            times = times + np.array([6, 18], 'timedelta64[h]')
        return times

    def get_data(self, coordinates, coordinates_index):
        """Summary

        Parameters
        ----------
        coordinates : TYPE
            Description
        coordinates_index : TYPE
            Description

        Returns
        -------
        TYPE
            Description
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
    """Summary

    Attributes
    ----------
    property : TYPE
        Description
    source : TYPE
        Description
    """
    
    source = tl.Unicode('https://n5eil01u.ecs.nsidc.org/opendap/SMAP/'
                        'SPL4SMLM.003/2015.03.31/'
                        'SMAP_L4_SM_lmc_00000000T000000_Vv3030_001.h5')

    property = tl.Enum(['clsm_dzsf', 'mwrtm_bh', 'clsm_cdcr2', 'mwrtm_poros',
                        'clsm_dzgt3', 'clsm_dzgt2', 'mwrtm_rghhmax',
                        'mwrtm_rghpolmix', 'clsm_dzgt1', 'clsm_wp', 'mwrtm_lewt',
                        'clsm_dzgt4', 'clsm_cdcr1', 'cell_elevation',
                        'mwrtm_rghwmin', 'clsm_dzrz', 'mwrtm_vegcls', 'mwrtm_bv',
                        'mwrtm_rghwmax', 'mwrtm_rghnrh', 'clsm_dztsurf',
                        'mwrtm_rghhmin', 'mwrtm_wangwp', 'mwrtm_wangwt',
                        'clsm_dzgt5', 'clsm_dzpr', 'clsm_poros',
                        'cell_land_fraction', 'mwrtm_omega', 'mwrtm_soilcls',
                        'clsm_dzgt6', 'mwrtm_rghnrv', 'mwrtm_clay', 'mwrtm_sand'])

    @tl.default('layerkey')
    def _layerkey_default(self):
        return self.property

    def get_native_coordinates(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        try:
            return self.load_cached_obj('native.coordinates')
        except:
            pass

        ds = self.dataset
        lons = np.array(ds[self.lonkey][:, :])
        lats = np.array(ds[self.latkey][:, :])
        lons[lons == self.no_data_vals[0]] = np.nan
        lats[lats == self.no_data_vals[0]] = np.nan
        lons = np.nanmean(lons, axis=0)
        lats = np.nanmean(lats, axis=1)
        coords = podpac.Coordinate(lat=lats, lon=lons,
                                   order=['lat', 'lon'])
        self.cache_obj(coords, 'native.coordinates')
        return coords

class SMAPPorosity(SMAPProperties):
    """Summary

    Attributes
    ----------
    property : TYPE
        Description
    """
    property = tl.Unicode('clsm_poros')

class SMAPWilt(SMAPProperties):
    """Summary
    
    Attributes
    ----------
    property : TYPE
        Description
    """
    property = tl.Unicode('clsm_wp')

class SMAPDateFolder(podpac.OrderedCompositor):
    """Summary
    
    Attributes
    ----------
    auth_class : TYPE
        Description
    auth_session : TYPE
        Description
    base_url : TYPE
        Description
    cache_native_coordinates : TYPE
        Description
    date_time_url_re : TYPE
        Description
    date_url_re : TYPE
        Description
    file_url_re : TYPE
        Description
    file_url_re2 : TYPE
        Description
    folder_date : TYPE
        Description
    latlon_delta : TYPE
        Description
    latlon_url_re : TYPE
        Description
    layerkey : TYPE
        Description
    password : TYPE
        Description
    product : TYPE
        Description
    username : TYPE
        Description
    """

    auth_session = tl.Instance(authentication.EarthDataSession)
    auth_class = tl.Type(authentication.EarthDataSession)
    username = tl.Unicode(None, allow_none=True)
    password = tl.Unicode(None, allow_none=True)

    @tl.default('auth_session')
    def _auth_session_default(self):
        session = self.auth_class(username=self.username, password=self.password)
        return session

    base_url = tl.Unicode(SMAP_BASE_URL)
    product = tl.Enum(SMAP_PRODUCT_MAP.coords['product'].data.tolist())
    folder_date = tl.Unicode(u'')

    file_url_re = re.compile('.*_[0-9]{8}T[0-9]{6}_.*\.h5')
    file_url_re2 = re.compile('.*_[0-9]{8}_.*\.h5')
    date_time_url_re = re.compile('[0-9]{8}T[0-9]{6}')
    date_url_re = re.compile('[0-9]{8}')
    latlon_url_re = re.compile('[0-9]{3}[E,W][0-9]{2}[N,S]')

    latlon_delta = tl.Float(default_value=1.5)

    cache_native_coordinates = tl.Bool(False)

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
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return '/'.join([self.base_url, self.product, self.folder_date])

    @tl.default('sources')
    def sources_default(self):
        """Summary

        Returns
        -------
        TYPE
            Description
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

        src_objs = np.array([SMAPSource(source=b + s,
                                        interpolation_param=tol,
                                        auth_session=self.auth_session,
                                        layerkey=self.layerkey)
                             for s in sources])
        return src_objs

    @tl.default('is_source_coordinates_complete')
    def src_crds_complete_default(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.product not in SMAP_INCOMPLETE_SOURCE_COORDINATES

    def get_source_coordinates(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        try:
            return self.load_cached_obj('source.coordinates')
        except:
            pass
        times, latlon, _ = self.get_available_coords_sources()

        if latlon is not None and latlon.size > 0:
            crds = podpac.Coordinate(
                time_lat_lon=(times,
                              podpac.Coord(latlon[:, 0], delta=self.latlon_delta),
                              podpac.Coord(latlon[:, 1], delta=self.latlon_delta)
                              )
                )
        else:
            crds = podpac.Coordinate(time=times)
        self.cache_obj(crds, 'source.coordinates')
        return crds

    def get_shared_coordinates(self):
        """Summary

        Returns
        -------
        TYPE
            Description
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
        TYPE
            Description

        Raises
        ------
        RuntimeError
            Description
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
        else:
            return times[I], None, sources[I]

    @property
    def keys(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.sources[0].keys


class SMAP(podpac.OrderedCompositor):
    """Summary

    Attributes
    ----------
    auth_class : TYPE
        Description
    auth_session : TYPE
        Description
    base_url : TYPE
        Description
    date_url_re : TYPE
        Description
    layerkey : TYPE
        Description
    password : TYPE
        Description
    product : TYPE
        Description
    username : TYPE
        Description
    """
    
    base_url = tl.Unicode(SMAP_BASE_URL)
    product = tl.Enum(SMAP_PRODUCT_MAP.coords['product'].data.tolist())
    date_url_re = re.compile('[0-9]{4}\.[0-9]{2}\.[0-9]{2}')

    auth_session = tl.Instance(authentication.EarthDataSession)
    auth_class = tl.Type(authentication.EarthDataSession)
    username = tl.Unicode(None, allow_none=True)
    password = tl.Unicode(None, allow_none=True)

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
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.product

    @tl.default('sources')
    def sources_default(self):
        """Summary

        Returns
        -------
        TYPE
            Description
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
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return podpac.Coordinate(time=self.get_available_times_dates()[0])

    def get_available_times_dates(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        
        Raises
        ------
        RuntimeError
            Description
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
        """Summary

        Returns
        -------
        TYPE
            Description
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
        """Summary

        Returns
        -------
        TYPE
            Description
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
        d['attrs'] = OrderedDict()
        d['attrs']['product'] = self.product
        if self.interpolation:
            d['attrs']['interpolation'] = self.interpolation
        return d

    @property
    def keys(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.sources[0].keys
       
   
class SMAPBestAvailable(podpac.OrderedCompositor):
    """Summary
    """
    
    @tl.default('sources')
    def sources_default(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        src_objs = np.array([
            SMAP(interpolation=self.interpolation, product='SPL2SMAP_S.001'),
            SMAP(interpolation=self.interpolation, product='SPL4SMAU.003')
        ])
        return src_objs
 
  

if __name__ == '__main__':
    import podpac

    #from podpac.core.authentication import EarthDataSession
    #eds = EarthDataSession()
    #eds.update_login()
        # follow the prompts
    from podpac.core.data.type import WCS
    coords = podpac.Coordinate(time='2015-04-06T06',
                               lat=(-34.5, -35.25, 30), lon=(145.75, 146.5, 30),
                               order=['time', 'lat', 'lon'])
   

    #from podpac.datalib.smap import SMAP
    smap = SMAP(product='SPL3SMP.004')
    #smap = SMAP(product='SPL4SMAU.003')
    #nc = smap.native_coordinates
    #pnc, srcs = smap.get_partial_native_coordinates_sources()
    sources = smap.sources
    o = smap.execute(coords)
    
    #%% Get data from DASSP
   
    coordinates_world = \
        podpac.Coordinate(lat=(-90, 90, 1.),
                          lon=(-180, 180, 1.),
                          time=['2017-11-18T00:00:00', '2017-11-19T00:00:00'],
                          order=['lat', 'lon', 'time'])
    sentinel = SMAP(interpolation='nearest_preview', product='SPL2SMAP_S.001')
    smap = SMAP(product='SPL4SMAU.003')
    pnc2, srcs2 = smap.get_partial_native_coordinates_sources()
    o2 = smap.execute(coordinates_world)
    pnc3, srcs3 = sentinel.get_partial_native_coordinates_sources()
    o3 = sentinel.execute(coordinates_world)
    s = sentinel.sources[121]
    s.source_coordinates.intersect(coordinates_world)
    
    
    source = ('https://n5eil01u.ecs.nsidc.org:443/opendap'
              '/SMAP/SPL2SMAP_S.001/2017.02.08'
              '/SMAP_L2_SM_SP_1AIWDV_20170208T011127_20170208T004300_079E30N_R15180_001.h5')
    sm = SMAPSource(source=source)
    sm.native_coordinates
    o = sm.execute(sm.native_coordinates)
    
    smd = SMAPDateFolder(product='SPL2SMAP_S.001', folder_date='2017.02.08')
    crds = smd.get_source_coordinates()
    c = podpac.Coordinate(lat=0, lon=0)
    a = crds.intersect(c)
    c2 = podpac.Coordinate(lat=30, lon=119)
    a2 = crds.intersect(c2)
    c3 = podpac.Coordinate(lat=30.5, lon=119.5)
    a3 = crds.intersect(c3)
    
    
    from podpac.core import authentication
    ed_session = authentication.EarthDataSession()
    #ed_session.update_login()
    
    from pydap.cas.urs import setup_session
    from pydap.client import open_url
    source = 'https://n5eil01u.ecs.nsidc.org/opendap/hyrax/SMAP/SPL4SMGP.003/2015.04.07/SMAP_L4_SM_gph_20150407T013000_Vv3030_001.h5'
    source = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/2016/06/MERRA2_400.tavg1_2d_slv_Nx.20160601.nc4'
    source = 'https://n5eil01u.ecs.nsidc.org/opendap/hyrax/SMAP/SPL4SMGP.003/2015.04.07/SMAP_L4_SM_gph_20150407T193000_Vv3030_001.h5'
    
    # Seems like we have to check a url in order to not get stuck in redirect land
    #ed_session.get(source + '.dds')
    #session = setup_session(ed_session.auth[0], ed_session.auth[1],
                            #check_url=source)
    #dataset = open_url(source, session=ed_session)

    sm = SMAPSource(source=source) #, auth_session=ed_session)
    sm.dataset
    sm.native_coordinates
    
    coords = podpac.Coordinate(lat=sm.native_coordinates['lat'].coordinates[::10],
                              lon=sm.native_coordinates['lon'].coordinates[::10],
                              time=sm.native_coordinates['time'], 
                              order=['lat', 'lon', 'time'])
    
    o = sm.execute(coords)
    
    smap = SMAP(interpolation='nearest_preview', product='SPL4SMAU.003')
    smap.source_coordinates
    smap.shared_coordinates 
    o = smap.execute(coordinates_world)
    print("done")
    
    

    
    porosity = SMAPPorosity()
    o = porosity.execute(coordinates_world)    
    
    wilt = SMAPWilt()
    o = wilt.execute(coordinates_world)
    
    
    source = ('https://n5eil01u.ecs.nsidc.org/opendap/hyrax/SMAP'
              '/SPL4SMGP.003/2015.04.07'
              '/SMAP_L4_SM_gph_20150407T013000_Vv3030_001.h5')
    #source2 = ('https://n5eil01u.ecs.nsidc.org/opendap/hyrax/SMAP/SPL3SMP.004/'
              #'2015.04.11/SMAP_L3_SM_P_20150411_R14010_001.h5')
    #source = ('https://n5eil01u.ecs.nsidc.org/opendap/hyrax/SMAP/SPL4SMAU.003/'
    #          '2015.04.03/SMAP_L4_SM_aup_20150403T030000_Vv3030_001.h5')
    smap = SMAPSource(source=source, interpolation='nearest_preview')
    coords = smap.native_coordinates
    print (coords)
    print (coords['time'].area_bounds)
    coord_pt = podpac.Coordinate(lat=10., lon=-67., order=['lat', 'lon'])  # Not covered
    o = smap.execute(coord_pt)
    ##coord_pt = podpac.Coordinate(lat=66., lon=-72.)  
    ##o = smap.execute(coord_pt)
    
    ##coords = podpac.Coordinate(lat=[45., 66., 50], lon=[-80., -70., 20])  
    lat, lon = smap.native_coordinates.coords['lat'], smap.native_coordinates.coords['lon']
    lat = lat[::10][np.isfinite(lat[::10])]
    lon = lon[::10][np.isfinite(lon[::10])]
    coords = podpac.Coordinate(lat=lat, lon=lon, order=['lat', 'lon'])
    
    o = smap.execute(coords)    
    
    #t_coords = podpac.Coordinate(time=np.datetime64('2015-12-11T06'))
    #o2 = smap.execute(t_coords)
        
    smap = SMAP(product='SPL4SMAU.003')
    
    coords = smap.native_coordinates
    print (coords)
    #print (coords['time'].area_bounds)
    
    coords = podpac.Coordinate(time=coords.coords['time'][:3],
                               lat=[45., 66., 50], lon=[-80., -70., 20],
                               order=['time', 'lat', 'lon'])  

    o = smap.execute(coords)    
    coords2 = podpac.Coordinate(time=coords.coords['time'][1:2],
                               lat=[45., 66., 50], lon=[-80., -70., 20],
                               order=['time', 'lat', 'lon'])      
    o2 = smap.execute(coords2) 
    

    print ('Done')

