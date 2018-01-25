from __future__ import division, unicode_literals, print_function, absolute_import

import sys
import os
import copy
from collections import OrderedDict
import requests
from bs4 import BeautifulSoup
import re
from six import string_types
import numpy as np
import xarray as xr
import traitlets as tl
import h5py


# Internal dependencies
import podpac
from podpac.core.data import type as datatype
from podpac.core import authentication

# Optional Dependencies
try:
    import boto3
except:
    boto3 = None

# Helper functions

def smap2np_date(date):
    if isinstance(date, string_types):
        ymd = '-'.join([date[:4], date[4:6], date[6:8]])
        if len(date) == 15:
            HMS = ' ' + ':'.join([date[9:11], date[11:13], date[13:15]])
        else:
            HMS = ''
        date = np.array([ymd + HMS], dtype='datetime64')
    return date

def np2smap_date(date):
    if isinstance(date, np.datetime64):
        date = str(date).replace('-', '.')
    return date

SMAP_PRODUCT_MAP = xr.DataArray([
        ['cell_lat', 'cell_lon', 'Analysis_Data_', 'sm_surface_analysis'],
        ['cell_lat', 'cell_lon', 'Geophysical_Data_', 'sm_surface'],
        ['{rdk}latitude', '{rdk}longitude', 'Soil_Moisture_Retrieval_Data_',
            'soil_moisture'],
        ['{rdk}latitude', '{rdk}longitude', 'Soil_Moisture_Retrieval_Data_',
            'soil_moisture'],
        ['{rdk}AM_latitude', '{rdk}AM_longitude', 'Soil_Moisture_Retrieval_Data_',
            'soil_moisture'],
        ['cell_lat', 'cell_lon', 'Land_Model_Constants_Data_', ''],
    ],
    dims = ['product', 'attr'],
    coords = {'product': ['SPL4SMAU.003', 'SPL4SMGP.003', 'SPL3SMA.003', 'SPL3SMAP.003',
                          'SPL3SMP.004', 'SPL4SMLM.003'],
              'attr':['latkey', 'lonkey', 'rootdatakey', 'layerkey']
              }
)
SMAP_BASE_URL = 'https://n5eil01u.ecs.nsidc.org/opendap/hyrax/SMAP'


class SMAPSource(datatype.PyDAP):
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
    rootdatakey = tl.Unicode(u'Soil_Moisture_Retrieval_Data_')    
    @tl.default('rootdatakey')
    def _rootdatakey_default(self):
        return SMAP_PRODUCT_MAP.sel(product=self.product, attr='rootdatakey').item()

    layerkey = tl.Unicode()
    @tl.default('layerkey')
    def _layerkey_default(self):
        return SMAP_PRODUCT_MAP.sel(product=self.product, attr='layerkey').item()

    no_data_vals = [-9999.0]
  
    @property
    def product(self):
        src = self.source.split('/')
        return src[src.index('SMAP')+1]

    @property
    def datakey(self):
        return self.rootdatakey + self.layerkey
    
    @property
    def latkey(self):
        return SMAP_PRODUCT_MAP.sel(product=self.product,
                   attr='latkey').item().format(rdk=self.rootdatakey)
    
    @property
    def lonkey(self):
        return SMAP_PRODUCT_MAP.sel(product=self.product,
                   attr='lonkey').item().format(rdk=self.rootdatakey)

    def get_native_coordinates(self):
        try:
            return self.load_cached_obj('native.coordinates')
        except:
            pass
        times = self.get_available_times()
        ds = self.dataset
        lons = np.array(ds[self.lonkey][:, :])
        lats = np.array(ds[self.latkey][:, :])
        lons[lons==self.no_data_vals[0]] = np.nan
        lats[lats==self.no_data_vals[0]] = np.nan
        lons = np.nanmean(lons, axis=0)
        lats = np.nanmean(lats, axis=1)
        coords = podpac.Coordinate(lat=lats, lon=lons, time=np.array(times), 
                                   order=['time', 'lat', 'lon'])
        self.cache_obj(coords, 'native.coordinates')
        return coords
        
    def get_available_times(self):
        times = self.source.split('_')[4]
        times = smap2np_date(times) 
        if 'SM_P_' in self.source:
            times = times + np.array([6, 18], 'timedelta64[h]')
        return times
    
    def get_data(self, coordinates, coordinates_slice):
        # We actually ignore the time slice
        s = tuple([slc for d, slc in zip(coordinates.dims, coordinates_slice)
                   if 'time' not in d]) 
        if 'SM_P_' in self.source:
            d = self.initialize_coord_array(coordinates, 'nan')
            am_key = self.rootdatakey + 'AM_' + self.layerkey
            pm_key = self.rootdatakey + 'PM_' + self.layerkey + '_pm'
            try:
                t = self.native_coordinates.coords['time'][0]
                d.loc[dict(time=t)] = np.array(self.dataset[am_key][s])
            except: pass
            try: 
                t = self.native_coordinates.coords['time'][1]
                d.loc[dict(time=t)] = np.array(self.dataset[pm_key][s])
            except: pass
        else:
            data = np.array(self.dataset[self.datakey][s])
            d = self.initialize_coord_array(coordinates, 'data', 
                                            fillval=data.reshape(coordinates.shape))
        return d    

class SMAPProperties(SMAPSource):
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
        try:
            return self.load_cached_obj('native.coordinates')
        except:
            pass
        ds = self.dataset
        lons = np.array(ds[self.lonkey][:, :])
        lats = np.array(ds[self.latkey][:, :])
        lons[lons==self.no_data_vals[0]] = np.nan
        lats[lats==self.no_data_vals[0]] = np.nan
        lons = np.nanmean(lons, axis=0)
        lats = np.nanmean(lats, axis=1)
        coords = podpac.Coordinate(lat=lats, lon=lons, 
                                   order=['lat', 'lon'])
        self.cache_obj(coords, 'native.coordinates')
        return coords    
    
class SMAPPorosity(SMAPProperties):
    property = tl.Unicode('clsm_poros')
    
class SMAPWilt(SMAPProperties):
    property = tl.Unicode('clsm_wp')    

class SMAPDateFolder(podpac.OrderedCompositor):
    auth_session = tl.Instance(authentication.EarthDataSession)
    auth_class = tl.Type(authentication.EarthDataSession)
    username = tl.Unicode(None, allow_none=True)
    password = tl.Unicode(None, allow_none=True)    
    
    @tl.default('auth_session')
    def _auth_session_default(self):
        session = self.auth_class(
                username=self.username, password=self.password)
        return session
    
    base_url = tl.Unicode(SMAP_BASE_URL)
    product = tl.Enum(SMAP_PRODUCT_MAP.coords['product'].data.tolist())
    folder_date = tl.Unicode(u'')

    file_url_re = re.compile('.*_[0-9]{8}T[0-9]{6}_.*\.h5')
    date_url_re = re.compile('[0-9]{8}T[0-9]{6}')
    
    cache_native_coordinates = tl.Bool(False)

    @property
    def source(self):
        return '/'.join([self.base_url, self.product, self.folder_date])

    @tl.default('sources')
    def sources_default(self):
        try: 
            sources = self.load_cached_obj('sources')
        except:
            _, sources = self.get_available_times_sources()
            self.cache_obj(sources, 'sources')
        b = self.source + '/'
        tol = self.source_coordinates['time'].delta / 2
        src_objs = np.array([SMAPSource(source=b + s,
                                        interpolation_tolerance=tol,
                                        auth_session=self.auth_session)
                             for s in sources])
        return src_objs
    
    @tl.default('is_source_coordinates_complete')
    def src_crds_complete_default(self):
        return True

    def get_source_coordinates(self):
        try: 
            return self.load_cached_obj('source.coordinates')
        except:
            pass
        times, _ = self.get_available_times_sources()
        time_crds = podpac.Coordinate(time=times)
        self.cache_obj(time_crds, 'source.coordinates')
        return time_crds

    def get_shared_coordinates(self):
        try: 
            return self.load_cached_obj('shared.coordinates')
        except:
            pass
        coords = copy.deepcopy(self.sources[0].native_coordinates)
        del coords._coords['time']
        self.cache_obj(coords, 'shared.coordinates')
        return coords

    def get_available_times_sources(self):
        url = self.source
        soup = BeautifulSoup(self.auth_session.get(url).text, 'lxml')
        a = soup.find_all('a')
        file_regex = self.file_url_re
        date_regex = self.date_url_re
        times = []
        sources = []
        for aa in a:
            m = file_regex.match(aa.get_text())
            if m:
                sources.append(m.group())
                date_time = date_regex.search(m.group()).group()
                times.append(np.datetime64(
                    '%s-%s-%sT%s:%s:%s' % (date_time[:4], date_time[4:6],
                        date_time[6:8], date_time[9:11], date_time[11:13],
                        date_time[13:15])
                    ))
        times = np.array(times)
        sources = np.array(sources)
        I = np.argsort(times)
        return times[I], sources[I]


class SMAP(podpac.OrderedCompositor):
    base_url = tl.Unicode(SMAP_BASE_URL)
    product = tl.Enum(SMAP_PRODUCT_MAP.coords['product'].data.tolist())
    date_url_re = re.compile('[0-9]{4}\.[0-9]{2}\.[0-9]{2}')
    
    auth_session = tl.Instance(authentication.EarthDataSession)
    auth_class = tl.Type(authentication.EarthDataSession)
    username = tl.Unicode(None, allow_none=True)
    password = tl.Unicode(None, allow_none=True)
    
    @tl.default('auth_session')
    def _auth_session_default(self):
        session = self.auth_class(
                username=self.username, password=self.password)
        return session    
    
    @property
    def source(self):
        return self.product

    @tl.default('sources')
    def sources_default(self):
        try: 
            dates = self.load_cached_obj('dates')
        except: 
            dates = self.get_available_times_dates()[1]
            self.cache_obj(dates, 'dates')
        src_objs = np.array([
            SMAPDateFolder(product=self.product, folder_date=date,
                           shared_coordinates=self.shared_coordinates,
                           auth_session=self.auth_session)
            for date in dates])
        return src_objs

    def get_source_coordinates(self):
        return podpac.Coordinate(time=self.get_available_times_dates()[0])

    def get_available_times_dates(self):
        url = '/'.join([self.base_url, self.product])
        soup = BeautifulSoup(self.auth_session.get(url).text, 'lxml')
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
        if os.path.exists(self.cache_path('shared.coordinates')):
            return self.load_cached_obj('shared.coordinates')
        coords = SMAPDateFolder(product=self.product,
                                folder_date=self.get_available_times_dates()[1][0],
                           ).shared_coordinates
        self.cache_obj(coords, 'shared.coordinates')
        return coords

    @property
    def base_ref(self):
        return '%s_%s' % (self.__class__.__name__, self.product)
    
    @property
    def definition(self):
        d = OrderedDict()
        d['node'] = self.podpac_path
        d['attrs'] = OrderedDict()
        d['attrs']['product'] = self.product
        if self.interpolation:
            d['attrs']['interpolation'] = self.interpolation
        return d
    
class SMAPSentinelSource(podpac.DataSource):
    source = tl.Unicode(help="Filename for the sensor data")
    no_data_vals = tl.List([-9999.0])
    dataset = tl.Any()

    @tl.default('dataset')
    def dataset_default(self):
        return h5py.File(self.source, 'r')

    def get_time(self):
        f = self.dataset
        time = f['Metadata/Extent'].attrs['rangeEndingDateTime']
        time = time.decode()
        
        return time
     
    def get_native_coordinates(self):
        f = self.dataset
        
        times = self.get_time()
        lat = np.array(f['Soil_Moisture_Retrieval_Data_1km/latitude_1km'])
        lon = np.array(f['Soil_Moisture_Retrieval_Data_1km/longitude_1km'])
        
        coords = podpac.Coordinate(time=times,
                                   lat=np.nanmean(lat, 1),
                                   lon=np.nanmean(lon, 0),
                                   order=['lat', 'lon', 'time'])

        return coords    
    
    def get_data(self, coordinates, coordinates_slice):
        """
        This should return an UnitsDataArray
        coordinates and coordinates slice may be strided or subsets of the 
        source data, but all coordinates will match 1:1 with the subset data
        """
        f = self.dataset
        
        # We actually ignore the time slice
        s = tuple([slc for d, slc in zip(coordinates.dims, coordinates_slice)
                       if 'time' not in d])         
        
        data = np.array(f['Soil_Moisture_Retrieval_Data_1km/soil_moisture_1km'])[s]
        d = self.initialize_coord_array(coordinates, 'data', 
                                fillval=data.reshape(coordinates.shape))
        return d
    

class SMAPSentinelS3Date(podpac.OrderedCompositor):
    folder_date = tl.Unicode(u'')

    tile_re = re.compile('[0-9]{3}[E,W][0-9]{2}[N,S]')
    s3_file_filter = tl.Unicode('SMAPSentinel/SMAP_L2_SM_SP_1BIWDV_')
    cache_native_coordinates = tl.Bool(False)
    
    objkeys = tl.List()

    @tl.default('objkeys')
    def objkeys_default(self):
        try: 
            objkeys = self.load_cached_obj('objkeys')
        except: 
            s3 = boto3.resource('s3').Bucket(self.s3_bucket)
            objs = list(s3.objects.filter(Prefix=self.s3_file_filter))
            objkeys = [o.key for o in objs \
                       if o.key.endswith('.h5')]
            self.cache_obj(objkeys, 'objkeys') 
        return objkeys    

    @property
    def source(self):
        return 'SMAPSentinalS3Data_' + self.folder_date

    @tl.default('sources')
    def sources_default(self):
        sources = [o for o in self.objkeys \
                   if self.s3_file_filter + self.folder_date in o]
        
        tol = np.timedelta64(12, 'h')
        src_objs = np.array([datatype.S3Source(
                node_class=SMAPSentinelSource, 
                source=s,
                node_kwargs=dict(interpolation_tolerance=tol, 
                                 interpolation=self.interpolation))
            for s in sources])
        return src_objs
    
    @tl.default('is_source_coordinates_complete')
    def src_crds_complete_default(self):
        return False

    def get_source_coordinates(self):
        try: 
            return self.load_cached_obj('source.coordinates')
        except:
            pass
        tiles = self.get_available_tiles()
        crds = podpac.Coordinate(lat_lon=tiles)
        crds['lat_lon']._cached_delta = np.array([3, 3])
        self.cache_obj(crds, 'source.coordinates')
        return crds

    def get_available_tiles(self):
        sources = [o for o in self.objkeys \
                   if self.s3_file_filter + self.folder_date in o]
        lats = []
        lons = []
        strings = []
        tile_re = self.tile_re
        for aa in sources:
            m = tile_re.search(aa)
            if m:
                string = m.group()
                lons.append(float(string[:3]) * (1 - 2 *(string[3] == 'W')))
                lats.append(float(string[4:6]) * (1 - 2 *(string[6] == 'S')))
                strings.append(string)
                 
        tiles = [np.array(lats), np.array(lons)]    
        return tiles
        

class SMAPSentinelS3(podpac.OrderedCompositor):
    s3_bucket = tl.Unicode()
    date_url_re = re.compile('[0-9]{8}T[0-9]{6}')
    s3_file_filter = tl.Unicode('SMAPSentinel/SMAP_L2_SM_SP')
    objkeys = tl.List()
    
    @tl.default('objkeys')
    def objkeys_default(self):
        try: 
            objkeys = self.load_cached_obj('objkeys')
        except: 
            s3 = boto3.resource('s3').Bucket(self.s3_bucket)
            objs = list(s3.objects.filter(Prefix=self.s3_file_filter))
            objkeys = [o.key for o in objs if o.key.endswith('.h5')]
            objkeys.sort()
            self.cache_obj(objkeys, 'objkeys') 
        return objkeys    
    
    @tl.default('s3_bucket')
    def s3_bucket_default(self):
        return podpac.settings.S3_BUCKET_NAME
    
    @property
    def source(self):
        return "SentinelS3"

    @tl.default('sources')
    def sources_default(self):
        try: 
            dates = self.load_cached_obj('dates')
        except: 
            dates = self.get_available_times_dates()[1]
            self.cache_obj(dates, 'dates')
        src_objs = np.array([
            SMAPSentinelS3Date(folder_date=date,
                               objkeys=self.objkeys,
                               interpolation=self.interpolation)
            for date in dates])
        return src_objs

    def get_source_coordinates(self):
        try: 
            times = self.load_cached_obj('times')
        except: 
            times = self.get_available_times_dates()[0]
            self.cache_obj(times, 'times')
            
        return podpac.Coordinate(time=times)

    def get_available_times_dates(self):
        objkeys = self.objkeys
            
        regex = self.date_url_re
        times = []
        dates = []
        for aa in objkeys:
            m = regex.search(aa).group()
            if m:
                times.append(np.datetime64(m[:4] + '-' + m[4:6] + '-' + m[6:8]))
                dates.append(m[:8])
        times = np.unique(times)
        times.sort()
        dates = np.unique(dates)
        dates.sort()
        return np.array(times), dates.tolist()

    @property
    def base_ref(self):
        return '%s_%s' % (self.__class__.__name__, self.product)
    
    @property
    def definition(self):
        d = OrderedDict()
        d['node'] = self.podpac_path
        d['attrs'] = OrderedDict()
        d['attrs']['product'] = self.product
        if self.interpolation:
            d['attrs']['interpolation'] = self.interpolation
        return d    

class SMAPBestAvailable(podpac.OrderedCompositor):
    @tl.default('sources')
    def sources_default(self):
        src_objs = np.array([
            SMAPSentinelS3(interpolation=self.interpolation),
            SMAP(interpolation=self.interpolation, product='SPL4SMAU.003')
        ])
        return src_objs
    

if __name__ == '__main__':
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
    
    #o = sm.execute(coords)
    
    coordinates_world = \
        podpac.Coordinate(lat=(-90, 90, 1.),
                          lon=(-180, 180, 1.),
                          time='2017-10-10T12:00:00', 
                          order=['lat', 'lon', 'time'])    
    smap = SMAP(interpolation='nearest_preview', product='SPL4SMAU.003')
    smap.source_coordinates
    smap.shared_coordinates
    o = smap.execute(coordinates_world)
    print("done")
    
    
    s5 = SMAPSentinelS3(interpolation='nearest_preview')
    smapba = SMAPBestAvailable(interpolation='nearest_preview')
    
 
    s55 = s5.sources[6]
    s5591 = s55.sources[91]
    s5592 = s55.sources[92]
    print (s5591.native_coordinates)
    print (s5592.native_coordinates)
    #nc = copy.deepcopy(s5591.native_coordinates)
    #del nc._coords['time']
    a91 = s5591.execute(s5591.native_coordinates)
    #a92n = s5592.execute(s5592.native_coordinates)
    #a92 = s5592.execute(nc)
    a92 = s5592.execute(s5591.native_coordinates)
    
    coords = copy.deepcopy(s5591.native_coordinates)
    coords['time']._cached_delta = np.array([np.timedelta64(90, 'm')])
    coords["time"]._cached_bounds = None
    
    a = s5.execute(coords)
    b = smapba.execute(coords)
    c = smap.execute(coords)
    
    sd = SMAPSentinelSource(source=(r"\\OLYMPUS\Projects\1010028-Pipeline"
                            r"\Technical Work\Testing\Data\SMAPSentinel"
                            r"\SMAP_L2_SM_SP_1AIWDV_20170801T000000_20170731T235532_090E24N_T15110_002.h5"))
     
    
    #sdf = SMAPDateFolder(product='SPL4SMGP.003', folder_date='2016.04.07')
    
    #coords = sdf.native_coordinates
    #print (coords)
    ##print (coords['time'].area_bounds)
    
    #coords = podpac.Coordinate(time=coords.coords['time'][:3],
                               #lat=[45., 66., 50], lon=[-80., -70., 20],
                               #order=['time', 'lat', 'lon'])  

    
    #o = sdf.execute(coords)    
    #coords2 = podpac.Coordinate(time=coords.coords['time'][1:2],
                               #lat=[45., 66., 50], lon=[-80., -70., 20],
                               #order=['time', 'lat', 'lon'])      
    #o2 = sdf.execute(coords2)    
    
    #t_coords = podpac.Coordinate(time=np.datetime64('2015-12-11T06'))
    #o2 = smap.execute(t_coords)    
    

    
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

