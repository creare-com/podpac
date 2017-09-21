from __future__ import division, unicode_literals, print_function, absolute_import

import os
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import xarray as xr
import traitlets as tl

# Internal dependencies
import podpac
from podpac.core.data import type as datatype

# Helper functions

def smap2np_date(date):
    if isinstance(date, (str, unicode)):
        ymd = '-'.join([times[:4], times[4:6], times[6:8]])
        if len(date) == 15:
            HMS = ' ' + ':'.join(times[9:11], times[11:13], times[13:15])
        else:
            HMS = ''
        date = np.array(ymd + HMS, dtype='datetime64')
    return date

def np2smap_date(date):
    if isinstance(date, np.datetime64):
        date = str(date).replace('-', '.')
    return date

SMAP_PRODUCT_MAP = xr.DataArray([
        ['cell_lat', 'cell_lon', 'Geophysical_Data_', 'sm_surface'],
        ['{rdk}latitude', '{rdk}longitude', 'Soil_Moisture_Retrieval_Data_',
            'soil_moisture'],
        ['{rdk}latitude', '{rdk}longitude', 'Soil_Moisture_Retrieval_Data_',
            'soil_moisture'],
        ['{rdk}AM_latitude', '{rdk}AM_longitude', 'Soil_Moisture_Retrieval_Data_',
            'soil_moisture'],
    ],
    dims = ['product', 'attr'],
    coords = {'product': ['SPL4SMGP.003', 'SPL3SMA.003', 'SPL3SMAP.003',
                          'SPL3SMP.004'],
              'attr':['latkey', 'lonkey', 'rootdatakey', 'layerkey']
              }
)

class SMAPSource(datatype.PyDAP):
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

    @tl.default('native_coordinates')
    def get_native_coordinates(self):
        if os.path.exists(self.cache_path('native.coordinates')):
            return self.load_cached_obj('native.coordinates')
        
        times = self.get_available_times()
        ds = self.dataset
        lons = np.array(ds[self.lonkey][:, :])
        lats = np.array(ds[self.latkey][:, :])
        lons[lons==self.no_data_vals[0]] = np.nan
        lats[lats==self.no_data_vals[0]] = np.nan
        lons = np.nanmean(lons, axis=0)
        lats = np.nanmean(lats, axis=1)
        coords = podpac.Coordinate(lat=lats, lon=lons, time=np.array(times), 
                                   order=['lat', 'lon', 'time'])
        self.cache_obj(coords, 'native.coordinates')
        return coords
        
    def get_available_times(self):
        times = self.source.split('_')[4]
        times = np.array('-'.join([times[:4], times[4:6], times[6:8]]),
                         dtype='datetime64')
        if 'SM_P_' in self.source:
            times = times + np.array([6, 18], 'timedelta64[h]')
        return times
    
    def get_data(self, coordinates, coordinates_slice):
        s = tuple(coordinates_slice)
        if 'SM_P_' in self.source:
            d = self.initialize_coord_array(coordinates, 'nan')
            am_key = self.rootdatakey + 'AM_' + self.layerkey
            pm_key = self.rootdatakey + 'PM_' + self.layerkey + '_pm'
            d[dict(time=0)] = np.array(self.dataset[am_key][s])
            d[dict(time=1)] = np.array(self.dataset[pm_key][s])
        else:
            data = np.array(self.dataset[self.datakey][s])
            d = self.initialize_coord_array(coordinates, 'data', 
                                            fillval=data.reshape(coordinates.shape))
        return d    

class SMAPDateFolder(podpac.OrderedCompositor):
    pass

class SMAP(podpac.OrderedCompositor):
    base_url = tl.Unicode(u'https://n5eil01u.ecs.nsidc.org/opendap/hyrax/SMAP')
    product = tl.Enum(['SPL4SMGP.003', 'SPL3SMA.003', 'SPL3SMAP.003', 
        'SPL3SMP.004'])
    date_url_re = re.compile('[0-9]{4}\.[0-9]{2}\.[0-9]{2}')
    rootdatakey = tl.Unicode(u'Soil_Moisture_Retrieval_Data_')

    tl.default('source_coordinate')
    def get_source_coordinates(self):
        return podpac.Coordinates(time=self.get_available_times())

    def get_available_times(self):
        url = '/'.join([self.base_url, self.product])
        soup = BeautifulSoup(requests.get(url).text, 'lxml')
        a = soup.find_all('a')
        regex = self.date_url_re
        times = []
        for aa in a:
            m = regex.match(aa.get_text())
            if m:
                times.append(np.datetime64(m.group().replace('.', '-')))
        times.sort()
        return np.array(times)

    def get_fn(self, date):
        date = self.np2smap_date(date)
        url = '/'.join([self.base_url, self.product, date])
        soup = BeautifulSoup(requests.get(url).text, 'lxml')
        a = soup.find_all('a')
        p = self.product
        regex = re.compile('SMAP_{level}_.*_{date}_R.*_.*\.h5'.format(
            level=p[2:4], date=date.replace('.', '')))
        for aa in a:
            fn = regex.search(aa.get_text())
            if fn:
                return fn.group()
        else:
            raise Exeption('No file found')
        
    def get_fn_url(self, date):
        date = self.np2smap_date(date)
        return '/'.join([self.base_url, self.product, date, self.get_fn(date)])
    
if __name__ == '__main__':
    source = ('https://n5eil01u.ecs.nsidc.org/opendap/hyrax/SMAP'
              '/SPL4SMGP.003/2015.04.07'
              '/SMAP_L4_SM_gph_20150407T013000_Vv3030_001.h5')
    source2 = ('https://n5eil01u.ecs.nsidc.org/opendap/hyrax/SMAP/SPL3SMP.004/'
              '2015.04.11/SMAP_L3_SM_P_20150411_R14010_001.h5')
    smap = SMAPSource(source=source, interpolation='nearest_preview')
    coords = smap.native_coordinates
    print (coords)
    print (coords['time'].area_bounds)
    #coord_pt = podpac.Coordinate(lat=10., lon=-67.)  # Not covered
    #o = smap.execute(coord_pt)
    #coord_pt = podpac.Coordinate(lat=66., lon=-72.)  
    #o = smap.execute(coord_pt)
    
    #coords = podpac.Coordinate(lat=[45., 66., 50], lon=[-80., -70., 20])  
    lat, lon = smap.native_coordinates.coords['lat'], smap.native_coordinates.coords['lon']
    lat = lat[::10][np.isfinite(lat[::10])]
    lon = lon[::10][np.isfinite(lon[::10])]
    coords = podpac.Coordinate(lat=lat, lon=lon, order=['lat', 'lon'])
    
    #o = smap.execute(coords)    
    
    t_coords = podpac.Coordinate(time=np.datetime64('2015-12-11T06'))
    o2 = smap.execute(t_coords)
    print ('Done')

