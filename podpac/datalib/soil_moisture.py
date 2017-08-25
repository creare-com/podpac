from __future__ import division, unicode_literals, print_function, absolute_import

import os
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import traitlets as tl

# Internal dependencies
import podpac
from podpac.core.data import type as datatype

class SMAPSource(datatype.PyDAP):
    base_url = tl.Unicode(u'https://n5eil01u.ecs.nsidc.org/opendap/hyrax/SMAP')
    product = tl.Enum(['SPL3SMA.003', 'SPL3SMAP.003', 'SPL3SMP.004'])
    date_url_re = re.compile('[0-9]{4}\.[0-9]{2}\.[0-9]{2}')
    rootdatakey = tl.Unicode(u'Soil_Moisture_Retrieval_Data_')    
    layerkey = tl.Unicode(u'soil_moisture')
    no_data_vals = [-9999.0]
    
    
    @property
    def datakey(self):
        return self.rootdatakey + self.layerkey
    
    @property
    def latkey(self):
        rk = self.rootdatakey
        if 'SM_P_' in self.source:
            rk = rk + 'AM_'
        return rk + 'latitude'

    @property
    def lonkey(self):
        rk = self.rootdatakey
        if 'SM_P_' in self.source:
            rk = rk + 'AM_'
        return rk + 'longitude'

    @tl.default('native_coordinates')
    def set_native_coordinates(self):
        if os.path.exists(self.cache_path('native.coordinates')):
            return self.load_cached_obj('native.coordinates')
        
        times = self.get_available_times()
        ds = self.dataset
        lons = ds[self.lonkey][:, :]
        lats = ds[self.latkey][:, :]
        lons[lons==self.no_data_vals[0]] = np.nan
        lats[lats==self.no_data_vals[0]] = np.nan
        lons = np.nanmean(lons, axis=0)
        lats = np.nanmean(lats, axis=1)
        coords = podpac.Coordinate(lat=lats, lon=lons, time=np.array(times))
        self.cache_obj(coords, 'native.coordinates')
        return coords
        
    def get_available_times(self):
        times = self.source.split('_')[4]
        times = np.array('-'.join([times[:4], times[4:6], times[6:]]),
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
            d[dict(time=0)] = self.dataset[am_key][s]
            d[dict(time=1)] = self.dataset[pm_key][s]
        else:
            data = self.dataset[self.datakey][s]
            d = self.initialize_coord_array(coordinates, 'data', 
                                            fillval=data.reshape(coordinates.shape))
        return d    

class SMAP(datatype.PyDAP):
    base_url = tl.Unicode(u'https://n5eil01u.ecs.nsidc.org/opendap/hyrax/SMAP')
    product = tl.Enum(['SPL3SMA.003', 'SPL3SMAP.003', 'SPL3SMP.004'])
    date_url_re = re.compile('[0-9]{4}\.[0-9]{2}\.[0-9]{2}')
    rootdatakey = tl.Unicode(u'Soil_Moisture_Retrieval_Data_')

    @property
    def latkey(self):
        rk = self.rootdatakey
        if self.product == 'SPL3SMP.004':
            rk = rk + 'AM_'
        return rk + 'latitude'

    @property
    def lonkey(self):
        rk = self.rootdatakey
        if self.product == 'SPL3SMP.004':
            rk = rk + 'AM_'
        return rk + 'longitude'

    @tl.default('native_coordinates')
    def set_native_coordinates(self):
        times = self.get_available_times()
        if self.source is None:
            source = self.get_fn_url(times[0])
        else: 
            source = self.source
        ds = self.open_dataset(source)
        lons = ds[self.lonkey][0, :].squeeze()
        lats = ds[self.latkey][:, 0].squeeze()
        return podpac.Coordinate(lat=lats, lon=lons, time=np.array(times))
        
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

    @staticmethod
    def np2smap_date(date):
        if isinstance(date, np.datetime64):
            date = str(date).replace('-', '.')
        return date
        
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
    source = ('https://n5eil01u.ecs.nsidc.org/opendap/hyrax/SMAP/SPL3SMP.004/'
              '2015.04.11/SMAP_L3_SM_P_20150411_R14010_001.h5')
    smap = SMAPSource(source=source)
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
    coords = podpac.Coordinate(lat=lat, lon=lon)
    o = smap.execute(coords)    
    
    print ('Done')

