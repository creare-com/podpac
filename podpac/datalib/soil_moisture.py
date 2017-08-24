from __future__ import division, unicode_literals, print_function, absolute_import

import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import traitlets as tl

# Internal dependencies
import podpac
from podpac.core.data import type as datatype

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
        times = [regex.match(aa.get_text()).group() 
                 for aa in a if regex.match(aa.get_text())]
        return times
        
    def get_fn(self, date):
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
        return '/'.join([self.base_url, self.product, date, self.get_fn(date)])
    
if __name__ == '__main__':
    smap = SMAP(product='SPL3SMP.004')
    coords = smap.native_coordinates
    print ('Done')

