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

class AirMOSS_Source(datatype.PyDAP):
    base_url = tl.Unicode(    u'https://thredds.daac.ornl.gov/thredds/dodsC/'
                              u'ornldaac/1421')
    base_dir_url = tl.Unicode(u'https://thredds.daac.ornl.gov/thredds/catalog/'
                              u'ornldaac/1421/catalog.html')
    product = tl.Enum(['L4RZSM'])
    date_url_re = re.compile('[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}')
    datakey = tl.Unicode(u'sm1')    
    no_data_vals = [-9999.0]
    
    @tl.default('native_coordinates')
    def set_native_coordinates(self):
        if os.path.exists(self.cache_path('native.coordinates')):
            return self.load_cached_obj('native.coordinates')
        
        ds = self.dataset
        base_date = ds['time'].attributes['units']
        base_date = self.date_url_re.search(base_date).group()
        times = (ds['time'][:]).astype('timedelta64[h]')\
            + np.array(base_date, 'datetime64')
        lons = (ds['lon'][0], ds['lon'][-1], 
                ds['lon'][1] - ds['lon'][0])
        lats = (ds['lat'][0], ds['lat'][-1], 
                ds['lat'][1] - ds['lat'][0])

        coords = podpac.Coordinate(time=np.array(times), lat=lats, lon=lons,
                                   coord_order=['time', 'lat', 'lon'])
        self.cache_obj(coords, 'native.coordinates')
        
        return coords
    
    def get_data(self, coordinates, coordinates_slice):
        data = self.dataset[self.datakey].array[tuple(coordinates_slice)]
        d = self.initialize_coord_array(coordinates, 'data', 
                                        fillval=data.reshape(coordinates.shape))
        return d    

if __name__ == '__main__':
    source = ('https://thredds.daac.ornl.gov/thredds/dodsC/ornldaac/1421/'
              'L4RZSM_BermsP_20121025_v5.nc4')
    am = AirMOSS_Source(source=source, interpolation='nearest_preview')
    coords = am.native_coordinates
    print (coords)
    print (coords['time'].area_bounds)

    lat, lon = am.native_coordinates.coords['lat'], am.native_coordinates.coords['lon']
    lat = lat[::10][np.isfinite(lat[::10])]
    lon = lon[::10][np.isfinite(lon[::10])]
    coords = podpac.Coordinate(lat=lat, lon=lon, coord_order=['lat', 'lon'])
    o = am.execute(coords)    
  
    print ('Done')

