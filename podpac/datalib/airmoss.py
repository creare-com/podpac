"""
Airmoss summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import re
from collections import OrderedDict

import requests
from bs4 import BeautifulSoup
import numpy as np
import traitlets as tl

# Internal dependencies
import podpac
from podpac.core.data import types as datatype

class AirMOSS_Source(datatype.PyDAP):
    """Summary

    Attributes
    ----------
    datakey : TYPE
        Description
    date_url_re : TYPE
        Description
    nan_vals : list
        Description
    product : TYPE
        Description
    """

    product = tl.Enum(['L4RZSM'], default_value='L4RZSM')
    date_url_re = re.compile('[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}')
    datakey = tl.Unicode(u'sm1')
    nan_vals = [-9999.0]

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
        base_date = ds['time'].attributes['units']
        base_date = self.date_url_re.search(base_date).group()
        times = (ds['time'][:]).astype('timedelta64[h]') + np.array(base_date, 'datetime64')

        lons = podpac.crange(ds['lon'][0], ds['lon'][-1], ds['lon'][1] - ds['lon'][0])
        lats = podpac.crange(ds['lat'][0], ds['lat'][-1], ds['lat'][1] - ds['lat'][0])
        coords = podpac.Coordinates([times, lats, lons], dims=['time', 'lat', 'lon'])
        self.cache_obj(coords, 'native.coordinates')

        return coords

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
        data = self.dataset[self.datakey].array[tuple(coordinates_index)]
        d = self.create_output_array(coordinates, data=data.reshape(coordinates.shape))
        return d


class AirMOSS_Site(podpac.OrderedCompositor):
    """Summary

    Attributes
    ----------
    base_dir_url : TYPE
        Description
    base_url : TYPE
        Description
    date_url_re : TYPE
        Description
    product : TYPE
        Description
    site : TYPE
        Description
    """

    product = tl.Enum(['L4RZSM'], default_value='L4RZSM')
    base_url = tl.Unicode(u'https://thredds.daac.ornl.gov/thredds/dodsC/ornldaac/1421')
    base_dir_url = tl.Unicode(u'https://thredds.daac.ornl.gov/thredds/catalog/ornldaac/1421/catalog.html')
    site = tl.Unicode('')
    date_url_re = re.compile('[0-9]{8}')

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
        times = self.get_available_dates()
        lons = podpac.crange(ds['lon'][0], ds['lon'][-1], ds['lon'][1] - ds['lon'][0])
        lats = podpac.crange(ds['lat'][0], ds['lat'][-1], ds['lat'][1] - ds['lat'][0])
        coords = podpac.Coordinates([times, lats, lons], dims=['time', 'lat', 'lon'])
        self.cache_obj(coords, 'native.coordinates')

        return coords

    def get_available_dates(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        soup = BeautifulSoup(requests.get(self.base_dir_url).text, 'lxml')
        a = soup.find_all('a')
        regex = self.date_url_re

        times = []
        for aa in a:
            text = aa.get_text()
            if self.site in text:
                m = regex.search(text)
                if m:
                    t = m.group()
                    times.append(np.datetime64('-'.join([t[:4], t[4:6], t[6:]])))
        times.sort()
        return np.array(times)



class AirMOSS(podpac.OrderedCompositor):
    """Summary

    Attributes
    ----------
    product : TYPE
        Description
    site_url_re : TYPE
        Description
    """

    product = tl.Enum(['L4RZSM'], default_value='L4RZSM')
    site_url_re = tl.Any()

    @tl.default('site_url_re')
    def get_site_url_re(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return re.compile(self.product + '_.*_' + '[0-9]{8}.*\.nc4')

    def get_available_sites(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        soup = BeautifulSoup(requests.get(self.base_dir_url).text, 'lxml')
        a = soup.find_all('a')
        regex = self.site_url_re

        sites = OrderedDict()
        for aa in a:
            text = aa.get_text()
            m = regex.match(text)
            if m:
                site = text.split('_')[1]
                sites[site] = 1 + sites.get(site, 0)

        return sites

if __name__ == '__main__':
    ams = AirMOSS_Site(interpolation='nearest_preview',
                       site='BermsP')
    print(ams.native_coordinates)

    source = 'https://thredds.daac.ornl.gov/thredds/dodsC/ornldaac/1421/L4RZSM_BermsP_20121025_v5.nc4'
    am = AirMOSS_Source(source=source, interpolation='nearest_preview')
    coords = am.native_coordinates
    print(coords)
    print(coords['time'].area_bounds)

    lat, lon = am.native_coordinates.coords['lat'], am.native_coordinates.coords['lon']
    lat = lat[::10][np.isfinite(lat[::10])]
    lon = lon[::10][np.isfinite(lon[::10])]
    coords = podpac.Coordinates([lat, lon], order=['lat', 'lon'])
    o = am.eval(coords)

    print('Done')
