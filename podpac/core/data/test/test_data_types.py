from __future__ import division, unicode_literals, print_function, absolute_import

import pytest
import numpy as np
import xarray as xr

import podpac
from podpac.core.data.type import NumpyArray

class TestBasicInterpolation(object):
    def setup_method(self, method):
        self.coord_src = podpac.coordinate(
            lat=(45, 0, 16),
            lon=(-70., -65., 16),
            time=(0, 1, 2),
            order=['lat', 'lon', 'time'])
        
        LON, LAT, TIME = np.meshgrid(
            self.coord_src['lon'].coordinates,
            self.coord_src['lat'].coordinates,
            self.coord_src['time'].coordinates)        
        
        self.latSource = LAT
        self.lonSource = LON
        self.timeSource = TIME
        
        self.nasLat = NumpyArray(
            source=LAT.astype(float),
            native_coordinates=self.coord_src,
            interpolation='bilinear')
        
        self.nasLon = NumpyArray(
            source=LON.astype(float),
            native_coordinates=self.coord_src, 
            interpolation='bilinear')

        self.nasTime = NumpyArray(source=TIME.astype(float),
            native_coordinates=self.coord_src, 
            interpolation='bilinear')

    def test_raster_to_raster(self):
        coord_dst = podpac.coordinate(
            lat=(5., 40., 50),
            lon=(-68., -66., 100),
            order=['lat', 'lon'])

        oLat = self.nasLat.execute(coord_dst)
        oLon = self.nasLon.execute(coord_dst)
        
        LON, LAT = np.meshgrid(
            coord_dst['lon'].coordinates,
            coord_dst['lat'].coordinates)
        
        np.testing.assert_array_almost_equal(oLat.data[..., 0], LAT)
        np.testing.assert_array_almost_equal(oLon.data[..., 0], LON)
        
    def test_raster_to_points(self):
        coord_dst = podpac.coordinate(lat_lon=((5., 40), (-68., -66), 60))
        oLat = self.nasLat.execute(coord_dst)
        oLon = self.nasLon.execute(coord_dst)
        
        LAT = coord_dst.coords['lat_lon']['lat']
        LON = coord_dst.coords['lat_lon']['lon']
        
        np.testing.assert_array_almost_equal(oLat.data[..., 0], LAT)
        np.testing.assert_array_almost_equal(oLon.data[..., 0], LON)    
