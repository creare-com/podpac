from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np
import pytest

import podpac
from podpac.core.algorithm.algorithm import Arange
from podpac.core.algorithm.signal import Convolution, TimeConvolution, SpatialConvolution

class TestSignal(object):
    # TODO break this into unit tests
    # TODO add assertions
    def test_signal(self):
        lat = podpac.clinspace(45, 66, 30)
        lon = podpac.clinspace(-80, 70, 40)
        time = podpac.crange('2017-09-01', '2017-10-31', '1,D')

        coords = podpac.Coordinates([time, lat, lon], dims=['time','lat', 'lon'])
        coords_spatial = podpac.Coordinates([lat, lon], dims=['lat', 'lon'])
        coords_time = podpac.Coordinates([time], dims=['time'])
        
        kernel3 = np.array([[[1, 2, 1]]])
        kernel2 = np.array([[1, 2, 1]])
        kernel1 = np.array([1, 2, 1])
        
        o = Arange().eval(coords)
        
        node = Convolution(source=Arange(), kernel=kernel3)
        o3d_full = node.eval(coords)
        
        node = Convolution(source=Arange(), kernel=kernel2)
        o2d_spatial1 = node.eval(coords_spatial)
        
        node = SpatialConvolution(source=Arange(), kernel=kernel2)
        o3d_spatial = node.eval(coords)
        o2d_spatial2 = node.eval(coords_spatial)
        
        node = TimeConvolution(source=Arange(), kernel=kernel1)
        o3d_time = node.eval(coords)
        o3d_time = node.eval(coords_time)
        
        node = SpatialConvolution(source=Arange(), kernel_type='gaussian, 3, 1')
        o3d_spatial = node.eval(coords)
        o2d_spatial2 = node.eval(coords_spatial)
        node = SpatialConvolution(source=Arange(), kernel_type='mean, 3')
        o3d_spatial = node.eval(coords)
        o2d_spatial2 = node.eval(coords_spatial)
        
        node = TimeConvolution(source=Arange(), kernel_type='gaussian, 3, 1')
        o3d_time = node.eval(coords)
        node = TimeConvolution(source=Arange(), kernel_type='mean, 3')
        o3d_time = node.eval(coords)
        
        node = Convolution(source=Arange(), kernel=kernel2)
        with pytest.raises(Exception): # TODO which exception
            node.eval(coords)
        
        node = Convolution(source=Arange(), kernel=kernel1)
        with pytest.raises(Exception): # TODO which exception?
            node.eval(coords_spatial)
        
        with pytest.raises(Exception): # TODO which exception?
            node = SpatialConvolution(source=Arange(), kernel=kernel3)
        
        with pytest.raises(Exception): # TODO which exception?
            node = SpatialConvolution(source=Arange(), kernel=kernel1)
        
        with pytest.raises(Exception): # TODO which exception?
            node = TimeConvolution(source=Arange(), kernel=kernel3)
        
        with pytest.raises(Exception): # TODO which exception?
            node = TimeConvolution(source=Arange(), kernel=kernel2)
        
        node = TimeConvolution(source=Arange(), kernel=kernel1)
        with pytest.raises(Exception): # TODO which exception?
            node.eval(coords_spatial)
        
