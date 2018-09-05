from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np
import pytest

from podpac.core.coordinates import Coordinates, UniformCoordinates1d
from podpac.core.algorithm.algorithm import Arange
from podpac.core.algorithm.signal import Convolution, TimeConvolution, SpatialConvolution

class TestSignal(object):
    # TODO break this into unit tests
    # TODO add assertions
    def test_signal(self):
        lat = UniformCoordinates1d(45, 66, size=30, name='lat')
        lon = UniformCoordinates1d(-80, 70, size=40, name='lon')
        time = UniformCoordinates1d('2017-09-01', '2017-10-31', '1,D', name='time')

        coords = Coordinates([time, lat, lon])
        coords_spatial = Coordinates([lat, lon])
        coords_time = Coordinates([time])
        
        kernel3 = np.array([[[1, 2, 1]]])
        kernel2 = np.array([[1, 2, 1]])
        kernel1 = np.array([1, 2, 1])
        
        o = Arange().execute(coords)
        
        node = Convolution(source=Arange(), kernel=kernel3)
        o3d_full = node.execute(coords)
        
        node = Convolution(source=Arange(), kernel=kernel2)
        o2d_spatial1 = node.execute(coords_spatial)
        
        node = SpatialConvolution(source=Arange(), kernel=kernel2)
        o3d_spatial = node.execute(coords)
        o2d_spatial2 = node.execute(coords_spatial)
        
        node = TimeConvolution(source=Arange(), kernel=kernel1)
        o3d_time = node.execute(coords)
        o3d_time = node.execute(coords_time)
        
        node = SpatialConvolution(source=Arange(), kernel_type='gaussian, 3, 1')
        o3d_spatial = node.execute(coords)
        o2d_spatial2 = node.execute(coords_spatial)
        node = SpatialConvolution(source=Arange(), kernel_type='mean, 3')
        o3d_spatial = node.execute(coords)
        o2d_spatial2 = node.execute(coords_spatial)
        
        node = TimeConvolution(source=Arange(), kernel_type='gaussian, 3, 1')
        o3d_time = node.execute(coords)
        node = TimeConvolution(source=Arange(), kernel_type='mean, 3')
        o3d_time = node.execute(coords)
        
        node = Convolution(source=Arange(), kernel=kernel2)
        with pytest.raises(Exception): # TODO which exception
            node.execute(coords)
        
        node = Convolution(source=Arange(), kernel=kernel1)
        with pytest.raises(Exception): # TODO which exception?
            node.execute(coords_spatial)
        
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
            node.execute(coords_spatial)
        
