from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np
import xarray as xr
from pint.errors import DimensionalityError
from pint import UnitRegistry
ureg = UnitRegistry()

from podpac.core.coordinate import Coordinate
from podpac.core.algorithm.algorithm import Algorithm, SinCoords, Arithmetic
from podpac.core.algorithm.coord_select import ExpandCoordinates, SelectCoordinates
from podpac.core.algorithm.algorithm import Arange
from podpac.core.algorithm.signal import Convolution, TimeConvolution, SpatialConvolution

class TestSignal(object):
    def test_signal(self):
        coords = Coordinate(
            time=('2017-09-01', '2017-10-31', '1,D'),
            lat=[45., 66., 30], lon=[-80., -70., 40],
            order=['time', 'lat', 'lon'])
        
        coords_spatial = Coordinate(
            lat=[45., 66., 30], lon=[-80., -70., 40],
            order=['lat', 'lon'])
        
        coords_time = Coordinate(
            time=('2017-09-01', '2017-10-31', '1,D'),
            order=['time'])
        
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
        try: node.execute(coords)
        except: pass # should fail because the input node is 3 dimensions
        else: raise Exception("expected an exception")
        
        node = Convolution(source=Arange(), kernel=kernel1)
        try: node.execute(coords_spatial)
        except: pass # should fail because the input node is 1 dimensions
        else: raise Exception("expected an exception")
        
        try: node = SpatialConvolution(source=Arange(), kernel=kernel3)
        except: pass # should fail because the kernel is 3 dimensions
        else: raise Exception("expected an exception")
        
        try: node = SpatialConvolution(source=Arange(), kernel=kernel1)
        except: pass # should fail because the kernel is 1 dimension
        else: raise Exception("expected an exception")
        
        try: node = TimeConvolution(source=Arange(), kernel=kernel3)
        except: pass # should fail because the kernel is 3 dimensions
        else: raise Exception("expected an exception")
        
        try: node = TimeConvolution(source=Arange(), kernel=kernel2)
        except: pass # should fail because the kernel is 2 dimensions
        else: raise Exception("expected an exception")
        
        node = TimeConvolution(source=Arange(), kernel=kernel1)
        try: node.execute(coords_spatial)
        except: pass # should fail because the source has no time dimension
        else: raise Exception("expected an exception")
