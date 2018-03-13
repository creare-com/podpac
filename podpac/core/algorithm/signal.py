
import traitlets as tl
import numpy as np
import xarray as xr
import scipy.signal

from podpac.core.coordinate import Coordinate
from podpac.core.node import Node
from podpac.core.algorithm.algorithm import Algorithm

#NOTE: another option for the convolution kernel input would be to accept an
#      xarray and add and transpose dimensions as necessary.

class Convolution(Algorithm):
    input_node = tl.Instance(Node)
    kernel = tl.Instance(np.ndarray)
    
    @property
    def full_kernel(self):
        return self.kernel

    def algorithm(self):
        res = scipy.signal.convolve(self.input_node.output, self.full_kernel, mode='same')
        return res

class TimeConvolution(Convolution):
    @tl.validate('kernel')
    def validate_kernel(self, proposal):
        if proposal['value'].ndim != 1:
            raise ValueError('kernel must have ndim=1 (got ndim=%d)' % (
                proposal['value'].ndim))

        return proposal['value']

    @property
    def full_kernel(self):
        if 'time' not in self.input_node.output.dims:
            raise ValueError('cannot compute time convolution from'
                             'time-indepedendent input')
        
        kernel = np.array([[self.kernel]])
        kernel = xr.DataArray(kernel, dims=('lat', 'lon', 'time'))
        kernel = kernel.transpose(*self.input_node.output.dims)
        return kernel.data

class SpatialConvolution(Convolution):
    @tl.validate('kernel')
    def validate_kernel(self, proposal):
        if proposal['value'].ndim != 2:
            raise ValueError('kernel must have ndim=2 (got ndim=%d)' % (
                proposal['value'].ndim))

        return proposal['value']

    @property
    def full_kernel(self):
        x = self.input_node.output
        if 'time' not in self.input_node.output.dims:
            return self.kernel

        kernel = np.array([self.kernel])
        kernel = xr.DataArray(kernel, dims=('time', 'lat', 'lon'))
        kernel = kernel.transpose(*self.input_node.output.dims)
        return kernel.data

if __name__ == '__main__':
    from podpac.core.algorithm.algorithm import Arange

    coords = Coordinate(
        time=('2017-09-01', '2017-10-31', '1,D'),
        lat=[45., 66., 30], lon=[-80., -70., 40],
        order=['time', 'lat', 'lon'])

    coords_spatial = Coordinate(
        lat=[45., 66., 30], lon=[-80., -70., 40],
        order=['lat', 'lon'])

    kernel3 = np.array([[[1, 2, 1]]])
    kernel2 = np.array([[1, 2, 1]])
    kernel1 = np.array([1, 2, 1])

    o = Arange().execute(coords)

    node = Convolution(input_node=Arange(), kernel=kernel3)
    o3d_full = node.execute(coords)

    node = Convolution(input_node=Arange(), kernel=kernel2)
    o2d_spatial1 = node.execute(coords_spatial)
    
    node = SpatialConvolution(input_node=Arange(), kernel=kernel2)
    o3d_spatial = node.execute(coords)
    o2d_spatial2 = node.execute(coords_spatial)

    node = TimeConvolution(input_node=Arange(), kernel=kernel1)
    o3d_time = node.execute(coords)

    node = Convolution(input_node=Arange(), kernel=kernel2)
    try: node.execute(coords)
    except: pass # should fail becaus the input node is 3 dimensions
    else: raise Exception("expected an exception")

    node = Convolution(input_node=Arange(), kernel=kernel)
    try: node.execute(coords_spatial)
    except: pass # should fail becaus the input node is 2 dimensions
    else: raise Exception("expected an exception")

    try: node = SpatialConvolution(input_node=Arange(), kernel=kernel)
    except: pass # should fail because the kernel is 3 dimensions
    else: raise Exception("expected an exception")

    try: node = SpatialConvolution(input_node=Arange(), kernel=kernel1)
    except: pass # should fail because the kernel is 1 dimension
    else: raise Exception("expected an exception")

    try: node = TimeConvolution(input_node=Arange(), kernel=kernel)
    except: pass # should fail because the kernel is 3 dimensions
    else: raise Exception("expected an exception")

    try: node = TimeConvolution(input_node=Arange(), kernel=kernel2)
    except: pass # should fail because the kernel is 2 dimensions
    else: raise Exception("expected an exception")

    node = TimeConvolution(input_node=Arange(), kernel=kernel1)
    try: node.execute(coords_spatial)
    except: pass # should fail because the input_node has no time dimension
    else: raise Exception("expected an exception")
