"""
Signal Summary

NOTE: another option for the convolution kernel input would be to accept an
    xarray and add and transpose dimensions as necessary.

NOTE: At the moment this module is quite brittle... it makes assumptions about
    the input node (i.e. alt coordinates would likely break this)
"""

from collections import OrderedDict

import traitlets as tl
import numpy as np
import xarray as xr
import scipy.signal

from podpac.core.coordinate import Coordinate, UniformCoord
from podpac.core.coordinate import add_coord
from podpac.core.node import Node
from podpac.core.algorithm.algorithm import Algorithm

class Convolution(Algorithm):
    """Summary
    
    Attributes
    ----------
    evaluated_coordinates : TYPE
        Description
    expanded_coordinates : TYPE
        Description
    input_node : TYPE
        Description
    kernel : TYPE
        Description
    kernel_ndim : TYPE
        Description
    kernel_type : TYPE
        Description
    output : TYPE
        Description
    output_coordinates : TYPE
        Description
    params : TYPE
        Description
    """
    
    input_node = tl.Instance(Node)
    kernel = tl.Instance(np.ndarray)
    kernel_type = tl.Unicode()
    kernel_ndim = tl.Int()
    output_coordinates = tl.Instance(Coordinate)
    expanded_coordinates = tl.Instance(Coordinate)
   
    @property
    def native_coordinates(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        return self.input_node.native_coordinates
 
    def execute(self, coordinates, params=None, output=None):
        """Summary
        
        Parameters
        ----------
        coordinates : TYPE
            Description
        params : None, optional
            Description
        output : None, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        self.evaluated_coordinates = coordinates
        self.params = params
        self.output = output
        # This is needed to get the full_kernel
        self.output_coordinates = self.input_node.get_output_coords(coordinates)

        # This should be aligned with coordinates' dimension order
        # The size of this kernel is used to figure out the expanded size
        shape = self.full_kernel.shape

        # expand the coordinates
        exp_coords = OrderedDict()
        exp_slice = []
        for c, s in zip(coordinates._coords, shape):
            coord = coordinates[c]
            if s == 1:
                exp_coords[c] = coord
                exp_slice.append(slice(None))
                continue
            if not isinstance(coord, UniformCoord):
                exp_slice.append(slice(None))
                continue
            s_start = -s // 2
            s_end = s // 2 - ((s + 1) % 2)
            # The 1e-07 is for floating point error because if endpoint is slightly
            # in front of delta * N then the endpoint is excluded
            exp_coords[c] = UniformCoord(
                start=add_coord(coord.start, s_start * coord.delta),
                stop=add_coord(coord.stop, s_end * coord.delta + 1e-07*coord.delta),
                delta=coord.delta)
            exp_slice.append(slice(-s_start, -s_end))
        exp_coords = Coordinate(exp_coords)
        self.expanded_coordinates = exp_coords
        exp_slice = tuple(exp_slice)

        # execute using expanded coordinates
        out = super(Convolution, self).execute(exp_coords, params, output)

        # reduce down to originally requested coordinates
        self.output = out[exp_slice]

        return self.output

    @tl.default('kernel')
    def _kernel_default(self):
        kernel_type = self.kernel_type
        if not kernel_type:
            raise ValueError("Need to supply either 'kernel' as a numpy array,"
                             " or 'kernel_type' as a string.")
        ktype = kernel_type.split(',')[0]
        size = int(kernel_type.split(',')[1])
        args = [float(a) for a in kernel_type.split(',')[2:]]
        if ktype == 'mean':
            k = np.ones([size] * self.kernel_ndim)
        else:
            f = getattr(scipy.signal, ktype)
            k1d = f(size, *args)
            k = k1d.copy()
            for i in range(self.kernel_ndim - 1):
                k = np.tensordot(k, k1d, 0)
        
        return k / k.sum()
 
    @property
    def full_kernel(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        return self.kernel

    def algorithm(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if np.isnan(np.max(self.input_node.output)):
            method = 'direct'
        else: method = 'auto'
        res = scipy.signal.convolve(self.input_node.output,
                                    self.full_kernel,
                                    mode='same', method=method)
        return res


class TimeConvolution(Convolution):
    """Summary
    
    Attributes
    ----------
    kernel_ndim : TYPE
        Description
    """
    
    kernel_ndim = tl.Int(1)
    @tl.validate('kernel')
    def validate_kernel(self, proposal):
        """Summary
        
        Parameters
        ----------
        proposal : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        
        Raises
        ------
        ValueError
            Description
        """
        if proposal['value'].ndim != 1:
            raise ValueError('kernel must have ndim=1 (got ndim=%d)' % (
                proposal['value'].ndim))

        return proposal['value']

    @property
    def full_kernel(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        
        Raises
        ------
        ValueError
            Description
        """
        if 'time' not in self.output_coordinates.dims:
            raise ValueError('cannot compute time convolution from time-indepedendent input')
        if 'lat' not in self.output_coordinates.dims and 'lon' not in self.output_coordinates.dims:
            return self.kernel
 
        kernel = np.array([[self.kernel]])
        kernel = xr.DataArray(kernel, dims=('lat', 'lon', 'time'))
        kernel = kernel.transpose(*self.output_coordinates.dims)
        return kernel.data


class SpatialConvolution(Convolution):
    """Summary
    
    Attributes
    ----------
    kernel_ndim : TYPE
        Description
    """
    
    kernel_ndim = tl.Int(2)
    @tl.validate('kernel')
    def validate_kernel(self, proposal):
        """Summary
        
        Parameters
        ----------
        proposal : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        
        Raises
        ------
        ValueError
            Description
        """
        if proposal['value'].ndim != 2:
            raise ValueError('kernel must have ndim=2 (got ndim=%d)' % (proposal['value'].ndim))

        return proposal['value']

    @property
    def full_kernel(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if 'time' not in self.output_coordinates.dims:
            return self.kernel

        kernel = np.array([self.kernel]).T
        kernel = xr.DataArray(kernel, dims=('lat', 'lon', 'time'))
        kernel = kernel.transpose(*self.output_coordinates.dims)

        return kernel.data

if __name__ == '__main__':
    from podpac.core.algorithm.algorithm import Arange
    import podpac

    coords = podpac.coordinate(
        time=('2017-09-01', '2017-10-31', '1,D'),
        lat=[45., 66., 30], lon=[-80., -70., 40],
        order=['time', 'lat', 'lon'])

    coords_spatial = podpac.coordinate(
        lat=[45., 66., 30], lon=[-80., -70., 40],
        order=['lat', 'lon'])
    
    coords_time = podpac.coordinate(
        time=('2017-09-01', '2017-10-31', '1,D'),
        order=['time'])

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
    o3d_time = node.execute(coords_time)

    node = SpatialConvolution(input_node=Arange(), kernel_type='gaussian, 3, 1')
    o3d_spatial = node.execute(coords)
    o2d_spatial2 = node.execute(coords_spatial)
    node = SpatialConvolution(input_node=Arange(), kernel_type='mean, 3')
    o3d_spatial = node.execute(coords)
    o2d_spatial2 = node.execute(coords_spatial)

    node = TimeConvolution(input_node=Arange(), kernel_type='gaussian, 3, 1')
    o3d_time = node.execute(coords)
    node = TimeConvolution(input_node=Arange(), kernel_type='mean, 3')
    o3d_time = node.execute(coords)

    node = Convolution(input_node=Arange(), kernel=kernel2)
    try: node.execute(coords)
    except: pass # should fail because the input node is 3 dimensions
    else: raise Exception("expected an exception")

    node = Convolution(input_node=Arange(), kernel=kernel1)
    try: node.execute(coords_spatial)
    except: pass # should fail because the input node is 1 dimensions
    else: raise Exception("expected an exception")

    try: node = SpatialConvolution(input_node=Arange(), kernel=kernel3)
    except: pass # should fail because the kernel is 3 dimensions
    else: raise Exception("expected an exception")

    try: node = SpatialConvolution(input_node=Arange(), kernel=kernel1)
    except: pass # should fail because the kernel is 1 dimension
    else: raise Exception("expected an exception")

    try: node = TimeConvolution(input_node=Arange(), kernel=kernel3)
    except: pass # should fail because the kernel is 3 dimensions
    else: raise Exception("expected an exception")

    try: node = TimeConvolution(input_node=Arange(), kernel=kernel2)
    except: pass # should fail because the kernel is 2 dimensions
    else: raise Exception("expected an exception")

    node = TimeConvolution(input_node=Arange(), kernel=kernel1)
    try: node.execute(coords_spatial)
    except: pass # should fail because the input_node has no time dimension
    else: raise Exception("expected an exception")
