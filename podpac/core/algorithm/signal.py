"""
Signal Summary

NOTE: another option for the convolution kernel input would be to accept an
    xarray and add and transpose dimensions as necessary.

NOTE: At the moment this module is quite brittle... it makes assumptions about
    the input node (i.e. alt coordinates would likely break this)
"""

import podpac
from collections import OrderedDict

import traitlets as tl
import numpy as np
import xarray as xr
import scipy.signal

from podpac.core.coordinate import Coordinate, UniformCoord
from podpac.core.coordinate import add_coord
from podpac.core.node import Node
from podpac.core.algorithm.algorithm import Algorithm
from podpac.core.utils import common_doc
from podpac.core.node import COMMON_NODE_DOC

COMMON_DOC = COMMON_NODE_DOC.copy()
COMMON_DOC['full_kernel'] = '''Kernel that contains all the dimensions of the input source, in the correct order.
        
        Returns
        -------
        np.ndarray
            The dimensionally full convolution kernel'''
COMMON_DOC['validate_kernel'] = '''Checks to make sure the kernel is valid. 
        
        Parameters
        ----------
        proposal : np.ndarray
            The proposed kernel
        
        Returns
        -------
        np.ndarray
            The valid kernel
        
        Raises
        ------
        ValueError
            If the kernel is not valid (i.e. incorrect dimensionality). '''

class Convolution(Algorithm):
    """Base algorithm node for computing convolutions. This node automatically resizes the request to avoid edge effects.
    
    Attributes
    ----------
    expanded_coordinates : podpac.Coordinate
        The expanded coordinates needed to avoid edge effects.
    source : podpac.Node
        Source node on which convolution will be performed. 
    kernel : np.ndarray
        The convolution kernel
    kernel_ndim : int
        Number of dimensions of the kernel
    kernel_type : str, optional
        If kernel is not defined, kernel_type will create a kernel based on the inputs. 
        The format for the created  kernels is '<kernel_type>, <kernel_size>, <kernel_params>'.
        Any kernel defined in `scipy.signal` as well as `mean` can be used. For example:
        kernel_type = 'mean, 8' or kernel_type = 'gaussian,16,8' are both valid. 
        Note: These kernels are automatically normalized such that kernel.sum() == 1
    output_coordinates : podpac.Coordinate
        The non-expanded coordinates
    """
    
    source = tl.Instance(Node)
    kernel = tl.Instance(np.ndarray)  # Would like to tag this, but arrays are not yet supported
    kernel_type = tl.Unicode().tag(attr=True)
    kernel_ndim = tl.Int().tag(attr=True)
    output_coordinates = tl.Instance(Coordinate)
    expanded_coordinates = tl.Instance(Coordinate)
   
    @property
    def native_coordinates(self):
        """Returns the native coordinates of the source node. 
        """
        return self.source.native_coordinates
 
    @common_doc(COMMON_DOC)
    def execute(self, coordinates, params=None, output=None, method=None):
        """Executes this nodes using the supplied coordinates and params. 
        
        Parameters
        ----------
        coordinates : podpac.Coordinate
            {evaluated_coordinates}
        params : dict, optional
            {execute_params} 
        output : podpac.UnitsDataArray, optional
            {execute_out}
        method : str, optional
            {execute_method}
        
        Returns
        -------
        {execute_return}
        """
        self.evaluated_coordinates = coordinates
        self._params = self.get_params(params)
        self.output = output
        # This is needed to get the full_kernel
        self.output_coordinates = self.source.get_output_coords(coordinates)

        # This should be aligned with coordinates' dimension order
        # The size of this kernel is used to figure out the expanded size
        shape = self.full_kernel.shape
        
        if len(shape) != len(self.output_coordinates.shape):
            raise ValueError("Kernel shape does not match source data shape")

        # expand the coordinates
        exp_coords = OrderedDict()
        exp_slice = []
        for c, s in zip(coordinates._coords, shape):
            coord = coordinates[c]
            if s == 1 or (not isinstance(coord, UniformCoord)):
                exp_coords[c] = coord
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
        out = super(Convolution, self).execute(exp_coords, params, output, method)

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
        """{full_kernel}
        """
        return self.kernel

    def algorithm(self):
        """Computes the convolution of the source and the kernel
        
        Returns
        -------
        np.ndarray
            Resultant array. 
        """
        if np.isnan(np.max(self.source.output)):
            method = 'direct'
        else: method = 'auto'
        res = scipy.signal.convolve(self.source.output,
                                    self.full_kernel,
                                    mode='same', method=method)
        return res


class TimeConvolution(Convolution):
    """Specialized convolution node that computes temporal convolutions only.
    
    Attributes
    ----------
    kernel_ndim : int
        Value is 1. Should not be modified.
    """
    
    kernel_ndim = tl.Int(1)
    @tl.validate('kernel')
    def validate_kernel(self, proposal):
        """{validate_kernel}
        """
        if proposal['value'].ndim != 1:
            raise ValueError('kernel must have ndim=1 (got ndim=%d)' % (
                proposal['value'].ndim))

        return proposal['value']

    @property
    def full_kernel(self):
        """{full_kernel}
        
        Raises
        ------
        ValueError
            If source data doesn't have time dimension.
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
    """Specialized convolution node that computes lat-lon convolutions only.
    
    Attributes
    ----------
    kernel_ndim : int
        Value is 2. Should not be modified.
    """
    
    kernel_ndim = tl.Int(2)
    @tl.validate('kernel')
    def validate_kernel(self, proposal):
        """{validate_kernel}
        """
        if proposal['value'].ndim != 2:
            raise ValueError('kernel must have ndim=2 (got ndim=%d)' % (proposal['value'].ndim))

        return proposal['value']

    @property
    def full_kernel(self):
        """{full_kernel}
        """
        if 'time' not in self.output_coordinates.dims:
            return self.kernel

        kernel = np.array([self.kernel]).T
        kernel = xr.DataArray(kernel, dims=('lat', 'lon', 'time'))
        kernel = kernel.transpose(*self.output_coordinates.dims)

        return kernel.data
