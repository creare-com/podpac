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

from podpac.core.coordinates import Coordinates, UniformCoordinates1d
from podpac.core.coordinates import add_coord
from podpac.core.node import Node
from podpac.core.algorithm.algorithm import Algorithm
from podpac.core.utils import common_doc
from podpac.core.node import COMMON_NODE_DOC, node_eval

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
    """Compute a general convolution over a source node.

    This node automatically resizes the requested coordinates to avoid edge effects.
    
    Attributes
    ----------
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
    """
    
    source = tl.Instance(Node)
    kernel = tl.Instance(np.ndarray).tag(attr=True)
    kernel_type = tl.Unicode().tag(attr=True)
    kernel_ndim = tl.Int().tag(attr=True)

    _expanded_coordinates = tl.Instance(Coordinates)
    _full_kernel = tl.Instance(np.ndarray)
 
    @common_doc(COMMON_DOC)
    @node_eval
    def eval(self, coordinates, output=None):
        """Evaluates this nodes using the supplied coordinates.
        
        Parameters
        ----------
        coordinates : podpac.Coordinates
            {requested_coordinates}
        output : podpac.UnitsDataArray, optional
            {eval_output}
        
        Returns
        -------
        {eval_return}
        """
        # This should be aligned with coordinates' dimension order
        # The size of this kernel is used to figure out the expanded size
        self._full_kernel = self.get_full_kernel(coordinates)
        
        if len(self._full_kernel.shape) != len(coordinates.shape):
            raise ValueError("shape mismatch, kernel does not match source data (%s != %s)" % (
                self._full_kernel.shape, coordinates.shape))

        # expand the coordinates
        exp_coords = []
        exp_slice = []
        for dim, s in zip(coordinates.dims, self._full_kernel.shape):
            coord = coordinates[dim]
            if s == 1 or not isinstance(coord, UniformCoordinates1d):
                exp_coords.append(coord)
                exp_slice.append(slice(None))
                continue

            s_start = -s // 2
            s_end = s // 2 - ((s + 1) % 2)
            # The 1e-07 is for floating point error because if endpoint is slightly
            # in front of step * N then the endpoint is excluded
            exp_coords.append(UniformCoordinates1d(
                add_coord(coord.start, s_start * coord.step),
                add_coord(coord.stop, s_end * coord.step + 1e-07*coord.step),
                coord.step,
                **coord.properties))
            exp_slice.append(slice(-s_start, -s_end))
        exp_slice = tuple(exp_slice)
        self._expanded_coordinates = Coordinates(exp_coords)

        # evaluate source using expanded coordinates, convolve, and then slice out original coordinates
        self.outputs['source'] = self.source.eval(self._expanded_coordinates, method=method)
        
        if np.isnan(np.max(self.outputs['source'])):
            method = 'direct'
        else:
            method = 'auto'

        result = scipy.signal.convolve(self.outputs['source'], self._full_kernel, mode='same', method=method)
        result = result[exp_slice]

        # evaluate using expanded coordinates and then reduce down to originally requested coordinates
        out = super(Convolution, self).eval(exp_coords)
        result = out[exp_slice]
        if output is None:
            output = result
        else:
            output[:] = result

        return output

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
 
    def get_full_kernel(self, coordinates):
        """{full_kernel}
        """
        return self.kernel


class TimeConvolution(Convolution):
    """Compute a temporal convolution over a source node.
    
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

    def get_full_kernel(self, coordinates):
        """{full_kernel}
        
        Raises
        ------
        ValueError
            If source data doesn't have time dimension.
        """
        if 'time' not in coordinates.dims:
            raise ValueError('cannot compute time convolution from time-indepedendent input')
        if 'lat' not in coordinates.dims and 'lon' not in coordinates.dims:
            return self.kernel
 
        kernel = np.array([[self.kernel]])
        kernel = xr.DataArray(kernel, dims=('lat', 'lon', 'time'))
        kernel = kernel.transpose(*coordinates.dims)
        return kernel.data


class SpatialConvolution(Convolution):
    """Compute a lat-lon convolution over a source node.
    
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

    def get_full_kernel(self, coordinates):
        """{full_kernel}
        """
        if 'time' not in coordinates.dims:
            return self.kernel

        kernel = np.array([self.kernel]).T
        kernel = xr.DataArray(kernel, dims=('lat', 'lon', 'time'))
        kernel = kernel.transpose(*coordinates.dims)

        return kernel.data
