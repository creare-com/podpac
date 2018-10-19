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
    output_coordinates : podpac.Coordinates
        The non-expanded coordinates
    """
    
    source = tl.Instance(Node)
    kernel = tl.Instance(np.ndarray)  # Would like to tag this, but arrays are not yet supported
    kernel_type = tl.Unicode().tag(attr=True)
    kernel_ndim = tl.Int().tag(attr=True)

    _expanded_coordinates = tl.Instance(Coordinates)
 
    @common_doc(COMMON_DOC)
    def eval(self, coordinates, output=None, method=None):
        """Evaluates this nodes using the supplied coordinates.
        
        Parameters
        ----------
        coordinates : podpac.Coordinates
            {requested_coordinates}
        output : podpac.UnitsDataArray, optional
            {eval_output}
        method : str, optional
            {eval_method}
        
        Returns
        -------
        {eval_return}
        """
        self._requested_coordinates = coordinates
        self._output_coordinates = coordinates
        self.output = output

        # This should be aligned with coordinates' dimension order
        # The size of this kernel is used to figure out the expanded size
        shape = self.full_kernel.shape
        
        if len(shape) != len(self._output_coordinates.shape):
            raise ValueError("Kernel shape does not match source data shape")

        # expand the coordinates
        exp_coords = []
        exp_slice = []
        for dim, s in zip(coordinates.dims, shape):
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
        exp_coords = Coordinates(exp_coords)
        exp_slice = tuple(exp_slice)

        # evaluate using expanded coordinates and then reduce down to originally requested coordinates
        self._expanded_coordinates = exp_coords
        out = super(Convolution, self).eval(exp_coords, output, method)
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
        if 'time' not in self._output_coordinates.dims:
            raise ValueError('cannot compute time convolution from time-indepedendent input')
        if 'lat' not in self._output_coordinates.dims and 'lon' not in self._output_coordinates.dims:
            return self.kernel
 
        kernel = np.array([[self.kernel]])
        kernel = xr.DataArray(kernel, dims=('lat', 'lon', 'time'))
        kernel = kernel.transpose(*self._output_coordinates.dims)
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
        if 'time' not in self._output_coordinates.dims:
            return self.kernel

        kernel = np.array([self.kernel]).T
        kernel = xr.DataArray(kernel, dims=('lat', 'lon', 'time'))
        kernel = kernel.transpose(*self._output_coordinates.dims)

        return kernel.data
