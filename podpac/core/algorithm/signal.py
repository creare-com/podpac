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

from podpac.core.settings import settings
from podpac.core.coordinates import Coordinates, UniformCoordinates1d
from podpac.core.coordinates import add_coord
from podpac.core.node import Node
from podpac.core.algorithm.algorithm import Algorithm
from podpac.core.utils import common_doc, ArrayTrait, NodeTrait
from podpac.core.node import COMMON_NODE_DOC, node_eval

COMMON_DOC = COMMON_NODE_DOC.copy()
COMMON_DOC[
    "full_kernel"
] = """Kernel that contains all the dimensions of the input source, in the correct order.
        
        Returns
        -------
        np.ndarray
            The dimensionally full convolution kernel"""
COMMON_DOC[
    "validate_kernel"
] = """Checks to make sure the kernel is valid. 
        
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
            If the kernel is not valid (i.e. incorrect dimensionality). """


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

    source = NodeTrait
    kernel = ArrayTrait(dtype=float).tag(attr=True)

    def _first_init(self, kernel=None, kernel_type=None, kernel_ndim=None, **kwargs):
        if kernel is not None:
            if kernel_type is not None:
                raise TypeError("Convolution expected 'kernel' or 'kernel_type', not both")

        if kernel is None:
            if kernel_type is None:
                raise TypeError("Convolution requires 'kernel' array or 'kernel_type' string")
            if kernel_ndim is None:
                raise TypeError("Convolution requires 'kernel_ndim' when supplying a 'kernel_type' string")

            kernel = self._make_kernel(kernel_type, kernel_ndim)

        kwargs["kernel"] = kernel
        return super(Convolution, self)._first_init(**kwargs)

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
        full_kernel = self._get_full_kernel(coordinates)

        if full_kernel.ndim != coordinates.ndim:
            raise ValueError(
                "Cannot evaluate coordinates, kernel and coordinates ndims mismatch (%d != %d)"
                % (full_kernel.ndim, coordinates.ndim)
            )

        # expand the coordinates
        exp_coords = []
        exp_slice = []
        for dim, s in zip(coordinates.dims, full_kernel.shape):
            coord = coordinates[dim]
            if s == 1 or not isinstance(coord, UniformCoordinates1d):
                exp_coords.append(coord)
                exp_slice.append(slice(None))
                continue

            s_start = -s // 2
            s_end = s // 2 - ((s + 1) % 2)
            # The 1e-07 is for floating point error because if endpoint is slightly
            # in front of step * N then the endpoint is excluded
            exp_coords.append(
                UniformCoordinates1d(
                    add_coord(coord.start, s_start * coord.step),
                    add_coord(coord.stop, s_end * coord.step + 1e-07 * coord.step),
                    coord.step,
                    **coord.properties,
                )
            )
            exp_slice.append(slice(-s_start, -s_end))
        exp_slice = tuple(exp_slice)
        expanded_coordinates = Coordinates(exp_coords)

        if settings["DEBUG"]:
            self._expanded_coordinates = expanded_coordinates

        # evaluate source using expanded coordinates, convolve, and then slice out original coordinates
        source = self.source.eval(expanded_coordinates)

        if np.any(np.isnan(source)):
            method = "direct"
        else:
            method = "auto"

        result = scipy.signal.convolve(source, full_kernel, mode="same", method=method)
        result = result[exp_slice]

        if output is None:
            output = self.create_output_array(coordinates, data=result)
        else:
            output[:] = result

        return output

    @staticmethod
    def _make_kernel(kernel_type, ndim):
        ktype = kernel_type.split(",")[0]
        size = int(kernel_type.split(",")[1])
        if ktype == "mean":
            k = np.ones([size] * ndim)
        else:
            args = [float(a) for a in kernel_type.split(",")[2:]]
            f = getattr(scipy.signal, ktype)
            k1d = f(size, *args)
            k = k1d.copy()
            for i in range(ndim - 1):
                k = np.tensordot(k, k1d, 0)

        return k / k.sum()

    def _get_full_kernel(self, coordinates):
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

    kernel = ArrayTrait(dtype=float, ndim=1).tag(attr=True)

    def _first_init(self, kernel=None, kernel_type=None, kernel_ndim=1, **kwargs):
        return super(TimeConvolution, self)._first_init(
            kernel=kernel, kernel_type=kernel_type, kernel_ndim=kernel_ndim, **kwargs
        )

    def _get_full_kernel(self, coordinates):
        """{full_kernel}
        
        Raises
        ------
        ValueError
            If source data doesn't have time dimension.
        """
        if "time" not in coordinates.dims:
            raise ValueError("cannot compute time convolution with time-independent coordinates")
        if "lat" not in coordinates.dims and "lon" not in coordinates.dims:
            return self.kernel

        kernel = np.array([[self.kernel]])
        kernel = xr.DataArray(kernel, dims=("lat", "lon", "time"))
        kernel = kernel.transpose(*coordinates.dims)
        return kernel.data


class SpatialConvolution(Convolution):
    """Compute a lat-lon convolution over a source node.
    
    Attributes
    ----------
    kernel_ndim : int
        Value is 2. Should not be modified.
    """

    kernel = ArrayTrait(dtype=float, ndim=2).tag(attr=True)

    def _first_init(self, kernel=None, kernel_type=None, kernel_ndim=2, **kwargs):
        return super(SpatialConvolution, self)._first_init(
            kernel=kernel, kernel_type=kernel_type, kernel_ndim=kernel_ndim, **kwargs
        )

    def _get_full_kernel(self, coordinates):
        """{full_kernel}
        """
        if "lat" not in coordinates.dims or "lon" not in coordinates.dims:
            raise ValueError("cannot compute spatial convolution with coordinate dims %s" % coordinates.dims)
        if "time" not in coordinates.dims:
            return self.kernel

        kernel = np.array([self.kernel]).T
        kernel = xr.DataArray(kernel, dims=("lat", "lon", "time"))
        kernel = kernel.transpose(*coordinates.dims)

        return kernel.data
