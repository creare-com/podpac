"""
Stats Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import warnings
from operator import mul
from functools import reduce

import xarray as xr
import numpy as np
import scipy.stats
import traitlets as tl
from six import string_types

from podpac.core.coordinates import Coordinates
from podpac.core.node import Node
from podpac.core.algorithm.algorithm import Algorithm
from podpac.core.utils import common_doc
from podpac.core.node import COMMON_NODE_DOC

COMMON_DOC = COMMON_NODE_DOC.copy()

# =============================================================================
# Reduce Nodes
# =============================================================================

class Reduce(Algorithm):
    """Base node for reduction algorithms
    
    Attributes
    ----------
    dims : list
        List of strings that give the dimensions which should be reduced
    input_coordinates : podpac.Coordinates
        The input coordinates before reduction
    source : podpac.Node
        The source node that will be reduced. 
    """
    
    input_coordinates = tl.Instance(Coordinates)
    source = tl.Instance(Node)

    dims = tl.List().tag(attr=True)
    iter_chunk_size = tl.Union([tl.Int(), tl.Unicode()], allow_none=True, default_value=None)

    def _first_init(self, **kwargs):
        if 'dims' in kwargs and isinstance(kwargs['dims'], string_types):
            kwargs['dims'] = [kwargs['dims']]
        return super(Reduce, self)._first_init(**kwargs)

    @property
    def native_coordinates(self):
        """Pass through for source.native_coordinates or self.native_coordinates_source.native_coordinates (preferred)        """
        return self.source.native_coordinates

    def get_dims(self, out):
        """
        Translates requested reduction dimensions.
        
        Parameters
        ----------
        out : UnitsDataArray
            The output array
        
        Returns
        -------
        list
            List of dimensions after reduction
        """
        # Using self.output.dims does a lot of work for us comparing
        # native_coordinates to evaluated coordinates
        input_dims = list(out.dims)
        
        if not self.dims:
            return input_dims

        return [dim for dim in self.dims if dim in input_dims]
    
    def dims_axes(self, output):
        """Finds the indices for the dimensions that will be reduced. This is passed to numpy. 
        
        Parameters
        ----------
        output : UnitsDataArray
            The output array with the reduced dimensions
        
        Returns
        -------
        list
            List of integers for the dimensions that will be reduces
        """
        axes = [i for i in range(len(output.dims)) if output.dims[i] in self.dims]
        return axes

    @property
    def chunk_size(self):
        """Size of chunks for parallel processing or large arrays that do not fit in memory
        
        Returns
        -------
        int
            Size of chunks
        """

        if self.iter_chunk_size == 'auto':
            return 1024**2 # TODO

        return self.iter_chunk_size

    @property
    def chunk_shape(self):
        """Shape of chunks for parallel processing or large arrays that do not fit in memory.
        
        Returns
        -------
        list
            List of integers giving the shape of each chunk.
        """
        if self.chunk_size is None:
            # return self.input_coordinates.shape
            return None

        chunk_size = self.chunk_size
        coords = self.input_coordinates
        
        d = {k:coords[k].size for k in coords.dims if k not in self.dims}
        s = reduce(mul, d.values(), 1)
        for dim in self.dims:
            n = chunk_size // s
            if n == 0:
                d[dim] = 1
            elif n < coords[dim].size:
                d[dim] = n
            else:
                d[dim] = coords[dim].size
            s *= d[dim]

        return [d[dim] for dim in coords.dims]

    def _reshape(self, x):
        """
        Transpose and reshape a DataArray to put the reduce dimensions together
        as axis 0. This is useful for example for scipy.stats.skew and kurtosis
        which only calculate over a single axis, by default 0.
        
        Parameters
        ----------
        x : xr.DataArray
            Input DataArray
        
        Returns
        -------
        a : np.array
            Transposed and reshaped array
        """

        if self.dims is None:
            return x.data.flatten()

        n = len(self.dims)
        dims = list(self.dims) + [d for d in x.dims if d not in self.dims]
        x = x.transpose(*dims)
        a = x.data.reshape(-1, *x.shape[n:])
        return a

    def iteroutputs(self, method=None):
        """Generator for the chunks of the output
        
        Yields
        ------
        UnitsDataArray
            Output for this chunk
        """
        for chunk in self.input_coordinates.iterchunks(self.chunk_shape):
            yield self.source.execute(chunk, method=method)

    @common_doc(COMMON_DOC)
    def execute(self, coordinates, output=None, method=None):
        """Executes this nodes using the supplied coordinates. 
        
        Parameters
        ----------
        coordinates : podpac.Coordinates
            {requested_coordinates}
        output : podpac.UnitsDataArray, optional
            {execute_out}
        method : str, optional
            {execute_method}
        
        Returns
        -------
        {execute_return}
        """
        self.input_coordinates = coordinates
        self.output = output
        
        self.requested_coordinates = coordinates
        test_out = self.get_output_coords(coords=coordinates)
        self.dims = self.get_dims(test_out)
 
        self.requested_coordinates = self.requested_coordinates.drop(self.dims)
        if self.output is None:
            self.output = self.initialize_coord_array(self.requested_coordinates)

        if self.chunk_size and self.chunk_size < reduce(mul, coordinates.shape, 1):
            result = self.reduce_chunked(self.iteroutputs(method), method)
        else:
            if self.implicit_pipeline_evaluation:
                self.source.execute(coordinates, method=method)
            result = self.reduce(self.source.output)

        if self.output.shape is (): # or self.requested_coordinates is None
            self.output.data = result
        else:
            self.output[:] = result #.transpose(*self.output.dims) # is this necessary?
        
        self.evaluated = True

        return self.output

    def reduce(self, x):
        """
        Reduce a full array, e.g. x.mean(self.dims).
        
        Must be defined in each child.
        
        Parameters
        ----------
        x : UnitsDataArray
            Array that needs to be reduced.
        
        Raises
        ------
        NotImplementedError
            Must be defined in each child.
        """

        raise NotImplementedError

    def reduce_chunked(self, xs, method=None):
        """
        Reduce a list of xs with a memory-effecient iterative algorithm.
        
        Optionally defined in each child.
        
        Parameters
        ----------
        xs : list, generator
            List of UnitsDataArray's that need to be reduced together.
        
        Returns
        -------
        UnitsDataArray
            Reduced output.
        """

        warnings.warn("No reduce_chunked method defined, using one-step reduce")
        x = self.source.execute(self.input_coordinates, method=method)
        return self.reduce(x)

    @property
    def evaluated_hash(self):
        """Unique hash used for caching
        
        Returns
        -------
        str
            Hash string used for caching
        
        Raises
        ------
        Exception
            If node has not been evaluated, no hash can be determined
        """
        if self.input_coordinates is None:
            raise Exception("node not evaluated")
            
        return self.get_hash(self.input_coordinates)

    @property
    def latlon_bounds_str(self):
        """String for lat-lon bounds used in hash. 
        
        Returns
        -------
        str
            String containg lat/lon bounds
        """
        return self.input_coordinates.latlon_bounds_str

class Min(Reduce):
    """Computes the minimum across dimension(s)
    """
    
    def reduce(self, x):
        """Computes the minimum across dimension(s)
        
        Parameters
        ----------
        x : UnitsDataArray
            Source data.
        
        Returns
        -------
        UnitsDataArray
            Minimum of the source data over self.dims
        """
        return x.min(dim=self.dims)
    
    def reduce_chunked(self, xs, method=None):
        """Computes the minimum across a chunk
        
        Parameters
        ----------
        xs : iterable
            Iterable of sources
        
        Returns
        -------
        UnitsDataArray
            Minimum of the source data over self.dims
        """
        # note: np.fmin ignores NaNs, np.minimum propagates NaNs
        y = xr.full_like(self.output, np.nan)
        for x in xs:
            y = np.fmin(y, x.min(dim=self.dims))
        return y


class Max(Reduce):
    """Computes the maximum across dimension(s)
    """
    
    def reduce(self, x):
        """Computes the maximum across dimension(s)
        
        Parameters
        ----------
        x : UnitsDataArray
            Source data.
        
        Returns
        -------
        UnitsDataArray
            Maximum of the source data over self.dims
        """
        return x.max(dim=self.dims)

    def reduce_chunked(self, xs, method=None):
        """Computes the maximum across a chunk
        
        Parameters
        ----------
        xs : iterable
            Iterable of sources
        
        Returns
        -------
        UnitsDataArray
            Maximum of the source data over self.dims
        """
        # note: np.fmax ignores NaNs, np.maximum propagates NaNs
        y = xr.full_like(self.output, np.nan)
        for x in xs:
            y = np.fmax(y, x.max(dim=self.dims))
        return y


class Sum(Reduce):
    """Computes the sum across dimension(s)
    """
    
    def reduce(self, x):
        """Computes the sum across dimension(s)
        
        Parameters
        ----------
        x : UnitsDataArray
            Source data.
        
        Returns
        -------
        UnitsDataArray
            Sum of the source data over self.dims
        """
        return x.sum(dim=self.dims)

    def reduce_chunked(self, xs, method=None):
        """Computes the sum across a chunk
        
        Parameters
        ----------
        xs : iterable
            Iterable of sources
        
        Returns
        -------
        UnitsDataArray
            Sum of the source data over self.dims
        """
        s = xr.zeros_like(self.output)
        for x in xs:
            s += x.sum(dim=self.dims)
        return s


class Count(Reduce):
    """Counts the finite values across dimension(s)
    """
    
    def reduce(self, x):
        """Counts the finite values across dimension(s)
        
        Parameters
        ----------
        x : UnitsDataArray
            Source data.
        
        Returns
        -------
        UnitsDataArray
            Number of finite values of the source data over self.dims
        """
        return np.isfinite(x).sum(dim=self.dims)

    def reduce_chunked(self, xs, method=None):
        """Counts the finite values across a chunk
        
        Parameters
        ----------
        xs : iterable
            Iterable of sources
        
        Returns
        -------
        UnitsDataArray
            Number of finite values of the source data over self.dims
        """
        n = xr.zeros_like(self.output)
        for x in xs:
            n += np.isfinite(x).sum(dim=self.dims)
        return n


class Mean(Reduce):
    """Computes the mean across dimension(s)
    """
    
    def reduce(self, x):
        """Computes the mean across dimension(s)
        
        Parameters
        ----------
        x : UnitsDataArray
            Source data.
        
        Returns
        -------
        UnitsDataArray
            Mean of the source data over self.dims
        """
        return x.mean(dim=self.dims)

    def reduce_chunked(self, xs, method=None):
        """Computes the mean across a chunk
        
        Parameters
        ----------
        xs : iterable
            Iterable of sources
        
        Returns
        -------
        UnitsDataArray
            Mean of the source data over self.dims
        """
        s = xr.zeros_like(self.output) # alt: s = np.zeros(self.shape)
        n = xr.zeros_like(self.output)
        for x in xs:
            # TODO efficency
            s += x.sum(dim=self.dims)
            n += np.isfinite(x).sum(dim=self.dims)
        output = s / n
        return output


class Variance(Reduce):
    """Computes the variance across dimension(s)
    """
    
    def reduce(self, x):
        """Computes the variance across dimension(s)
        
        Parameters
        ----------
        x : UnitsDataArray
            Source data.
        
        Returns
        -------
        UnitsDataArray
            Variance of the source data over self.dims
        """
        return x.var(dim=self.dims)

    def reduce_chunked(self, xs, method=None):
        """Computes the variance across a chunk
        
        Parameters
        ----------
        xs : iterable
            Iterable of sources
        
        Returns
        -------
        UnitsDataArray
            Variance of the source data over self.dims
        """
        n = xr.zeros_like(self.output)
        m = xr.zeros_like(self.output)
        m2 = xr.zeros_like(self.output)

        # Welford, adapted to handle multiple data points in each iteration
        for x in xs:
            n += np.isfinite(x).sum(dim=self.dims)
            d = x - m
            m += (d/n).sum(dim=self.dims)
            d2 = x - m
            m2 += (d*d2).sum(dim=self.dims)

        return m2 / n


class Skew(Reduce):
    """
    Computes the skew across dimension(s)

    TODO NaN behavior when there is NO data (currently different in reduce and reduce_chunked)
    """

    def reduce(self, x):
        """Computes the skew across dimension(s)
        
        Parameters
        ----------
        x : UnitsDataArray
            Source data.
        
        Returns
        -------
        UnitsDataArray
            Skew of the source data over self.dims
        """
        # N = np.isfinite(x).sum(dim=self.dims)
        # M1 = x.mean(dim=self.dims)
        # E = x - M1
        # E2 = E**2
        # E3 = E2*E
        # M2 = (E2).sum(dim=self.dims)
        # M3 = (E3).sum(dim=self.dims)
        # skew = self.skew(M3, M2, N)

        a = self._reshape(x)
        skew = scipy.stats.skew(a, nan_policy='omit')
        return skew
        
    def reduce_chunked(self, xs, method=None):
        """Computes the skew across a chunk
        
        Parameters
        ----------
        xs : iterable
            Iterable of sources
        
        Returns
        -------
        UnitsDataArray
            Skew of the source data over self.dims
        """
        N = xr.zeros_like(self.output)
        M1 = xr.zeros_like(self.output)
        M2 = xr.zeros_like(self.output)
        M3 = xr.zeros_like(self.output)
        check_empty = True

        for x in xs:
            Nx = np.isfinite(x).sum(dim=self.dims)
            M1x = x.mean(dim=self.dims)
            Ex = x - M1x
            Ex2 = Ex**2
            Ex3 = Ex2*Ex
            M2x = (Ex2).sum(dim=self.dims)
            M3x = (Ex3).sum(dim=self.dims)

            # premask to omit NaNs
            b = Nx.data > 0
            Nx = Nx.data[b]
            M1x = M1x.data[b]
            M2x = M2x.data[b]
            M3x = M3x.data[b]
            
            Nb = N.data[b]
            M1b = M1.data[b]
            M2b = M2.data[b]

            # merge
            d = M1x - M1b
            n = Nb + Nx
            NNx = Nb * Nx

            M3.data[b] += (M3x +
                           d**3 * NNx * (Nb-Nx) / n**2 +
                           3 * d * (Nb*M2x - Nx*M2b) / n)
            M2.data[b] += M2x + d**2 * NNx / n
            M1.data[b] += d * Nx / n
            N.data[b] = n

        # calculate skew
        skew = np.sqrt(N) * M3 / np.sqrt(M2**3)
        return skew

class Kurtosis(Reduce):
    """Computes the kurtosis across dimension(s)
    TODO NaN behavior when there is NO data (currently different in reduce and reduce_chunked)
    """

    def reduce(self, x):
        """Computes the kurtosis across dimension(s)
        
        Parameters
        ----------
        x : UnitsDataArray
            Source data.
        
        Returns
        -------
        UnitsDataArray
            Kurtosis of the source data over self.dims
        """
        # N = np.isfinite(x).sum(dim=self.dims)
        # M1 = x.mean(dim=self.dims)        
        # E = x - M1
        # E2 = E**2
        # E4 = E2**2
        # M2 = (E2).sum(dim=self.dims)
        # M4 = (E4).sum(dim=self.dims)
        # kurtosis = N * M4 / M2**2 - 3

        a = self._reshape(x)
        kurtosis = scipy.stats.kurtosis(a, nan_policy='omit')
        return kurtosis

    def reduce_chunked(self, xs, method=None):
        """Computes the kurtosis across a chunk
        
        Parameters
        ----------
        xs : iterable
            Iterable of sources
        
        Returns
        -------
        UnitsDataArray
            Kurtosis of the source data over self.dims
        """
        N = xr.zeros_like(self.output)
        M1 = xr.zeros_like(self.output)
        M2 = xr.zeros_like(self.output)
        M3 = xr.zeros_like(self.output)
        M4 = xr.zeros_like(self.output)

        for x in xs:
            Nx = np.isfinite(x).sum(dim=self.dims)
            M1x = x.mean(dim=self.dims)
            Ex = x - M1x
            Ex2 = Ex**2
            Ex3 = Ex2*Ex
            Ex4 = Ex2**2
            M2x = (Ex2).sum(dim=self.dims)
            M3x = (Ex3).sum(dim=self.dims)
            M4x = (Ex4).sum(dim=self.dims)
            
            # premask to omit NaNs
            b = Nx.data > 0
            Nx = Nx.data[b]
            M1x = M1x.data[b]
            M2x = M2x.data[b]
            M3x = M3x.data[b]
            M4x = M4x.data[b]
            
            Nb = N.data[b]
            M1b = M1.data[b]
            M2b = M2.data[b]
            M3b = M3.data[b]

            # merge
            d = M1x - M1b
            n = Nb + Nx
            NNx = Nb * Nx

            M4.data[b] += (M4x +
                           d**4 * NNx * (Nb**2 - NNx + Nx**2) / n**3 +
                           6 * d**2 * (Nb**2*M2x + Nx**2*M2b) / n**2 +
                           4 * d * (Nb*M3x - Nx*M3b) / n)

            M3.data[b] += (M3x +
                           d**3 * NNx * (Nb-Nx) / n**2 +
                           3 * d * (Nb*M2x - Nx*M2b) / n)
            M2.data[b] += M2x + d**2 * NNx / n
            M1.data[b] += d * Nx / n
            N.data[b] = n

        # calculate kurtosis
        kurtosis = N * M4 / M2**2 - 3
        return kurtosis


class StandardDeviation(Variance):
    """Computes the standard deviation across dimension(s)
    """
    
    def reduce(self, x):
        """Computes the standard deviation across dimension(s)
        
        Parameters
        ----------
        x : UnitsDataArray
            Source data.
        
        Returns
        -------
        UnitsDataArray
            Standard deviation of the source data over self.dims
        """
        return x.std(dim=self.dims)

    def reduce_chunked(self, xs, method=None):
        """Computes the standard deviation across a chunk
        
        Parameters
        ----------
        xs : iterable
            Iterable of sources
        
        Returns
        -------
        UnitsDataArray
            Standard deviation of the source data over self.dims
        """
        var = super(StandardDeviation, self).reduce_chunked(xs, method)
        return np.sqrt(var)

# =============================================================================
# Orthogonally chunked reduce
# =============================================================================

class Reduce2(Reduce):
    """
    Extended Reduce class that enables chunks that are smaller than the reduced
    output array.
    
    The base Reduce node ensures that each chunk is at least as big as the
    reduced output, which works for statistics that can be calculated in O(1)
    space. For statistics that require O(n) space, the node must iterate
    through the Coordinates orthogonally to the reduce dimension, using chunks
    that only cover a portion of the output array.
    
    Note that the above nodes *could* be implemented to allow small chunks.
    """

    @property
    def chunk_shape(self):
        """Shape of chunks for parallel processing or large arrays that do not fit in memory.
        
        Returns
        -------
        list
            List of integers giving the shape of each chunk.
        """
        if self.chunk_size is None:
            # return self.input_coordinates.shape
            return None

        chunk_size = self.chunk_size
        coords = self.input_coordinates
        
        # here, the minimum size is the reduce-dimensions size
        d = {k:coords[k].size for k in self.dims}
        s = reduce(mul, d.values(), 1)
        for dim in coords.dims[::-1]:
            if dim in self.dims:
                continue
            n = chunk_size // s
            if n == 0:
                d[dim] = 1
            elif n < coords[dim].size:
                d[dim] = n
            else:
                d[dim] = coords[dim].size
            s *= d[dim]

        return [d[dim] for dim in coords.dims]

    def iteroutputs(self, method):
        """Generator for the chunks of the output
        
        Yields
        ------
        UnitsDataArray
            Output for this chunk
        """
        for chunk, slices in self.input_coordinates.iterchunks(self.chunk_shape, return_slices=True):
            yield self.source.execute(chunk, method=method), slices

    def reduce_chunked(self, xs, method=None):
        """
        Reduce a list of xs with a memory-effecient iterative algorithm.
        
        Optionally defined in each child.
        
        Parameters
        ----------
        xs : list, generator
            List of UnitsDataArray's that need to be reduced together.
        
        Returns
        -------
        UnitsDataArray
            Reduced output.
        """
        # special case for full reduce
        if not self.requested_coordinates.dims:
            x, xslices = next(xs)
            return self.reduce(x)

        I = [self.input_coordinates.dims.index(dim) for dim in self.requested_coordinates.dims]
        y = xr.full_like(self.output, np.nan)
        for x, xslices in xs:
            yslc = [xslices[i] for i in I]
            y.data[yslc] = self.reduce(x)
        return y


class Median(Reduce2):
    """Computes the median across dimension(s)
    """
    
    def reduce(self, x):
        """Computes the median across dimension(s)
        
        Parameters
        ----------
        x : UnitsDataArray
            Source data.
        
        Returns
        -------
        UnitsDataArray
            Median of the source data over self.dims
        """
        return x.median(dim=self.dims)


class Percentile(Reduce2):
    """Computes the percentile across dimension(s)
    
    Attributes
    ----------
    percentile : TYPE
        Description
    """
    
    percentile = tl.Float(default=50.0).tag(attr=True)

    def reduce(self, x):
        """Computes the percentile across dimension(s)
        
        Parameters
        ----------
        x : UnitsDataArray
            Source data.
        
        Returns
        -------
        UnitsDataArray
            Percentile of the source data over self.dims
        """

        return np.nanpercentile(x, self.percentile, self.dims_axes(x))

# =============================================================================
# Time-Grouped Reduce
# =============================================================================

# TODO native_coordinates_source as an attribute

class GroupReduce(Algorithm):
    """
    Group a time-dependent source node by a datetime accessor and reduce.
    
    Attributes
    ----------
    custom_reduce_fn : function
        required if reduce_fn is 'custom'.
    groupby : str
        datetime sub-accessor. Currently 'dayofyear' is the enabled option.
    native_coordinates_source : podpac.Node
        Node that acts as the source for the native_coordinates of this node. 
    reduce_fn : str
        builtin xarray groupby reduce function, or 'custom'.
    source : podpac.Node
        Source node, must have native_coordinates
    """

    source = tl.Instance(Node)
    native_coordinates_source = tl.Instance(Node, allow_none=True)

    # see https://github.com/pydata/xarray/blob/eeb109d9181c84dfb93356c5f14045d839ee64cb/xarray/core/accessors.py#L61
    groupby = tl.CaselessStrEnum(['dayofyear']) # could add season, month, etc

    reduce_fn = tl.CaselessStrEnum(['all', 'any', 'count', 'max', 'mean',
                                    'median', 'min', 'prod', 'std',
                                    'sum', 'var', 'custom'])
    custom_reduce_fn = tl.Any()

    @property
    def native_coordinates(self):
        """Pass through for source.native_coordinates or self.native_coordinates_source.native_coordinates (preferred)
        """
        try:
            if self.native_coordinates_source:
                return self.native_coordinates_source.native_coordinates
            else:
                return self.source.native_coordinates
        except:
            raise Exception("no native coordinates found")

    @property
    def source_coordinates(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        # intersect grouped time coordinates using groupby DatetimeAccessor
        native_time = xr.DataArray(self.native_coordinates.coords['time'])
        eval_time = xr.DataArray(self.requested_coordinates.coords['time'])
        N = getattr(native_time.dt, self.groupby)
        E = getattr(eval_time.dt, self.groupby)
        native_time_mask = np.in1d(N, E)

        # use requested spatial coordinates and filtered native times
        coords = Coordinates(
            time=native_time.data[native_time_mask],
            lat=self.requested_coordinates['lat'],
            lon=self.requested_coordinates['lon'],
            order=('time', 'lat', 'lon'))

        return coords

    @common_doc(COMMON_DOC)
    def execute(self, coordinates, output=None, method=None):
        """Executes this nodes using the supplied coordinates. 
        
        Parameters
        ----------
        coordinates : podpac.Coordinates
            {requested_coordinates}
        output : podpac.UnitsDataArray, optional
            {execute_out}
        method : str, optional
            {execute_method}
        
        Returns
        -------
        {execute_return}
        
        Raises
        ------
        ValueError
            If source it not time-depended (required by this node).
        """
        self.requested_coordinates = coordinates
        self.output = output

        if self.output is None:
            self.output = self.initialize_output_array()

        if 'time' not in self.native_coordinates.dims:
            raise ValueError("GroupReduce source node must be time-dependent")
        
        if self.implicit_pipeline_evaluation:
            self.source.execute(self.source_coordinates, method=method)

        # group
        grouped = self.source.output.groupby('time.%s' % self.groupby)
        
        # reduce
        if self.reduce_fn is 'custom':
            out = grouped.apply(self.custom_reduce_fn, 'time')
        else:
            # standard, e.g. grouped.median('time')
            out = getattr(grouped, self.reduce_fn)('time')

        # map
        eval_time = xr.DataArray(self.requested_coordinates.coords['time'])
        E = getattr(eval_time.dt, self.groupby)
        out = out.sel(**{self.groupby:E}).rename({self.groupby: 'time'})
        self.output[:] = out.transpose(*self.output.dims).data

        self.evaluated = True
        return self.output

    def base_ref(self):
        """
        Default pipeline node reference/name in pipeline node definitions
        
        Returns
        -------
        str
            Default pipeline node reference/name in pipeline node definitions
        """
        return '%s.%s.%s' % (self.source.base_ref,self.groupby,self.reduce_fn)

class DayOfYear(GroupReduce):
    """
    Group a time-dependent source node by day of year and reduce. Convenience node for GroupReduce.
    
    Attributes
    ----------
    custom_reduce_fn : function
        required if reduce_fn is 'custom'.
    native_coordinates_source : podpac.Node
        Node that acts as the source for the native_coordinates of this node. 
    reduce_fn : str
        builtin xarray groupby reduce function, or 'custom'.
    source : podpac.Node
        Source node, must have native_coordinates
    """

    groupby = 'dayofyear'
