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

import podpac
from podpac.core.coordinates import Coordinates
from podpac.core.node import Node
from podpac.core.algorithm.algorithm import UnaryAlgorithm
from podpac.core.utils import common_doc, NodeTrait
from podpac.core.node import COMMON_NODE_DOC, node_eval

COMMON_DOC = COMMON_NODE_DOC.copy()


class Reduce(UnaryAlgorithm):
    """Base node for statistical algorithms
    
    Attributes
    ----------
    dims : list
        List of strings that give the dimensions which should be reduced
    source : podpac.Node
        The source node that will be reduced. 
    """

    dims = tl.List().tag(attr=True)

    _reduced_coordinates = tl.Instance(Coordinates, allow_none=True)
    _dims = tl.List(trait_type=str)

    def _first_init(self, **kwargs):
        if "dims" in kwargs and isinstance(kwargs["dims"], string_types):
            kwargs["dims"] = [kwargs["dims"]]
        return super(Reduce, self)._first_init(**kwargs)

    def _get_dims(self, out):
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
        axes = [i for i in range(len(output.dims)) if output.dims[i] in self._dims]
        return axes

    @property
    def chunk_size(self):
        """Size of chunks for parallel processing or large arrays that do not fit in memory
        
        Returns
        -------
        int
            Size of chunks
        """

        chunk_size = podpac.settings["CHUNK_SIZE"]
        if chunk_size == "auto":
            return 1024 ** 2  # TODO
        else:
            return chunk_size

    def _get_chunk_shape(self, coords):
        """Shape of chunks for parallel processing or large arrays that do not fit in memory.
        
        Returns
        -------
        list
            List of integers giving the shape of each chunk.
        """
        if self.chunk_size is None:
            return None

        chunk_size = self.chunk_size

        d = {k: coords[k].size for k in coords.dims if k not in self._dims}
        s = reduce(mul, d.values(), 1)
        for dim in self._dims:
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

        if self._dims is None:
            return x.data.flatten()

        n = len(self._dims)
        dims = list(self._dims) + [d for d in x.dims if d not in self._dims]
        x = x.transpose(*dims)
        a = x.data.reshape(-1, *x.shape[n:])
        return a

    def iteroutputs(self, coordinates):
        """Generator for the chunks of the output
        
        Yields
        ------
        UnitsDataArray
            Output for this chunk
        """
        chunk_shape = self._get_chunk_shape(coordinates)
        for chunk in coordinates.iterchunks(chunk_shape):
            yield self.source.eval(chunk)

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

        self._requested_coordinates = coordinates

        if self.dims:
            self._dims = [dim for dim in self.dims if dim in coordinates.dims]
        else:
            self._dims = list(coordinates.dims)
        self._reduced_coordinates = coordinates.drop(self._dims)

        if output is None:
            output = self.create_output_array(self._reduced_coordinates)

        if self.chunk_size and self.chunk_size < reduce(mul, coordinates.shape, 1):
            try:
                result = self.reduce_chunked(self.iteroutputs(coordinates), output)
            except NotImplementedError:
                warnings.warn("No reduce_chunked method defined, using one-step reduce")
                source_output = self.source.eval(coordinates)
                result = self.reduce(source_output)
        else:
            source_output = self.source.eval(coordinates)
            result = self.reduce(source_output)

        if output.shape == ():
            output.data = result
        else:
            output[:] = result

        return output

    def reduce(self, x):
        """
        Reduce a full array, e.g. x.mean(dims).
        
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

    def reduce_chunked(self, xs, output):
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

        raise NotImplementedError


class ReduceOrthogonal(Reduce):
    """
    Extended Reduce class that enables chunks that are smaller than the reduced
    output array.
    
    The base Reduce node ensures that each chunk is at least as big as the
    reduced output, which works for statistics that can be calculated in O(1)
    space. For statistics that require O(n) space, the node must iterate
    through the Coordinates orthogonally to the reduce dimension, using chunks
    that only cover a portion of the output array.
    """

    def _get_chunk_shape(self, coords):
        """Shape of chunks for parallel processing or large arrays that do not fit in memory.
        
        Returns
        -------
        list
            List of integers giving the shape of each chunk.
        """
        if self.chunk_size is None:
            return None

        chunk_size = self.chunk_size

        # here, the minimum size is the reduce-dimensions size
        d = {k: coords[k].size for k in self._dims}
        s = reduce(mul, d.values(), 1)
        for dim in coords.dims[::-1]:
            if dim in self._dims:
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

    def iteroutputs(self, coordinates):
        """Generator for the chunks of the output
        
        Yields
        ------
        UnitsDataArray
            Output for this chunk
        """

        chunk_shape = self._get_chunk_shape(coordinates)
        for chunk, slices in coordinates.iterchunks(chunk_shape, return_slices=True):
            yield self.source.eval(chunk), slices

    def reduce_chunked(self, xs, output):
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
        if not self._reduced_coordinates.dims:
            x, xslices = next(xs)
            return self.reduce(x)

        y = xr.full_like(output, np.nan)
        for x, xslices in xs:
            yslc = tuple(xslices[x.dims.index(dim)] for dim in self._reduced_coordinates.dims)
            y.data[yslc] = self.reduce(x)
        return y


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
            Minimum of the source data over dims
        """
        return x.min(dim=self._dims)

    def reduce_chunked(self, xs, output):
        """Computes the minimum across a chunk
        
        Parameters
        ----------
        xs : iterable
            Iterable of sources
        
        Returns
        -------
        UnitsDataArray
            Minimum of the source data over dims
        """
        # note: np.fmin ignores NaNs, np.minimum propagates NaNs
        y = xr.full_like(output, np.nan)
        for x in xs:
            y = np.fmin(y, x.min(dim=self._dims))
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
            Maximum of the source data over dims
        """
        return x.max(dim=self._dims)

    def reduce_chunked(self, xs, output):
        """Computes the maximum across a chunk
        
        Parameters
        ----------
        xs : iterable
            Iterable of sources
        
        Returns
        -------
        UnitsDataArray
            Maximum of the source data over dims
        """
        # note: np.fmax ignores NaNs, np.maximum propagates NaNs
        y = xr.full_like(output, np.nan)
        for x in xs:
            y = np.fmax(y, x.max(dim=self._dims))
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
            Sum of the source data over dims
        """
        return x.sum(dim=self._dims)

    def reduce_chunked(self, xs, output):
        """Computes the sum across a chunk
        
        Parameters
        ----------
        xs : iterable
            Iterable of sources
        
        Returns
        -------
        UnitsDataArray
            Sum of the source data over dims
        """
        s = xr.zeros_like(output)
        for x in xs:
            s += x.sum(dim=self._dims)
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
            Number of finite values of the source data over dims
        """
        return np.isfinite(x).sum(dim=self._dims)

    def reduce_chunked(self, xs, output):
        """Counts the finite values across a chunk
        
        Parameters
        ----------
        xs : iterable
            Iterable of sources
        
        Returns
        -------
        UnitsDataArray
            Number of finite values of the source data over dims
        """
        n = xr.zeros_like(output)
        for x in xs:
            n += np.isfinite(x).sum(dim=self._dims)
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
            Mean of the source data over dims
        """
        return x.mean(dim=self._dims)

    def reduce_chunked(self, xs, output):
        """Computes the mean across a chunk
        
        Parameters
        ----------
        xs : iterable
            Iterable of sources
        
        Returns
        -------
        UnitsDataArray
            Mean of the source data over dims
        """
        s = xr.zeros_like(output)
        n = xr.zeros_like(output)
        for x in xs:
            # TODO efficency
            s += x.sum(dim=self._dims)
            n += np.isfinite(x).sum(dim=self._dims)
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
            Variance of the source data over dims
        """
        return x.var(dim=self._dims)

    def reduce_chunked(self, xs, output):
        """Computes the variance across a chunk
        
        Parameters
        ----------
        xs : iterable
            Iterable of sources
        
        Returns
        -------
        UnitsDataArray
            Variance of the source data over dims
        """
        n = xr.zeros_like(output)
        m = xr.zeros_like(output)
        m2 = xr.zeros_like(output)

        # Welford, adapted to handle multiple data points in each iteration
        for x in xs:
            n += np.isfinite(x).sum(dim=self._dims)
            d = x - m
            m += (d / n).sum(dim=self._dims)
            d2 = x - m
            m2 += (d * d2).sum(dim=self._dims)

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
            Skew of the source data over dims
        """
        # N = np.isfinite(x).sum(dim=self._dims)
        # M1 = x.mean(dim=self._dims)
        # E = x - M1
        # E2 = E**2
        # E3 = E2*E
        # M2 = (E2).sum(dim=self._dims)
        # M3 = (E3).sum(dim=self._dims)
        # skew = self.skew(M3, M2, N)

        a = self._reshape(x)
        skew = scipy.stats.skew(a, nan_policy="omit")
        return skew

    def reduce_chunked(self, xs, output):
        """Computes the skew across a chunk
        
        Parameters
        ----------
        xs : iterable
            Iterable of sources
        
        Returns
        -------
        UnitsDataArray
            Skew of the source data over dims
        """
        N = xr.zeros_like(output)
        M1 = xr.zeros_like(output)
        M2 = xr.zeros_like(output)
        M3 = xr.zeros_like(output)
        check_empty = True

        for x in xs:
            Nx = np.isfinite(x).sum(dim=self._dims)
            M1x = x.mean(dim=self._dims)
            Ex = x - M1x
            Ex2 = Ex ** 2
            Ex3 = Ex2 * Ex
            M2x = (Ex2).sum(dim=self._dims)
            M3x = (Ex3).sum(dim=self._dims)

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

            M3.data[b] += M3x + d ** 3 * NNx * (Nb - Nx) / n ** 2 + 3 * d * (Nb * M2x - Nx * M2b) / n
            M2.data[b] += M2x + d ** 2 * NNx / n
            M1.data[b] += d * Nx / n
            N.data[b] = n

        # calculate skew
        skew = np.sqrt(N) * M3 / np.sqrt(M2 ** 3)
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
            Kurtosis of the source data over dims
        """
        # N = np.isfinite(x).sum(dim=self._dims)
        # M1 = x.mean(dim=self._dims)
        # E = x - M1
        # E2 = E**2
        # E4 = E2**2
        # M2 = (E2).sum(dim=self._dims)
        # M4 = (E4).sum(dim=self._dims)
        # kurtosis = N * M4 / M2**2 - 3

        a = self._reshape(x)
        kurtosis = scipy.stats.kurtosis(a, nan_policy="omit")
        return kurtosis

    def reduce_chunked(self, xs, output):
        """Computes the kurtosis across a chunk
        
        Parameters
        ----------
        xs : iterable
            Iterable of sources
        
        Returns
        -------
        UnitsDataArray
            Kurtosis of the source data over dims
        """
        N = xr.zeros_like(output)
        M1 = xr.zeros_like(output)
        M2 = xr.zeros_like(output)
        M3 = xr.zeros_like(output)
        M4 = xr.zeros_like(output)

        for x in xs:
            Nx = np.isfinite(x).sum(dim=self._dims)
            M1x = x.mean(dim=self._dims)
            Ex = x - M1x
            Ex2 = Ex ** 2
            Ex3 = Ex2 * Ex
            Ex4 = Ex2 ** 2
            M2x = (Ex2).sum(dim=self._dims)
            M3x = (Ex3).sum(dim=self._dims)
            M4x = (Ex4).sum(dim=self._dims)

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

            M4.data[b] += (
                M4x
                + d ** 4 * NNx * (Nb ** 2 - NNx + Nx ** 2) / n ** 3
                + 6 * d ** 2 * (Nb ** 2 * M2x + Nx ** 2 * M2b) / n ** 2
                + 4 * d * (Nb * M3x - Nx * M3b) / n
            )

            M3.data[b] += M3x + d ** 3 * NNx * (Nb - Nx) / n ** 2 + 3 * d * (Nb * M2x - Nx * M2b) / n
            M2.data[b] += M2x + d ** 2 * NNx / n
            M1.data[b] += d * Nx / n
            N.data[b] = n

        # calculate kurtosis
        kurtosis = N * M4 / M2 ** 2 - 3
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
            Standard deviation of the source data over dims
        """
        return x.std(dim=self._dims)

    def reduce_chunked(self, xs, output):
        """Computes the standard deviation across a chunk
        
        Parameters
        ----------
        xs : iterable
            Iterable of sources
        
        Returns
        -------
        UnitsDataArray
            Standard deviation of the source data over dims
        """
        var = super(StandardDeviation, self).reduce_chunked(xs, output)
        return np.sqrt(var)


class Median(ReduceOrthogonal):
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
            Median of the source data over dims
        """
        return x.median(dim=self._dims)


class Percentile(ReduceOrthogonal):
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
            Percentile of the source data over dims
        """

        return np.nanpercentile(x, self.percentile, self.dims_axes(x))


# =============================================================================
# Time-Grouped Reduce
# =============================================================================


class GroupReduce(UnaryAlgorithm):
    """
    Group a time-dependent source node and then compute a statistic for each result.
    
    Attributes
    ----------
    custom_reduce_fn : function
        required if reduce_fn is 'custom'.
    groupby : str
        datetime sub-accessor. Currently 'dayofyear' is the enabled option.
    reduce_fn : str
        builtin xarray groupby reduce function, or 'custom'.
    source : podpac.Node
        Source node
    """

    coordinates_source = NodeTrait(allow_none=True)

    # see https://github.com/pydata/xarray/blob/eeb109d9181c84dfb93356c5f14045d839ee64cb/xarray/core/accessors.py#L61
    groupby = tl.CaselessStrEnum(["dayofyear"])  # could add season, month, etc

    reduce_fn = tl.CaselessStrEnum(
        ["all", "any", "count", "max", "mean", "median", "min", "prod", "std", "sum", "var", "custom"]
    )
    custom_reduce_fn = tl.Any()

    _source_coordinates = tl.Instance(Coordinates)

    @tl.default("coordinates_source")
    def _default_coordinates_source(self):
        return self.source

    def _get_source_coordinates(self, requested_coordinates):
        # get available time coordinates
        # TODO do these two checks during node initialization
        available_coordinates = self.coordinates_source.find_coordinates()
        if len(available_coordinates) != 1:
            raise ValueError("Cannot evaluate this node; too many available coordinates")
        avail_coords = available_coordinates[0]
        if "time" not in avail_coords.udims:
            raise ValueError("GroupReduce coordinates source node must be time-dependent")

        # intersect grouped time coordinates using groupby DatetimeAccessor
        avail_time = xr.DataArray(avail_coords.coords["time"])
        eval_time = xr.DataArray(requested_coordinates.coords["time"])
        N = getattr(avail_time.dt, self.groupby)
        E = getattr(eval_time.dt, self.groupby)
        native_time_mask = np.in1d(N, E)

        # use requested spatial coordinates and filtered available times
        coords = Coordinates(
            time=avail_time.data[native_time_mask],
            lat=requested_coordinates["lat"],
            lon=requested_coordinates["lon"],
            order=("time", "lat", "lon"),
        )

        return coords

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
        
        Raises
        ------
        ValueError
            If source it not time-depended (required by this node).
        """

        self._source_coordinates = self._get_source_coordinates(coordinates)

        if output is None:
            output = self.create_output_array(coordinates)

        source_output = self.source.eval(self._source_coordinates)

        # group
        grouped = source_output.groupby("time.%s" % self.groupby)

        # reduce
        if self.reduce_fn == "custom":
            out = grouped.apply(self.custom_reduce_fn, "time")
        else:
            # standard, e.g. grouped.median('time')
            out = getattr(grouped, self.reduce_fn)("time")

        # map
        eval_time = xr.DataArray(coordinates.coords["time"])
        E = getattr(eval_time.dt, self.groupby)
        out = out.sel(**{self.groupby: E}).rename({self.groupby: "time"})
        output[:] = out.transpose(*output.dims).data

        return output

    def base_ref(self):
        """
        Default node reference/name in node definitions
        
        Returns
        -------
        str
            Default node reference/name in node definitions
        """
        return "%s.%s.%s" % (self.source.base_ref, self.groupby, self.reduce_fn)


class DayOfYear(GroupReduce):
    """
    Group a time-dependent source node by day of year and compute a statistic for each group.
    
    Attributes
    ----------
    custom_reduce_fn : function
        required if reduce_fn is 'custom'.
    reduce_fn : str
        builtin xarray groupby reduce function, or 'custom'.
    source : podpac.Node
        Source node
    """

    groupby = "dayofyear"
