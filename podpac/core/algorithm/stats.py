"""
Stats Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import warnings
from operator import mul
from functools import reduce
import logging

import xarray as xr
import numpy as np
import scipy.stats
import traitlets as tl
from six import string_types

# Internal dependencies
import podpac
from podpac.core.coordinates import Coordinates
from podpac.core.node import Node
from podpac.core.algorithm.algorithm import UnaryAlgorithm, Algorithm
from podpac.core.utils import common_doc, NodeTrait
from podpac.core.node import COMMON_NODE_DOC

COMMON_DOC = COMMON_NODE_DOC.copy()

# Set up logging
_log = logging.getLogger(__name__)


class Reduce(UnaryAlgorithm):
    """Base node for statistical algorithms

    Attributes
    ----------
    dims : list
        List of strings that give the dimensions which should be reduced
    source : podpac.Node
        The source node that will be reduced.
    """

    from podpac.core.utils import DimsTrait

    dims = DimsTrait(allow_none=True, default_value=None).tag(attr=True)

    _reduced_coordinates = tl.Instance(Coordinates, allow_none=True)
    _dims = tl.List(trait=tl.Unicode())

    def _first_init(self, **kwargs):
        if "dims" in kwargs and isinstance(kwargs["dims"], string_types):
            kwargs["dims"] = [kwargs["dims"]]
        return super(Reduce, self)._first_init(**kwargs)

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

    def iteroutputs(self, coordinates, _selector):
        """Generator for the chunks of the output

        Yields
        ------
        UnitsDataArray
            Output for this chunk
        """
        chunk_shape = self._get_chunk_shape(coordinates)
        for chunk in coordinates.iterchunks(chunk_shape):
            yield self.source.eval(chunk, _selector=_selector)

    @common_doc(COMMON_DOC)
    def _eval(self, coordinates, output=None, _selector=None):
        """Evaluates this nodes using the supplied coordinates.

        Parameters
        ----------
        coordinates : podpac.Coordinates
            {requested_coordinates}
        output : podpac.UnitsDataArray, optional
            {eval_output}
        _selector: callable(coordinates, request_coordinates)
            {eval_selector}

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
                result = self.reduce_chunked(self.iteroutputs(coordinates, _selector), output)
            except NotImplementedError:
                warnings.warn("No reduce_chunked method defined, using one-step reduce")
                source_output = self.source.eval(coordinates, _selector=_selector)
                result = self.reduce(source_output)
        else:
            source_output = self.source.eval(coordinates, _selector=_selector)
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

    def iteroutputs(self, coordinates, selector):
        """Generator for the chunks of the output

        Yields
        ------
        UnitsDataArray
            Output for this chunk
        """

        chunk_shape = self._get_chunk_shape(coordinates)
        for chunk, slices in coordinates.iterchunks(chunk_shape, return_slices=True):
            yield self.source.eval(chunk, _selector=selector), slices

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
            yslc = tuple(xslices[self._requested_coordinates.dims.index(dim)] for dim in self._reduced_coordinates.dims)
            y.data[yslc] = self.reduce(x)
        return y


class Min(Reduce):
    """Computes the minimum across dimension(s)"""

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
    """Computes the maximum across dimension(s)"""

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
    """Computes the sum across dimension(s)"""

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
    """Counts the finite values across dimension(s)"""

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
    """Computes the mean across dimension(s)"""

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
    """Computes the variance across dimension(s)"""

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
    """Computes the standard deviation across dimension(s)"""

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

    Example
    ---------
    coords.dims == ['lat', 'lon', 'time']
    median = Median(source=node, dims=['lat', 'lon'])
    o = median.eval(coords)
    o.dims == ['time']
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

    percentile = tl.Float(default_value=50.0).tag(attr=True)

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

_REDUCE_FUNCTIONS = ["all", "any", "count", "max", "mean", "median", "min", "prod", "std", "sum", "var", "custom"]


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

    _repr_keys = ["source", "groupby", "reduce_fn"]
    coordinates_source = NodeTrait(allow_none=True).tag(attr=True)

    # see https://github.com/pydata/xarray/blob/eeb109d9181c84dfb93356c5f14045d839ee64cb/xarray/core/accessors.py#L61
    groupby = tl.CaselessStrEnum(["dayofyear", "weekofyear", "season", "month"], allow_none=True).tag(attr=True)
    reduce_fn = tl.CaselessStrEnum(_REDUCE_FUNCTIONS).tag(attr=True)
    custom_reduce_fn = tl.Any(allow_none=True, default_value=None).tag(attr=True)

    _source_coordinates = tl.Instance(Coordinates)

    @tl.default("coordinates_source")
    def _default_coordinates_source(self):
        return self.source

    @common_doc(COMMON_DOC)
    def _eval(self, coordinates, output=None, _selector=None):
        """Evaluates this nodes using the supplied coordinates.

        Parameters
        ----------
        coordinates : podpac.Coordinates
            {requested_coordinates}
        output : podpac.UnitsDataArray, optional
            {eval_output}
        selector: callable(coordinates, request_coordinates)
            {eval_selector}

        Returns
        -------
        {eval_return}

        Raises
        ------
        ValueError
            If source it not time-depended (required by this node).
        """

        source_output = self.source.eval(coordinates)

        # group
        grouped = source_output.groupby("time.%s" % self.groupby)

        # reduce
        if self.reduce_fn == "custom":
            out = grouped.apply(self.custom_reduce_fn, "time")
        else:
            # standard, e.g. grouped.median('time')
            out = getattr(grouped, self.reduce_fn)("time")

        out = out.rename({self.groupby: "time"})
        if output is None:
            coords = podpac.coordinates.merge_dims(
                [coordinates.drop("time"), Coordinates([out.coords["time"]], ["time"])]
            )
            coords = coords.transpose(*out.dims)
            output = self.create_output_array(coords, data=out.data)
        else:
            output.data[:] = out.data[:]

        ## map
        # eval_time = xr.DataArray(coordinates.coords["time"])
        # E = getattr(eval_time.dt, self.groupby)
        # out = out.sel(**{self.groupby: E}).rename({self.groupby: "time"})
        # output[:] = out.transpose(*output.dims).data

        return output

    @tl.default('base_ref')
    def _default_base_ref(self):
        """
        Default node reference/name in node definitions

        Returns
        -------
        str
            Default node reference/name in node definitions
        """
        return "%s.%s.%s" % (self.source.base_ref, self.groupby, self.reduce_fn)


class ResampleReduce(UnaryAlgorithm):
    """
    Resample a time-dependent source node using a statistical operation to achieve the result.

    Attributes
    ----------
    custom_reduce_fn : function
        required if reduce_fn is 'custom'.
    resample : str
        datetime sub-accessor. Currently 'dayofyear' is the enabled option.
    reduce_fn : str
        builtin xarray groupby reduce function, or 'custom'.
    source : podpac.Node
        Source node
    """

    _repr_keys = ["source", "resample", "reduce_fn"]
    coordinates_source = NodeTrait(allow_none=True).tag(attr=True)

    # see https://github.com/pydata/xarray/blob/eeb109d9181c84dfb93356c5f14045d839ee64cb/xarray/core/accessors.py#L61
    resample = tl.Unicode().tag(attr=True)
    reduce_fn = tl.CaselessStrEnum(_REDUCE_FUNCTIONS).tag(attr=True)
    custom_reduce_fn = tl.Any(allow_none=True, default_value=None).tag(attr=True)

    _source_coordinates = tl.Instance(Coordinates)

    @tl.default("coordinates_source")
    def _default_coordinates_source(self):
        return self.source

    @common_doc(COMMON_DOC)
    def _eval(self, coordinates, output=None, _selector=None):
        """Evaluates this nodes using the supplied coordinates.

        Parameters
        ----------
        coordinates : podpac.Coordinates
            {requested_coordinates}
        output : podpac.UnitsDataArray, optional
            {eval_output}
        _selector: callable(coordinates, request_coordinates)
            {eval_selector}

        Returns
        -------
        {eval_return}

        Raises
        ------
        ValueError
            If source it not time-dependent (required by this node).
        """

        source_output = self.source.eval(coordinates, _selector=_selector)

        # group
        grouped = source_output.resample(time=self.resample)

        # reduce
        if self.reduce_fn == "custom":
            out = grouped.reduce(self.custom_reduce_fn)
        else:
            # standard, e.g. grouped.median('time')
            out = getattr(grouped, self.reduce_fn)()

        if output is None:
            output = podpac.UnitsDataArray(out)
            output.attrs = source_output.attrs
        else:
            output.data[:] = out.data[:]

        ## map
        # eval_time = xr.DataArray(coordinates.coords["time"])
        # E = getattr(eval_time.dt, self.groupby)
        # out = out.sel(**{self.groupby: E}).rename({self.groupby: "time"})
        # output[:] = out.transpose(*output.dims).data

        return output

    @tl.default('base_ref')
    def _default_base_ref(self):
        """
        Default node reference/name in node definitions

        Returns
        -------
        str
            Default node reference/name in node definitions
        """
        return "%s.%s.%s" % (self.source.base_ref, self.resample, self.reduce_fn)


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


class DayOfYearWindow(Algorithm):
    """
    This applies a function over a moving window around day-of-year in the requested coordinates.
    It includes the ability to rescale the input/outputs. Note if, the input coordinates include multiple years, the
    moving window will include all of the data inside the day-of-year window.

    Users need to implement the 'function' method.

    Attributes
    -----------
    source: podpac.Node
        The source node from which the statistics will be computed
    window: int, optional
        Default is 0. The size of the window over which to compute the distrubtion. This is always centered about the
        day-of-year. The total number of days is always an odd number. For example, window=2 and window=3 will compute
        the beta distribution for [x-1, x, x + 1] and report it as the result for x, where x is a day of the year.
    scale_max: podpac.Node, optional
        Default is None. A source dataset that can be used to scale the maximum value of the source function so that it
        will fall between [0, 1]. If None, uses self.scale_float[0].
    scale_min: podpac.Node, optional
        Default is None. A source dataset that can be used to scale the minimum value of the source function so that it
        will fall between [0, 1]. If None, uses self.scale_float[1].
    scale_float: list, optional
        Default is []. Floating point numbers used to scale the max [0] and min [1] of the source so that it falls
        between [0, 1]. If scale_max or scale_min are defined, this property is ignored. If these are defined, the data
        will be rescaled only if rescale=True below.
        If None and scale_max/scale_min are not defined, the data is not scaled in any way.
    rescale: bool, optional
        Rescales the output data after being scaled from scale_float or scale_min/max
    """

    source = tl.Instance(podpac.Node).tag(attr=True)
    window = tl.Int(0).tag(attr=True)
    scale_max = tl.Instance(podpac.Node, default_value=None, allow_none=True).tag(attr=True)
    scale_min = tl.Instance(podpac.Node, default_value=None, allow_none=True).tag(attr=True)
    scale_float = tl.List(default_value=None, allow_none=True).tag(attr=True)
    rescale = tl.Bool(False).tag(attr=True)

    def algorithm(self, inputs, coordinates):
        win = self.window // 2
        source = inputs["source"]

        # Scale the source to range [0, 1], required for the beta distribution
        if "scale_max" in inputs:
            scale_max = inputs["scale_max"]
        elif self.scale_float and self.scale_float[1] is not None:
            scale_max = self.scale_float[1]
        else:
            scale_max = None

        if "scale_min" in inputs:
            scale_min = inputs["scale_min"]
        elif self.scale_float and self.scale_float[0] is not None:
            scale_min = self.scale_float[0]
        else:
            scale_min = None

        _log.debug("scale_min: {}\nscale_max: {}".format(scale_min, scale_max))
        if scale_min is not None and scale_max is not None:
            source = (source.copy() - scale_min) / (scale_max - scale_min)
            with np.errstate(invalid="ignore"):
                source.data[(source.data < 0) | (source.data > 1)] = np.nan

        # Make the output coordinates with day-of-year as time
        coords = xr.Dataset({"time": coordinates["time"].coordinates})
        dsdoy = np.sort(np.unique(coords.time.dt.dayofyear))
        latlon_coords = coordinates.drop("time")
        time_coords = podpac.Coordinates([dsdoy], ["time"])
        coords = podpac.coordinates.merge_dims([latlon_coords, time_coords])
        coords = coords.transpose(*coordinates.dims)
        output = self.create_output_array(coords)

        # if all-nan input, no need to calculate
        if np.all(np.isnan(source)):
            return output

        # convert source time coords to day-of-year as well
        sdoy = source.time.dt.dayofyear

        # loop over each day of year and compute window
        for i, doy in enumerate(dsdoy):
            _log.debug("Working on doy {doy} ({i}/{ld})".format(doy=doy, i=i + 1, ld=len(dsdoy)))

            # If either the start or end runs over the year, we need to do an OR on the bool index
            # ----->s....<=e------   .in -out
            # ..<=e----------->s..
            do_or = False

            start = doy - win
            if start < 1:
                start += 365
                do_or = True

            end = doy + win
            if end > 365:
                end -= 365
                do_or = True

            if do_or:
                I = (sdoy >= start) | (sdoy <= end)
            else:
                I = (sdoy >= start) & (sdoy <= end)

            # Scipy's beta function doesn's support multi-dimensional arrays, so we have to loop over lat/lon/alt
            lat_f = lon_f = alt_f = [None]
            dims = ["lat", "lon", "alt"]
            if "lat" in source.dims:
                lat_f = source["lat"].data
            if "lon" in source.dims:
                lon_f = source["lon"].data
            if "alt" in source.dims:
                alt_f = source["alt"].data

            for alt in alt_f:
                for lat in lat_f:
                    for lon in lon_f:
                        # _log.debug(f'lat, lon, alt = {lat}, {lon}, {alt})
                        loc_dict = {k: v for k, v in zip(dims, [lat, lon, alt]) if v is not None}

                        data = source.sel(time=I, **loc_dict).dropna("time").data
                        if np.all(np.isnan(data)):
                            continue

                        # Fit function to the particular point
                        output.loc[loc_dict][{"time": i}] = self.function(data, output.loc[loc_dict][{"time": i}])

        # Rescale the outputs
        if self.rescale:
            output = self.rescale_outputs(output, scale_max, scale_min)
        return output

    def function(self, data, output):
        raise NotImplementedError(
            "Child classes need to implement this function. It is applied over the data and needs"
            " to populate the output."
        )

    def rescale_outputs(self, output, scale_max, scale_min):
        output = (output * (scale_max - scale_min)) + scale_min
        return output
