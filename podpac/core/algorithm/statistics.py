from __future__ import division, unicode_literals, print_function, absolute_import

import warnings
from operator import mul

import xarray as xr
import numpy as np
import scipy.stats
import traitlets as tl

from podpac.core.coordinate import Coordinate
from podpac.core.node import Node
from podpac.core.algorithm.algorithm import Algorithm

class Reduce(Algorithm):
    input_coordinates = tl.Instance(Coordinate)
    input_node = tl.Instance(Node)

    @property
    def dims(self):
        """
        Validates and translates requested reduction dimensions.
        """

        input_dims = self.input_coordinates.dims

        if self.params is None or 'dims' not in self.params:
            return input_dims

        valid_dims = []
        if 'time' in input_dims:
            valid_dims.append('time')
        if 'lat_lon' in input_dims:
            valid_dims.append('lat_lon')
        if 'lat' in input_dims and 'lon' in input_dims:
            valid_dims.append('lat_lon')
        if 'alt' in input_dims:
            valid_dims.append('alt')

        params_dims = self.params['dims']
        if not isinstance(params_dims, (list, tuple)):
            params_dims = [params_dims]

        dims = []
        for dim in params_dims:
            if dim not in valid_dims:
                raise ValueError("Invalid Reduce dimension: %s" % dim)    
            elif dim in input_dims:
                dims.append(dim)
            elif dim == 'lat_lon':
                dims.append('lat')
                dims.append('lon')
        
        return dims

    def get_reduced_coordinates(self, coordinates):
        dims = self.dims
        kwargs = {}
        order = []
        for dim in coordinates.dims:
            if dim in self.dims:
                continue
            
            kwargs[dim] = coordinates[dim]
            order.append(dim)
        
        if order:
            kwargs['order'] = order

        return Coordinate(**kwargs)

    @property
    def chunk_size(self):
        if 'chunk_size' not in self.params:
            # return self.input_coordinates.shape
            return None

        if self.params['chunk_size'] == 'auto':
            return 1024**2 # TODO

        return self.params['chunk_size']

    @property
    def chunk_shape(self):
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

        Arguments
        ---------
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

    def iteroutputs(self):
        for chunk in self.input_coordinates.iterchunks(self.chunk_shape):
            yield self.input_node.execute(chunk, self.params)

    def execute(self, coordinates, params=None, output=None):
        self.input_coordinates = coordinates
        self.params = params
        self.output = output

        self.evaluated_coordinates = self.get_reduced_coordinates(coordinates)

        if self.output is None:
            self.output = self.initialize_output_array()
            
        if self.chunk_size and self.chunk_size < reduce(mul, coordinates.shape, 1):
            result = self.reduce_chunked(self.iteroutputs())
        else:
            if self.implicit_pipeline_evaluation:
                self.input_node.execute(coordinates, params)
            result = self.reduce(self.input_node.output)

        if self.output.shape is (): # or self.evaluated_coordinates is None
            self.output.data = result
        else:
            self.output[:] = result #.transpose(*self.output.dims) # is this necessary?
        
        self.evaluated = True

        return self.output

    def reduce(self, x):
        """
        Reduce a full array, e.g. x.mean(self.dims).

        Must be defined in each child.
        """

        raise NotImplementedError

    def reduce_chunked(self, xs):
        """
        Reduce a list of xs with a memory-effecient iterative algorithm.

        Optionally defined in each child.
        """

        warnings.warn("No reduce_chunked method defined, using one-step reduce")
        x = self.input_node.execute(self.input_coordinates, self.params)
        return self.reduce(x)

    @property
    def evaluated_hash(self):
        if self.input_coordinates is None:
            raise Exception("node not evaluated")
            
        return self.get_hash(self.input_coordinates, self.params)

    @property
    def latlon_bounds_str(self):
        return self.input_coordinates.latlon_bounds_str

class Min(Reduce):
    def reduce(self, x):
        return x.min(dim=self.dims)
    
    def reduce_chunked(self, xs):
        # note: np.fmin ignores NaNs, np.minimum propagates NaNs
        y = xr.full_like(self.output, np.nan)
        for x in xs:
            y = np.fmin(y, x.min(dim=self.dims))
        return y

class Max(Reduce):
    def reduce(self, x):
        return x.max(dim=self.dims)

    def reduce_chunked(self, xs):
        # note: np.fmax ignores NaNs, np.maximum propagates NaNs
        y = xr.full_like(self.output, np.nan)
        for x in xs:
            y = np.fmax(y, x.max(dim=self.dims))
        return y

class Sum(Reduce):
    def reduce(self, x):
        return x.sum(dim=self.dims)

    def reduce_chunked(self, xs):
        s = xr.zeros_like(self.output)
        for x in xs:
            s += x.sum(dim=self.dims)
        return s

class Count(Reduce):
    def reduce(self, x):
        return np.isfinite(x).sum(dim=self.dims)

    def reduce_chunked(self, xs):
        n = xr.zeros_like(self.output)
        for x in xs:
            n += np.isfinite(x).sum(dim=self.dims)
        return n

class Mean(Reduce):
    def reduce(self, x):
        return x.mean(dim=self.dims)

    def reduce_chunked(self, xs):
        s = xr.zeros_like(self.output) # alt: s = np.zeros(self.shape)
        n = xr.zeros_like(self.output)
        for x in xs:
            # TODO efficency
            s += x.sum(dim=self.dims)
            n += np.isfinite(x).sum(dim=self.dims)
        output = s / n
        return output

class Variance(Reduce):
    def reduce(self, x):
        return x.var(dim=self.dims)

    def reduce_chunked(self, xs):
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

class Mean2(Mean):
    def reduce_chunked(self, xs):
        N = xr.zeros_like(self.output)
        M1 = xr.zeros_like(self.output)
        
        for x in xs:
            Nx = np.isfinite(x).sum(dim=self.dims)
            M1x = x.mean(dim=self.dims)
            d = M1x - M1
            n = N + Nx
            
            M1 += d/n
            N = n

        return M1

class Variance2(Variance):
    def reduce_chunked(self, xs):
        N = xr.zeros_like(self.output)
        M1 = xr.zeros_like(self.output)
        M2 = xr.zeros_like(self.output)

        for x in xs:
            Nx = np.isfinite(x).sum(dim=self.dims)
            M1x = x.mean(dim=self.dims)
            M2x = x.var(dim=self.dims)
            
            d = M1x - M1
            n = N + Nx
            
            M2 += M2x + d**2*N*Nx/n
            M1 += d*Nx/n
            N = n

        return M2 / N

class Skew(Reduce):
    # TODO NaN behavior when there is NO data (currently different in reduce and reduce_chunked)

    def reduce(self, x):
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
        
    def reduce_chunked(self, xs):
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
    # TODO NaN behavior when there is NO data (currently different in reduce and reduce_chunked)

    def reduce(self, x):
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

    def reduce_chunked(self, xs):
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
    def reduce(self, x):
        return x.std(dim=self.dims)

    def reduce_chunked(self, xs):
        var = super(StandardDeviation, self).reduce_chunked(xs)
        return np.sqrt(var)

class Reduce2(Reduce):
    """
    Extended Reduce class that enables chunks that are smaller than the reduced
    output array.

    The base Reduce node ensures that each chunk is at least as big as the
    reduced output, which works for statistics that can be calculated in O(1)
    space. For statistics that require O(n) space, the node must iterate
    through the Coordinate orthogonally to the reduce dimension, using chunks
    that only cover a portion of the output array.

    Note that the above nodes *could* be implemented to allow small chunks.
    """

    @property
    def chunk_shape(self):
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

    def iteroutputs(self):
        chunks = self.input_coordinates.iterchunks(self.chunk_shape, return_slice=True)
        for slc, chunk in chunks:
            yield slc, self.input_node.execute(chunk, self.params)

class Median(Reduce2):
    def reduce(self, x):
        return x.median(dim=self.dims)

    def reduce_chunked(self, xs):
        I = [self.input_coordinates.dims.index(dim)
             for dim in
             self.evaluated_coordinates.dims]
        y = xr.full_like(self.output, np.nan)
        for xslc, x in xs:
            yslc = [xslc[i] for i in I]
            y.data[yslc] = x.median(dim=self.dims)
        return y

if __name__ == '__main__':
    from podpac.datalib.smap import SMAP

    smap = SMAP(product='SPL4SMAU.003')
    coords = smap.native_coordinates
    
    coords = Coordinate(
        time=coords.coords['time'][:6],
        lat=[45., 66., 5], lon=[-80., -70., 4],
        order=['time', 'lat', 'lon'])

    # smap_mean = Mean(input_node=SMAP(product='SPL4SMAU.003'))
    # print("lat_lon mean")
    # mean_ll = smap_mean.execute(coords, {'dims':'lat_lon'})
    # mean_ll_chunked = smap_mean.execute(coords, {'dims':'lat_lon', 'chunk_size': 2000})
    
    # print("time mean")
    # mean_time = smap_mean.execute(coords, {'dims':'time'})
    # mean_time_chunked = smap_mean.execute(coords, {'dims':'time', 'chunk_size': 2000})

    # print ("full mean")
    # mean_full = smap_mean.execute(coords, {'dims':['lat_lon', 'time']})
    # mean_full_chunked = smap_mean.execute(coords, {'dims': ['lat_lon', 'time'], 'chunk_size': 1000})
    # mean_full2 = smap_mean.execute(coords, {})

    # print("lat_lon count")
    # smap_count = Count(input_node=SMAP(product='SPL4SMAU.003'))
    # count_ll = smap_count.execute(coords, {'dims':'lat_lon'})
    # count_ll_chunked = smap_count.execute(coords, {'dims':'lat_lon', 'chunk_size': 1000})

    # print("lat_lon sum")
    # smap_sum = Sum(input_node=SMAP(product='SPL4SMAU.003'))
    # sum_ll = smap_sum.execute(coords, {'dims':'lat_lon'})
    # sum_ll_chunked = smap_sum.execute(coords, {'dims':'lat_lon', 'chunk_size': 1000})

    # print("lat_lon min")
    # smap_min = Min(input_node=SMAP(product='SPL4SMAU.003'))
    # min_ll = smap_min.execute(coords, {'dims':'lat_lon'})
    # min_ll_chunked = smap_min.execute(coords, {'dims':'lat_lon', 'chunk_size': 1000})
    # min_time = smap_min.execute(coords, {'dims':'time'})
    # min_time_chunked = smap_min.execute(coords, {'dims':'time', 'chunk_size': 1000})

    # print("lat_lon max")
    # smap_max = Max(input_node=SMAP(product='SPL4SMAU.003'))
    # max_ll = smap_max.execute(coords, {'dims':'lat_lon'})
    # max_ll_chunked = smap_max.execute(coords, {'dims':'lat_lon', 'chunk_size': 1000})
    # max_time = smap_max.execute(coords, {'dims':'time'})
    # max_time_chunked = smap_max.execute(coords, {'dims':'time', 'chunk_size': 1000})

    # smap_var = Variance(input_node=SMAP(product='SPL4SMAU.003'))
    # var_ll_chunked = smap_var.execute(coords, {'dims':'lat_lon', 'chunk_size': 6})
    # var_ll = smap_var.execute(coords, {'dims':'lat_lon'})
    # var_time = smap_var.execute(coords, {'dims':'time'})
    # var_time_chunked = smap_var.execute(coords, {'dims':'time', 'chunk_size': 1})

    # smap_var2 = Variance2(input_node=SMAP(product='SPL4SMAU.003'))
    # var2_ll = smap_var2.execute(coords, {'dims':'lat_lon'})
    # var2_ll_chunked = smap_var2.execute(coords, {'dims':'lat_lon', 'chunk_size': 6})
    # var2_time = smap_var2.execute(coords, {'dims':'time'})
    # var2_time_chunked = smap_var2.execute(coords, {'dims':'time', 'chunk_size': 1})

    # smap = SMAP(product='SPL4SMAU.003')
    # o = smap.execute(coords, {})

    smap_skew = Skew(input_node=SMAP(product='SPL4SMAU.003'))
    skew_ll = smap_skew.execute(coords, {'dims':'lat_lon'})
    skew_ll_chunked = smap_skew.execute(coords, {'dims':'lat_lon', 'chunk_size': 40})
    skew_ll_chunked1 = smap_skew.execute(coords, {'dims':'lat_lon', 'chunk_size': 1})
    skew_time = smap_skew.execute(coords, {'dims':'time'})
    skew_time_chunked = smap_skew.execute(coords, {'dims':'time', 'chunk_size': 40})
    skew_time_chunked1 = smap_skew.execute(coords, {'dims':'time', 'chunk_size': 1})

    smap_kurtosis = Kurtosis(input_node=SMAP(product='SPL4SMAU.003'))
    kurtosis_ll = smap_kurtosis.execute(coords, {'dims':'lat_lon'})
    kurtosis_ll_chunked = smap_kurtosis.execute(coords, {'dims':'lat_lon', 'chunk_size': 40})
    kurtosis_ll_chunked1 = smap_kurtosis.execute(coords, {'dims':'lat_lon', 'chunk_size': 1})
    kurtosis_time = smap_kurtosis.execute(coords, {'dims':'time'})
    kurtosis_time_chunked = smap_kurtosis.execute(coords, {'dims':'time', 'chunk_size': 40})
    kurtosis_time_chunked1 = smap_kurtosis.execute(coords, {'dims':'time', 'chunk_size': 1})

    # smap_std = StandardDeviation(input_node=SMAP(product='SPL4SMAU.003'))
    # std_ll = smap_std.execute(coords, {'dims':'lat_lon'})
    # std_ll_chunked = smap_std.execute(coords, {'dims':'lat_lon', 'chunk_size': 1000})
    # std_time = smap_std.execute(coords, {'dims':'time'})
    # std_time_chunked = smap_std.execute(coords, {'dims':'time', 'chunk_size': 1000})

    # smap_median = Median(input_node=SMAP(product='SPL4SMAU.003'))
    # median_ll = smap_median.execute(coords, {'dims':'lat_lon'})
    # median_ll_chunked = smap_median.execute(coords, {'dims':'lat_lon', 'chunk_size': 1})
    # median_ll_chunked2 = smap_median.execute(coords, {'dims':'lat_lon', 'chunk_size': 10})
    # median_time = smap_median.execute(coords, {'dims':'time'})
    # median_time_chunked = smap_median.execute(coords, {'dims':'time', 'chunk_size': 1})
    # median_time_chunked2 = smap_median.execute(coords, {'dims':'time', 'chunk_size': 10})

    print ("Done")