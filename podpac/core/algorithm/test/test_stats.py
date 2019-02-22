from __future__ import division, unicode_literals, print_function, absolute_import

import pytest
import numpy as np
import xarray as xr
import scipy.stats

import podpac
from podpac.core.data.types import Array
from podpac.core.algorithm.stats import Min, Max, Sum, Count, Mean, Variance, Skew, Kurtosis, StandardDeviation
from podpac.core.algorithm.stats import Median, Percentile
from podpac.core.algorithm.stats import GroupReduce, DayOfYear

def setup_module():
    global coords, source, data
    coords = podpac.Coordinates(
        [podpac.clinspace(0, 1, 10), podpac.clinspace(0, 1, 10), podpac.crange('2018-01-01', '2018-01-10', '1,D')],
        dims=['lat', 'lon', 'time'])

    a = np.random.random(coords.shape)
    a[3, 0, 0] = np.nan
    a[0, 3, 0] = np.nan
    a[0, 0, 3] = np.nan
    source = Array(source=a, native_coordinates=coords)
    data = source.eval(coords)

class TestReduce(object):
    """ Tests the Reduce class """

    def setup_method(self):
        # save chunk size
        self.saved_chunk_size = podpac.settings['CHUNK_SIZE']
        podpac.settings['CHUNK_SIZE'] = None

    def teardown_method(self):
        podpac.settings['CHUNK_SIZE'] = self.saved_chunk_size

    def test_auto_chunk(self):
        podpac.settings['CHUNK_SIZE'] = 'auto'

        # any reduce node would do here
        node = Min(source=source)
        node.eval(coords)

    def test_not_implemented(self):
        from podpac.core.algorithm.stats import Reduce

        node = Reduce(source=source)
        with pytest.raises(NotImplementedError):
            node.eval(coords)

    def test_chunked_fallback(self):
        from podpac.core.algorithm.stats import Reduce

        class First(Reduce):
            def reduce(self, x):
                return x.isel(**{dim:0 for dim in self.dims})

        node = First(source=source, dims='time')
        
        # use reduce function
        podpac.settings['CHUNK_SIZE'] = None
        output = node.eval(coords)
        
        # fall back on reduce function with warning
        with pytest.warns(UserWarning):
            podpac.settings['CHUNK_SIZE'] = 100
            output_chunked = node.eval(coords)

        # should be the same
        xr.testing.assert_allclose(output, output_chunked)

class BaseTests(object):
    """ Common tests for Reduce subclasses """

    def setup_method(self):
        # save chunk size
        self.saved_chunk_size = podpac.settings['CHUNK_SIZE']
        podpac.settings['CHUNK_SIZE'] = None

    def teardown_method(self):
        podpac.settings['CHUNK_SIZE'] = self.saved_chunk_size

    def test_full(self):
        node = self.NodeClass(source=source)
        output = node.eval(coords)
        # xr.testing.assert_allclose(output, self.expected_full)
        np.testing.assert_allclose(output.data, self.expected_full.data)

        node = self.NodeClass(source=source, dims=coords.dims)
        output = node.eval(coords)
        # xr.testing.assert_allclose(output, self.expected_full)
        np.testing.assert_allclose(output.data, self.expected_full.data)

    def test_full_chunked(self):
        podpac.settings['CHUNK_SIZE'] = 100
        node = self.NodeClass(source=source, dims=coords.dims)
        output = node.eval(coords)
        # xr.testing.assert_allclose(output, self.expected_full)
        np.testing.assert_allclose(output.data, self.expected_full.data)

    def test_lat_lon(self):
        node = self.NodeClass(source=source, dims=['lat', 'lon'])
        output = node.eval(coords)
        # xr.testing.assert_allclose(output, self.expected_latlon)
        np.testing.assert_allclose(output.data, self.expected_latlon.data)

    @pytest.mark.xfail(reason="bug, to fix")
    def test_lat_lon_chunked(self):
        podpac.settings['CHUNK_SIZE'] = 100
        node = self.NodeClass(source=source, dims=['lat', 'lon'])
        output = node.eval(coords)
        # xr.testing.assert_allclose(output, self.expected_latlon)
        np.testing.assert_allclose(output.data, self.expected_latlon.data)

    def test_time(self):
        node = self.NodeClass(source=source, dims='time')
        output = node.eval(coords)
        # xr.testing.assert_allclose(output, self.expected_time)
        np.testing.assert_allclose(output.data, self.expected_time.data)

    def test_time_chunked(self):
        podpac.settings['CHUNK_SIZE'] = 100
        node = self.NodeClass(source=source, dims='time')
        output = node.eval(coords)
        # xr.testing.assert_allclose(output, self.expected_time)
        np.testing.assert_allclose(output.data, self.expected_time.data)

class TestMin(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = Min
        cls.expected_full = data.min()
        cls.expected_latlon = data.min(dim=['lat', 'lon'])
        cls.expected_time = data.min(dim='time')

class TestMax(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = Max
        cls.expected_full = data.max()
        cls.expected_latlon = data.max(dim=['lat', 'lon'])
        cls.expected_time = data.max(dim='time')

class TestSum(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = Sum
        cls.expected_full = data.sum()
        cls.expected_latlon = data.sum(dim=['lat', 'lon'])
        cls.expected_time = data.sum(dim='time')

class TestCount(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = Count
        cls.expected_full = np.isfinite(data).sum()
        cls.expected_latlon = np.isfinite(data).sum(dim=['lat', 'lon'])
        cls.expected_time = np.isfinite(data).sum(dim='time')

class TestMean(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = Mean
        cls.expected_full = data.mean()
        cls.expected_latlon = data.mean(dim=['lat', 'lon'])
        cls.expected_time = data.mean(dim='time')

class TestVariance(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = Variance
        cls.expected_full = data.var()
        cls.expected_latlon = data.var(dim=['lat', 'lon'])
        cls.expected_time = data.var(dim='time')

class TestStandardDeviation(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = StandardDeviation
        cls.expected_full = data.std()
        cls.expected_latlon = data.std(dim=['lat', 'lon'])
        cls.expected_time = data.std(dim='time')

class TestSkew(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = Skew
        n, m, l = data.shape
        cls.expected_full = xr.DataArray(scipy.stats.skew(data.data.reshape(n*m*l), nan_policy='omit'))
        cls.expected_latlon = scipy.stats.skew(data.data.reshape((n*m, l)), axis=0, nan_policy='omit')
        cls.expected_time = scipy.stats.skew(data, axis=2, nan_policy='omit')

class TestKurtosis(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = Kurtosis
        n, m, l = data.shape
        cls.expected_full = xr.DataArray(scipy.stats.kurtosis(data.data.reshape(n*m*l), nan_policy='omit'))
        cls.expected_latlon = scipy.stats.kurtosis(data.data.reshape((n*m, l)), axis=0, nan_policy='omit')
        cls.expected_time = scipy.stats.kurtosis(data, axis=2, nan_policy='omit')

class TestMedian(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = Median
        cls.expected_full = data.median()
        cls.expected_latlon = data.median(dim=['lat', 'lon'])
        cls.expected_time = data.median(dim='time')

# class TestPercentile(BaseTests):
#     @classmethod
#     def setup_class(cls):
#         cls.NodeClass = Percentile

class TestGroupReduce(object):
    pass

class TestDayOfYear(object):
    pass
