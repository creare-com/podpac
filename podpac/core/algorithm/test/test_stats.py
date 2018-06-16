from __future__ import division, unicode_literals, print_function, absolute_import

import pytest
import numpy as np
import xarray as xr

from podpac.core.coordinate import Coordinate
from podpac.core.data.type import NumpyArray
from podpac.core.algorithm.stats import Min, Max, Sum, Count, Mean, Variance, Skew, Kurtosis, StandardDeviation
from podpac.core.algorithm.stats import Median, Percentile
from podpac.core.algorithm.stats import GroupReduce, DayOfYear

def setup_module():
    global coords, source, data
    coords = Coordinate(
        lat=(0, 1, 10),
        lon=(0, 1, 10),
        time=('2018-01-01', '2018-01-10', '1,D'),
        order=['lat', 'lon', 'time'])

    a = np.random.random(coords.shape)
    a[3, 0, 0] = np.nan
    a[0, 3, 0] = np.nan
    a[0, 0, 3] = np.nan
    source = NumpyArray(source=a, native_coordinates=coords)
    data = source.execute(coords)

class TestReduce(object):
    def test_invalid_dims(self):
        # any reduce node would do here
        node = Min(source=source)
        
        # valid dim
        node.execute(coords, {'dims': 'lat'})
        
        # invalid dim
        with pytest.raises(ValueError):
            node.execute(coords, {'dims': 'alt'})

    def test_auto_chunk(self):
        # any reduce node would do here
        node = Min(source=source)
        node.execute(coords, {'iter_chunk_size': 'auto'})

    def test_not_implemented(self):
        from podpac.core.algorithm.stats import Reduce

        node = Reduce(source=source)
        with pytest.raises(NotImplementedError):
            node.execute(coords)

    def test_chunked_fallback(self):
        from podpac.core.algorithm.stats import Reduce

        class First(Reduce):
            def reduce(self, x):
                return x.isel(**{dim:0 for dim in self.dims})

        node = First(source=source)
        
        # use reduce function
        output = node.execute(coords, {'dims': 'time'})
        
        # fall back on reduce function with warning
        with pytest.warns(UserWarning):
            output_chunked = node.execute(coords, {'dims': 'time', 'iter_chunk_size': 100})

        # should be the same
        xr.testing.assert_allclose(output, output_chunked)

class BaseTests(object):
    def test_full(self):
        output = self.node.execute(coords)
        xr.testing.assert_allclose(output, self.expected_full)

        output = self.node.execute(coords, {'dims': coords.dims})
        xr.testing.assert_allclose(output, self.expected_full)

        output = self.node.execute(coords, {'iter_chunk_size': 100})
        xr.testing.assert_allclose(output, self.expected_full)

    def test_lat_lon(self):
        output = self.node.execute(coords, {'dims': ['lat', 'lon']})
        xr.testing.assert_allclose(output, self.expected_latlon)

        output = self.node.execute(coords, {'dims': ['lat', 'lon'], 'iter_chunk_size': 100})
        xr.testing.assert_allclose(output, self.expected_latlon)

    def test_time(self):
        output = self.node.execute(coords, {'dims': 'time'})
        xr.testing.assert_allclose(output, self.expected_time)

        output = self.node.execute(coords, {'dims': 'time', 'iter_chunk_size': 100})
        xr.testing.assert_allclose(output, self.expected_time)

@pytest.mark.skip(reason="TODO: Python 3 issues")
class TestMin(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Min(source=source)
        cls.expected_full = data.min()
        cls.expected_latlon = data.min(dim=['lat', 'lon'])
        cls.expected_time = data.min(dim='time')

@pytest.mark.skip(reason="TODO: Python 3 issues")
class TestMax(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Max(source=source)
        cls.expected_full = data.max()
        cls.expected_latlon = data.max(dim=['lat', 'lon'])
        cls.expected_time = data.max(dim='time')

@pytest.mark.skip(reason="TODO: Python 3 issues")
class TestSum(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Sum(source=source)
        cls.expected_full = data.sum()
        cls.expected_latlon = data.sum(dim=['lat', 'lon'])
        cls.expected_time = data.sum(dim='time')

# xr.testing.assert_allclose should ignore the DataArray attrs, but that seems to be why these are failing
# @pytest.mark.xfail(reason="possibly a bug in xarray.testing.assert_allclose")
@pytest.mark.skip(reason="TODO: Python 3 issues")
class TestCount(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Count(source=source)
        cls.expected_full = np.isfinite(data).sum()
        cls.expected_latlon = np.isfinite(data).sum(dim=['lat', 'lon'])
        cls.expected_time = np.isfinite(data).sum(dim='time')

@pytest.mark.skip(reason="TODO: Python 3 issues")
class TestMean(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Mean(source=source)
        cls.expected_full = data.mean()
        cls.expected_latlon = data.mean(dim=['lat', 'lon'])
        cls.expected_time = data.mean(dim='time')

@pytest.mark.skip(reason="TODO: Python 3 issues")
class TestVariance(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Variance(source=source)
        cls.expected_full = data.var()
        cls.expected_latlon = data.var(dim=['lat', 'lon'])
        cls.expected_time = data.var(dim='time')

@pytest.mark.skip(reason="TODO: Python 3 issues")
class TestStandardDeviation(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = StandardDeviation(source=source)
        cls.expected_full = data.std()
        cls.expected_latlon = data.std(dim=['lat', 'lon'])
        cls.expected_time = data.std(dim='time')

@pytest.mark.skip("TODO")
class TestSkew(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Skew(source=source)

@pytest.mark.skip("TODO")
class TestKurtosis(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Kurtosis(source=source)

@pytest.mark.skip("not-yet-working")
class TestMedian(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Median(source=source)
        cls.expected_full = data.median()
        cls.expected_latlon = data.median(dim=['lat', 'lon'])
        cls.expected_time = data.median(dim='time')

@pytest.mark.skip("TODO")
class TestPercentile(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Percentile(source=source)

@pytest.mark.skip("TODO")
class TestGroupReduce(object):
    def test(self):
        pass

@pytest.mark.skip("TODO")
class TestDayOfYear(object):
    def test(self):
        pass