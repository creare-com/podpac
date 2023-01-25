from __future__ import division, unicode_literals, print_function, absolute_import

import pytest
import numpy as np
import xarray as xr
import scipy.stats
import traitlets as tl

import podpac
from podpac.core.algorithms.utility import Arange
from podpac.core.data.array_source import Array
from podpac.core.algorithms.stats import Reduce
from podpac.core.algorithms.stats import Min, Max, Sum, Count, Mean, Variance, Skew, Kurtosis, StandardDeviation
from podpac.core.algorithms.generic import Arithmetic
from podpac.core.algorithms.stats import Median, Percentile
from podpac.core.algorithms.stats import GroupReduce, DayOfYear, DayOfYearWindow


def setup_module():
    global coords, source, data, multisource, bdata
    coords = podpac.Coordinates(
        [podpac.clinspace(0, 1, 10), podpac.clinspace(0, 1, 10), podpac.crange("2018-01-01", "2018-01-10", "1,D")],
        dims=["lat", "lon", "time"],
    )

    a = np.random.random(coords.shape)
    a[3, 0, 0] = np.nan
    a[0, 3, 0] = np.nan
    a[0, 0, 3] = np.nan
    source = Array(source=a, coordinates=coords)
    data = source.eval(coords)

    ab = np.stack([a, 2 * a], -1)
    multisource = Array(source=ab, coordinates=coords, outputs=["a", "b"])
    bdata = 2 * data


class TestReduce(object):
    """Tests the Reduce class"""

    def test_auto_chunk(self):
        # any reduce node would do here
        node = Min(source=source)

        with podpac.settings:
            podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"] = False
            podpac.settings["CHUNK_SIZE"] = "auto"
            node.eval(coords)

    def test_chunked_fallback(self):
        with podpac.settings:
            podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"] = False

            class First(Reduce):
                def reduce(self, x):
                    return x.isel(**{dim: 0 for dim in self.dims})

            node = First(source=source, dims="time")

            # use reduce function
            podpac.settings["CHUNK_SIZE"] = None
            output = node.eval(coords)

            # fall back on reduce function with warning
            with pytest.warns(UserWarning):
                podpac.settings["CHUNK_SIZE"] = 500
                output_chunked = node.eval(coords)

            # should be the same
            xr.testing.assert_allclose(output, output_chunked)


class BaseTests(object):
    """Common tests for Reduce subclasses"""

    def test_full(self):
        with podpac.settings:
            podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"] = False
            podpac.settings["CHUNK_SIZE"] = None

            node = self.NodeClass(source=source)
            output = node.eval(coords)
            # xr.testing.assert_allclose(output, self.expected_full)
            np.testing.assert_allclose(output.data, self.expected_full.data)

            node = self.NodeClass(source=source, dims=coords.dims)
            output = node.eval(coords)
            # xr.testing.assert_allclose(output, self.expected_full)
            np.testing.assert_allclose(output.data, self.expected_full.data)

    def test_full_chunked(self):
        with podpac.settings:
            node = self.NodeClass(source=source, dims=coords.dims)
            podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"] = False
            podpac.settings["CHUNK_SIZE"] = 500
            output = node.eval(coords)
            # xr.testing.assert_allclose(output, self.expected_full)
            np.testing.assert_allclose(output.data, self.expected_full.data)

    def test_lat_lon(self):
        with podpac.settings:
            podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"] = False
            podpac.settings["CHUNK_SIZE"] = None
            node = self.NodeClass(source=source, dims=["lat", "lon"])
            output = node.eval(coords)
            # xr.testing.assert_allclose(output, self.expected_latlon)
            np.testing.assert_allclose(output.data, self.expected_latlon.data)

    def test_lat_lon_chunked(self):
        with podpac.settings:
            podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"] = False
            podpac.settings["CHUNK_SIZE"] = 500
            node = self.NodeClass(source=source, dims=["lat", "lon"])
            output = node.eval(coords)
            # xr.testing.assert_allclose(output, self.expected_latlon)
            np.testing.assert_allclose(output.data, self.expected_latlon.data)

    def test_time(self):
        with podpac.settings:
            podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"] = False
            podpac.settings["CHUNK_SIZE"] = None
            node = self.NodeClass(source=source, dims="time")
            output = node.eval(coords)
            # xr.testing.assert_allclose(output, self.expected_time)
            np.testing.assert_allclose(output.data, self.expected_time.data)

    def test_time_chunked(self):
        with podpac.settings:
            podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"] = False
            podpac.settings["CHUNK_SIZE"] = 500
            node = self.NodeClass(source=source, dims="time")
            output = node.eval(coords)
            # xr.testing.assert_allclose(output, self.expected_time)
            np.testing.assert_allclose(output.data, self.expected_time.data)

    def test_multiple_outputs(self):
        with podpac.settings:
            podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"] = False
            podpac.settings["CHUNK_SIZE"] = None
            node = self.NodeClass(source=multisource, dims=["lat", "lon"])
            output = node.eval(coords)
            assert output.dims == ("time", "output")
            np.testing.assert_array_equal(output["output"], ["a", "b"])
            np.testing.assert_allclose(output.sel(output="a"), self.expected_latlon)
            np.testing.assert_allclose(output.sel(output="b"), self.expected_latlon_b)


class TestMin(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = Min
        cls.expected_full = data.min()
        cls.expected_latlon = data.min(dim=["lat", "lon"])
        cls.expected_latlon_b = bdata.min(dim=["lat", "lon"])
        cls.expected_time = data.min(dim="time")


class TestMax(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = Max
        cls.expected_full = data.max()
        cls.expected_latlon = data.max(dim=["lat", "lon"])
        cls.expected_latlon_b = bdata.max(dim=["lat", "lon"])
        cls.expected_time = data.max(dim="time")


class TestSum(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = Sum
        cls.expected_full = data.sum()
        cls.expected_latlon = data.sum(dim=["lat", "lon"])
        cls.expected_latlon_b = bdata.sum(dim=["lat", "lon"])
        cls.expected_time = data.sum(dim="time")


class TestCount(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = Count
        cls.expected_full = np.isfinite(data).sum()
        cls.expected_latlon = np.isfinite(data).sum(dim=["lat", "lon"])
        cls.expected_latlon_b = np.isfinite(bdata).sum(dim=["lat", "lon"])
        cls.expected_time = np.isfinite(data).sum(dim="time")


class TestMean(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = Mean
        cls.expected_full = data.mean()
        cls.expected_latlon = data.mean(dim=["lat", "lon"])
        cls.expected_latlon_b = bdata.mean(dim=["lat", "lon"])
        cls.expected_time = data.mean(dim="time")

    def test_chunk_sizes(self):
        for n in [20, 21, 1000, 1001]:
            podpac.settings["CHUNK_SIZE"] = n
            node = self.NodeClass(source=source, dims=coords.dims)
            output = node.eval(coords)
            # xr.testing.assert_allclose(output, self.expected_full)
            np.testing.assert_allclose(output.data, self.expected_full.data)


class TestVariance(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = Variance
        cls.expected_full = data.var()
        cls.expected_latlon = data.var(dim=["lat", "lon"])
        cls.expected_latlon_b = bdata.var(dim=["lat", "lon"])
        cls.expected_time = data.var(dim="time")


class TestStandardDeviation(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = StandardDeviation
        cls.expected_full = data.std()
        cls.expected_latlon = data.std(dim=["lat", "lon"])
        cls.expected_latlon_b = bdata.std(dim=["lat", "lon"])
        cls.expected_time = data.std(dim="time")
        cls.expected_latlon_b = 2 * cls.expected_latlon


class TestSkew(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = Skew
        n, m, l = data.shape
        cls.expected_full = xr.DataArray(scipy.stats.skew(data.data.reshape(n * m * l), nan_policy="omit"))
        cls.expected_latlon = scipy.stats.skew(data.data.reshape((n * m, l)), axis=0, nan_policy="omit")
        cls.expected_latlon_b = scipy.stats.skew(bdata.data.reshape((n * m, l)), axis=0, nan_policy="omit")
        cls.expected_time = scipy.stats.skew(data, axis=2, nan_policy="omit")


class TestKurtosis(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = Kurtosis
        n, m, l = data.shape
        cls.expected_full = xr.DataArray(scipy.stats.kurtosis(data.data.reshape(n * m * l), nan_policy="omit"))
        cls.expected_latlon = scipy.stats.kurtosis(data.data.reshape((n * m, l)), axis=0, nan_policy="omit")
        cls.expected_latlon_b = scipy.stats.kurtosis(bdata.data.reshape((n * m, l)), axis=0, nan_policy="omit")
        cls.expected_time = scipy.stats.kurtosis(data, axis=2, nan_policy="omit")


class TestMedian(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.NodeClass = Median
        cls.expected_full = data.median()
        cls.expected_latlon = data.median(dim=["lat", "lon"])
        cls.expected_latlon_b = bdata.median(dim=["lat", "lon"])
        cls.expected_time = data.median(dim="time")


@pytest.mark.skip("TODO")
class TestPercentile(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Percentile(source=source)
        # TODO can we replace dims_axes with reshape (or vice versa)


class TestGroupReduce(object):
    pass


class TestResampleReduce(object):
    pass


class TestDayOfYear(object):
    pass


class F(DayOfYearWindow):
    cache_output = tl.Bool(False)
    force_eval = tl.Bool(True)

    def function(self, data, output):
        return len(data)


class FM(DayOfYearWindow):
    cache_output = tl.Bool(False)
    force_eval = tl.Bool(True)

    def function(self, data, output):
        return np.mean(data)


class TestDayOfYearWindow(object):
    def test_doy_window1(self):
        coords = podpac.coordinates.concat(
            [
                podpac.Coordinates([podpac.crange("1999-12-29", "2000-01-02", "1,D", "time")]),
                podpac.Coordinates([podpac.crange("2001-12-30", "2002-01-03", "1,D", "time")]),
            ]
        )

        node = Arange()
        nodedoywindow = F(source=node, window=1, cache_output=False, force_eval=True)
        o = nodedoywindow.eval(coords)

        np.testing.assert_array_equal(o, [2, 2, 1, 1, 2, 2])

    def test_doy_window2(self):
        coords = podpac.coordinates.concat(
            [
                podpac.Coordinates([podpac.crange("1999-12-29", "2000-01-03", "1,D", "time")]),
                podpac.Coordinates([podpac.crange("2001-12-30", "2002-01-02", "1,D", "time")]),
            ]
        )

        node = Arange()
        nodedoywindow = F(source=node, window=2, cache_output=False, force_eval=True)
        o = nodedoywindow.eval(coords)

        np.testing.assert_array_equal(o, [6, 5, 3, 3, 5, 6])

    def test_doy_window2_mean_rescale_float(self):
        coords = podpac.coordinates.concat(
            [
                podpac.Coordinates([podpac.crange("1999-12-29", "2000-01-03", "1,D", "time")]),
                podpac.Coordinates([podpac.crange("2001-12-30", "2002-01-02", "1,D", "time")]),
            ]
        )

        node = Arange()
        nodedoywindow = FM(source=node, window=2, cache_output=False, force_eval=True)
        o = nodedoywindow.eval(coords)

        nodedoywindow_s = FM(
            source=node, window=2, cache_output=False, force_eval=True, scale_float=[0, coords.size], rescale=True
        )
        o_s = nodedoywindow_s.eval(coords)

        np.testing.assert_array_almost_equal(o, o_s)

    def test_doy_window2_mean_rescale_max_min(self):
        with podpac.settings:
            podpac.settings.set_unsafe_eval(True)

            coords = podpac.coordinates.concat(
                [
                    podpac.Coordinates([podpac.crange("1999-12-29", "2000-01-03", "1,D", "time")]),
                    podpac.Coordinates([podpac.crange("2001-12-30", "2002-01-02", "1,D", "time")]),
                ]
            )

            node = Arange()
            node_max = Arithmetic(source=node, eqn="(source < 5) + source")
            node_min = Arithmetic(source=node, eqn="-1*(source < 5) + source")

            nodedoywindow_s = FM(
                source=node,
                window=2,
                cache_output=False,
                force_eval=True,
                scale_max=node_max,
                scale_min=node_min,
                rescale=False,
            )
            o_s = nodedoywindow_s.eval(coords)

            np.testing.assert_array_almost_equal([0.5] * o_s.size, o_s)
