import os.path
from collections import OrderedDict

import numpy as np
import pytest
import rasterio
import rasterio.cache

from podpac.core.coordinates import Coordinates
from podpac.core.units import UnitsDataArray
from podpac.core.data.rasterio_source import Rasterio
from podpac import clinspace


class TestRasterio(object):
    """test rasterio data source"""

    source = os.path.join(os.path.dirname(__file__), "assets/RGB.byte.tif")
    band = 1

    def test_init(self):
        """test basic init of class"""

        Rasterio(source=self.source, band=self.band)

    def test_dataset(self):
        """test dataset attribute and trait default"""

        node = Rasterio(source=self.source, band=self.band)
        try:
            raster_reader = rasterio._io.RasterReader  # Rasterio < v1.0
        except AttributeError:
            raster_reader = rasterio.io.DatasetReader  # Rasterio >= v1.0
        assert isinstance(node.dataset, raster_reader)

        node.close_dataset()

    def test_coordinates(self):
        """test default coordinates implementations"""

        node = Rasterio(source=self.source)
        assert isinstance(node.coordinates, Coordinates)
        assert len(node.coordinates["lat"]) == 718

    def test_get_data(self):
        """test default get_data method"""

        node = Rasterio(source=self.source)
        output = node.eval(node.coordinates)
        assert isinstance(output, UnitsDataArray)

    def test_band_count(self):
        """test band descriptions methods"""
        node = Rasterio(source=self.source)
        assert node.band_count == 3

    def test_band_descriptions(self):
        """test band count method"""
        node = Rasterio(source=self.source)
        assert isinstance(node.band_descriptions, OrderedDict)
        assert list(node.band_descriptions.keys()) == [0, 1, 2]

    def test_band_keys(self):
        """test band keys methods"""
        node = Rasterio(source=self.source)
        assert set(node.band_keys.keys()) == {
            "STATISTICS_STDDEV",
            "STATISTICS_MINIMUM",
            "STATISTICS_MEAN",
            "STATISTICS_MAXIMUM",
        }

    def test_get_band_numbers(self):
        """test band numbers methods"""
        node = Rasterio(source=self.source)
        numbers = node.get_band_numbers("STATISTICS_MINIMUM", "0")
        np.testing.assert_array_equal(numbers, [1, 2, 3])

    def test_get_window_coords(self):
        """test get_window_coords method"""
        c1 = Coordinates([clinspace(31, 30, 16, "lat"), clinspace(-0.25, 1.5, 64, "lon")])
        c2 = Coordinates([clinspace(30.75, 30.25, 16, "lat"), clinspace(-0.0, 1.25, 64, "lon")])

        c3 = Coordinates([clinspace(31, 30, 16, "lat"), clinspace(-0.5, 1.25, 64, "lon")])

        node = Rasterio()
        _, new_coords_1 = node._get_window_coords(c2, c1)  # tests when 1 coord completely contains the other
        _, new_coords_2 = node._get_window_coords(c3, c1)  # tests when 1 coord does not completely contian the other

        expected_values = {0: {"lon": 46, "lat": 10}, 1: {"lon": 55, "lat": 16}}
        for i, data in enumerate([new_coords_1, new_coords_2]):
            for a in ["lon", "lat"]:
                assert np.isnan(data[a].coordinates).sum() == 0  # nan check
                assert len(data[a]) == expected_values[i][a]  # guard against old issue of return being trimmed


class TestRasterioFailSafe(object):
    """Test that a failed open or read doesn't break later calls."""

    source = os.path.join(os.path.dirname(__file__), "assets/RGB.byte.tif")

    @pytest.fixture
    def cache_clears(self, monkeypatch: pytest.MonkeyPatch) -> list:
        """Record calls that clear GDAL's HTTP cache."""
        calls = []

        def clear(pattern: str) -> None:
            calls.append(pattern)

        monkeypatch.setattr(rasterio.cache, "invalidate", clear)
        return calls

    @pytest.fixture
    def node(self) -> Rasterio:
        """Single band rasterio node."""
        return Rasterio(source=self.source, band=1)

    @pytest.fixture
    def expected(self) -> np.ndarray:
        """Output of a read with no failures, from a separate node so ``node`` stays unopened."""
        reference = Rasterio(source=self.source, band=1)
        return reference.eval(reference.coordinates).data

    def throw_once(self, monkeypatch: pytest.MonkeyPatch, target: object, name: str) -> None:
        """Make ``target.name`` raise a RasterioIOError once then remaining calls return original method."""
        original = getattr(target, name)
        thrown = False

        def _throw_sequence(*args, **kwargs):
            nonlocal thrown
            if not thrown:
                thrown = True
                raise rasterio.errors.RasterioIOError("simulated dropped response")

            return original(*args, **kwargs)

        monkeypatch.setattr(target, name, _throw_sequence)

    def test_read_failure_resets(self, monkeypatch, cache_clears, node, expected):
        """A failed read raises, drops the handle and GDAL's cache, and the next eval works."""
        handle = node.dataset
        self.throw_once(monkeypatch, rasterio.io.DatasetReader, "read")
        with pytest.raises(rasterio.errors.RasterioIOError):
            node.eval(node.coordinates)
        assert cache_clears == [node.source]
        assert handle.closed
        data = node.eval(node.coordinates).data
        np.testing.assert_array_equal(data, expected)
        assert node.dataset is not handle

    def test_open_failure_resets(self, monkeypatch, node, expected):
        """A failed open raises, and the next access works."""
        self.throw_once(monkeypatch, rasterio, "open", 1)
        with pytest.raises(rasterio.errors.RasterioIOError):
            node.dataset
        data = node.eval(node.coordinates).data
        np.testing.assert_array_equal(data, expected)

    def test_close_dataset_reopens(self, node, expected):
        """The dataset is reopened after close_dataset."""
        node.dataset
        node.close_dataset()
        data = node.eval(node.coordinates).data
        np.testing.assert_array_equal(data, expected)
