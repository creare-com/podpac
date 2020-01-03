import os.path
from collections import OrderedDict

import numpy as np
import rasterio
import pytest
from traitlets import TraitError

from podpac.core.coordinates import Coordinates
from podpac.core.units import UnitsDataArray
from podpac.core.data.file import Rasterio


class MockRasterio(Rasterio):
    """mock rasterio data source """

    source = os.path.join(os.path.dirname(__file__), "assets/RGB.byte.tif")
    band = 1

    def get_native_coordinates(self):
        return self.native_coordinates


class TestRasterio(object):
    """test rasterio data source"""

    source = os.path.join(os.path.dirname(__file__), "assets/RGB.byte.tif")
    band = 1

    def test_init(self):
        """test basic init of class"""

        node = Rasterio(source=self.source, band=self.band)
        assert isinstance(node, Rasterio)

        node = MockRasterio()
        assert isinstance(node, MockRasterio)

    def test_traits(self):
        """ check each of the rasterio traits """

        with pytest.raises(TraitError):
            Rasterio(source=5, band=self.band)

        with pytest.raises(TraitError):
            Rasterio(source=self.source, band="test")

    def test_dataset(self):
        """test dataset attribute and trait default """

        node = Rasterio(source=self.source, band=self.band)
        try:
            RasterReader = rasterio._io.RasterReader  # Rasterio < v1.0
        except:
            RasterReader = rasterio.io.DatasetReader  # Rasterio >= v1.0
        assert isinstance(node.dataset, RasterReader)

        node.close_dataset()

    def test_default_native_coordinates(self):
        """test default native coordinates implementations"""

        node = Rasterio(source=self.source)
        native_coordinates = node.get_native_coordinates()
        assert isinstance(native_coordinates, Coordinates)
        assert len(native_coordinates["lat"]) == 718

    def test_get_data(self):
        """test default get_data method"""

        node = Rasterio(source=self.source)
        native_coordinates = node.get_native_coordinates()
        output = node.eval(native_coordinates)

        assert isinstance(output, UnitsDataArray)

    def test_band_descriptions(self):
        """test band count method"""
        node = Rasterio(source=self.source)
        bands = node.band_descriptions
        assert bands and isinstance(bands, OrderedDict)

    def test_band_count(self):
        """test band descriptions methods"""
        node = Rasterio(source=self.source)
        count = node.band_count
        assert count and isinstance(count, int)

    def test_band_keys(self):
        """test band keys methods"""
        node = Rasterio(source=self.source)
        keys = node.band_keys
        assert keys and isinstance(keys, dict)

    def test_get_band_numbers(self):
        """test band numbers methods"""
        node = Rasterio(source=self.source)
        numbers = node.get_band_numbers("STATISTICS_MINIMUM", "0")
        assert isinstance(numbers, np.ndarray)
        np.testing.assert_array_equal(numbers, np.arange(3) + 1)
