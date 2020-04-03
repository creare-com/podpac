import os.path
from collections import OrderedDict

import numpy as np
import rasterio
import pytest
from traitlets import TraitError

from podpac.core.coordinates import Coordinates
from podpac.core.units import UnitsDataArray
from podpac.core.data.rasterio_source import Rasterio


class TestRasterio(object):
    """test rasterio data source"""

    source = os.path.join(os.path.dirname(__file__), "assets/RGB.byte.tif")
    band = 1

    def test_init(self):
        """test basic init of class"""

        node = Rasterio(source=self.source, band=self.band)

    def test_dataset(self):
        """test dataset attribute and trait default """

        node = Rasterio(source=self.source, band=self.band)
        try:
            RasterReader = rasterio._io.RasterReader  # Rasterio < v1.0
        except:
            RasterReader = rasterio.io.DatasetReader  # Rasterio >= v1.0
        assert isinstance(node.dataset, RasterReader)

        node.close_dataset()

    def test_native_coordinates(self):
        """test default native coordinates implementations"""

        node = Rasterio(source=self.source)
        assert isinstance(node.native_coordinates, Coordinates)
        assert len(node.native_coordinates["lat"]) == 718

    def test_get_data(self):
        """test default get_data method"""

        node = Rasterio(source=self.source)
        output = node.eval(node.native_coordinates)
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
