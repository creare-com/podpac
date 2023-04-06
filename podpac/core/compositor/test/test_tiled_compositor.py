import pytest
import numpy as np

import podpac
from podpac.core.data.array_source import Array
from podpac.core.compositor.tile_compositor import TileCompositor


class TestTileCompositor(object):
    def test_composition(self):
        a = Array(source=np.arange(5) + 100, coordinates=podpac.Coordinates([[0, 1, 2, 3, 4]], dims=["lat"]))
        b = Array(source=np.arange(5) + 200, coordinates=podpac.Coordinates([[5, 6, 7, 8, 9]], dims=["lat"]))
        c = Array(source=np.arange(5) + 300, coordinates=podpac.Coordinates([[10, 11, 12, 13, 14]], dims=["lat"]))

        node = TileCompositor(sources=[a, b, c])

        output = node.eval(podpac.Coordinates([[3.5, 4.5, 5.5]], dims=["lat"]))
        np.testing.assert_array_equal(output["lat"], [3, 4, 5, 6])
        np.testing.assert_array_equal(output, [103, 104, 200, 201])

    def test_interpolation(self):
        a = Array(source=np.arange(5) + 100, coordinates=podpac.Coordinates([[0, 1, 2, 3, 4]], dims=["lat"]))
        b = Array(source=np.arange(5) + 200, coordinates=podpac.Coordinates([[5, 6, 7, 8, 9]], dims=["lat"]))
        c = Array(source=np.arange(5) + 300, coordinates=podpac.Coordinates([[10, 11, 12, 13, 14]], dims=["lat"]))

        node = TileCompositor(sources=[a, b, c], interpolation="bilinear")

        output = node.eval(podpac.Coordinates([[3.5, 4.5, 5.5]], dims=["lat"]))
        np.testing.assert_array_equal(output["lat"], [3.5, 4.5, 5.5])
        np.testing.assert_array_equal(output, [103.5, 152.0, 200.5])

    def test_composition_stacked_multiindex_names(self):
        a = Array(
            source=np.arange(5) + 100,
            coordinates=podpac.Coordinates([[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]], dims=["lat_lon"]),
        )
        b = Array(
            source=np.arange(5) + 200,
            coordinates=podpac.Coordinates([[[5, 6, 7, 8, 9], [5, 6, 7, 8, 9]]], dims=["lat_lon"]),
        )

        node = TileCompositor(sources=[a, b])

        output = node.eval(podpac.Coordinates([[[3, 4, 5, 6], [3, 4, 5, 6]]], dims=["lat_lon"]))

        # this is checking that the 'lat' and 'lon' multiindex names are still there
        np.testing.assert_array_equal(output["lat"], [3, 4, 5, 6])
        np.testing.assert_array_equal(output["lon"], [3, 4, 5, 6])
        np.testing.assert_array_equal(output, [103, 104, 200, 201])

    def test_get_source_data(self):
        a = Array(source=np.arange(5) + 100, coordinates=podpac.Coordinates([[0, 1, 2, 3, 4]], dims=["lat"]))
        b = Array(source=np.arange(5) + 200, coordinates=podpac.Coordinates([[5, 6, 7, 8, 9]], dims=["lat"]))
        c = Array(source=np.arange(5) + 300, coordinates=podpac.Coordinates([[10, 11, 12, 13, 14]], dims=["lat"]))

        node = TileCompositor(sources=[a, b, c])

        data = node.get_source_data()
        np.testing.assert_array_equal(data["lat"], np.arange(15))
        np.testing.assert_array_equal(data, np.hstack([source.source for source in node.sources]))

        # with bounds
        data = node.get_source_data({"lat": (2.5, 6.5)})
        np.testing.assert_array_equal(data["lat"], [3, 4, 5, 6])
        np.testing.assert_array_equal(data, [103, 104, 200, 201])

        # error
        with podpac.settings:
            podpac.settings.set_unsafe_eval(True)
            d = podpac.algorithm.Arithmetic(eqn="a+2", a=a)
        node = TileCompositor(sources=[a, b, c, d])
        with pytest.raises(ValueError, match="Cannot get composited source data"):
            node.get_source_data()
