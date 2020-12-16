import numpy as np
import pytest

from .coordinates_for_tests import COORDINATES
from podpac.datalib.terraintiles import TerrainTiles, get_tile_urls
from podpac import Coordinates, clinspace


@pytest.mark.integration
class TestTerrainTiles(object):
    def test_common_coordinates(self):
        node = TerrainTiles()
        for ck, c in COORDINATES.items():
            print("Evaluating: ", ck)
            o = node.eval(c)
            assert np.any(np.isfinite(o.data))

    def test_terrain_tiles(self):
        c = Coordinates([clinspace(40, 43, 1000), clinspace(-76, -72, 1000)], dims=["lat", "lon"])
        c2 = Coordinates(
            [clinspace(40, 43, 1000), clinspace(-76, -72, 1000), ["2018-01-01", "2018-01-02"]],
            dims=["lat", "lon", "time"],
        )

        node = TerrainTiles(tile_format="geotiff", zoom=8)
        output = node.eval(c)
        assert np.any(np.isfinite(output))

        output = node.eval(c2)
        assert np.any(np.isfinite(output))

        node = TerrainTiles(tile_format="geotiff", zoom=8, cache_ctrl=["ram", "disk"])
        output = node.eval(c)
        assert np.any(np.isfinite(output))

        # tile urls
        print(np.array(get_tile_urls("geotiff", 1)))
        print(np.array(get_tile_urls("geotiff", 9, coordinates=c)))
