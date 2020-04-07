import itertools

import pytest
import numpy as np
import traitlets as tl

import podpac
from podpac.utils import cached_property
from podpac.data import DataSource
from podpac.core.compositor.tile_compositor import TileCompositor, UniformTileCompositor, UniformTileMixin


class MockTile(UniformTileMixin, podpac.data.DataSource):
    x = tl.Int()  # used as a modifier to distinguish between tiles in the tests
    data = np.arange(16).reshape(1, 4, 4)

    def get_data(self, coordinates, coordinates_index):
        return self.create_output_array(coordinates, data=self.data[coordinates_index] + self.x)


class MockTileCompositor(UniformTileCompositor):
    shape = (3, 3, 3)

    @cached_property
    def sources(self):
        return [
            MockTile(tile=(i, j, k), grid=self, x=20 * n)
            for n, (i, j, k) in enumerate(itertools.product(range(3), range(3), range(3)))
        ]

    def get_native_coordinates(self):
        return podpac.Coordinates(
            [["2018-01-01", "2018-01-03", "2018-01-05"], podpac.clinspace(0, 11, 12), podpac.clinspace(0, 11, 12)],
            dims=["time", "lat", "lon"],
        )


class TestUniformTileMixin(object):
    def test_tile_coordinates_index(self):
        class MyTile(UniformTileMixin, DataSource):
            pass

        grid = MockTileCompositor()
        tile = MyTile(grid=grid, tile=(1, 1, 0))

        assert tile.width == grid.tile_width
        assert tile.native_coordinates == podpac.Coordinates(
            ["2018-01-03", podpac.clinspace(4, 7, 4), podpac.clinspace(0, 3, 4)], dims=["time", "lat", "lon"]
        )


class TestTileCompositor(object):
    def test_sources(self):
        node = TileCompositor()
        with pytest.raises(NotImplementedError):
            node.sources

        node = MockTileCompositor()
        assert len(node.sources) == 27
        assert all(isinstance(tile, MockTile) for tile in node.sources)

    def test_native_coordinates(self):
        node = TileCompositor()
        with pytest.raises(NotImplementedError):
            node.native_coordinates

        node = MockTileCompositor()
        assert node.native_coordinates == podpac.Coordinates(
            [["2018-01-01", "2018-01-03", "2018-01-05"], podpac.clinspace(0, 11, 12), podpac.clinspace(0, 11, 12)],
            dims=["time", "lat", "lon"],
        )


class TestUniformTileCompositor(object):
    def test_tile_width(self):
        node = MockTileCompositor()
        assert node.tile_width == (1, 4, 4)

    def test_get_data_native_coordinates(self):
        node = MockTileCompositor()

        # all native_coordinates
        output = node.eval(node.native_coordinates)
        assert np.all(np.isfinite(output))
        np.testing.assert_array_equal(output[0, :4, :4], np.arange(16).reshape(4, 4) + 0)
        np.testing.assert_array_equal(output[0, :4, 4:8], np.arange(16).reshape(4, 4) + 20)
        np.testing.assert_array_equal(output[0, 4:8, :4], np.arange(16).reshape(4, 4) + 60)
        np.testing.assert_array_equal(output[1, :4, :4], np.arange(16).reshape(4, 4) + 180)

        # single point
        output = node.eval(node.native_coordinates[2, 2, 2])
        np.testing.assert_array_equal(output, [[[370]]])

        # partial tiles
        output = node.eval(node.native_coordinates[1, 2:6, 2:4])
        np.testing.assert_array_equal(output, [[[190, 191], [194, 195], [242, 243], [246, 247]]])

    def test_get_data_spatial_interpolation(self):
        # exact times, interpolated lat/lon
        c1 = podpac.Coordinates(["2018-01-01", [0.25, 0.75, 1.25], [0.25, 0.75, 1.25]], dims=["time", "lat", "lon"])
        c2 = podpac.Coordinates(["2018-01-03", [0.25, 0.75, 1.25], [0.25, 0.75, 1.25]], dims=["time", "lat", "lon"])

        node = MockTileCompositor(interpolation="nearest")
        np.testing.assert_array_equal(node.eval(c1), [[[0, 1, 1], [4, 5, 5], [4, 5, 5]]])
        np.testing.assert_array_equal(node.eval(c2), [[[180, 181, 181], [184, 185, 185], [184, 185, 185]]])

        node = MockTileCompositor(interpolation="bilinear")
        np.testing.assert_array_equal(node.eval(c1), [[[1.25, 1.75, 2.25], [3.25, 3.75, 4.25], [5.25, 5.75, 6.25]]])
        np.testing.assert_array_equal(
            node.eval(c2), [[[181.25, 181.75, 182.25], [183.25, 183.75, 184.25], [185.25, 185.75, 186.25]]]
        )

    def test_get_data_time_interpolation(self):
        # exact lat/lon, interpolated times
        c1 = podpac.Coordinates(["2018-01-01T01:00:00", [1, 2], [1, 2]], dims=["time", "lat", "lon"])
        c2 = podpac.Coordinates(["2018-01-02T23:00:00", [1, 2], [1, 2]], dims=["time", "lat", "lon"])
        c3 = podpac.Coordinates(["2018-01-03T01:00:00", [1, 2], [1, 2]], dims=["time", "lat", "lon"])

        node = MockTileCompositor(interpolation="nearest")
        np.testing.assert_array_equal(node.eval(c1), [[[5, 6], [9, 10]]])
        np.testing.assert_array_equal(node.eval(c2), [[[185, 186], [189, 190]]])
        np.testing.assert_array_equal(node.eval(c3), [[[185, 186], [189, 190]]])

        # TODO
        # node = MockTileCompositor(interpolation='bilinear')
        # np.testing.assert_array_equal(node.eval(c1), TODO)
        # np.testing.assert_array_equal(node.eval(c2), TODO)
        # np.testing.assert_array_equal(node.eval(c3), TODO)

    def test_get_data_interpolation(self):
        # interpolated lat/lon and time
        c1 = podpac.Coordinates(
            ["2018-01-01T01:00:00", [0.25, 0.75, 1.25], [0.25, 0.75, 1.25]], dims=["time", "lat", "lon"]
        )
        c2 = podpac.Coordinates(
            ["2018-01-02T23:00:00", [0.25, 0.75, 1.25], [0.25, 0.75, 1.25]], dims=["time", "lat", "lon"]
        )
        c3 = podpac.Coordinates(
            ["2018-01-03T01:00:00", [0.25, 0.75, 1.25], [0.25, 0.75, 1.25]], dims=["time", "lat", "lon"]
        )

        node = MockTileCompositor(interpolation="nearest")
        np.testing.assert_array_equal(node.eval(c1), [[[0, 1, 1], [4, 5, 5], [4, 5, 5]]])
        np.testing.assert_array_equal(node.eval(c2), [[[180, 181, 181], [184, 185, 185], [184, 185, 185]]])
        np.testing.assert_array_equal(node.eval(c3), [[[180, 181, 181], [184, 185, 185], [184, 185, 185]]])

        # TODO
        # node = MockTileCompositor(interpolation='bilinear')
        # np.testing.assert_array_equal(node.eval(c1), TODO)
        # np.testing.assert_array_equal(node.eval(c2), TODO)
        # np.testing.assert_array_equal(node.eval(c3), TODO)
