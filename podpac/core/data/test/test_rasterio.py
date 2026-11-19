import os.path
from collections import OrderedDict

import numpy as np
import pytest
import rasterio
from unittest.mock import patch
from rasterio.errors import RasterioIOError
from rasterio import transform as RasterioTransform
from rasterio.enums import Resampling
from podpac.core.coordinates import Coordinates, merge_dims
from podpac.core.units import UnitsDataArray
from podpac.core.data.rasterio_source import Rasterio
from podpac.core.interpolation.interpolation import Interpolate
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

    def test_open_s3_env_keys(self, monkeypatch):
        """test GDAL finds env keys"""
        monkeypatch.setenv("GDAL_HTTP_PROXY", "invalid://proxy")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "value")
        source = Rasterio(aws_get_auth_from_env=True)
        # Raises on proxy instead of InvalidCredentials
        with pytest.raises(RasterioIOError, match="proxy"):
            source.open_dataset("s3://bucket/key.tif")

    def test_open_s3_env_token_file(self, monkeypatch, tmp_path):
        """test GDAL resolves the token itself"""
        monkeypatch.setenv("GDAL_HTTP_PROXY", "invalid://proxy")
        token_file = tmp_path / "token"
        token_file.write_text("token")
        monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", str(token_file))
        monkeypatch.setenv("AWS_ROLE_ARN", "arn")
        source = Rasterio(aws_get_auth_from_env=True)
        # Raises a RasterioIOError not a botocore error
        with pytest.raises(RasterioIOError):
            source.open_dataset("s3://bucket/key.tif")


class TestRasterioOverviews(object):
    """test prefer_overviews reads the overview matching the request resolution"""

    @pytest.fixture
    def source(self, tmp_path):
        """raster with averaged overviews, so overview pixels differ from full-resolution pixels"""
        path = str(tmp_path / "source.tif")
        transform = RasterioTransform.from_origin(-180, 90, 360 / 512, 180 / 256)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=256,
            width=512,
            count=1,
            dtype="float64",
            crs="EPSG:4326",
            transform=transform,
            tiled=True,
            blockxsize=128,
            blockysize=128,
        ) as dataset:
            dataset.write(np.arange(256 * 512, dtype=float).reshape(256, 512) ** 1.5, 1)
            dataset.build_overviews([2, 4], Resampling.average)
        return path

    @pytest.fixture
    def overview(self, source, tmp_path):
        """the factor 4 overview of source (level 1), as its own file"""
        path = str(tmp_path / "overview.tif")
        with rasterio.open(source, overview_level=1) as dataset:
            data = dataset.read(1)
            transform = dataset.transform
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype="float64",
            crs="EPSG:4326",
            transform=transform,
        ) as dataset:
            dataset.write(data, 1)
        return path

    @pytest.fixture
    def coarse(self):
        """request about 6x coarser than source, which selects the factor 4 overview (level 1)"""
        return Coordinates([clinspace(80, -80, 40, "lat"), clinspace(-170, 170, 80, "lon")])

    def assert_overview_level(self, node, coordinates, level):
        assert node._overview_level(coordinates.transform(node.coordinates.crs)) == level

    def assert_interpolated_equal(self, node, expected_node, coordinates, interpolation):
        output = Interpolate(source=node, interpolation=interpolation).eval(coordinates)
        expected = Interpolate(source=expected_node, interpolation=interpolation).eval(coordinates)
        np.testing.assert_allclose(output.data, expected.data, rtol=1e-12)

    def test_nearest(self, source, overview, coarse):
        """nearest interpolation reads the overview"""
        self.assert_interpolated_equal(
            Rasterio(source=source, prefer_overviews=True), Rasterio(source=overview), coarse, "nearest"
        )

    def test_bilinear(self, source, overview, coarse):
        """bilinear interpolation reads the overview, not full resolution"""
        self.assert_interpolated_equal(
            Rasterio(source=source, prefer_overviews=True), Rasterio(source=overview), coarse, "bilinear"
        )

    def test_no_interpolation(self, source, overview, coarse):
        """without an interpolator, the node returns the overview pixels within the request bounds"""
        output = Rasterio(source=source, prefer_overviews=True).eval(coarse)
        expected = Rasterio(source=overview).eval(coarse)
        np.testing.assert_array_equal(output.data, expected.data)
        np.testing.assert_allclose(output["lat"].data, expected["lat"].data, rtol=1e-12)
        np.testing.assert_allclose(output["lon"].data, expected["lon"].data, rtol=1e-12)

    def test_no_interpolation_outside(self, source):
        """without an interpolator, a request outside the source returns an empty output"""
        outside = Coordinates([clinspace(80, 70, 40, "lat"), clinspace(200, 210, 40, "lon")])
        output = Rasterio(source=source, prefer_overviews=True).eval(outside)
        assert output.size == 0

    def test_overview_opened_once(self, source, coarse):
        """the chosen overview level is opened once for its grid, then closed"""
        node = Rasterio(source=source, prefer_overviews=True)
        Interpolate(source=node, interpolation="nearest").eval(coarse)
        with patch.object(node, "open_dataset") as open_dataset:
            Interpolate(source=node, interpolation="nearest").eval(coarse)
        open_dataset.assert_not_called()

    def test_evenly_spaced(self, source, overview):
        """a request on every third overview pixel reads those pixels"""
        with rasterio.open(overview) as dataset:
            data = dataset.read(1)
            transform = dataset.transform
        rows, cols = np.arange(0, 60, 3), np.arange(0, 120, 3)
        coordinates = Coordinates(
            [transform.f + transform.e * (rows + 0.5), transform.c + transform.a * (cols + 0.5)], dims=["lat", "lon"]
        )
        output = Interpolate(source=Rasterio(source=source, prefer_overviews=True), interpolation="nearest").eval(
            coordinates
        )
        np.testing.assert_array_equal(output.data, data[np.ix_(rows, cols)])

    def test_round_off_factor(self, source):
        """a request with small epsilon under 4x coarser than the source, picks the factor 4 overview"""
        step = 4 * (180 / 256) * (1 - 1e-12)
        lat = 89 - step * np.arange(40)
        lon = -179 + step * np.arange(80)
        self.assert_overview_level(
            Rasterio(source=source, prefer_overviews=True), Coordinates([lat, lon], dims=["lat", "lon"]), 1
        )

    def test_stacked(self, source):
        """a stacked point request picks the overview from its lat and lon steps"""
        points = Coordinates([[np.linspace(-80, 80, 40), np.linspace(-170, 170, 40)]], dims=["lat_lon"])
        self.assert_overview_level(Rasterio(source=source, prefer_overviews=True), points, 1)

    def test_extra_dim(self, source, overview, coarse):
        """a source with a time dim picks the overview from lat and lon only and keeps time"""

        class TimeRasterio(Rasterio):
            def get_coordinates(self):
                spatial = super().get_coordinates()
                return merge_dims([spatial, Coordinates([["2020-01-01"]], dims=["time"], crs=spatial.crs)])

        request = merge_dims([coarse, Coordinates([["2020-01-01"]], dims=["time"], crs=coarse.crs)])
        node = TimeRasterio(source=source, prefer_overviews=True)
        self.assert_overview_level(node, request, 1)
        self.assert_interpolated_equal(node, TimeRasterio(source=overview), request, "nearest")

    def test_full_resolution(self, source):
        """a full-resolution request reads full resolution"""
        fine = Coordinates([clinspace(45, 44, 3, "lat"), clinspace(10, 11, 3, "lon")])
        self.assert_interpolated_equal(
            Rasterio(source=source, prefer_overviews=True), Rasterio(source=source), fine, "bilinear"
        )
