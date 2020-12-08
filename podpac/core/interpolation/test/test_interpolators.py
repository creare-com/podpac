"""
Test interpolation methods


"""
# pylint: disable=C0111,W0212,R0903

import pytest
import traitlets as tl
import numpy as np

from podpac.core.utils import ArrayTrait
from podpac.core.units import UnitsDataArray
from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.data.rasterio_source import rasterio
from podpac.core.data.datasource import DataSource
from podpac.core.interpolation.interpolation import Interpolate
from podpac.core.interpolation.interpolation_manager import InterpolationManager, InterpolationException
from podpac.core.interpolation.nearest_neighbor_interpolator import NearestNeighbor, NearestPreview
from podpac.core.interpolation.rasterio_interpolator import RasterioInterpolator
from podpac.core.interpolation.scipy_interpolator import ScipyGrid, ScipyPoint
from podpac.core.interpolation.xarray_interpolator import XarrayInterpolator


class MockArrayDataSource(DataSource):
    cache_output = False
    data = ArrayTrait().tag(attr=True)
    coordinates = tl.Instance(Coordinates).tag(attr=True)

    def get_data(self, coordinates, coordinates_index):
        return self.create_output_array(coordinates, data=self.data[coordinates_index])


class TestNearest(object):
    def test_nearest_preview_select(self):

        # test straight ahead functionality
        reqcoords = Coordinates([[-0.5, 1.5, 3.5], [0.5, 2.5, 4.5]], dims=["lat", "lon"])
        srccoords = Coordinates([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dims=["lat", "lon"])

        interp = InterpolationManager("nearest_preview")

        coords, cidx = interp.select_coordinates(srccoords, reqcoords)

        assert len(coords) == len(srccoords) == len(cidx)
        assert len(coords["lat"]) == len(reqcoords["lat"])
        assert len(coords["lon"]) == len(reqcoords["lon"])
        assert np.all(coords["lat"].coordinates == np.array([0, 2, 4]))

        # test when selection is applied serially
        # this is equivalent to above
        reqcoords = Coordinates([[-0.5, 1.5, 3.5], [0.5, 2.5, 4.5]], dims=["lat", "lon"])
        srccoords = Coordinates([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dims=["lat", "lon"])

        interp = InterpolationManager(
            [{"method": "nearest_preview", "dims": ["lat"]}, {"method": "nearest_preview", "dims": ["lon"]}]
        )

        coords, cidx = interp.select_coordinates(srccoords, reqcoords)

        # test when coordinates are stacked and unstacked
        # TODO: how to handle stacked/unstacked coordinate asynchrony?
        # reqcoords = Coordinates([[-.5, 1.5, 3.5], [.5, 2.5, 4.5]], dims=['lat', 'lon'])
        # srccoords = Coordinates([([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])], dims=['lat_lon'])

        # interp = InterpolationManager('nearest_preview')

        # srccoords, srccoords_index = srccoords.intersect(reqcoords, outer=True, return_index=True)
        # coords, cidx = interp.select_coordinates(reqcoords, srccoords, srccoords_index)

        # assert len(coords) == len(srcoords) == len(cidx)
        # assert len(coords['lat']) == len(reqcoords['lat'])
        # assert len(coords['lon']) == len(reqcoords['lon'])
        # assert np.all(coords['lat'].coordinates == np.array([0, 2, 4]))

    def test_nearest_select_issue226(self):
        reqcoords = Coordinates([[-0.5, 1.5, 3.5], [0.5, 2.5, 4.5]], dims=["lat", "lon"])
        srccoords = Coordinates([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dims=["lat", "lon"])

        interp = InterpolationManager("nearest")

        coords, cidx = interp.select_coordinates(srccoords, reqcoords)

        assert len(coords) == len(srccoords) == len(cidx)
        assert len(coords["lat"]) == len(reqcoords["lat"])
        assert len(coords["lon"]) == len(reqcoords["lon"])
        assert np.all(coords["lat"].coordinates == np.array([0, 2, 4]))

        # test when selection is applied serially
        # this is equivalent to above
        reqcoords = Coordinates([[-0.5, 1.5, 3.5], [0.5, 2.5, 4.5]], dims=["lat", "lon"])
        srccoords = Coordinates([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dims=["lat", "lon"])

        interp = InterpolationManager([{"method": "nearest", "dims": ["lat"]}, {"method": "nearest", "dims": ["lon"]}])

        coords, cidx = interp.select_coordinates(srccoords, reqcoords)

        assert len(coords) == len(srccoords) == len(cidx)
        assert len(coords["lat"]) == len(reqcoords["lat"])
        assert len(coords["lon"]) == len(reqcoords["lon"])
        assert np.all(coords["lat"].coordinates == np.array([0, 2, 4]))

    def test_nearest_interpolation(self):
        interpolation = "nearest"

        # unstacked 1D
        data = np.random.rand(5)
        coords_src = Coordinates([np.linspace(0, 10, 5)], dims=["lat"])
        coords_dst = Coordinates([[1, 1.2, 1.5, 5, 9]], dims=["lat"])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation="nearest")
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert output.values[0] == data[0]
        assert output.values[1] == data[0]
        assert output.values[2] == data[1]

        # unstacked N-D
        data = np.random.rand(5, 5)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=["lat", "lon"])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation=interpolation)
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert output.values[0, 0] == data[1, 1]

        # source = stacked, dest = stacked
        data = np.random.rand(5)
        coords_src = Coordinates([(np.linspace(0, 10, 5), np.linspace(0, 10, 5))], dims=["lat_lon"])
        coords_dst = Coordinates([(np.linspace(1, 9, 3), np.linspace(1, 9, 3))], dims=["lat_lon"])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation=interpolation)
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert all(output.values == data[[0, 2, 4]])

        # source = stacked, dest = unstacked
        data = np.random.rand(5)
        coords_src = Coordinates([(np.linspace(0, 10, 5), np.linspace(0, 10, 5))], dims=["lat_lon"])
        coords_dst = Coordinates([np.linspace(1, 9, 3), np.linspace(1, 9, 3)], dims=["lat", "lon"])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation=interpolation)
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(output.values == data[np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])])

        # source = unstacked, dest = stacked
        data = np.random.rand(5, 5)
        coords_src = Coordinates([np.linspace(0, 10, 5), np.linspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([(np.linspace(1, 9, 3), np.linspace(1, 9, 3))], dims=["lat_lon"])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation=interpolation)
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(output.values == data[[0, 2, 4], [0, 2, 4]])

        # source = unstacked and non-uniform, dest = stacked
        data = np.random.rand(5, 5)
        coords_src = Coordinates([[0, 1.1, 1.2, 6.1, 10], [0, 1.1, 4, 7.1, 9.9]], dims=["lat", "lon"])
        coords_dst = Coordinates([(np.linspace(1, 9, 3), np.linspace(1, 9, 3))], dims=["lat_lon"])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation=interpolation)
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(output.values == data[[1, 3, 4], [1, 2, 4]])

        # lat_lon_time_alt --> lon, alt_time, lat
        data = np.random.rand(5)
        coords_src = Coordinates([[[0, 1, 2, 3, 4]] * 4], dims=[["lat", "lon", "time", "alt"]])
        coords_dst = Coordinates(
            [[1, 2.4, 3.9], [[1, 2.4, 3.9], [1, 2.4, 3.9]], [1, 2.4, 3.9]], dims=["lon", "alt_time", "lat"]
        )
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation=interpolation)
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(output.values[[0, 1, 2], [0, 1, 2], [0, 1, 2]] == data[[1, 2, 4]])

    def test_nearest_preview_interpolation(self):
        interpolation = "nearest_preview"

        # unstacked 1D
        data = np.random.rand(5)
        coords_src = Coordinates([np.linspace(0, 10, 5)], dims=["lat"])
        coords_dst = Coordinates([[1, 1.2, 1.5, 5, 9]], dims=["lat"])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation=interpolation)
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert output.values[0] == data[0]
        assert output.values[1] == data[0]
        assert output.values[2] == data[1]

        # unstacked N-D
        data = np.random.rand(5, 5)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=["lat", "lon"])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation=interpolation)
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert output.values[0, 0] == data[1, 1]

        # source = unstacked, dest = stacked
        data = np.random.rand(5, 5)
        coords_src = Coordinates([np.linspace(0, 10, 5), np.linspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([(np.linspace(1, 9, 3), np.linspace(1, 9, 3))], dims=["lat_lon"])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation=interpolation)
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(output.values == data[[0, 2, 4], [0, 2, 4]])

        # source = unstacked and non-uniform, dest = stacked
        data = np.random.rand(5, 5)
        coords_src = Coordinates([[0, 1.1, 1.2, 6.1, 10], [0, 1.1, 4, 7.1, 9.9]], dims=["lat", "lon"])
        coords_dst = Coordinates([(np.linspace(1, 9, 3), np.linspace(1, 9, 3))], dims=["lat_lon"])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation=interpolation)
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(output.values == data[[1, 3, 4], [1, 2, 4]])

    def test_spatial_tolerance(self):
        # unstacked 1D
        data = np.random.rand(5)
        coords_src = Coordinates([np.linspace(0, 10, 5)], dims=["lat"])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "nearest", "params": {"spatial_tolerance": 1.1}})

        coords_dst = Coordinates([[1, 1.2, 1.5, 5, 9]], dims=["lat"])
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert output.values[0] == data[0]
        assert np.isnan(output.values[1])
        assert output.values[2] == data[1]

        # stacked 1D
        data = np.random.rand(5)
        coords_src = Coordinates([[np.linspace(0, 10, 5), np.linspace(0, 10, 5)]], dims=[["lat", "lon"]])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "nearest", "params": {"spatial_tolerance": 1.1}})

        coords_dst = Coordinates([[[1, 1.2, 1.5, 5, 9], [1, 1.2, 1.5, 5, 9]]], dims=[["lat", "lon"]])
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert output.values[0] == data[0]
        assert np.isnan(output.values[1])
        assert output.values[2] == data[1]

    def test_time_tolerance(self):
        # unstacked 1D
        data = np.random.rand(5, 5)
        coords_src = Coordinates(
            [np.linspace(0, 10, 5), clinspace("2018-01-01", "2018-01-09", 5)], dims=["lat", "time"]
        )
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(
            source=source,
            interpolation={
                "method": "nearest",
                "params": {"spatial_tolerance": 1.1, "time_tolerance": np.timedelta64(1, "D")},
            },
        )

        coords_dst = Coordinates([[1, 1.2, 1.5, 5, 9], clinspace("2018-01-01", "2018-01-09", 3)], dims=["lat", "time"])
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert output.values[0, 0] == data[0, 0]
        assert output.values[0, 1] == data[0, 2]
        assert np.isnan(output.values[1, 0])
        assert np.isnan(output.values[1, 1])
        assert output.values[2, 0] == data[1, 0]
        assert output.values[2, 1] == data[1, 2]

    def test_stacked_source_unstacked_region_non_square(self):
        # unstacked 1D
        data = np.random.rand(5)
        coords_src = Coordinates(
            [[np.linspace(0, 10, 5), clinspace("2018-01-01", "2018-01-09", 5)]], dims=[["lat", "time"]]
        )
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "nearest", "interpolators": [NearestNeighbor]})
        coords_dst = Coordinates([[1, 1.2, 1.5, 5, 9], clinspace("2018-01-01", "2018-01-09", 3)], dims=["lat", "time"])
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(output.values == data[np.array([[0, 2, 4]] * 5)])

    def test_time_space_scale_grid(self):
        # Grid
        data = np.random.rand(5, 3, 2)
        data[2, 1, 0] = np.nan
        coords_src = Coordinates(
            [np.linspace(0, 10, 5), ["2018-01-01", "2018-01-02", "2018-01-03"], [0, 10]], dims=["lat", "time", "alt"]
        )
        coords_dst = Coordinates([5.1, "2018-01-02T11", 1], dims=["lat", "time", "alt"])

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(
            source=source,
            interpolation={
                "method": "nearest",
                "interpolators": [NearestNeighbor],
                "params": {
                    "spatial_scale": 1,
                    "time_scale": "1,D",
                    "alt_scale": 10,
                    "remove_nan": True,
                    "use_selector": False,
                },
            },
        )
        output = node.eval(coords_dst)
        assert output == data[2, 2, 0]

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(
            source=source,
            interpolation={
                "method": "nearest",
                "interpolators": [NearestNeighbor],
                "params": {
                    "spatial_scale": 1,
                    "time_scale": "1,s",
                    "alt_scale": 10,
                    "remove_nan": True,
                    "use_selector": False,
                },
            },
        )
        output = node.eval(coords_dst)
        assert output == data[2, 1, 1]

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(
            source=source,
            interpolation={
                "method": "nearest",
                "interpolators": [NearestNeighbor],
                "params": {
                    "spatial_scale": 1,
                    "time_scale": "1,s",
                    "alt_scale": 1,
                    "remove_nan": True,
                    "use_selector": False,
                },
            },
        )
        output = node.eval(coords_dst)
        assert output == data[3, 1, 0]

    def test_remove_nan(self):
        # Stacked
        data = np.random.rand(5)
        data[2] = np.nan
        coords_src = Coordinates(
            [[np.linspace(0, 10, 5), clinspace("2018-01-01", "2018-01-09", 5)]], dims=[["lat", "time"]]
        )
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(
            source=source,
            interpolation={"method": "nearest", "interpolators": [NearestNeighbor], "params": {"remove_nan": False}},
        )
        coords_dst = Coordinates([[5.1]], dims=["lat"])
        output = node.eval(coords_dst)
        assert np.isnan(output)

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(
            source=source,
            interpolation={
                "method": "nearest",
                "interpolators": [NearestNeighbor],
                "params": {"remove_nan": True, "use_selector": False},
            },
        )
        output = node.eval(coords_dst)
        # This fails because the selector selects the nan value... can we turn off the selector?
        assert output == data[3]

        # Grid
        data = np.random.rand(5, 3)
        data[2, 1] = np.nan
        coords_src = Coordinates([np.linspace(0, 10, 5), [1, 2, 3]], dims=["lat", "time"])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(
            source=source,
            interpolation={"method": "nearest", "interpolators": [NearestNeighbor], "params": {"remove_nan": False}},
        )
        coords_dst = Coordinates([5.1, 2.01], dims=["lat", "time"])
        output = node.eval(coords_dst)
        assert np.isnan(output)

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(
            source=source,
            interpolation={
                "method": "nearest",
                "interpolators": [NearestNeighbor],
                "params": {"remove_nan": True, "use_selector": False},
            },
        )
        output = node.eval(coords_dst)
        assert output == data[2, 2]

    def test_respect_bounds(self):
        data = np.random.rand(5)
        coords_src = Coordinates([[1, 2, 3, 4, 5]], ["alt"])
        coords_dst = Coordinates([[-0.5, 1.1, 2.6]], ["alt"])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(
            source=source,
            interpolation={
                "method": "nearest",
                "interpolators": [NearestNeighbor],
                "params": {"respect_bounds": False},
            },
        )
        output = node.eval(coords_dst)
        np.testing.assert_array_equal(output.data, data[[0, 0, 2]])

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(
            source=source,
            interpolation={"method": "nearest", "interpolators": [NearestNeighbor], "params": {"respect_bounds": True}},
        )
        output = node.eval(coords_dst)
        np.testing.assert_array_equal(output.data[1:], data[[0, 2]])
        assert np.isnan(output.data[0])


class TestInterpolateRasterioInterpolator(object):
    """test interpolation functions"""

    def test_interpolate_rasterio(self):
        """ regular interpolation using rasterio"""

        assert rasterio is not None

        data = np.arange(0, 15)
        data.resize((3, 5))

        coords_src = Coordinates([clinspace(0, 10, 3), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(1, 11, 3), clinspace(1, 11, 5)], dims=["lat", "lon"])

        # try one specific rasterio case to measure output
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "min", "interpolators": [RasterioInterpolator]})
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert output.data[0, 3] == 3.0
        assert output.data[0, 4] == 4.0

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "max", "interpolators": [RasterioInterpolator]})
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert output.data[0, 3] == 9.0
        assert output.data[0, 4] == 9.0

        # TODO boundary should be able to use a default
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(
            source=source,
            interpolation={"method": "bilinear", "interpolators": [RasterioInterpolator]},
            boundary={"lat": 2.5, "lon": 1.25},
        )
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        np.testing.assert_allclose(
            output, [[1.4, 2.4, 3.4, 4.4, 5.0], [6.4, 7.4, 8.4, 9.4, 10.0], [10.4, 11.4, 12.4, 13.4, 14.0]]
        )

    def test_interpolate_rasterio_descending(self):
        """should handle descending"""

        data = np.random.rand(5, 5)
        coords_src = Coordinates([clinspace(10, 0, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=["lat", "lon"])

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "nearest", "interpolators": [RasterioInterpolator]})
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(output.lon.values == coords_dst["lon"].coordinates)


class TestInterpolateScipyGrid(object):
    """test interpolation functions"""

    def test_interpolate_scipy_grid(self):

        data = np.arange(0, 25)
        data.resize((5, 5))

        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(1, 11, 5), clinspace(1, 11, 5)], dims=["lat", "lon"])

        # try one specific rasterio case to measure output
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "nearest", "interpolators": [ScipyGrid]})
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        print(output)
        assert output.data[0, 0] == 0.0
        assert output.data[0, 3] == 3.0
        assert output.data[1, 3] == 8.0
        assert np.isnan(output.data[0, 4])  # TODO: how to handle outside bounds

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "cubic_spline", "interpolators": [ScipyGrid]})
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert int(output.data[0, 0]) == 2
        assert int(output.data[2, 4]) == 16

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "bilinear", "interpolators": [ScipyGrid]})
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert int(output.data[0, 0]) == 2
        assert int(output.data[3, 3]) == 20
        assert np.isnan(output.data[4, 4])  # TODO: how to handle outside bounds

    def test_interpolate_irregular_arbitrary_2dims(self):
        """ irregular interpolation """

        # Note, this test also tests the looper helper

        # try >2 dims
        data = np.random.rand(5, 5, 3)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5), [2, 3, 5]], dims=["lat", "lon", "time"])
        coords_dst = Coordinates([clinspace(1, 11, 5), clinspace(1, 11, 5), [2, 3, 4]], dims=["lat", "lon", "time"])

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(
            source=source,
            interpolation=[{"method": "nearest", "interpolators": [ScipyGrid]}, {"method": "linear", "dims": ["time"]}],
        )
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(output.lon.values == coords_dst["lon"].coordinates)
        assert np.all(output.time.values == coords_dst["time"].coordinates)

        # assert output.data[0, 0] == data[]

    def test_interpolate_looper_helper(self):
        """ irregular interpolation """

        # Note, this test also tests the looper helper

        # try >2 dims
        data = np.random.rand(5, 5, 3, 2)
        result = data.copy()
        result[:, :, 2, :] = (result[:, :, 1, :] + result[:, :, 2, :]) / 2
        result = (result[..., 0:1] + result[..., 1:]) / 2
        result = result[[0, 1, 2, 3, 4]]
        result = result[:, [0, 1, 2, 3, 4]]
        result[-1] = np.nan
        result[:, -1] = np.nan
        coords_src = Coordinates(
            [clinspace(0, 10, 5), clinspace(0, 10, 5), [2, 3, 5], [0, 2]], dims=["lat", "lon", "time", "alt"]
        )
        coords_dst = Coordinates(
            [clinspace(1, 11, 5), clinspace(1, 11, 5), [2, 3, 4], [1]], dims=["lat", "lon", "time", "alt"]
        )

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(
            source=source,
            interpolation=[
                {"method": "nearest", "interpolators": [ScipyGrid]},
                {"method": "linear", "dims": ["time", "alt"]},
            ],
        )
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(output.lon.values == coords_dst["lon"].coordinates)
        assert np.all(output.time.values == coords_dst["time"].coordinates)
        assert np.all(output.alt.values == coords_dst["alt"].coordinates)
        np.testing.assert_array_almost_equal(result, output.data)

    def test_interpolate_irregular_arbitrary_descending(self):
        """should handle descending"""

        data = np.random.rand(5, 5)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=["lat", "lon"])

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "nearest", "interpolators": [ScipyGrid]})
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        np.testing.assert_array_equal(output.lat.values, coords_dst["lat"].coordinates)
        np.testing.assert_array_equal(output.lon.values, coords_dst["lon"].coordinates)

    def test_interpolate_irregular_arbitrary_swap(self):
        """should handle descending"""

        data = np.random.rand(5, 5)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=["lat", "lon"])

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "nearest", "interpolators": [ScipyGrid]})
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        np.testing.assert_array_equal(output.lat.values, coords_dst["lat"].coordinates)
        np.testing.assert_array_equal(output.lon.values, coords_dst["lon"].coordinates)

    def test_interpolate_irregular_lat_lon(self):
        """ irregular interpolation """

        data = np.random.rand(5, 5)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([[[0, 2, 4, 6, 8, 10], [0, 2, 4, 5, 6, 10]]], dims=["lat_lon"])

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "nearest", "interpolators": [ScipyGrid]})
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert "lat_lon" in output.dims
        np.testing.assert_array_equal(output["lat"].values, coords_dst["lat"].coordinates)
        np.testing.assert_array_equal(output["lon"].values, coords_dst["lon"].coordinates)
        assert output.values[0] == data[0, 0]
        assert output.values[1] == data[1, 1]
        assert output.values[-1] == data[-1, -1]


class TestInterpolateScipyPoint(object):
    def test_interpolate_scipy_point(self):
        """ interpolate point data to nearest neighbor with various coords_dst"""

        data = np.random.rand(6)
        coords_src = Coordinates([[[0, 2, 4, 6, 8, 10], [0, 2, 4, 5, 6, 10]]], dims=["lat_lon"])
        coords_dst = Coordinates([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]], dims=["lat_lon"])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "nearest", "interpolators": [ScipyPoint]})

        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert "lat_lon" in output.dims
        np.testing.assert_array_equal(output.lat.values, coords_dst["lat"].coordinates)
        np.testing.assert_array_equal(output.lon.values, coords_dst["lon"].coordinates)
        assert output.values[0] == data[0]
        assert output.values[-1] == data[3]

        coords_dst = Coordinates([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dims=["lat", "lon"])
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        np.testing.assert_array_equal(output.lat.values, coords_dst["lat"].coordinates)
        assert output.values[0, 0] == data[0]
        assert output.values[-1, -1] == data[3]


class TestXarrayInterpolator(object):
    """test interpolation functions"""

    def test_nearest_interpolation(self):
        interpolation = {
            "method": "nearest",
            "interpolators": [XarrayInterpolator],
            "params": {"fill_value": "extrapolate"},
        }

        # unstacked 1D
        data = np.random.rand(5)
        coords_src = Coordinates([np.linspace(0, 10, 5)], dims=["lat"])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation=interpolation)

        coords_dst = Coordinates([[1, 1.2, 1.5, 5, 9]], dims=["lat"])
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert output.values[0] == data[0]
        assert output.values[1] == data[0]
        assert output.values[2] == data[1]

        # unstacked N-D
        data = np.random.rand(5, 5)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=["lat", "lon"])

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation=interpolation)
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert output.values[0, 0] == data[1, 1]

        # stacked
        # TODO: implement stacked handling
        data = np.random.rand(5)
        coords_src = Coordinates([(np.linspace(0, 10, 5), np.linspace(0, 10, 5))], dims=["lat_lon"])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "nearest", "interpolators": [XarrayInterpolator]})
        coords_dst = Coordinates([(np.linspace(1, 9, 3), np.linspace(1, 9, 3))], dims=["lat_lon"])

        with pytest.raises(InterpolationException):
            output = node.eval(coords_dst)

        # TODO: implement stacked handling
        # source = stacked, dest = unstacked
        data = np.random.rand(5)
        coords_src = Coordinates([(np.linspace(0, 10, 5), np.linspace(0, 10, 5))], dims=["lat_lon"])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "nearest", "interpolators": [XarrayInterpolator]})
        coords_dst = Coordinates([np.linspace(1, 9, 3), np.linspace(1, 9, 3)], dims=["lat", "lon"])

        with pytest.raises(InterpolationException):
            output = node.eval(coords_dst)

        # source = unstacked, dest = stacked
        data = np.random.rand(5, 5)
        coords_src = Coordinates([np.linspace(0, 10, 5), np.linspace(0, 10, 5)], dims=["lat", "lon"])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "nearest", "interpolators": [XarrayInterpolator]})
        coords_dst = Coordinates([(np.linspace(1, 9, 3), np.linspace(1, 9, 3))], dims=["lat_lon"])

        output = node.eval(coords_dst)
        np.testing.assert_array_equal(output.data, data[[0, 2, 4], [0, 2, 4]])

    def test_interpolate_xarray_grid(self):

        data = np.arange(0, 25)
        data.resize((5, 5))

        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(1, 11, 5), clinspace(1, 11, 5)], dims=["lat", "lon"])

        # try one specific rasterio case to measure output
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "nearest", "interpolators": [XarrayInterpolator]})
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        # print(output)
        assert output.data[0, 0] == 0.0
        assert output.data[0, 3] == 3.0
        assert output.data[1, 3] == 8.0
        assert np.isnan(output.data[0, 4])  # TODO: how to handle outside bounds

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(
            source=source,
            interpolation={"method": "linear", "interpolators": [XarrayInterpolator], "params": {"fill_nan": True}},
        )
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert int(output.data[0, 0]) == 2
        assert int(output.data[2, 3]) == 15

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(
            source=source,
            interpolation={"method": "slinear", "interpolators": [XarrayInterpolator], "params": {"fill_nan": True}},
        )
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert int(output.data[0, 0]) == 2
        assert int(output.data[3, 3]) == 20
        assert np.isnan(output.data[4, 4])

        # Check extrapolation
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(
            source=source,
            interpolation={
                "method": "linear",
                "interpolators": [XarrayInterpolator],
                "params": {"fill_nan": True, "fill_value": "extrapolate"},
            },
        )
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert int(output.data[0, 0]) == 2
        assert int(output.data[4, 4]) == 26
        assert np.all(~np.isnan(output.data))

    def test_interpolate_irregular_arbitrary_2dims(self):
        """ irregular interpolation """
        # try >2 dims
        data = np.random.rand(5, 5, 3)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5), [2, 3, 5]], dims=["lat", "lon", "time"])
        coords_dst = Coordinates([clinspace(1, 11, 5), clinspace(1, 11, 5), [2, 3, 5]], dims=["lat", "lon", "time"])

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "nearest", "interpolators": [XarrayInterpolator]})
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(output.lon.values == coords_dst["lon"].coordinates)
        assert np.all(output.time.values == coords_dst["time"].coordinates)

        # assert output.data[0, 0] == data[]

    def test_interpolate_irregular_arbitrary_descending(self):
        """should handle descending"""
        data = np.random.rand(5, 5)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=["lat", "lon"])

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "nearest", "interpolators": [XarrayInterpolator]})
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(output.lon.values == coords_dst["lon"].coordinates)

    def test_interpolate_irregular_arbitrary_swap(self):
        """should handle descending"""

        data = np.random.rand(5, 5)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=["lat", "lon"])

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "nearest", "interpolators": [XarrayInterpolator]})
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(output.lon.values == coords_dst["lon"].coordinates)

    def test_interpolate_irregular_lat_lon(self):
        """ irregular interpolation """

        data = np.random.rand(5, 5)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([[[0, 2, 4, 6, 8, 10], [0, 2, 4, 5, 6, 10]]], dims=["lat_lon"])

        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(source=source, interpolation={"method": "nearest", "interpolators": [XarrayInterpolator]})
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat_lon.values == coords_dst.xcoords["lat_lon"])
        assert output.values[0] == data[0, 0]
        assert output.values[1] == data[1, 1]
        assert output.values[-1] == data[-1, -1]

    def test_interpolate_fill_nan(self):
        data = np.arange(0, 25).astype(float)
        data.resize((5, 5))
        data[2, 2] = np.nan

        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(1, 11, 5), clinspace(1, 11, 5)], dims=["lat", "lon"])

        # Ensure nan present
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(
            source=source,
            interpolation={"method": "linear", "interpolators": [XarrayInterpolator], "params": {"fill_nan": False}},
        )
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(np.isnan(output.data[1:3, 1:3]))

        # Ensure nan gone
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(
            source=source,
            interpolation={"method": "linear", "interpolators": [XarrayInterpolator], "params": {"fill_nan": True}},
        )
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        np.testing.assert_array_almost_equal(output.data[1:3, 1:3].ravel(), [8.4, 9.4, 13.4, 14.4])

        # Ensure nan gone, flip lat-lon on source
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lon", "lat"])
        source = MockArrayDataSource(data=data, coordinates=coords_src)
        node = Interpolate(
            source=source,
            interpolation={"method": "linear", "interpolators": [XarrayInterpolator], "params": {"fill_nan": True}},
        )
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        np.testing.assert_array_almost_equal(output.data[1:3, 1:3].T.ravel(), [8.4, 9.4, 13.4, 14.4])
