"""
Test interpolation methods


"""
# pylint: disable=C0111,W0212,R0903

import pytest
import traitlets as tl
import numpy as np

import podpac
from podpac.core.utils import ArrayTrait
from podpac.core.units import UnitsDataArray
from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.data.rasterio_source import rasterio
from podpac.core.data.datasource import DataSource
from podpac.core.interpolation.interpolation_manager import InterpolationManager, InterpolationException
from podpac.core.interpolation.nearest_neighbor_interpolator import NearestNeighbor, NearestPreview
from podpac.core.interpolation.rasterio_interpolator import RasterioInterpolator
from podpac.core.interpolation.scipy_interpolator import ScipyGrid, ScipyPoint
from podpac.core.interpolation.xarray_interpolator import XarrayInterpolator


class MockArrayDataSource(DataSource):
    data = ArrayTrait().tag(attr=True)
    coordinates = tl.Instance(Coordinates).tag(attr=True)

    def get_data(self, coordinates, coordinates_index):
        return self.create_output_array(coordinates, data=self.data[coordinates_index]).interpolate()


class MockArrayDataSourceXR(DataSource):
    data = ArrayTrait().tag(attr=True)
    coordinates = tl.Instance(Coordinates).tag(attr=True)

    def get_data(self, coordinates, coordinates_index):
        dataxr = self.create_output_array(self.coordinates, data=self.data)
        return self.create_output_array(coordinates, data=dataxr[coordinates_index].data).interpolate()


class TestNone(object):
    def test_none_select(self):
        reqcoords = Coordinates([[-0.5, 1.5, 3.5], [0.5, 2.5, 4.5]], dims=["lat", "lon"])
        srccoords = Coordinates([[-1, 0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dims=["lat", "lon"])

        # test straight ahead functionality
        interp = InterpolationManager("none")
        coords, cidx = interp.select_coordinates(srccoords, reqcoords)
        assert coords == srccoords[1:5, 1:-1]
        assert srccoords[cidx] == coords

        # test when selection is applied serially
        interp = InterpolationManager([{"method": "none", "dims": ["lat"]}, {"method": "none", "dims": ["lon"]}])

        coords, cidx = interp.select_coordinates(srccoords, reqcoords)
        assert coords == srccoords[1:5, 1:-1]
        assert srccoords[cidx] == coords

        # Test Case where rounding issues causes problem with endpoint
        reqcoords = Coordinates([[0, 2, 4], [0, 2, 4]], dims=["lat", "lon"])
        lat = np.arange(0, 6.1, 1.3333333333333334)
        lon = np.arange(0, 6.1, 1.333333333333334)  # Notice one decimal less on this number
        srccoords = Coordinates([lat, lon], dims=["lat", "lon"])

        # test straight ahead functionality
        interp = InterpolationManager("none")
        coords, cidx = interp.select_coordinates(srccoords, reqcoords)
        srccoords = Coordinates([lat, lon], dims=["lat", "lon"])
        assert srccoords[cidx] == coords

    def test_none_interpolation(self):
        node = podpac.data.Array(
            source=[0, 1, 2],
            coordinates=podpac.Coordinates([[1, 5, 9]], dims=["lat"]),
            interpolation="none",
        )
        o = node.eval(podpac.Coordinates([podpac.crange(1, 9, 1)], dims=["lat"]))
        np.testing.assert_array_equal(o.data, node.source)

    def test_none_heterogeneous(self):
        # Heterogeneous
        node = podpac.data.Array(
            source=[[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
            coordinates=podpac.Coordinates([[1, 5, 9, 13], [0, 1, 2]], dims=["lat", "lon"]),
            interpolation=[{"method": "none", "dims": ["lat"]}, {"method": "linear", "dims": ["lon"]}],
        )
        o = node.eval(podpac.Coordinates([podpac.crange(1, 9, 2), [0.5, 1.5]], dims=["lat", "lon"]))
        np.testing.assert_array_equal(
            o.data,
            [
                [0.5, 1.5],
                [
                    0.5,
                    1.5,
                ],
                [0.5, 1.5],
            ],
        )

        # Heterogeneous _flipped
        node = podpac.data.Array(
            source=[[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
            coordinates=podpac.Coordinates([[1, 5, 9, 13], [0, 1, 2]], dims=["lat", "lon"]),
            interpolation=[{"method": "linear", "dims": ["lon"]}, {"method": "none", "dims": ["lat"]}],
        )
        o = node.eval(podpac.Coordinates([podpac.crange(1, 9, 2), [0.5, 1.5]], dims=["lat", "lon"]))
        np.testing.assert_array_equal(
            o.data,
            [
                [0.5, 1.5],
                [
                    0.5,
                    1.5,
                ],
                [0.5, 1.5],
            ],
        )

        # Examples
        #  source                      eval
        #  lat_lon                     lat, lon
        node = podpac.data.Array(
            source=[0, 1, 2],
            coordinates=podpac.Coordinates([[[1, 5, 9], [1, 5, 9]]], dims=[["lat", "lon"]]),
            interpolation=[{"method": "none", "dims": ["lon", "lat"]}],
        )
        o = node.eval(podpac.Coordinates([podpac.crange(1, 9, 1), podpac.crange(1, 9, 1)], dims=["lon", "lat"]))
        np.testing.assert_array_equal(o.data, node.source)

        #  source                      eval
        #  lat, lon                    lat_lon
        node = podpac.data.Array(
            source=[[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
            coordinates=podpac.Coordinates([[1, 5, 9, 13], [0, 1, 2]], dims=["lat", "lon"]),
            interpolation=[{"method": "none", "dims": ["lat", "lon"]}],
        )
        o = node.eval(podpac.Coordinates([[podpac.crange(1, 9, 2), podpac.crange(1, 9, 2)]], dims=[["lat", "lon"]]))
        np.testing.assert_array_equal(o.data, node.source[:-1, 1:])


class TestNearest(object):
    def test_nearest_preview_select(self):
        reqcoords = Coordinates([[-0.5, 1.5, 3.5], [0.5, 2.5, 4.5]], dims=["lat", "lon"])
        srccoords = Coordinates([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dims=["lat", "lon"])

        # test straight ahead functionality
        interp = InterpolationManager("nearest_preview")
        coords, cidx = interp.select_coordinates(srccoords, reqcoords)
        np.testing.assert_array_equal(coords["lat"].coordinates, [0, 2, 4])
        np.testing.assert_array_equal(coords["lon"].coordinates, [0, 2, 4])
        assert srccoords[cidx] == coords

        # test when selection is applied serially
        interp = InterpolationManager(
            [{"method": "nearest_preview", "dims": ["lat"]}, {"method": "nearest_preview", "dims": ["lon"]}]
        )

        coords, cidx = interp.select_coordinates(srccoords, reqcoords)
        np.testing.assert_array_equal(coords["lat"].coordinates, [0, 2, 4])
        np.testing.assert_array_equal(coords["lon"].coordinates, [0, 2, 4])
        assert srccoords[cidx] == coords

        # Test reverse selection
        reqcoords = Coordinates([[-0.5, 1.5, 3.5], [0.5, 2.5, 4.5]], dims=["lat", "lon"])
        srccoords = Coordinates([[0, 1, 2, 3, 4, 5][::-1], [0, 1, 2, 3, 4, 5][::-1]], dims=["lat", "lon"])

        # test straight ahead functionality
        interp = InterpolationManager("nearest_preview")
        coords, cidx = interp.select_coordinates(srccoords, reqcoords)
        np.testing.assert_array_equal(coords["lat"].coordinates, [4, 2, 0])
        np.testing.assert_array_equal(coords["lon"].coordinates, [5, 3, 1])  # Yes, this is expected behavior
        assert srccoords[cidx] == coords

        coords, cidx = interp.select_coordinates(srccoords, reqcoords)
        np.testing.assert_array_equal(coords["lat"].coordinates, [4, 2, 0])
        np.testing.assert_array_equal(coords["lon"].coordinates, [5, 3, 1])
        assert srccoords[cidx] == coords

        # Test Case where rounding issues causes problem with endpoint
        reqcoords = Coordinates([[0, 2, 4], [0, 2, 4]], dims=["lat", "lon"])
        lat = np.arange(0, 6.1, 1.3333333333333334)
        lon = np.arange(0, 6.1, 1.333333333333334)  # Notice one decimal less on this number
        srccoords = Coordinates([lat, lon], dims=["lat", "lon"])

        # test straight ahead functionality
        interp = InterpolationManager("nearest_preview")
        coords, cidx = interp.select_coordinates(srccoords, reqcoords)
        np.testing.assert_almost_equal(coords["lat"].coordinates, lat[::2])
        np.testing.assert_array_equal(coords["lon"].coordinates, lon[:4])
        np.testing.assert_almost_equal(list(srccoords[cidx].bounds.values()), list(coords.bounds.values()))
        assert srccoords[cidx].shape == coords.shape

        coords, cidx = interp.select_coordinates(srccoords, reqcoords)
        np.testing.assert_almost_equal(coords["lat"].coordinates, lat[::2])
        np.testing.assert_array_equal(coords["lon"].coordinates, lon[:4])
        np.testing.assert_almost_equal(list(srccoords[cidx].bounds.values()), list(coords.bounds.values()))
        assert srccoords[cidx].shape == coords.shape

    # def test_nearest_preview_select_stacked(self):
    #     # TODO: how to handle stacked/unstacked coordinate asynchrony?
    #     reqcoords = Coordinates([[-.5, 1.5, 3.5], [.5, 2.5, 4.5]], dims=['lat', 'lon'])
    #     srccoords = Coordinates([([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])], dims=['lat_lon'])

    #     interp = InterpolationManager('nearest_preview')

    #     srccoords, srccoords_index = srccoords.intersect(reqcoords, outer=True, return_index=True)
    #     coords, cidx = interp.select_coordinates(reqcoords, srccoords, srccoords_index)

    #     assert len(coords) == len(srcoords) == len(cidx)
    #     assert len(coords['lat']) == len(reqcoords['lat'])
    #     assert len(coords['lon']) == len(reqcoords['lon'])
    #     assert np.all(coords['lat'].coordinates == np.array([0, 2, 4]))

    def test_nearest_select_issue226(self):
        reqcoords = Coordinates([[-0.5, 1.5, 3.5], [0.5, 2.5, 4.5]], dims=["lat", "lon"])
        srccoords = Coordinates([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dims=["lat", "lon"])

        # test straight ahead functionality
        interp = InterpolationManager("nearest")
        coords, cidx = interp.select_coordinates(srccoords, reqcoords)
        np.testing.assert_array_equal(coords["lat"].coordinates, [0, 2, 4])
        np.testing.assert_array_equal(coords["lon"].coordinates, [0, 3, 5])
        assert srccoords[cidx] == coords

        # test when selection is applied serially
        interp = InterpolationManager([{"method": "nearest", "dims": ["lat"]}, {"method": "nearest", "dims": ["lon"]}])
        coords, cidx = interp.select_coordinates(srccoords, reqcoords)
        np.testing.assert_array_equal(coords["lat"].coordinates, [0, 2, 4])
        np.testing.assert_array_equal(coords["lon"].coordinates, [0, 3, 5])
        assert srccoords[cidx] == coords

    def test_nearest_select_issue445(self):
        sc = Coordinates([clinspace(-59.9, 89.9, 100, name="lat"), clinspace(-179.9, 179.9, 100, name="lon")])
        node = podpac.data.Array(
            interpolation="nearest_preview", source=np.arange(sc.size).reshape(sc.shape), coordinates=sc
        )
        coords = Coordinates([-61, 72], dims=["lat", "lon"])
        out = node.eval(coords)
        assert out.shape == (1, 1)
        assert np.isnan(out.data[0, 0])

    def test_interpolation(self):

        for interpolation in ["nearest", "nearest_preview"]:

            # unstacked 1D
            source = np.random.rand(5)
            coords_src = Coordinates([np.linspace(0, 10, 5)], dims=["lat"])
            node = MockArrayDataSource(data=source, coordinates=coords_src, interpolation=interpolation)

            coords_dst = Coordinates([[1, 1.2, 1.5, 5, 9]], dims=["lat"])
            output = node.eval(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst["lat"].coordinates)
            assert output.values[0] == source[0] and output.values[1] == source[0] and output.values[2] == source[1]

            # unstacked N-D
            source = np.random.rand(5, 5)
            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
            coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=["lat", "lon"])

            node = MockArrayDataSource(data=source, coordinates=coords_src, interpolation=interpolation)
            output = node.eval(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst["lat"].coordinates)
            assert output.values[0, 0] == source[1, 1]

            # source = stacked, dest = stacked
            source = np.random.rand(5)
            coords_src = Coordinates([(np.linspace(0, 10, 5), np.linspace(0, 10, 5))], dims=["lat_lon"])
            node = MockArrayDataSource(
                data=source,
                coordinates=coords_src,
                interpolation={"method": "nearest", "interpolators": [NearestNeighbor]},
            )
            coords_dst = Coordinates([(np.linspace(1, 9, 3), np.linspace(1, 9, 3))], dims=["lat_lon"])
            output = node.eval(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst["lat"].coordinates)
            assert all(output.values == source[[0, 2, 4]])

            # source = stacked, dest = unstacked
            source = np.random.rand(5)
            coords_src = Coordinates([(np.linspace(0, 10, 5), np.linspace(0, 10, 5))], dims=["lat_lon"])
            node = MockArrayDataSource(
                data=source,
                coordinates=coords_src,
                interpolation={"method": "nearest", "interpolators": [NearestNeighbor]},
            )
            coords_dst = Coordinates([np.linspace(1, 9, 3), np.linspace(1, 9, 3)], dims=["lat", "lon"])

            output = node.eval(coords_dst)
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst["lat"].coordinates)
            assert np.all(output.values == source[np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])])

            # source = unstacked, dest = stacked
            source = np.random.rand(5, 5)
            coords_src = Coordinates([np.linspace(0, 10, 5), np.linspace(0, 10, 5)], dims=["lat", "lon"])
            node = MockArrayDataSource(
                data=source,
                coordinates=coords_src,
                interpolation={"method": "nearest", "interpolators": [NearestNeighbor]},
            )
            coords_dst = Coordinates([(np.linspace(1, 9, 3), np.linspace(1, 9, 3))], dims=["lat_lon"])

            output = node.eval(coords_dst)
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst["lat"].coordinates)
            assert np.all(output.values == source[[0, 2, 4], [0, 2, 4]])

            # source = unstacked and non-uniform, dest = stacked
            source = np.random.rand(5, 5)
            coords_src = Coordinates([[0, 1.1, 1.2, 6.1, 10], [0, 1.1, 4, 7.1, 9.9]], dims=["lat", "lon"])
            node = MockArrayDataSource(
                data=source,
                coordinates=coords_src,
                interpolation={"method": "nearest", "interpolators": [NearestNeighbor]},
            )
            coords_dst = Coordinates([(np.linspace(1, 9, 3), np.linspace(1, 9, 3))], dims=["lat_lon"])

            output = node.eval(coords_dst)
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst["lat"].coordinates)
            assert np.all(output.values == source[[1, 3, 4], [1, 2, 4]])

            # lat_lon_time_alt --> lon, alt_time, lat
            source = np.random.rand(5)
            coords_src = Coordinates([[[0, 1, 2, 3, 4]] * 4], dims=[["lat", "lon", "time", "alt"]])
            node = MockArrayDataSource(
                data=source,
                coordinates=coords_src,
                interpolation={"method": "nearest", "interpolators": [NearestNeighbor]},
            )
            coords_dst = Coordinates(
                [[1, 2.4, 3.9], [[1, 2.4, 3.9], [1, 2.4, 3.9]], [1, 2.4, 3.9]], dims=["lon", "alt_time", "lat"]
            )

            output = node.eval(coords_dst)
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst["lat"].coordinates)
            assert np.all(output.values[[0, 1, 2], [0, 1, 2], [0, 1, 2]] == source[[1, 2, 4]])

    def test_spatial_tolerance(self):
        # unstacked 1D
        source = np.random.rand(5)
        coords_src = Coordinates([np.linspace(0, 10, 5)], dims=["lat"])
        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "nearest", "params": {"spatial_tolerance": 1.1}},
        )

        coords_dst = Coordinates([[1, 1.2, 1.5, 5, 9]], dims=["lat"])
        output = node.eval(coords_dst)

        print(output)
        print(source)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert output.values[0] == source[0] and np.isnan(output.values[1]) and output.values[2] == source[1]

        # stacked 1D
        source = np.random.rand(5)
        coords_src = Coordinates([[np.linspace(0, 10, 5), np.linspace(0, 10, 5)]], dims=[["lat", "lon"]])
        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "nearest", "params": {"spatial_tolerance": 1.1}},
        )

        coords_dst = Coordinates([[[1, 1.2, 1.5, 5, 9], [1, 1.2, 1.5, 5, 9]]], dims=[["lat", "lon"]])
        output = node.eval(coords_dst)

        print(output)
        print(source)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert output.values[0] == source[0] and np.isnan(output.values[1]) and output.values[2] == source[1]

    def test_time_tolerance(self):

        # unstacked 1D
        source = np.random.rand(5, 5)
        coords_src = Coordinates(
            [np.linspace(0, 10, 5), clinspace("2018-01-01", "2018-01-09", 5)], dims=["lat", "time"]
        )
        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={
                "method": "nearest",
                "params": {"spatial_tolerance": 1.1, "time_tolerance": np.timedelta64(1, "D")},
            },
        )

        coords_dst = Coordinates([[1, 1.2, 1.5, 5, 9], clinspace("2018-01-01", "2018-01-09", 3)], dims=["lat", "time"])
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert (
            output.values[0, 0] == source[0, 0]
            and output.values[0, 1] == source[0, 2]
            and np.isnan(output.values[1, 0])
            and np.isnan(output.values[1, 1])
            and output.values[2, 0] == source[1, 0]
            and output.values[2, 1] == source[1, 2]
        )

    def test_stacked_source_unstacked_region_non_square(self):
        # unstacked 1D
        source = np.random.rand(5)
        coords_src = Coordinates(
            [[np.linspace(0, 10, 5), clinspace("2018-01-01", "2018-01-09", 5)]], dims=[["lat", "time"]]
        )
        node = MockArrayDataSource(
            data=source, coordinates=coords_src, interpolation={"method": "nearest", "interpolators": [NearestNeighbor]}
        )

        coords_dst = Coordinates([[1, 1.2, 1.5, 5, 9], clinspace("2018-01-01", "2018-01-09", 3)], dims=["lat", "time"])
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(output.values == source[np.array([[0, 2, 4]] * 5)])

    def test_time_space_scale_grid(self):
        # Grid
        source = np.random.rand(5, 3, 2)
        source[2, 1, 0] = np.nan
        coords_src = Coordinates(
            [np.linspace(0, 10, 5), ["2018-01-01", "2018-01-02", "2018-01-03"], [0, 10]], dims=["lat", "time", "alt"]
        )
        coords_dst = Coordinates([5.1, "2018-01-02T11", 1], dims=["lat", "time", "alt"])

        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
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
        assert output == source[2, 2, 0]

        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
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
        assert output == source[2, 1, 1]

        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
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
        assert output == source[3, 1, 0]

    def test_remove_nan(self):
        # Stacked
        source = np.random.rand(5)
        source[2] = np.nan
        coords_src = Coordinates(
            [[np.linspace(0, 10, 5), clinspace("2018-01-01", "2018-01-09", 5)]], dims=[["lat", "time"]]
        )
        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "nearest", "interpolators": [NearestNeighbor], "params": {"remove_nan": False}},
        )
        coords_dst = Coordinates([[5.1]], dims=["lat"])
        output = node.eval(coords_dst)
        assert np.isnan(output)

        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={
                "method": "nearest",
                "interpolators": [NearestNeighbor],
                "params": {"remove_nan": True, "use_selector": False},
            },
        )
        output = node.eval(coords_dst)
        assert (
            output == source[3]
        )  # This fails because the selector selects the nan value... can we turn off the selector?

        # Grid
        source = np.random.rand(5, 3)
        source[2, 1] = np.nan
        coords_src = Coordinates([np.linspace(0, 10, 5), [1, 2, 3]], dims=["lat", "time"])
        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "nearest", "interpolators": [NearestNeighbor], "params": {"remove_nan": False}},
        )
        coords_dst = Coordinates([5.1, 2.01], dims=["lat", "time"])
        output = node.eval(coords_dst)
        assert np.isnan(output)

        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={
                "method": "nearest",
                "interpolators": [NearestNeighbor],
                "params": {"remove_nan": True, "use_selector": False},
            },
        )
        output = node.eval(coords_dst)
        assert output == source[2, 2]

    def test_respect_bounds(self):
        source = np.random.rand(5)
        coords_src = Coordinates([[1, 2, 3, 4, 5]], ["alt"])
        coords_dst = Coordinates([[-0.5, 1.1, 2.6]], ["alt"])
        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={
                "method": "nearest",
                "interpolators": [NearestNeighbor],
                "params": {"respect_bounds": False},
            },
        )
        output = node.eval(coords_dst)
        np.testing.assert_array_equal(output.data, source[[0, 0, 2]])

        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "nearest", "interpolators": [NearestNeighbor], "params": {"respect_bounds": True}},
        )
        output = node.eval(coords_dst)
        np.testing.assert_array_equal(output.data[1:], source[[0, 2]])
        assert np.isnan(output.data[0])

    def test_2Dstacked(self):
        # With Time
        source = np.random.rand(5, 4, 2)
        coords_src = Coordinates(
            [
                [
                    np.arange(5)[:, None] + 0.1 * np.ones((5, 4)),
                    np.arange(4)[None, :] + 0.1 * np.ones((5, 4)),
                ],
                [0.4, 0.7],
            ],
            ["lat_lon", "time"],
        )
        coords_dst = Coordinates([np.arange(4) + 0.2, np.arange(1, 4) - 0.2, [0.5]], ["lat", "lon", "time"])
        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={
                "method": "nearest",
                "interpolators": [NearestNeighbor],
            },
        )
        output = node.eval(coords_dst)
        np.testing.assert_array_equal(output, source[:4, 1:, :1])

        # Using 'xarray' coordinates type
        node = MockArrayDataSourceXR(
            data=source,
            coordinates=coords_src,
            coordinate_index_type="xarray",
            interpolation={
                "method": "nearest",
                "interpolators": [NearestNeighbor],
            },
        )
        output = node.eval(coords_dst)
        np.testing.assert_array_equal(output, source[:4, 1:, :1])

        # Using 'slice' coordinates type
        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            coordinate_index_type="slice",
            interpolation={
                "method": "nearest",
                "interpolators": [NearestNeighbor],
            },
        )
        output = node.eval(coords_dst)
        np.testing.assert_array_equal(output, source[:4, 1:, :1])

        # Without Time
        source = np.random.rand(5, 4)
        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src.drop("time"),
            interpolation={
                "method": "nearest",
                "interpolators": [NearestNeighbor],
            },
        )
        output = node.eval(coords_dst)
        np.testing.assert_array_equal(output, source[:4, 1:])

    # def test_3Dstacked(self):
    #     # With Time
    #     source = np.random.rand(5, 4, 2)
    #     coords_src = Coordinates([[
    #         np.arange(5)[:, None, None] + 0.1 * np.ones((5, 4, 2)),
    #         np.arange(4)[None, :, None] + 0.1 * np.ones((5, 4, 2)),
    #         np.arange(2)[None, None, :] + 0.1 * np.ones((5, 4, 2))]], ["lat_lon_time"])
    #     coords_dst = Coordinates([np.arange(4)+0.2, np.arange(1, 4)-0.2, [0.5]], ["lat", "lon", "time"])
    #     node = MockArrayDataSource(
    #         data=source,
    #         coordinates=coords_src,
    #         interpolation={
    #             "method": "nearest",
    #             "interpolators": [NearestNeighbor],
    #         },
    #     )
    #     output = node.eval(coords_dst)
    #     np.testing.assert_array_equal(output, source[:4, 1:, :1])

    #     # Using 'xarray' coordinates type
    #     node = MockArrayDataSourceXR(
    #         data=source,
    #         coordinates=coords_src,
    #         coordinate_index_type='xarray',
    #         interpolation={
    #             "method": "nearest",
    #             "interpolators": [NearestNeighbor],
    #         },
    #     )
    #     output = node.eval(coords_dst)
    #     np.testing.assert_array_equal(output, source[:4, 1:, :1])

    #     # Using 'slice' coordinates type
    #     node = MockArrayDataSource(
    #         data=source,
    #         coordinates=coords_src,
    #         coordinate_index_type='slice',
    #         interpolation={
    #             "method": "nearest",
    #             "interpolators": [NearestNeighbor],
    #         },
    #     )
    #     output = node.eval(coords_dst)
    #     np.testing.assert_array_equal(output, source[:4, 1:, :1])

    #     # Without Time
    #     source = np.random.rand(5, 4)
    #     node = MockArrayDataSource(
    #         data=source,
    #         coordinates=coords_src.drop('time'),
    #         interpolation={
    #             "method": "nearest",
    #             "interpolators": [NearestNeighbor],
    #         },
    #     )
    #     output = node.eval(coords_dst)
    #     np.testing.assert_array_equal(output, source[:4, 1:])


class TestInterpolateRasterioInterpolator(object):
    """test interpolation functions"""

    def test_interpolate_rasterio(self):
        """regular interpolation using rasterio"""

        assert rasterio is not None

        source = np.arange(0, 15)
        source.resize((3, 5))

        coords_src = Coordinates([clinspace(0, 10, 3), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(1, 11, 3), clinspace(1, 11, 5)], dims=["lat", "lon"])

        # try one specific rasterio case to measure output
        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "min", "interpolators": [RasterioInterpolator]},
        )
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert output.data[0, 3] == 3.0
        assert output.data[0, 4] == 4.0

        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "max", "interpolators": [RasterioInterpolator]},
        )
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert output.data[0, 3] == 9.0
        assert output.data[0, 4] == 9.0

        # TODO boundary should be able to use a default
        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
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

        source = np.random.rand(5, 5)
        coords_src = Coordinates([clinspace(10, 0, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=["lat", "lon"])

        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "nearest", "interpolators": [RasterioInterpolator]},
        )
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(output.lon.values == coords_dst["lon"].coordinates)


class TestInterpolateScipyGrid(object):
    """test interpolation functions"""

    def test_interpolate_scipy_grid(self):

        source = np.arange(0, 25)
        source.resize((5, 5))

        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(1, 11, 5), clinspace(1, 11, 5)], dims=["lat", "lon"])

        # try one specific rasterio case to measure output
        node = MockArrayDataSource(
            data=source, coordinates=coords_src, interpolation={"method": "nearest", "interpolators": [ScipyGrid]}
        )
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        print(output)
        assert output.data[0, 0] == 0.0
        assert output.data[0, 3] == 3.0
        assert output.data[1, 3] == 8.0
        assert np.isnan(output.data[0, 4])  # TODO: how to handle outside bounds

        node = MockArrayDataSource(
            data=source, coordinates=coords_src, interpolation={"method": "cubic_spline", "interpolators": [ScipyGrid]}
        )
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert int(output.data[0, 0]) == 2
        assert int(output.data[2, 4]) == 16

        node = MockArrayDataSource(
            data=source, coordinates=coords_src, interpolation={"method": "bilinear", "interpolators": [ScipyGrid]}
        )
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert int(output.data[0, 0]) == 2
        assert int(output.data[3, 3]) == 20
        assert np.isnan(output.data[4, 4])  # TODO: how to handle outside bounds

    def test_interpolate_irregular_arbitrary_2dims(self):
        """irregular interpolation"""

        # Note, this test also tests the looper helper

        # try >2 dims
        source = np.random.rand(5, 5, 3)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5), [2, 3, 5]], dims=["lat", "lon", "time"])
        coords_dst = Coordinates([clinspace(1, 11, 5), clinspace(1, 11, 5), [2, 3, 4]], dims=["lat", "lon", "time"])

        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation=[{"method": "nearest", "interpolators": [ScipyGrid]}, {"method": "linear", "dims": ["time"]}],
        )
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(output.lon.values == coords_dst["lon"].coordinates)
        assert np.all(output.time.values == coords_dst["time"].coordinates)

        # assert output.data[0, 0] == source[]

    def test_interpolate_looper_helper(self):
        """irregular interpolation"""

        # Note, this test also tests the looper helper

        # try >2 dims
        source = np.random.rand(5, 5, 3, 2)
        result = source.copy()
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

        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
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

        source = np.random.rand(5, 5)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=["lat", "lon"])

        node = MockArrayDataSource(
            data=source, coordinates=coords_src, interpolation={"method": "nearest", "interpolators": [ScipyGrid]}
        )
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        np.testing.assert_array_equal(output.lat.values, coords_dst["lat"].coordinates)
        np.testing.assert_array_equal(output.lon.values, coords_dst["lon"].coordinates)

    def test_interpolate_irregular_arbitrary_swap(self):
        """should handle descending"""

        source = np.random.rand(5, 5)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=["lat", "lon"])

        node = MockArrayDataSource(
            data=source, coordinates=coords_src, interpolation={"method": "nearest", "interpolators": [ScipyGrid]}
        )
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        np.testing.assert_array_equal(output.lat.values, coords_dst["lat"].coordinates)
        np.testing.assert_array_equal(output.lon.values, coords_dst["lon"].coordinates)

    def test_interpolate_irregular_lat_lon(self):
        """irregular interpolation"""

        source = np.random.rand(5, 5)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([[[0, 2, 4, 6, 8, 10], [0, 2, 4, 5, 6, 10]]], dims=["lat_lon"])

        node = MockArrayDataSource(
            data=source, coordinates=coords_src, interpolation={"method": "nearest", "interpolators": [ScipyGrid]}
        )
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert "lat_lon" in output.dims
        np.testing.assert_array_equal(output["lat"].values, coords_dst["lat"].coordinates)
        np.testing.assert_array_equal(output["lon"].values, coords_dst["lon"].coordinates)
        assert output.values[0] == source[0, 0]
        assert output.values[1] == source[1, 1]
        assert output.values[-1] == source[-1, -1]


class TestInterpolateScipyPoint(object):
    def test_interpolate_scipy_point(self):
        """interpolate point data to nearest neighbor with various coords_dst"""

        source = np.random.rand(6)
        coords_src = Coordinates([[[0, 2, 4, 6, 8, 10], [0, 2, 4, 5, 6, 10]]], dims=["lat_lon"])
        coords_dst = Coordinates([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]], dims=["lat_lon"])
        node = MockArrayDataSource(
            data=source, coordinates=coords_src, interpolation={"method": "nearest", "interpolators": [ScipyPoint]}
        )

        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert "lat_lon" in output.dims
        np.testing.assert_array_equal(output.lat.values, coords_dst["lat"].coordinates)
        np.testing.assert_array_equal(output.lon.values, coords_dst["lon"].coordinates)
        assert output.values[0] == source[0]
        assert output.values[-1] == source[3]

        coords_dst = Coordinates([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dims=["lat", "lon"])
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        np.testing.assert_array_equal(output.lat.values, coords_dst["lat"].coordinates)
        assert output.values[0, 0] == source[0]
        assert output.values[-1, -1] == source[3]


class TestXarrayInterpolator(object):
    """test interpolation functions"""

    def test_nearest_interpolation(self):

        interpolation = {
            "method": "nearest",
            "interpolators": [XarrayInterpolator],
            "params": {"fill_value": "extrapolate"},
        }

        # unstacked 1D
        source = np.random.rand(5)
        coords_src = Coordinates([np.linspace(0, 10, 5)], dims=["lat"])
        node = MockArrayDataSource(data=source, coordinates=coords_src, interpolation=interpolation)

        coords_dst = Coordinates([[1, 1.2, 1.5, 5, 9]], dims=["lat"])
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert output.values[0] == source[0] and output.values[1] == source[0] and output.values[2] == source[1]

        # unstacked N-D
        source = np.random.rand(5, 5)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=["lat", "lon"])

        node = MockArrayDataSource(data=source, coordinates=coords_src, interpolation=interpolation)
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert output.values[0, 0] == source[1, 1]

        # stacked
        # TODO: implement stacked handling
        source = np.random.rand(5)
        coords_src = Coordinates([(np.linspace(0, 10, 5), np.linspace(0, 10, 5))], dims=["lat_lon"])
        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "nearest", "interpolators": [XarrayInterpolator]},
        )
        coords_dst = Coordinates([(np.linspace(1, 9, 3), np.linspace(1, 9, 3))], dims=["lat_lon"])

        with pytest.raises(InterpolationException):
            output = node.eval(coords_dst)

        # TODO: implement stacked handling
        # source = stacked, dest = unstacked
        source = np.random.rand(5)
        coords_src = Coordinates([(np.linspace(0, 10, 5), np.linspace(0, 10, 5))], dims=["lat_lon"])
        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "nearest", "interpolators": [XarrayInterpolator]},
        )
        coords_dst = Coordinates([np.linspace(1, 9, 3), np.linspace(1, 9, 3)], dims=["lat", "lon"])

        with pytest.raises(InterpolationException):
            output = node.eval(coords_dst)

        # source = unstacked, dest = stacked
        source = np.random.rand(5, 5)
        coords_src = Coordinates([np.linspace(0, 10, 5), np.linspace(0, 10, 5)], dims=["lat", "lon"])
        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "nearest", "interpolators": [XarrayInterpolator]},
        )
        coords_dst = Coordinates([(np.linspace(1, 9, 3), np.linspace(1, 9, 3))], dims=["lat_lon"])

        output = node.eval(coords_dst)
        np.testing.assert_array_equal(output.data, source[[0, 2, 4], [0, 2, 4]])

    def test_interpolate_xarray_grid(self):

        source = np.arange(0, 25)
        source.resize((5, 5))

        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(1, 11, 5), clinspace(1, 11, 5)], dims=["lat", "lon"])

        # try one specific rasterio case to measure output
        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "nearest", "interpolators": [XarrayInterpolator]},
        )
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        # print(output)
        assert output.data[0, 0] == 0.0
        assert output.data[0, 3] == 3.0
        assert output.data[1, 3] == 8.0
        assert np.isnan(output.data[0, 4])  # TODO: how to handle outside bounds

        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "linear", "interpolators": [XarrayInterpolator], "params": {"fill_nan": True}},
        )
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert int(output.data[0, 0]) == 2
        assert int(output.data[2, 3]) == 15

        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "slinear", "interpolators": [XarrayInterpolator], "params": {"fill_nan": True}},
        )
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert int(output.data[0, 0]) == 2
        assert int(output.data[3, 3]) == 20
        assert np.isnan(output.data[4, 4])

        # Check extrapolation
        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
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
        """irregular interpolation"""

        # try >2 dims
        source = np.random.rand(5, 5, 3)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5), [2, 3, 5]], dims=["lat", "lon", "time"])
        coords_dst = Coordinates([clinspace(1, 11, 5), clinspace(1, 11, 5), [2, 3, 5]], dims=["lat", "lon", "time"])

        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "nearest", "interpolators": [XarrayInterpolator]},
        )
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(output.lon.values == coords_dst["lon"].coordinates)
        assert np.all(output.time.values == coords_dst["time"].coordinates)

        # assert output.data[0, 0] == source[]

    def test_interpolate_irregular_arbitrary_descending(self):
        """should handle descending"""

        source = np.random.rand(5, 5)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=["lat", "lon"])

        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "nearest", "interpolators": [XarrayInterpolator]},
        )
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(output.lon.values == coords_dst["lon"].coordinates)

    def test_interpolate_irregular_arbitrary_swap(self):
        """should handle descending"""

        source = np.random.rand(5, 5)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=["lat", "lon"])

        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "nearest", "interpolators": [XarrayInterpolator]},
        )
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(output.lon.values == coords_dst["lon"].coordinates)

    def test_interpolate_irregular_lat_lon(self):
        """irregular interpolation"""

        source = np.random.rand(5, 5)
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([[[0, 2, 4, 6, 8, 10], [0, 2, 4, 5, 6, 10]]], dims=["lat_lon"])

        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "nearest", "interpolators": [XarrayInterpolator]},
        )
        output = node.eval(coords_dst)

        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat_lon.values == coords_dst.xcoords["lat_lon"])
        assert output.values[0] == source[0, 0]
        assert output.values[1] == source[1, 1]
        assert output.values[-1] == source[-1, -1]

    def test_interpolate_fill_nan(self):
        source = np.arange(0, 25).astype(float)
        source.resize((5, 5))
        source[2, 2] = np.nan

        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        coords_dst = Coordinates([clinspace(1, 11, 5), clinspace(1, 11, 5)], dims=["lat", "lon"])

        # Ensure nan present
        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "linear", "interpolators": [XarrayInterpolator], "params": {"fill_nan": False}},
        )
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        assert np.all(np.isnan(output.data[1:3, 1:3]))

        # Ensure nan gone
        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "linear", "interpolators": [XarrayInterpolator], "params": {"fill_nan": True}},
        )
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        np.testing.assert_array_almost_equal(output.data[1:3, 1:3].ravel(), [8.4, 9.4, 13.4, 14.4])

        # Ensure nan gone, flip lat-lon on source
        coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lon", "lat"])
        node = MockArrayDataSource(
            data=source,
            coordinates=coords_src,
            interpolation={"method": "linear", "interpolators": [XarrayInterpolator], "params": {"fill_nan": True}},
        )
        output = node.eval(coords_dst)
        assert isinstance(output, UnitsDataArray)
        assert np.all(output.lat.values == coords_dst["lat"].coordinates)
        np.testing.assert_array_almost_equal(output.data[1:3, 1:3].T.ravel(), [8.4, 9.4, 13.4, 14.4])
