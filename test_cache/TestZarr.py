import unittest
import numpy as np
import zarr
import podpac
import sys
import os
from podpac.data import Zarr


# Here is the ZarrCaching class:
# Add the directory containing your module to the Python path
sys.path.append(os.path.dirname('/home/cfoye/Projects/SoilMap/podpac/test_cache/'))

# Now you can import your module
from ZarrClass import ZarrCaching

class TestZarrCaching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create some toy data
        data = np.random.rand(100, 100, 100)

        # Create Native Coords
        lat = np.linspace(-90, 90, 100)
        lon = np.linspace(-180, 180, 100)
        time = np.arange("2023-06-02", 100, dtype='datetime64[D]')
        native_coords = podpac.Coordinates([lat, lon, time], dims=["lat", "lon", "time"])

        # Create a podpac array node with data and native coords
        cls.source_node = podpac.data.Array(source=data, coordinates=native_coords)

        cls.zarr_caching = ZarrCaching(cls.source_node, "/tmp/mydata.zarr", "/tmp/mybool.zarr")
        cls.zarr_caching.create_empty_zarr()

    def test_get_source_data(self):
        lat = np.linspace(-80, 80, 50)
        lon = np.linspace(-150, 150, 50)
        time = np.arange("2023-06-02", 10, dtype='datetime64[D]')
        request_coords = podpac.Coordinates([lat, lon, time], dims=["lat", "lon", "time"])

        data = self.zarr_caching.get_source_data(request_coords)
        self.assertEqual(data.shape, (90, 84, 10))

    def test_fill_zarr(self):
        lat = np.linspace(-80, 80, 50)
        lon = np.linspace(-150, 150, 50)
        time = np.arange("2023-06-02", 10, dtype='datetime64[D]')
        request_coords = podpac.Coordinates([lat, lon, time], dims=["lat", "lon", "time"])

        data = self.zarr_caching.get_source_data(request_coords)
        self.zarr_caching.fill_zarr(data, request_coords)

        # check the filled zarr array
        zarr_data = self.zarr_caching.group_data['data']
        self.assertTrue(np.any(zarr_data[:] != 0.0))

    def test_fill_entire_zarr(self):
        request_coords = self.zarr_caching.source_node.coordinates
        data = self.zarr_caching.get_source_data(request_coords)
        self.zarr_caching.fill_zarr(data, request_coords)

        # check the filled zarr array
        zarr_data = self.zarr_caching.group_data['data']
        self.assertTrue(np.all(zarr_data[:] != 0.0))

    def test_subselect_has(self):
        lat = np.linspace(-90, 90, 100)
        lon = np.linspace(-180, 180, 100)
        time = np.arange("2023-06-02", 100, dtype='datetime64[D]')
        request_coords = podpac.Coordinates([lat, lon, time], dims=["lat", "lon", "time"])

        coords = self.zarr_caching.subselect_has(request_coords)
        self.assertTrue(isinstance(coords, podpac.Coordinates))

    def test_eval(self):
        lat = np.linspace(-80, 80, 50)
        lon = np.linspace(-150, 150, 50)
        time = np.arange("2023-06-02", 10, dtype='datetime64[D]')
        request_coords = podpac.Coordinates([lat, lon, time], dims=["lat", "lon", "time"])

        data = self.zarr_caching.eval(request_coords)
        self.assertEqual(data.shape, (90, 84, 10))

if __name__ == "__main__":
    unittest.main()
