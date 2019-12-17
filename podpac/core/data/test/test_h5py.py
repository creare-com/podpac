import os.path

import numpy as np

from podpac.core.data.file import H5PY


class TestH5PY(object):
    source = os.path.join(os.path.dirname(__file__), "assets/h5raster.hdf5")

    def test_init(self):
        node = H5PY(source=self.source, data_key="data/init", lat_key="coords/lat", lon_key="coords/lon")
        node.dataset
        node.close_dataset()

    def test_native_coordinates(self):
        node = H5PY(
            source=self.source, data_key="data/init", lat_key="coords/lat", lon_key="coords/lon", dims=["lat", "lon"]
        )

        nc = node.native_coordinates
        assert node.native_coordinates.shape == (3, 4)
        np.testing.assert_array_equal(node.native_coordinates["lat"].coordinates, [45.1, 45.2, 45.3])
        np.testing.assert_array_equal(node.native_coordinates["lon"].coordinates, [-100.1, -100.2, -100.3, -100.4])

    def test_data(self):
        node = H5PY(
            source=self.source, data_key="data/init", lat_key="coords/lat", lon_key="coords/lon", dims=["lat", "lon"]
        )

        o = node.eval(node.native_coordinates)
        np.testing.assert_array_equal(o.data.ravel(), np.arange(12))

    def test_keys(self):
        node = H5PY(source=self.source, data_key="data/init", lat_key="coords/lat", lon_key="coords/lon")
        assert node.keys == ["/coords/lat", "/coords/lon", "/data/init"]

    def test_attrs(self):
        node = H5PY(source=self.source, data_key="data/init", lat_key="coords/lat", lon_key="coords/lon")
        assert node.attrs() == {}
        assert node.attrs("data") == {"test": "test"}
        assert node.attrs("coords/lat") == {"unit": "degrees"}
        assert node.attrs("coords/lon") == {"unit": "degrees"}
        assert node.attrs("coords") == {"crs": "EPSG:4326s"}
