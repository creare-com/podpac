import os.path
import numpy as np

import pytest

from podpac.core.data.dataset_source import Dataset


class TestDataset(object):
    """ test xarray dataset source
    """

    source = os.path.join(os.path.dirname(__file__), "assets/dataset.nc")
    lat = [0, 1, 2]
    lon = [10, 20, 30, 40]
    time = np.array(["2018-01-01", "2018-01-02"], dtype=np.datetime64)
    data = np.arange(24).reshape((3, 4, 2)).transpose((2, 0, 1))
    other = 2 * data

    def test_init_and_close(self):
        node = Dataset(source=self.source, time_key="day")
        node.close_dataset()

    def test_dims(self):
        node = Dataset(source=self.source, time_key="day")
        assert node.dims == ["time", "lat", "lon"]

        # un-mapped keys
        # node = Dataset(source=self.source)
        # with pytest.raises(ValueError, match="Unexpected dimension"):
        #     node.dims

    def test_available_data_keys(self):
        node = Dataset(source=self.source, time_key="day")
        assert node.available_data_keys == ["data", "other"]

    def test_native_coordinates(self):
        # specify dimension keys
        node = Dataset(source=self.source, time_key="day")
        nc = node.coordinates
        assert nc.dims == ("time", "lat", "lon")
        np.testing.assert_array_equal(nc["lat"].coordinates, self.lat)
        np.testing.assert_array_equal(nc["lon"].coordinates, self.lon)
        np.testing.assert_array_equal(nc["time"].coordinates, self.time)
        node.close_dataset()

    def test_get_data(self):
        # specify data key
        node = Dataset(source=self.source, time_key="day", data_key="data")
        out = node.eval(node.coordinates)
        np.testing.assert_array_equal(out, self.data)
        node.close_dataset()

        node = Dataset(source=self.source, time_key="day", data_key="other")
        out = node.eval(node.coordinates)
        np.testing.assert_array_equal(out, self.other)
        node.close_dataset()

    def test_get_data_multiple(self):
        node = Dataset(source=self.source, time_key="day", data_key=["data", "other"])
        out = node.eval(node.coordinates)
        assert out.dims == ("time", "lat", "lon", "output")
        np.testing.assert_array_equal(out["output"], ["data", "other"])
        np.testing.assert_array_equal(out.sel(output="data"), self.data)
        np.testing.assert_array_equal(out.sel(output="other"), self.other)
        node.close_dataset()

        # single
        node = Dataset(source=self.source, time_key="day", data_key=["other"])
        out = node.eval(node.coordinates)
        assert out.dims == ("time", "lat", "lon", "output")
        np.testing.assert_array_equal(out["output"], ["other"])
        np.testing.assert_array_equal(out.sel(output="other"), self.other)
        node.close_dataset()

        # alternate output names
        node = Dataset(source=self.source, time_key="day", data_key=["data", "other"], outputs=["a", "b"])
        out = node.eval(node.coordinates)
        assert out.dims == ("time", "lat", "lon", "output")
        np.testing.assert_array_equal(out["output"], ["a", "b"])
        np.testing.assert_array_equal(out.sel(output="a"), self.data)
        np.testing.assert_array_equal(out.sel(output="b"), self.other)
        node.close_dataset()

        # default
        node = Dataset(source=self.source, time_key="day")
        out = node.eval(node.coordinates)
        assert out.dims == ("time", "lat", "lon", "output")
        np.testing.assert_array_equal(out["output"], ["data", "other"])
        np.testing.assert_array_equal(out.sel(output="data"), self.data)
        np.testing.assert_array_equal(out.sel(output="other"), self.other)
        node.close_dataset()

    def test_extra_dim(self):
        # default
        node = Dataset(source=self.source)
        assert node.extra_dim is None

        # TODO
