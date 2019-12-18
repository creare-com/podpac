import os.path
import numpy as np

from podpac.core.data.file import CSV


class TestCSV(object):
    """ test csv data source
    """

    source = os.path.join(os.path.dirname(__file__), "assets/points.csv")
    lat = [0, 1, 1, 1, 1]
    lon = [0, 0, 2, 2, 2]
    alt = [0, 0, 0, 0, 4]
    time = np.array(
        [
            "2018-01-01T12:00:00",
            "2018-01-01T12:00:00",
            "2018-01-01T12:00:00",
            "2018-01-01T12:00:03",
            "2018-01-01T12:00:03",
        ],
        dtype=np.datetime64,
    )
    data = [0, 1, 2, 3, 4]

    def test_init(self):
        node = CSV(source=self.source)

    def test_close(self):
        node = CSV(source=self.source)
        node.close_dataset()

    def test_get_dims(self):
        node = CSV(source=self.source, alt_key="altitude")
        assert node.dims == ["lat", "lon", "time", "alt"]

    def test_available_keys(self):
        node = CSV(source=self.source, alt_key="altitude")
        assert node.available_keys == ["lat", "lon", "time", "altitude", "data"]

    def test_native_coordinates(self):
        node = CSV(source=self.source, alt_key="altitude")
        nc = node.native_coordinates
        assert nc.dims == ("lat_lon_time_alt",)
        np.testing.assert_array_equal(nc["lat"].coordinates, self.lat)
        np.testing.assert_array_equal(nc["lon"].coordinates, self.lon)
        np.testing.assert_array_equal(nc["time"].coordinates, self.time)
        np.testing.assert_array_equal(nc["alt"].coordinates, self.alt)

    def test_get_data(self):
        # specify data key
        node = CSV(source=self.source, alt_key="altitude", data_key="data")
        out = node.eval(node.native_coordinates)
        np.testing.assert_array_equal(out, self.data)

        # specify a different data key
        node = CSV(source=self.source, alt_key="altitude", data_key="altitude")
        out = node.eval(node.native_coordinates)
        np.testing.assert_array_equal(out, self.alt)

        # default data key
        node = CSV(source=self.source, alt_key="altitude")
        out = node.eval(node.native_coordinates)
        np.testing.assert_array_equal(out, self.data)

    def test_cols(self):
        node = CSV(source=self.source, lat_key=0, lon_key=1, time_key=2, alt_key=3, data_key=4)

        # native coordinantes
        nc = node.native_coordinates
        assert nc.dims == ("lat_lon_time_alt",)
        np.testing.assert_array_equal(nc["lat"].coordinates, self.lat)
        np.testing.assert_array_equal(nc["lon"].coordinates, self.lon)
        np.testing.assert_array_equal(nc["time"].coordinates, self.time)
        np.testing.assert_array_equal(nc["alt"].coordinates, self.alt)

        # eval
        out = node.eval(nc)
        np.testing.assert_array_equal(out, self.data)

    def test_no_column_names(self):
        # TODO
        pass
