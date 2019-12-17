import os.path
import numpy as np

from podpac.core.data.file import CSV


class TestCSV(object):
    """ test csv data source
    """

    source = os.path.join(os.path.dirname(__file__), "assets/points.csv")

    def test_init(self):
        # column numbers
        node = CSV(source=self.source, lat_key=0, lon_key=1, time_key=2, alt_key=3, data_key=4)

        # column names
        node = CSV(source=self.source, lat_key="lat", lon_key="lon", time_key="time", alt_key="alt", data_key="data")

        # defaults
        node = CSV(source=self.source)

    def test_dims(self):
        # detect
        node = CSV(source=self.source)
        np.testing.assert_array_equal(node.dims, ["lat", "lon", "time", "alt"])

        # specify the dims (subset and order)
        node = CSV(source=self.source, lat_key="lat", lon_key="lon", data_key="data", dims=["lon", "lat"])
        np.testing.assert_array_equal(node.dims, ["lon", "lat"])

    def test_native_coordinates(self):
        # column numbers
        node = CSV(
            source=self.source,
            lat_key=0,
            lon_key=1,
            time_key=2,
            alt_key=3,
            data_key="data",
            dims=["lat", "lon", "time", "alt"],
        )
        nc = node.native_coordinates
        assert nc.dims == ("lat_lon_time_alt",)
        assert nc.size == 5
        np.testing.assert_array_equal(nc["lat"].coordinates, [0, 1, 1, 1, 1])
        np.testing.assert_array_equal(nc["lon"].coordinates, [0, 0, 2, 2, 2])
        np.testing.assert_array_equal(nc["alt"].coordinates, [0, 0, 0, 0, 4])
        np.testing.assert_array_equal(
            nc["time"].coordinates,
            np.array(
                [
                    "2018-01-01T12:00:00",
                    "2018-01-01T12:00:00",
                    "2018-01-01T12:00:00",
                    "2018-01-01T12:00:03",
                    "2018-01-01T12:00:03",
                ],
                dtype=np.datetime64,
            ),
        )

        # column names
        node = CSV(
            source=self.source,
            lat_key="lat",
            lon_key="lon",
            time_key="time",
            alt_key="alt",
            data_key="data",
            dims=["lat", "lon", "time", "alt"],
        )
        nc = node.native_coordinates
        assert nc.dims == ("lat_lon_time_alt",)
        assert nc.size == 5
        np.testing.assert_array_equal(nc["lat"].coordinates, [0, 1, 1, 1, 1])
        np.testing.assert_array_equal(nc["lon"].coordinates, [0, 0, 2, 2, 2])
        np.testing.assert_array_equal(nc["alt"].coordinates, [0, 0, 0, 0, 4])
        np.testing.assert_array_equal(
            nc["time"].coordinates,
            np.array(
                [
                    "2018-01-01T12:00:00",
                    "2018-01-01T12:00:00",
                    "2018-01-01T12:00:00",
                    "2018-01-01T12:00:03",
                    "2018-01-01T12:00:03",
                ],
                dtype=np.datetime64,
            ),
        )

        # specify the dims
        node = CSV(source=self.source, lat_key="lat", lon_key="lon", data_key="data", dims=["lon", "lat"])
        nc = node.native_coordinates
        assert nc.dims == ("lon_lat",)
        assert nc.size == 5
        np.testing.assert_array_equal(nc["lat"].coordinates, [0, 1, 1, 1, 1])
        np.testing.assert_array_equal(nc["lon"].coordinates, [0, 0, 2, 2, 2])

    def test_data(self):
        node = CSV(
            source=self.source,
            lat_key=0,
            lon_key=1,
            time_key=2,
            alt_key=3,
            data_key="data",
            dims=["lat", "lon", "time", "alt"],
        )
        d = node.eval(node.native_coordinates)
        np.testing.assert_array_equal(d, [0, 1, 2, 3, 4])
