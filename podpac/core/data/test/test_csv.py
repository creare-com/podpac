import os.path

import pytest
import numpy as np

from podpac.core.data.file import CSV


class TestCSV(object):
    """ test csv data source
    """

    source_single = os.path.join(os.path.dirname(__file__), "assets/points-single.csv")
    source_multiple = os.path.join(os.path.dirname(__file__), "assets/points-multiple.csv")
    source_no_header = os.path.join(os.path.dirname(__file__), "assets/points-no-header.csv")

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
    other = [10.5, 20.5, 30.5, 40.5, 50.5]

    # def test_init(self):
    #     node = CSV(source=self.source_single, alt_key="altitude", crs="+proj=merc +vunits=m")

    # def test_close(self):
    #     node = CSV(source=self.source_single, alt_key="altitude", crs="+proj=merc +vunits=m")
    #     node.close_dataset()

    # def test_get_dims(self):
    #     node = CSV(source=self.source_single, alt_key="altitude", crs="+proj=merc +vunits=m")
    #     assert node.dims == ["lat", "lon", "time", "alt"]

    #     node = CSV(source=self.source_multiple, alt_key="altitude", crs="+proj=merc +vunits=m")
    #     assert node.dims == ["lat", "lon", "time", "alt"]

    # def test_available_keys(self):
    #     node = CSV(source=self.source_single, alt_key="altitude", crs="+proj=merc +vunits=m")
    #     assert node.available_keys == ["data"]

    #     node = CSV(source=self.source_multiple, alt_key="altitude", crs="+proj=merc +vunits=m")
    #     assert node.available_keys == ["data", "other"]

    # def test_native_coordinates(self):
    #     node = CSV(source=self.source_single, alt_key="altitude", crs="+proj=merc +vunits=m")
    #     nc = node.native_coordinates
    #     assert nc.dims == ("lat_lon_time_alt",)
    #     np.testing.assert_array_equal(nc["lat"].coordinates, self.lat)
    #     np.testing.assert_array_equal(nc["lon"].coordinates, self.lon)
    #     np.testing.assert_array_equal(nc["time"].coordinates, self.time)
    #     np.testing.assert_array_equal(nc["alt"].coordinates, self.alt)

    def test_get_data(self):
        node = CSV(source=self.source_single, alt_key="altitude", data_key="data", crs="+proj=merc +vunits=m")
        out = node.eval(node.native_coordinates)
        np.testing.assert_array_equal(out, self.data)

        node = CSV(source=self.source_multiple, alt_key="altitude", data_key="data", crs="+proj=merc +vunits=m")
        out = node.eval(node.native_coordinates)
        np.testing.assert_array_equal(out, self.data)

        node = CSV(source=self.source_multiple, alt_key="altitude", data_key="other", crs="+proj=merc +vunits=m")
        out = node.eval(node.native_coordinates)
        np.testing.assert_array_equal(out, self.other)

        # default
        node = CSV(source=self.source_single, alt_key="altitude", crs="+proj=merc +vunits=m")
        out = node.eval(node.native_coordinates)
        np.testing.assert_array_equal(out, self.data)

    def test_get_data_multiple(self):
        # multiple data keys
        node = CSV(
            source=self.source_multiple, alt_key="altitude", output_keys=["data", "other"], crs="+proj=merc +vunits=m"
        )
        out = node.eval(node.native_coordinates)
        assert out.dims == ("lat_lon_time_alt", "output")
        np.testing.assert_array_equal(out["output"], ["data", "other"])
        np.testing.assert_array_equal(out.sel(output="data"), self.data)
        np.testing.assert_array_equal(out.sel(output="other"), self.other)

        # single data key
        node = CSV(source=self.source_multiple, alt_key="altitude", output_keys=["data"], crs="+proj=merc +vunits=m")
        out = node.eval(node.native_coordinates)
        assert out.dims == ("lat_lon_time_alt", "output")
        np.testing.assert_array_equal(out["output"], ["data"])
        np.testing.assert_array_equal(out.sel(output="data"), self.data)

        # alternate output names
        node = CSV(
            source=self.source_multiple,
            alt_key="altitude",
            output_keys=["data", "other"],
            outputs=["a", "b"],
            crs="+proj=merc +vunits=m",
        )
        out = node.eval(node.native_coordinates)
        assert out.dims == ("lat_lon_time_alt", "output")
        np.testing.assert_array_equal(out["output"], ["a", "b"])
        np.testing.assert_array_equal(out.sel(output="a"), self.data)
        np.testing.assert_array_equal(out.sel(output="b"), self.other)

        # default
        node = CSV(source=self.source_multiple, alt_key="altitude", crs="+proj=merc +vunits=m")
        out = node.eval(node.native_coordinates)
        assert out.dims == ("lat_lon_time_alt", "output")
        np.testing.assert_array_equal(out["output"], ["data", "other"])
        np.testing.assert_array_equal(out.sel(output="data"), self.data)
        np.testing.assert_array_equal(out.sel(output="other"), self.other)

    def test_cols(self):
        node = CSV(
            source=self.source_multiple,
            lat_key=0,
            lon_key=1,
            time_key=2,
            alt_key=3,
            data_key=5,
            crs="+proj=merc +vunits=m",
        )

        # native_coordinates
        nc = node.native_coordinates
        assert nc.dims == ("lat_lon_time_alt",)
        np.testing.assert_array_equal(nc["lat"].coordinates, self.lat)
        np.testing.assert_array_equal(nc["lon"].coordinates, self.lon)
        np.testing.assert_array_equal(nc["time"].coordinates, self.time)
        np.testing.assert_array_equal(nc["alt"].coordinates, self.alt)

        # eval
        out = node.eval(nc)
        np.testing.assert_array_equal(out, self.other)

    def test_cols_multiple(self):
        node = CSV(
            source=self.source_multiple,
            lat_key=0,
            lon_key=1,
            time_key=2,
            alt_key=3,
            output_keys=[4, 5],
            outputs=["a", "b"],
            crs="+proj=merc +vunits=m",
        )

        # native coordinantes
        nc = node.native_coordinates
        assert nc.dims == ("lat_lon_time_alt",)
        np.testing.assert_array_equal(nc["lat"].coordinates, self.lat)
        np.testing.assert_array_equal(nc["lon"].coordinates, self.lon)
        np.testing.assert_array_equal(nc["time"].coordinates, self.time)
        np.testing.assert_array_equal(nc["alt"].coordinates, self.alt)

        # eval
        out = node.eval(nc)
        assert out.dims == ("lat_lon_time_alt", "output")
        np.testing.assert_array_equal(out["output"], ["a", "b"])
        np.testing.assert_array_equal(out.sel(output="a"), self.data)
        np.testing.assert_array_equal(out.sel(output="b"), self.other)

    def test_header(self):
        node = CSV(
            source=self.source_no_header,
            lat_key=0,
            lon_key=1,
            time_key=2,
            alt_key=3,
            data_key=4,
            header=None,
            crs="+proj=merc +vunits=m",
        )

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

    def test_base_definition(self):
        node = CSV(source=self.source_single, alt_key="altitude", crs="+proj=merc +vunits=m")
        d = node.base_definition
        if "attrs" in d:
            assert "header" not in d["attrs"]

        node = CSV(
            source=self.source_no_header,
            lat_key=0,
            lon_key=1,
            time_key=2,
            alt_key=3,
            data_key=4,
            header=None,
            crs="+proj=merc +vunits=m",
        )
        d = node.base_definition
        assert "attrs" in d
        assert "header" in d["attrs"]
