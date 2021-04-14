import os.path

import pytest
import numpy as np

from podpac.core.data.csv_source import CSVRaw


class TestCSV(object):
    """test csv data source"""

    source_single = os.path.join(os.path.dirname(__file__), "assets/points-single.csv")
    source_multiple = os.path.join(os.path.dirname(__file__), "assets/points-multiple.csv")
    source_no_header = os.path.join(os.path.dirname(__file__), "assets/points-no-header.csv")
    source_one_dim = os.path.join(os.path.dirname(__file__), "assets/points-one-dim.csv")
    source_no_data = os.path.join(os.path.dirname(__file__), "assets/points-no-data.csv")

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

    def test_init(self):
        node = CSVRaw(source=self.source_single, alt_key="altitude", crs="+proj=merc +vunits=m")

    def test_close(self):
        node = CSVRaw(source=self.source_single, alt_key="altitude", crs="+proj=merc +vunits=m")

    def test_get_dims(self):
        node = CSVRaw(source=self.source_single, alt_key="altitude", crs="+proj=merc +vunits=m")
        assert node.dims == ["lat", "lon", "time", "alt"]

        node = CSVRaw(source=self.source_multiple, alt_key="altitude", crs="+proj=merc +vunits=m")
        assert node.dims == ["lat", "lon", "time", "alt"]

    def test_available_data_keys(self):
        node = CSVRaw(source=self.source_single, alt_key="altitude", crs="+proj=merc +vunits=m")
        assert node.available_data_keys == ["data"]

        node = CSVRaw(source=self.source_multiple, alt_key="altitude", crs="+proj=merc +vunits=m")
        assert node.available_data_keys == ["data", "other"]

        node = CSVRaw(source=self.source_no_data, alt_key="altitude", crs="+proj=merc +vunits=m")
        with pytest.raises(ValueError, match="No data keys found"):
            node.available_data_keys

    def test_data_key(self):
        # default
        node = CSVRaw(source=self.source_single, alt_key="altitude", crs="+proj=merc +vunits=m")
        assert node.data_key == "data"

        # specify
        node = CSVRaw(source=self.source_single, data_key="data", alt_key="altitude", crs="+proj=merc +vunits=m")
        assert node.data_key == "data"

        # invalid
        with pytest.raises(ValueError, match="Invalid data_key"):
            node = CSVRaw(source=self.source_single, data_key="misc", alt_key="altitude", crs="+proj=merc +vunits=m")

    def test_data_key_col(self):
        # specify column
        node = CSVRaw(source=self.source_single, data_key=4, alt_key="altitude", crs="+proj=merc +vunits=m")
        assert node.data_key == 4

        # invalid (out of range)
        with pytest.raises(ValueError, match="Invalid data_key"):
            node = CSVRaw(source=self.source_single, data_key=5, alt_key="altitude", crs="+proj=merc +vunits=m")

        # invalid (dimension key)
        with pytest.raises(ValueError, match="Invalid data_key"):
            node = CSVRaw(source=self.source_single, data_key=0, alt_key="altitude", crs="+proj=merc +vunits=m")

    def test_data_key_multiple_outputs(self):
        # default
        node = CSVRaw(source=self.source_multiple, alt_key="altitude", crs="+proj=merc +vunits=m")
        assert node.data_key == ["data", "other"]

        # specify multiple
        node = CSVRaw(
            source=self.source_multiple, data_key=["other", "data"], alt_key="altitude", crs="+proj=merc +vunits=m"
        )
        assert node.data_key == ["other", "data"]

        # specify one
        node = CSVRaw(source=self.source_multiple, data_key="other", alt_key="altitude", crs="+proj=merc +vunits=m")
        assert node.data_key == "other"

        # specify multiple: invalid item
        with pytest.raises(ValueError, match="Invalid data_key"):
            node = CSVRaw(
                source=self.source_multiple, data_key=["data", "misc"], alt_key="altitude", crs="+proj=merc +vunits=m"
            )

        # specify one: invalid
        with pytest.raises(ValueError, match="Invalid data_key"):
            node = CSVRaw(source=self.source_multiple, data_key="misc", alt_key="altitude", crs="+proj=merc +vunits=m")

    def test_data_key_col_multiple_outputs(self):
        # specify multiple
        node = CSVRaw(source=self.source_multiple, data_key=[4, 5], alt_key="altitude", crs="+proj=merc +vunits=m")
        assert node.data_key == [4, 5]
        assert node.outputs == ["data", "other"]

        # specify one
        node = CSVRaw(source=self.source_multiple, data_key=4, alt_key="altitude", crs="+proj=merc +vunits=m")

        assert node.data_key == 4
        assert node.outputs is None

        # specify multiple: invalid item
        with pytest.raises(ValueError, match="Invalid data_key"):
            node = CSVRaw(source=self.source_multiple, data_key=[4, 6], alt_key="altitude", crs="+proj=merc +vunits=m")

            # specify one: invalid with pytest.raises(ValueError, match="Invalid data_key"):
            node = CSVRaw(source=self.source_multiple, data_key=6, alt_key="altitude", crs="+proj=merc +vunits=m")

    def test_coordinates(self):
        node = CSVRaw(source=self.source_single, alt_key="altitude", crs="+proj=merc +vunits=m")
        nc = node.coordinates
        assert nc.dims == ("lat_lon_time_alt",)
        np.testing.assert_array_equal(nc["lat"].coordinates, self.lat)
        np.testing.assert_array_equal(nc["lon"].coordinates, self.lon)
        np.testing.assert_array_equal(nc["time"].coordinates, self.time)
        np.testing.assert_array_equal(nc["alt"].coordinates, self.alt)

        # one dim (unstacked)
        node = CSVRaw(source=self.source_one_dim)
        nc = node.coordinates
        assert nc.dims == ("time",)

    def test_get_data(self):
        node = CSVRaw(source=self.source_single, alt_key="altitude", data_key="data", crs="+proj=merc +vunits=m")
        out = node.eval(node.coordinates)
        np.testing.assert_array_equal(out, self.data)

        node = CSVRaw(source=self.source_multiple, alt_key="altitude", data_key="data", crs="+proj=merc +vunits=m")
        out = node.eval(node.coordinates)
        np.testing.assert_array_equal(out, self.data)

        node = CSVRaw(source=self.source_multiple, alt_key="altitude", data_key="other", crs="+proj=merc +vunits=m")
        out = node.eval(node.coordinates)
        np.testing.assert_array_equal(out, self.other)

        # default
        node = CSVRaw(source=self.source_single, alt_key="altitude", crs="+proj=merc +vunits=m")
        out = node.eval(node.coordinates)
        np.testing.assert_array_equal(out, self.data)

    def test_get_data_multiple(self):
        # multiple data keys
        node = CSVRaw(
            source=self.source_multiple, alt_key="altitude", data_key=["data", "other"], crs="+proj=merc +vunits=m"
        )
        out = node.eval(node.coordinates)
        assert out.dims == ("lat_lon_time_alt", "output")
        np.testing.assert_array_equal(out["output"], ["data", "other"])
        np.testing.assert_array_equal(out.sel(output="data"), self.data)
        np.testing.assert_array_equal(out.sel(output="other"), self.other)

        # single data key
        node = CSVRaw(source=self.source_multiple, alt_key="altitude", data_key=["data"], crs="+proj=merc +vunits=m")
        out = node.eval(node.coordinates)
        assert out.dims == ("lat_lon_time_alt", "output")
        np.testing.assert_array_equal(out["output"], ["data"])
        np.testing.assert_array_equal(out.sel(output="data"), self.data)

        # alternate output names
        node = CSVRaw(
            source=self.source_multiple,
            alt_key="altitude",
            data_key=["data", "other"],
            outputs=["a", "b"],
            crs="+proj=merc +vunits=m",
        )
        out = node.eval(node.coordinates)
        assert out.dims == ("lat_lon_time_alt", "output")
        np.testing.assert_array_equal(out["output"], ["a", "b"])
        np.testing.assert_array_equal(out.sel(output="a"), self.data)
        np.testing.assert_array_equal(out.sel(output="b"), self.other)

        # default
        node = CSVRaw(source=self.source_multiple, alt_key="altitude", crs="+proj=merc +vunits=m")
        out = node.eval(node.coordinates)
        assert out.dims == ("lat_lon_time_alt", "output")
        np.testing.assert_array_equal(out["output"], ["data", "other"])
        np.testing.assert_array_equal(out.sel(output="data"), self.data)
        np.testing.assert_array_equal(out.sel(output="other"), self.other)

    def test_cols(self):
        node = CSVRaw(
            source=self.source_multiple,
            lat_key=0,
            lon_key=1,
            time_key=2,
            alt_key=3,
            data_key=5,
            crs="+proj=merc +vunits=m",
        )

        # coordinates
        nc = node.coordinates
        assert nc.dims == ("lat_lon_time_alt",)
        np.testing.assert_array_equal(nc["lat"].coordinates, self.lat)
        np.testing.assert_array_equal(nc["lon"].coordinates, self.lon)
        np.testing.assert_array_equal(nc["time"].coordinates, self.time)
        np.testing.assert_array_equal(nc["alt"].coordinates, self.alt)

        # eval
        out = node.eval(nc)
        np.testing.assert_array_equal(out, self.other)

    def test_cols_multiple(self):
        node = CSVRaw(
            source=self.source_multiple,
            lat_key=0,
            lon_key=1,
            time_key=2,
            alt_key=3,
            data_key=[4, 5],
            outputs=["a", "b"],
            crs="+proj=merc +vunits=m",
        )

        # native coordinantes
        nc = node.coordinates
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
        node = CSVRaw(
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
        nc = node.coordinates
        assert nc.dims == ("lat_lon_time_alt",)
        np.testing.assert_array_equal(nc["lat"].coordinates, self.lat)
        np.testing.assert_array_equal(nc["lon"].coordinates, self.lon)
        np.testing.assert_array_equal(nc["time"].coordinates, self.time)
        np.testing.assert_array_equal(nc["alt"].coordinates, self.alt)

        # eval
        out = node.eval(nc)
        np.testing.assert_array_equal(out, self.data)
