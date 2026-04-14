import os.path

import pytest
import numpy as np

from podpac.core.data.csv_source import CSV

_CRS = "+proj=merc +vunits=m"
INVALID_DATA_KEY = "Invalid data_key"

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
    T1 = "2018-01-01T12:00:00"
    T2 = "2018-01-01T12:00:03"
    time = np.array(
        [
            T1,
            T1,
            T1,
            T2,
            T2,
        ],
        dtype=np.datetime64,
    )
    data = [0, 1, 2, 3, 4]
    other = [10.5, 20.5, 30.5, 40.5, 50.5]

    def test_init(self):
        CSV(source=self.source_single, alt_key="altitude", crs=_CRS)

    def test_close(self):
        CSV(source=self.source_single, alt_key="altitude", crs=_CRS)

    def test_get_dims(self):
        node = CSV(source=self.source_single, alt_key="altitude", crs=_CRS)
        assert node.dims == ["lat", "lon", "time", "alt"]

        node = CSV(source=self.source_multiple, alt_key="altitude", crs=_CRS)
        assert node.dims == ["lat", "lon", "time", "alt"]

    def test_available_data_keys(self):
        node = CSV(source=self.source_single, alt_key="altitude", crs=_CRS)
        assert node.available_data_keys == ["data"]

        node = CSV(source=self.source_multiple, alt_key="altitude", crs=_CRS)
        assert node.available_data_keys == ["data", "other"]

        node = CSV(source=self.source_no_data, alt_key="altitude", crs=_CRS)
        with pytest.raises(ValueError, match="No data keys found"):
            _ = node.available_data_keys

    def test_data_key(self):
        # default
        node = CSV(source=self.source_single, alt_key="altitude", crs=_CRS)
        assert node.data_key == "data"

        # specify
        node = CSV(source=self.source_single, data_key="data", alt_key="altitude", crs=_CRS)
        assert node.data_key == "data"

        # invalid
        with pytest.raises(ValueError, match=INVALID_DATA_KEY):
            CSV(source=self.source_single, data_key="misc", alt_key="altitude", crs=_CRS)

    def test_data_key_col(self):
        # specify column
        node = CSV(source=self.source_single, data_key=4, alt_key="altitude", crs=_CRS)
        assert node.data_key == 4

        # invalid (out of range)
        with pytest.raises(ValueError, match=INVALID_DATA_KEY):
            CSV(source=self.source_single, data_key=5, alt_key="altitude", crs=_CRS)

        # invalid (dimension key)
        with pytest.raises(ValueError, match=INVALID_DATA_KEY):
            CSV(source=self.source_single, data_key=0, alt_key="altitude", crs=_CRS)

    def test_data_key_multiple_outputs(self):
        # default
        node = CSV(source=self.source_multiple, alt_key="altitude", crs=_CRS)
        assert node.data_key == ["data", "other"]

        # specify multiple
        node = CSV(
            source=self.source_multiple, data_key=["other", "data"], alt_key="altitude", crs=_CRS
        )
        assert node.data_key == ["other", "data"]

        # specify one
        node = CSV(source=self.source_multiple, data_key="other", alt_key="altitude", crs=_CRS)
        assert node.data_key == "other"

        # specify multiple: invalid item
        with pytest.raises(ValueError, match=INVALID_DATA_KEY):
            CSV(
                source=self.source_multiple, data_key=["data", "misc"], alt_key="altitude", crs=_CRS
            )

        # specify one: invalid
        with pytest.raises(ValueError, match=INVALID_DATA_KEY):
            CSV(source=self.source_multiple, data_key="misc", alt_key="altitude", crs=_CRS)

    def test_data_key_col_multiple_outputs(self):
        # specify multiple
        node = CSV(source=self.source_multiple, data_key=[4, 5], alt_key="altitude", crs=_CRS)
        assert node.data_key == [4, 5]
        assert node.outputs == ["data", "other"]

        # specify one
        node = CSV(source=self.source_multiple, data_key=4, alt_key="altitude", crs=_CRS)

        assert node.data_key == 4
        assert node.outputs is None

        # specify multiple: invalid item
        with pytest.raises(ValueError, match=INVALID_DATA_KEY):
            CSV(source=self.source_multiple, data_key=[4, 6], alt_key="altitude", crs=_CRS)

        # specify one: invalid with pytest.raises(ValueError, match=INVALID_DATA_KEY):
        with pytest.raises(ValueError, match=INVALID_DATA_KEY):
            CSV(source=self.source_multiple, data_key=6, alt_key="altitude", crs=_CRS)

    def test_coordinates(self):
        node = CSV(source=self.source_single, alt_key="altitude", crs=_CRS)
        nc = node.coordinates
        assert nc.dims == ("lat_lon_time_alt",)
        np.testing.assert_array_equal(nc["lat"].coordinates, self.lat)
        np.testing.assert_array_equal(nc["lon"].coordinates, self.lon)
        np.testing.assert_array_equal(nc["time"].coordinates, self.time)
        np.testing.assert_array_equal(nc["alt"].coordinates, self.alt)

        # one dim (unstacked)
        node = CSV(source=self.source_one_dim)
        nc = node.coordinates
        assert nc.dims == ("time",)

    def test_get_data(self):
        node = CSV(source=self.source_single, alt_key="altitude", data_key="data", crs=_CRS)
        out = node.eval(node.coordinates)
        np.testing.assert_array_equal(out, self.data)

        node = CSV(source=self.source_multiple, alt_key="altitude", data_key="data", crs=_CRS)
        out = node.eval(node.coordinates)
        np.testing.assert_array_equal(out, self.data)

        node = CSV(source=self.source_multiple, alt_key="altitude", data_key="other", crs=_CRS)
        out = node.eval(node.coordinates)
        np.testing.assert_array_equal(out, self.other)

        # default
        node = CSV(source=self.source_single, alt_key="altitude", crs=_CRS)
        out = node.eval(node.coordinates)
        np.testing.assert_array_equal(out, self.data)

    def test_get_data_multiple(self):
        # multiple data keys
        node = CSV(
            source=self.source_multiple, alt_key="altitude", data_key=["data", "other"], crs=_CRS
        )
        out = node.eval(node.coordinates)
        assert out.dims == ("lat_lon_time_alt", "output")
        np.testing.assert_array_equal(out["output"], ["data", "other"])
        np.testing.assert_array_equal(out.sel(output="data"), self.data)
        np.testing.assert_array_equal(out.sel(output="other"), self.other)

        # single data key
        node = CSV(source=self.source_multiple, alt_key="altitude", data_key=["data"], crs=_CRS)
        out = node.eval(node.coordinates)
        assert out.dims == ("lat_lon_time_alt", "output")
        np.testing.assert_array_equal(out["output"], ["data"])
        np.testing.assert_array_equal(out.sel(output="data"), self.data)

        # alternate output names
        node = CSV(
            source=self.source_multiple,
            alt_key="altitude",
            data_key=["data", "other"],
            outputs=["a", "b"],
            crs=_CRS,
        )
        out = node.eval(node.coordinates)
        assert out.dims == ("lat_lon_time_alt", "output")
        np.testing.assert_array_equal(out["output"], ["a", "b"])
        np.testing.assert_array_equal(out.sel(output="a"), self.data)
        np.testing.assert_array_equal(out.sel(output="b"), self.other)

        # default
        node = CSV(source=self.source_multiple, alt_key="altitude", crs=_CRS)
        out = node.eval(node.coordinates)
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
            crs=_CRS,
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
        node = CSV(
            source=self.source_multiple,
            lat_key=0,
            lon_key=1,
            time_key=2,
            alt_key=3,
            data_key=[4, 5],
            outputs=["a", "b"],
            crs=_CRS,
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
        node = CSV(
            source=self.source_no_header,
            lat_key=0,
            lon_key=1,
            time_key=2,
            alt_key=3,
            data_key=4,
            header=None,
            crs=_CRS,
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
