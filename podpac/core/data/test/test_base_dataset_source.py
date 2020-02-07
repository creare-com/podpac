import numpy as np
import pytest

from podpac.core.data.file import DatasetSource

LAT = [0, 1, 2]
LON = [10, 20]
TIME = [100, 200]
ALT = [1, 2, 3, 4]
DATA = np.arange(48).reshape((3, 2, 2, 4))
OTHER = 2 * np.arange(48).reshape((3, 2, 2, 4))


class MockDatasetSourceSingle(DatasetSource):
    source = "mock-single"
    dataset = {"lat": LAT, "lon": LON, "time": TIME, "alt": ALT, "data": DATA}
    dims = ["lat", "lon", "time", "alt"]
    available_keys = ["data"]


class MockDatasetSourceMultiple(DatasetSource):
    source = "mock-multiple"
    dataset = {"lat": LAT, "lon": LON, "time": TIME, "alt": ALT, "data": DATA, "other": OTHER}
    dims = ["lat", "lon", "time", "alt"]
    available_keys = ["data", "other"]


class TestDatasetSource(object):
    def test_not_implemented(self):
        with pytest.raises(NotImplementedError):
            node = DatasetSource()

    def test_init(self):
        node = MockDatasetSourceSingle()
        node = MockDatasetSourceMultiple()

    def test_close(self):
        node = MockDatasetSourceSingle()
        node.close_dataset()

    def test_data_key_and_output_keys(self):
        # cannot both be defined
        with pytest.raises(TypeError, match=".* cannot have both"):
            node = MockDatasetSourceSingle(data_key="data", output_keys=["data"])

        # make a standard single-output node for datasets with a single non-dimension key
        node = MockDatasetSourceSingle()
        assert node.data_key == "data"
        assert node.output_keys is None
        assert node.outputs is None

        # make a multi-output node for datasets with multiple non-dimension keys
        node = MockDatasetSourceMultiple()
        assert node.data_key is None
        assert node.output_keys == ["data", "other"]
        assert node.outputs == ["data", "other"]

    def test_outputs(self):
        # standard single-output nodes have no "outputs"
        node = MockDatasetSourceSingle(data_key="data")
        assert node.outputs == None

        node = MockDatasetSourceMultiple(data_key="data")
        assert node.outputs == None

        # for multi-output nodes, use the dataset's keys (output_keys) by default
        node = MockDatasetSourceSingle(output_keys=["data"])
        assert node.outputs == ["data"]

        node = MockDatasetSourceMultiple(output_keys=["data", "other"])
        assert node.outputs == ["data", "other"]

        node = MockDatasetSourceMultiple(output_keys=["data"])
        assert node.outputs == ["data"]

        # alternate outputs names can be specified
        node = MockDatasetSourceSingle(output_keys=["data"], outputs=["a"])
        assert node.outputs == ["a"]

        node = MockDatasetSourceMultiple(output_keys=["data", "other"], outputs=["a", "b"])
        assert node.outputs == ["a", "b"]

        node = MockDatasetSourceMultiple(output_keys=["data"], outputs=["a"])
        assert node.outputs == ["a"]

        node = MockDatasetSourceMultiple(outputs=["a", "b"])
        assert node.outputs == ["a", "b"]

        # but the outputs and output_keys must match
        with pytest.raises(ValueError, match="outputs and output_keys size mismatch"):
            node = MockDatasetSourceMultiple(output_keys=["data"], outputs=["a", "b"])

        with pytest.raises(ValueError, match="outputs and output_keys size mismatch"):
            node = MockDatasetSourceMultiple(output_keys=["data", "other"], outputs=["a"])

        with pytest.raises(ValueError, match="outputs and output_keys size mismatch"):
            node = MockDatasetSourceMultiple(outputs=["a"])

        # and outputs cannot be provided for single-output nodes
        with pytest.raises(TypeError, match="outputs must be None for single-output nodes"):
            node = MockDatasetSourceSingle(data_key="data", outputs=["a"])

        with pytest.raises(TypeError, match="outputs must be None for single-output nodes"):
            node = MockDatasetSourceMultiple(data_key="data", outputs=["a"])

        with pytest.raises(TypeError, match="outputs must be None for single-output nodes"):
            node = MockDatasetSourceSingle(outputs=["a"])

    def test_output(self):
        with pytest.raises(TypeError, match="Invalid output"):
            node = MockDatasetSourceSingle(data_key="data", output="data")

        with pytest.raises(ValueError, match="Invalid output"):
            node = MockDatasetSourceMultiple(outputs=["a", "b"], output="other")

    def test_native_coordinates(self):
        node = MockDatasetSourceSingle()
        nc = node.native_coordinates
        assert nc.dims == ("lat", "lon", "time", "alt")
        np.testing.assert_array_equal(nc["lat"].coordinates, LAT)
        np.testing.assert_array_equal(nc["lon"].coordinates, LON)
        np.testing.assert_array_equal(nc["time"].coordinates, TIME)
        np.testing.assert_array_equal(nc["alt"].coordinates, ALT)

    def test_base_definition(self):
        node = MockDatasetSourceSingle()
        d = node.base_definition
        assert "attrs" in d
        assert "lat_key" in d["attrs"]
        assert "lon_key" in d["attrs"]
        assert "alt_key" in d["attrs"]
        assert "time_key" in d["attrs"]
        assert "data_key" in d["attrs"]
        assert "output_keys" not in d["attrs"]
        assert "outputs" not in d["attrs"]
        assert "cf_time" not in d["attrs"]
        assert "cf_units" not in d["attrs"]
        assert "cf_calendar" not in d["attrs"]
        assert "crs" not in d["attrs"]

        node = MockDatasetSourceMultiple()
        d = node.base_definition
        assert "attrs" in d
        assert "lat_key" in d["attrs"]
        assert "lon_key" in d["attrs"]
        assert "alt_key" in d["attrs"]
        assert "time_key" in d["attrs"]
        assert "output_keys" in d["attrs"]
        assert "data_key" not in d["attrs"]
        assert "outputs" not in d["attrs"]
        assert "cf_time" not in d["attrs"]
        assert "cf_units" not in d["attrs"]
        assert "cf_calendar" not in d["attrs"]
        assert "crs" not in d["attrs"]

        node = MockDatasetSourceMultiple(outputs=["a", "b"])
        d = node.base_definition
        assert "attrs" in d
        assert "lat_key" in d["attrs"]
        assert "lon_key" in d["attrs"]
        assert "alt_key" in d["attrs"]
        assert "time_key" in d["attrs"]
        assert "output_keys" in d["attrs"]
        assert "outputs" in d["attrs"]
        assert "data_key" not in d["attrs"]
        assert "cf_time" not in d["attrs"]
        assert "cf_units" not in d["attrs"]
        assert "cf_calendar" not in d["attrs"]
        assert "crs" not in d["attrs"]

        class MockDatasetSource1(DatasetSource):
            source = "temp"
            dims = ["lat", "lon"]
            available_keys = ["data"]

        node = MockDatasetSource1(crs="EPSG::3294")
        d = node.base_definition
        assert "attrs" in d
        assert "lat_key" in d["attrs"]
        assert "lon_key" in d["attrs"]
        assert "data_key" in d["attrs"]
        assert "crs" in d["attrs"]
        assert "time_key" not in d["attrs"]
        assert "alt_key" not in d["attrs"]
        assert "output_keys" not in d["attrs"]
        assert "outputs" not in d["attrs"]
        assert "cf_time" not in d["attrs"]
        assert "cf_units" not in d["attrs"]
        assert "cf_calendar" not in d["attrs"]

        class MockDatasetSource2(DatasetSource):
            source = "temp"
            dims = ["time", "alt"]
            available_keys = ["data"]

        node = MockDatasetSource2(cf_time=True)
        d = node.base_definition
        assert "attrs" in d
        assert "alt_key" in d["attrs"]
        assert "time_key" in d["attrs"]
        assert "data_key" in d["attrs"]
        assert "cf_time" in d["attrs"]
        assert "cf_units" in d["attrs"]
        assert "cf_calendar" in d["attrs"]
        assert "crs" not in d["attrs"]
        assert "lat_key" not in d["attrs"]
        assert "lon_key" not in d["attrs"]
        assert "output_keys" not in d["attrs"]
        assert "outputs" not in d["attrs"]
