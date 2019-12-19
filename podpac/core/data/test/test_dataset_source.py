import numpy as np
import pytest

from podpac.core.data.file import DatasetSource

LAT = [0, 1, 2]
LON = [10, 20]
TIME = [100, 200]
ALT = [1, 2, 3, 4]
DATA = np.arange(48).reshape((3, 2, 2, 4))
OTHER = 2 * np.arange(48).reshape((3, 2, 2, 4))


class MockFileSourceSingle(DatasetSource):
    source = "mock-single"
    dataset = {"lat": LAT, "lon": LON, "time": TIME, "alt": ALT, "data": DATA}

    @property
    def dims(self):
        return ["lat", "lon", "time", "alt"]

    @property
    def available_keys(self):
        return ["data"]


class MockFileSourceMultiple(DatasetSource):
    source = "mock-multiple"
    dataset = {"lat": LAT, "lon": LON, "time": TIME, "alt": ALT, "data": DATA, "other": OTHER}

    @property
    def dims(self):
        return ["lat", "lon", "time", "alt"]

    @property
    def available_keys(self):
        return ["data", "other"]


class TestDatasetSource(object):
    def test_not_implemented(self):
        with pytest.raises(NotImplementedError):
            node = DatasetSource()

    def test_init(self):
        node = MockFileSourceSingle()
        node = MockFileSourceMultiple()

    def test_close(self):
        node = MockFileSourceSingle()
        node.close_dataset()

    def test_data_key_and_output_keys(self):
        # cannot both be defined
        with pytest.raises(TypeError, match=".* cannot have both"):
            node = MockFileSourceSingle(data_key="data", output_keys=["data"])

        # make a standard single-output node for datasets with a single non-dimension key
        node = MockFileSourceSingle()
        assert node.data_key == "data"
        assert node.output_keys is None
        assert node.outputs is None

        # make a multi-output node for datasets with multiple non-dimension keys
        node = MockFileSourceMultiple()
        assert node.data_key is None
        assert node.output_keys == ["data", "other"]
        assert node.outputs == ["data", "other"]

    def test_outputs(self):
        # standard single-output nodes have no "outputs"
        node = MockFileSourceSingle(data_key="data")
        assert node.outputs == None

        node = MockFileSourceMultiple(data_key="data")
        assert node.outputs == None

        # for multi-output nodes, use the dataset's keys (output_keys) by default
        node = MockFileSourceSingle(output_keys=["data"])
        assert node.outputs == ["data"]

        node = MockFileSourceMultiple(output_keys=["data", "other"])
        assert node.outputs == ["data", "other"]

        node = MockFileSourceMultiple(output_keys=["data"])
        assert node.outputs == ["data"]

        # alternate outputs names can be specified
        node = MockFileSourceSingle(output_keys=["data"], outputs=["a"])
        assert node.outputs == ["a"]

        node = MockFileSourceMultiple(output_keys=["data", "other"], outputs=["a", "b"])
        assert node.outputs == ["a", "b"]

        node = MockFileSourceMultiple(output_keys=["data"], outputs=["a"])
        assert node.outputs == ["a"]

        node = MockFileSourceMultiple(outputs=["a", "b"])
        assert node.outputs == ["a", "b"]

        # but the outputs and output_keys must match
        with pytest.raises(ValueError, match="outputs and output_keys size mismatch"):
            node = MockFileSourceMultiple(output_keys=["data"], outputs=["a", "b"])

        with pytest.raises(ValueError, match="outputs and output_keys size mismatch"):
            node = MockFileSourceMultiple(output_keys=["data", "other"], outputs=["a"])

        with pytest.raises(ValueError, match="outputs and output_keys size mismatch"):
            node = MockFileSourceMultiple(outputs=["a"])

        # and outputs cannot be provided for single-output nodes
        with pytest.raises(TypeError, match="outputs must be None for single-output nodes"):
            node = MockFileSourceSingle(data_key="data", outputs=["a"])

        with pytest.raises(TypeError, match="outputs must be None for single-output nodes"):
            node = MockFileSourceMultiple(data_key="data", outputs=["a"])

        with pytest.raises(TypeError, match="outputs must be None for single-output nodes"):
            node = MockFileSourceSingle(outputs=["a"])

    def test_output(self):
        with pytest.raises(TypeError, match="Invalid output"):
            node = MockFileSourceSingle(data_key="data", output="data")

        with pytest.raises(ValueError, match="Invalid output"):
            node = MockFileSourceMultiple(outputs=["a", "b"], output="other")

    def test_native_coordinates(self):
        node = MockFileSourceSingle()
        nc = node.native_coordinates
        assert nc.dims == ("lat", "lon", "time", "alt")
        np.testing.assert_array_equal(nc["lat"].coordinates, LAT)
        np.testing.assert_array_equal(nc["lon"].coordinates, LON)
        np.testing.assert_array_equal(nc["time"].coordinates, TIME)
        np.testing.assert_array_equal(nc["alt"].coordinates, ALT)

    def test_definition(self):
        # TODO don't include attrs when not necessary, such as
        #   output_keys and outputs for standard nodes
        #   data_key for multi nodes
        #   basically anything that uses the default, which might be an easier way to implement it...
        pass
