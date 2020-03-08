import os

import numpy as np
import traitlets as tl
import pytest

import podpac
from podpac.core.data.file_source import BaseFileSource
from podpac.core.data.file_source import LoadFileMixin
from podpac.core.data.file_source import FileKeysMixin

LAT = [0, 1, 2]
LON = [10, 20]
TIME = [100, 200]
ALT = [1, 2, 3, 4]
DATA = np.arange(48).reshape((3, 2, 2, 4))
OTHER = 2 * np.arange(48).reshape((3, 2, 2, 4))


class TestBaseFileSource(object):
    def test_source_required(self):
        node = BaseFileSource()
        with pytest.raises(ValueError, match="'source' required"):
            node.source

    def test_dataset_not_implemented(self):
        node = BaseFileSource(source="mysource")
        with pytest.raises(NotImplementedError):
            node.dataset

    def test_close(self):
        node = BaseFileSource(source="mysource")
        node.close_dataset()


# ---------------------------------------------------------------------------------------------------------------------
# LoadFileMixin
# ---------------------------------------------------------------------------------------------------------------------


class MockLoadFile(LoadFileMixin, BaseFileSource):
    def open_dataset(self, f):
        return None


class TestLoadFile(object):
    def test_open_dataset_not_implemented(self):
        node = LoadFileMixin()
        with pytest.raises(NotImplementedError):
            node.open_dataset(None)

    def test_local(self):
        path = os.path.join(os.path.dirname(__file__), "assets/points-single.csv")
        node = MockLoadFile(source=path)
        node.dataset

    @pytest.mark.aws
    def test_s3(self):
        # TODO replace this with a better public s3 fileobj for testing
        path = "s3://modis-pds/MCD43A4.006/00/08/2020018/MCD43A4.A2020018.h00v08.006.2020027031229_meta.json"
        node = MockLoadFile(source=path)
        node.dataset

    @pytest.mark.aws  # TODO
    def test_ftp(self):
        node = MockLoadFile(source="ftp://speedtest.tele2.net/1KB.zip")
        node.dataset

    @pytest.mark.aws  # TODO
    def test_http(self):
        node = MockLoadFile(source="https://httpstat.us/200")
        node.dataset

    def test_file(self):
        path = os.path.join(os.path.dirname(__file__), "assets/points-single.csv")
        node = MockLoadFile(source="file:///%s" % path)
        node.dataset

    def test_cache(self):
        path = os.path.join(os.path.dirname(__file__), "assets/points-single.csv")

        with podpac.settings:
            podpac.settings["DEFAULT_CACHE"] = ["ram"]
            node = MockLoadFile(source="file:///%s" % path)
            node.dataset

            # node caches dataset object
            assert node.has_cache("dataset")

            # another node can get cached object
            node2 = MockLoadFile(source="file:///%s" % path)
            assert node2.has_cache("dataset")
            node2.dataset


# ---------------------------------------------------------------------------------------------------------------------
# FileKeysMixin
# ---------------------------------------------------------------------------------------------------------------------


class MockFileKeys(FileKeysMixin, BaseFileSource):
    source = "mock-single"
    dataset = {"lat": LAT, "lon": LON, "time": TIME, "alt": ALT, "data": DATA}
    keys = ["lat", "lon", "time", "alt", "data"]
    dims = ["lat", "lon", "time", "alt"]


class MockFileKeysMultipleAvailable(FileKeysMixin, BaseFileSource):
    source = "mock-multiple"
    dataset = {"lat": LAT, "lon": LON, "time": TIME, "alt": ALT, "data": DATA, "other": OTHER}
    keys = ["lat", "lon", "time", "alt", "data", "other"]
    dims = ["lat", "lon", "time", "alt"]


class MockFileKeysEmpty(FileKeysMixin, BaseFileSource):
    source = "mock-empty"
    dataset = {"lat": LAT, "lon": LON, "time": TIME, "alt": ALT}
    keys = ["lat", "lon", "time", "alt"]
    dims = ["lat", "lon", "time", "alt"]


class TestFileKeys(object):
    def test_not_implemented(self):
        class MySource(FileKeysMixin, BaseFileSource):
            pass

        node = MySource(source="mysource")

        with pytest.raises(NotImplementedError):
            node.keys

        with pytest.raises(NotImplementedError):
            node.dims

    def test_available_data_keys(self):
        node = MockFileKeys()
        assert node.available_data_keys == ["data"]

        node = MockFileKeysMultipleAvailable()
        assert node.available_data_keys == ["data", "other"]

        node = MockFileKeysEmpty()
        with pytest.raises(ValueError, match="No data keys found"):
            node.available_data_keys

    def test_data_key(self):
        node = MockFileKeys()
        assert node.data_key == "data"
        assert node.output_keys is None

    def test_output_keys(self):
        node = MockFileKeysMultipleAvailable()
        assert node.data_key is None
        assert node.output_keys == ["data", "other"]

    def test_data_key_and_output_keys_mutually_exclusive(self):
        with pytest.raises(TypeError, match=".* cannot have both"):
            node = MockFileKeys(data_key="data", output_keys=["data"])

    def test_no_outputs(self):
        node = MockFileKeys(data_key="data")
        assert node.outputs == None

        node = MockFileKeysMultipleAvailable(data_key="data")
        assert node.outputs == None

        with pytest.raises(TypeError, match="outputs must be None for single-output nodes"):
            node = MockFileKeys(data_key="data", outputs=["a"])

        with pytest.raises(TypeError, match="outputs must be None for single-output nodes"):
            node = MockFileKeysMultipleAvailable(data_key="data", outputs=["a"])

        with pytest.raises(TypeError, match="outputs must be None for single-output nodes"):
            node = MockFileKeys(outputs=["a"])

    def test_outputs(self):
        # for multi-output nodes, use the dataset's keys (output_keys) by default
        node = MockFileKeys(output_keys=["data"])
        assert node.outputs == ["data"]

        node = MockFileKeysMultipleAvailable(output_keys=["data", "other"])
        assert node.outputs == ["data", "other"]

        node = MockFileKeysMultipleAvailable(output_keys=["data"])
        assert node.outputs == ["data"]

        # alternate outputs names can be specified
        node = MockFileKeys(output_keys=["data"], outputs=["a"])
        assert node.outputs == ["a"]

        node = MockFileKeysMultipleAvailable(output_keys=["data", "other"], outputs=["a", "b"])
        assert node.outputs == ["a", "b"]

        node = MockFileKeysMultipleAvailable(output_keys=["data"], outputs=["a"])
        assert node.outputs == ["a"]

        node = MockFileKeysMultipleAvailable(outputs=["a", "b"])
        assert node.outputs == ["a", "b"]

        # but the outputs and output_keys must match
        with pytest.raises(ValueError, match="outputs and output_keys size mismatch"):
            node = MockFileKeysMultipleAvailable(output_keys=["data"], outputs=["a", "b"])

        with pytest.raises(ValueError, match="outputs and output_keys size mismatch"):
            node = MockFileKeysMultipleAvailable(output_keys=["data", "other"], outputs=["a"])

    def test_native_coordinates(self):
        node = MockFileKeys()
        nc = node.native_coordinates
        assert nc.dims == ("lat", "lon", "time", "alt")
        np.testing.assert_array_equal(nc["lat"].coordinates, LAT)
        np.testing.assert_array_equal(nc["lon"].coordinates, LON)
        np.testing.assert_array_equal(nc["time"].coordinates, TIME)
        np.testing.assert_array_equal(nc["alt"].coordinates, ALT)
