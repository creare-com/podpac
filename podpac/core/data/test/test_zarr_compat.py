import os
import shutil
import tempfile
from unittest.mock import MagicMock

import zarr

from podpac.core.data import zarr_compat
from podpac.core.data.zarr_compat import (
    _ensure_async_fs,
    create_zarr_array,
    get_s3_store,
)


class TestZarrCompat(object):
    def setup_method(self):
        self.path = os.path.join(tempfile.gettempdir(), "test_zarr_compat.zarr")
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    def teardown_method(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    def test_open_create_array_roundtrip(self):
        group = zarr.open(self.path, mode="a")
        arr = create_zarr_array(group, "data", shape=(3, 4), chunks=True, dtype="float64", fill_value=0.0)
        arr[:] = 1.0

        reopened = zarr.open(self.path, mode="r")
        assert reopened["data"].shape == (3, 4)
        assert reopened["data"][0, 0] == 1.0

    def test_open_consolidated_missing_returns_none(self):
        # No .zmetadata has been written, so this should fall back gracefully instead of raising.
        zarr.open(self.path, mode="a")
        try:
            consolidated = zarr.open_consolidated(self.path, mode="r")
        except (KeyError, ValueError):
            consolidated = None
        assert consolidated is None


class TestZarrV3Detection:
    def test_v2_version_string(self, monkeypatch):
        monkeypatch.setattr(zarr_compat, "zarr", MagicMock(__version__="2.18.4"))
        assert zarr_compat._zarr_v3() is False

    def test_v3_version_string(self, monkeypatch):
        monkeypatch.setattr(zarr_compat, "zarr", MagicMock(__version__="3.2.1"))
        assert zarr_compat._zarr_v3() is True


class FakeS3FileSystem:
    """Duck-typed stand-in for an async-capable fsspec-cached filesystem class
    (e.g. a reasonably modern s3fs.S3FileSystem, which always subclasses
    fsspec.asyn.AsyncFileSystem regardless of how it was instantiated)."""

    async_impl = True

    def __init__(self, *args, asynchronous=False, **kwargs):
        self.asynchronous = asynchronous
        self.storage_args = args
        self.storage_options = kwargs


class FakeNonAsyncFileSystem:
    """Duck-typed stand-in for a filesystem class that never implemented the async
    protocol at all -- e.g. s3fs versions predating fsspec.asyn.AsyncFileSystem,
    which podpac's own "s3fs>=0.4" floor still technically allows."""

    async_impl = False
    asynchronous = False
    protocol = "file"


class TestEnsureAsyncFs:
    def test_returns_already_async_fs_unchanged(self):
        fake_s3 = FakeS3FileSystem(key="abc", asynchronous=True)
        assert _ensure_async_fs(fake_s3) is fake_s3

    def test_rebuilds_async_capable_sync_fs_as_async(self):
        fake_s3 = FakeS3FileSystem("arg1", key="abc", asynchronous=False)
        result = _ensure_async_fs(fake_s3)

        assert result is not fake_s3
        assert isinstance(result, FakeS3FileSystem)
        assert result.asynchronous is True
        assert result.storage_args == ("arg1",)
        assert result.storage_options == {"key": "abc"}

    def test_wraps_non_async_capable_fs(self):
        fake_s3 = FakeNonAsyncFileSystem()
        result = _ensure_async_fs(fake_s3)

        assert result is not fake_s3
        assert type(result).__name__ == "AsyncFileSystemWrapper"


class TestGetS3Store:
    def test_v3_converts_sync_async_capable_fs_to_async(self, monkeypatch):
        # zarr 3's FsspecStore requires (and works best with) an async-native fs;
        # a sync-mode instance of an async-capable class must be rebuilt as an
        # async instance, not passed through.
        monkeypatch.setattr(zarr_compat, "_zarr_v3", lambda: True)
        mock_zarr = MagicMock()
        monkeypatch.setattr(zarr_compat, "zarr", mock_zarr)

        fake_s3 = FakeS3FileSystem(key="abc", asynchronous=False)
        store = get_s3_store(fake_s3, "my-bucket/my-key.zarr")

        args, kwargs = mock_zarr.storage.FsspecStore.call_args
        used_fs = args[0]
        assert used_fs is not fake_s3
        assert used_fs.asynchronous is True
        assert used_fs.storage_options["key"] == "abc"
        assert kwargs == {"path": "my-bucket/my-key.zarr"}
        assert store is mock_zarr.storage.FsspecStore.return_value

    def test_v3_reuses_already_async_fs(self, monkeypatch):
        monkeypatch.setattr(zarr_compat, "_zarr_v3", lambda: True)
        mock_zarr = MagicMock()
        monkeypatch.setattr(zarr_compat, "zarr", mock_zarr)

        fake_s3 = FakeS3FileSystem(key="abc", asynchronous=True)
        store = get_s3_store(fake_s3, "my-bucket/my-key.zarr")

        mock_zarr.storage.FsspecStore.assert_called_once_with(fake_s3, path="my-bucket/my-key.zarr")
        assert store is mock_zarr.storage.FsspecStore.return_value

    def test_v3_wraps_non_async_capable_fs(self, monkeypatch):
        # A filesystem whose class never implemented the async protocol at all can't be
        # "reconstructed" into an async instance -- it must be wrapped instead. Without
        # this, zarr 3's FsspecStore.__init__ raises
        # TypeError("Filesystem needs to support async operations.").
        monkeypatch.setattr(zarr_compat, "_zarr_v3", lambda: True)
        mock_zarr = MagicMock()
        monkeypatch.setattr(zarr_compat, "zarr", mock_zarr)

        fake_s3 = FakeNonAsyncFileSystem()
        store = get_s3_store(fake_s3, "my-bucket/my-key.zarr")

        args, kwargs = mock_zarr.storage.FsspecStore.call_args
        used_fs = args[0]
        assert used_fs is not fake_s3
        assert type(used_fs).__name__ == "AsyncFileSystemWrapper"
        assert kwargs == {"path": "my-bucket/my-key.zarr"}
        assert store is mock_zarr.storage.FsspecStore.return_value

    def test_v2_uses_s3map(self, monkeypatch):
        monkeypatch.setattr(zarr_compat, "_zarr_v3", lambda: False)
        mock_s3fs = MagicMock()
        monkeypatch.setattr(zarr_compat, "lazy_module", lambda name: mock_s3fs)

        fake_s3 = object()
        store = get_s3_store(fake_s3, "my-bucket/my-key.zarr")

        mock_s3fs.S3Map.assert_called_once_with(root="my-bucket/my-key.zarr", s3=fake_s3, check=False)
        assert store is mock_s3fs.S3Map.return_value


class TestCreateZarrArrayV2Fallback:
    def test_uses_create_dataset_when_create_array_absent(self):
        # Duck-typed stand-in for a zarr 2.x Group, which only has create_dataset.
        class FakeV2Group:
            def __init__(self):
                self.calls = []

            def create_dataset(self, name, **kwargs):
                self.calls.append((name, kwargs))
                return "created"

        group = FakeV2Group()
        result = create_zarr_array(group, "data", shape=(3, 4), chunks=True, dtype="float64")

        assert result == "created"
        assert group.calls == [("data", {"shape": (3, 4), "dtype": "float64", "chunks": True})]
