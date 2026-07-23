import os
import shutil
import tempfile

from podpac.core.data.zarr_compat import zarr_open, zarr_open_consolidated, zarr_group, create_zarr_array


class TestZarrCompat(object):
    def setup_method(self):
        self.path = os.path.join(tempfile.gettempdir(), "test_zarr_compat.zarr")
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    def teardown_method(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    def test_open_create_array_roundtrip(self):
        group = zarr_open(self.path, mode="a")
        arr = create_zarr_array(group, "data", shape=(3, 4), chunks=True, dtype="float64", fill_value=0.0)
        arr[:] = 1.0

        reopened = zarr_open(self.path, mode="r")
        assert reopened["data"].shape == (3, 4)
        assert reopened["data"][0, 0] == 1.0

    def test_open_consolidated_missing_returns_none(self):
        # No .zmetadata has been written, so this should fall back gracefully instead of raising.
        zarr_open(self.path, mode="a")
        assert zarr_open_consolidated(self.path, mode="r") is None

    def test_group(self):
        group = zarr_group()
        create_zarr_array(group, "data", shape=(2, 2), chunks=True, dtype="float64", fill_value=0.0)
        assert "data" in group
        assert group["data"].shape == (2, 2)
