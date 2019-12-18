import os.path

import zarr
import pytest
from traitlets import TraitError

from podpac.core.coordinates import Coordinates
from podpac.core.data.file import Zarr


class TestZarr(object):
    path = os.path.join(os.path.dirname(__file__), "assets", "zarr")

    def test_local(self):
        node = Zarr(source=self.path, data_key="a")
        node.close_dataset()

    def test_local_invalid_path(self):
        with pytest.raises(ValueError, match="No Zarr store found"):
            node = Zarr(source="/does/not/exist", data_key="a")

    def test_dims(self):
        node = Zarr(source=self.path)
        assert node.dims == ["lat", "lon"]

    def test_available_keys(self):
        node = Zarr(source=self.path)
        assert node.available_keys == ["a", "b"]

    def test_native_coordinates(self):
        node = Zarr(source=self.path, data_key="a")
        assert node.native_coordinates == Coordinates([[0, 1, 2], [10, 20, 30, 40]], dims=["lat", "lon"])

    def test_eval(self):
        coords = Coordinates([0, 10], dims=["lat", "lon"])

        a = Zarr(source=self.path, data_key="a")
        assert a.eval(coords)[0, 0] == 0.0

        b = Zarr(source=self.path, data_key="b")
        assert b.eval(coords)[0, 0] == 1.0

    @pytest.mark.aws
    def test_s3(self):
        path = "s3://podpac-internal-test/drought_parameters.zarr"
        node = Zarr(source=path, data_key="d0")
        node.close_dataset()

    def test_used_dataset_directly(self):
        dataset = zarr.open(self.path, "r")
        node = Zarr(dataset=dataset, data_key="a")
