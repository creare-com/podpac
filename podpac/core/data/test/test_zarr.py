import os.path

import zarr
import pytest
from traitlets import TraitError

from podpac.core.coordinates import Coordinates
from podpac.core.data.file import Zarr


class TestZarr(object):
    path = os.path.join(os.path.dirname(__file__), "assets", "zarr")

    def test_local(self):
        node = Zarr(source=self.path, data_key="a", dims=["lat", "lon"])

    def test_local_invalid_path(self):
        with pytest.raises(ValueError, match="No Zarr store found"):
            node = Zarr(source="/does/not/exist", data_key="a", dims=["lat", "lon"])

    def test_native_coordinates(self):
        node = Zarr(source=self.path, data_key="a", dims=["lat", "lon"])
        assert node.native_coordinates == Coordinates([[0, 1, 2], [10, 20, 30, 40]], dims=["lat", "lon"])

    def test_eval(self):
        a = Zarr(source=self.path, data_key="a", dims=["lat", "lon"])
        b = Zarr(source=self.path, data_key="b", dims=["lat", "lon"])

        coords = Coordinates([0, 10], dims=["lat", "lon"])
        assert a.eval(coords)[0, 0] == 0.0
        assert b.eval(coords)[0, 0] == 1.0

    @pytest.mark.aws
    def test_s3(self):
        path = "s3://podpac-internal-test/drought_parameters.zarr"
        node = Zarr(source=path, data_key="d0", dims=["lat", "lon", "time"])

    def test_group(self):
        dataset = zarr.open(self.path, "r")
        node = Zarr(dataset=dataset, data_key="a", dims=["lat", "lon"])
