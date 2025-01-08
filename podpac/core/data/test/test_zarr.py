import os.path

import zarr
import pytest
import numpy as np
from traitlets import TraitError

from podpac.core.coordinates import Coordinates
from podpac.core.data.zarr_source import Zarr


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

    def test_available_data_keys(self):
        node = Zarr(source=self.path)
        assert node.available_data_keys == ["a", "b"]

    def test_coordinates(self):
        node = Zarr(source=self.path, data_key="a")
        assert node.coordinates == Coordinates([[0, 1, 2], [10, 20, 30, 40]], dims=["lat", "lon"])

    def test_eval(self):
        coords = Coordinates([0, 10], dims=["lat", "lon"])

        a = Zarr(source=self.path, data_key="a")
        assert a.eval(coords)[0, 0] == 0.0

        b = Zarr(source=self.path, data_key="b")
        assert b.eval(coords)[0, 0] == 1.0

    def test_eval_multiple(self):
        coords = Coordinates([0, 10], dims=["lat", "lon"])

        z = Zarr(source=self.path, data_key=["a", "b"])
        out = z.eval(coords)
        assert out.dims == ("lat", "lon", "output")
        np.testing.assert_array_equal(out["output"], ["a", "b"])
        assert out.sel(output="a")[0, 0] == 0.0
        assert out.sel(output="b")[0, 0] == 1.0

        # single output key
        z = Zarr(source=self.path, data_key=["a"])
        out = z.eval(coords)
        assert out.dims == ("lat", "lon", "output")
        np.testing.assert_array_equal(out["output"], ["a"])
        assert out.sel(output="a")[0, 0] == 0.0

        # alternate output names
        z = Zarr(source=self.path, data_key=["a", "b"], outputs=["A", "B"])
        out = z.eval(coords)
        assert out.dims == ("lat", "lon", "output")
        np.testing.assert_array_equal(out["output"], ["A", "B"])
        assert out.sel(output="A")[0, 0] == 0.0
        assert out.sel(output="B")[0, 0] == 1.0

        # default
        z = Zarr(source=self.path)
        out = z.eval(coords)
        assert out.dims == ("lat", "lon", "output")
        np.testing.assert_array_equal(out["output"], ["a", "b"])
        assert out.sel(output="a")[0, 0] == 0.0
        assert out.sel(output="b")[0, 0] == 1.0

    @pytest.mark.skip("Unreachable source")
    def test_s3(self):
        # This file no longer exists
        path = "s3://podpac-internal-test/drought_parameters.zarr"
        node = Zarr(source=path, data_key="d0")
        node.close_dataset()
