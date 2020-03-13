import numpy as np
import traitlets as tl
import pytest

from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.units import UnitsDataArray
from podpac.core.node import Node
from podpac.core.data.array_source import Array


class TestArray(object):
    """Test Array datasource class"""

    data = np.random.rand(11, 11)
    coordinates = Coordinates([clinspace(-25, 25, 11), clinspace(-25, 25, 11)], dims=["lat", "lon"])

    def test_data_array(self):
        node = Array(data=self.data, native_coordinates=self.coordinates)

    def test_data_list(self):
        # list is coercable to array
        node = Array(data=[0, 1, 1], native_coordinates=self.coordinates)

    def test_get_data(self):
        """ defined get_data function"""

        node = Array(data=self.data, native_coordinates=self.coordinates)
        output = node.eval(self.coordinates)

        assert isinstance(output, UnitsDataArray)
        assert output.values[0, 0] == self.data[0, 0]
        assert output.values[4, 5] == self.data[4, 5]

    def test_get_data_multiple(self):
        data = np.random.rand(11, 11, 2)
        node = Array(data=data, native_coordinates=self.coordinates, outputs=["a", "b"])
        output = node.eval(self.coordinates)
        assert isinstance(output, UnitsDataArray)
        assert output.dims == ("lat", "lon", "output")
        np.testing.assert_array_equal(output["output"], ["a", "b"])
        np.testing.assert_array_equal(output.sel(output="a"), data[:, :, 0])
        np.testing.assert_array_equal(output.sel(output="b"), data[:, :, 1])

        node = Array(data=data, native_coordinates=self.coordinates, outputs=["a", "b"], output="b")
        output = node.eval(self.coordinates)
        assert isinstance(output, UnitsDataArray)
        assert output.dims == ("lat", "lon")
        np.testing.assert_array_equal(output, data[:, :, 1])

    def test_native_coordinates(self):
        node = Array(data=self.data, native_coordinates=self.coordinates)
        assert node.native_coordinates

        node = Array(data=self.data)
        with pytest.raises(tl.TraitError):
            node.native_coordinates

    def test_no_cache(self):
        node = Array()
        assert len(node.cache_ctrl._cache_stores) == 0
