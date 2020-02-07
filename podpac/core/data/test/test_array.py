import numpy as np
import pytest

from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.units import UnitsDataArray
from podpac.core.node import Node
from podpac.core.data.array_source import Array


class TestArray(object):
    """Test Array datasource class"""

    data = np.random.rand(11, 11)
    coordinates = Coordinates([clinspace(-25, 25, 11), clinspace(-25, 25, 11)], dims=["lat", "lon"])

    def test_source_trait(self):
        """ must be an ndarray """

        node = Array(source=self.data, native_coordinates=self.coordinates)

        # list is coercable to array
        node = Array(source=[0, 1, 1], native_coordinates=self.coordinates)

        # this list is not coercable to array
        # Starting with numpy 0.16, this is now allowed!
        # with pytest.raises(TraitError):
        # node = Array(source=[0, [0, 1]], native_coordinates=self.coordinates)

    def test_get_data(self):
        """ defined get_data function"""

        source = self.data
        node = Array(source=source, native_coordinates=self.coordinates)
        output = node.eval(self.coordinates)

        assert isinstance(output, UnitsDataArray)
        assert output.values[0, 0] == source[0, 0]
        assert output.values[4, 5] == source[4, 5]

    def test_get_data_multiple(self):
        data = np.random.rand(11, 11, 2)
        node = Array(source=data, native_coordinates=self.coordinates, outputs=["a", "b"])
        output = node.eval(self.coordinates)
        assert isinstance(output, UnitsDataArray)
        assert output.dims == ("lat", "lon", "output")
        np.testing.assert_array_equal(output["output"], ["a", "b"])
        np.testing.assert_array_equal(output.sel(output="a"), data[:, :, 0])
        np.testing.assert_array_equal(output.sel(output="b"), data[:, :, 1])

        node = Array(source=data, native_coordinates=self.coordinates, outputs=["a", "b"], output="b")
        output = node.eval(self.coordinates)
        assert isinstance(output, UnitsDataArray)
        assert output.dims == ("lat", "lon")
        np.testing.assert_array_equal(output, data[:, :, 1])

    def test_native_coordinates(self):
        """test that native coordinates get defined"""

        node = Array(source=self.data)
        with pytest.raises(NotImplementedError):
            node.get_native_coordinates()

        node = Array(source=self.data, native_coordinates=self.coordinates)
        assert node.native_coordinates

        node = Array(source=self.data, native_coordinates=self.coordinates)
        native_coordinates = node.native_coordinates
        get_native_coordinates = node.get_native_coordinates()
        assert native_coordinates
        assert get_native_coordinates
        assert native_coordinates == get_native_coordinates

    def test_base_definition(self):
        node = Array(source=self.data, native_coordinates=self.coordinates)
        d = node.base_definition
        source = np.array(d["source"])
        np.testing.assert_array_equal(source, self.data)

    def test_definition(self):
        node = Array(source=self.data, native_coordinates=self.coordinates)
        node2 = Node.from_definition(node.definition)
        assert isinstance(node2, Array)
        np.testing.assert_array_equal(node2.source, self.data)

    def test_json(self):
        node = Array(source=self.data, native_coordinates=self.coordinates)
        node2 = Node.from_json(node.json)
        assert isinstance(node2, Array)
        np.testing.assert_array_equal(node2.source, self.data)
