import pydap
import numpy as np
import pytest
from traitlets import TraitError

from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.units import UnitsDataArray
from podpac.core.data.pydap_source import PyDAP

# Trying to fix test
pydap.client.open_url


class MockPyDAP(PyDAP):
    """mock pydap data source """

    source = "http://demo.opendap.org"
    username = "username"
    password = "password"
    datakey = "key"

    def get_native_coordinates(self):
        return self.native_coordinates


class TestPyDAP(object):
    """test pydap datasource"""

    source = "http://demo.opendap.org"
    username = "username"
    password = "password"
    datakey = "key"

    # mock parameters and data
    data = np.random.rand(11, 11)  # mocked from pydap endpoint
    coordinates = Coordinates([clinspace(-25, 25, 11), clinspace(-25, 25, 11)], dims=["lat", "lon"])

    def mock_pydap(self):
        def open_url(url, session=None):
            base = pydap.model.BaseType(name="key", data=self.data)
            dataset = pydap.model.DatasetType(name="dataset")
            dataset["key"] = base
            return dataset

        pydap.client.open_url = open_url

    def test_init(self):
        """test basic init of class"""

        node = PyDAP(source=self.source, datakey=self.datakey, username=self.username, password=self.password)
        assert isinstance(node, PyDAP)

        node = MockPyDAP()
        assert isinstance(node, MockPyDAP)

    def test_traits(self):
        """ check each of the pydap traits """

        with pytest.raises(TraitError):
            PyDAP(source=5, datakey=self.datakey)

        with pytest.raises(TraitError):
            PyDAP(source=self.source, datakey=5)

        nodes = [PyDAP(source=self.source, datakey=self.datakey), MockPyDAP()]

        # TODO: in traitlets, if you already define variable, it won't enforce case on
        # redefinition
        with pytest.raises(TraitError):
            nodes[0].username = 5

        with pytest.raises(TraitError):
            nodes[0].password = 5

        for node in nodes:
            with pytest.raises(TraitError):
                node.auth_class = "auth_class"

            with pytest.raises(TraitError):
                node.auth_session = "auth_class"

            with pytest.raises(TraitError):
                node.dataset = [1, 2, 3]

    def test_auth_session(self):
        """test auth_session attribute and traitlet default """

        # default to none if no username and password
        node = PyDAP(source=self.source, datakey=self.datakey)
        assert node.auth_session is None

        # default to none if no auth_class
        node = PyDAP(source=self.source, datakey=self.datakey, username=self.username, password=self.password)
        assert node.auth_session is None

    def test_dataset(self):
        """test dataset trait """
        self.mock_pydap()

        node = PyDAP(source=self.source, datakey=self.datakey)
        assert isinstance(node.dataset, pydap.model.DatasetType)

    def test_get_data(self):
        """test get_data function of pydap"""
        self.mock_pydap()

        node = PyDAP(source=self.source, datakey=self.datakey, native_coordinates=self.coordinates)
        output = node.eval(self.coordinates)
        assert isinstance(output, UnitsDataArray)
        assert output.values[0, 0] == self.data[0, 0]

        node = MockPyDAP(native_coordinates=self.coordinates)
        output = node.eval(self.coordinates)
        assert isinstance(output, UnitsDataArray)

    def test_native_coordinates(self):
        """test native coordinates of pydap datasource"""
        pass

    def test_keys(self):
        """test return of dataset keys"""
        self.mock_pydap()

        node = MockPyDAP(native_coordinates=self.coordinates)
        keys = node.keys
        assert "key" in keys
