import pydap
import numpy as np
import pytest
from traitlets import TraitError
import requests

from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.units import UnitsDataArray
from podpac.core.data.pydap_source import PyDAP
from podpac import settings

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

        node = PyDAP(source=self.source, datakey=self.datakey)
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

        for node in nodes:
            with pytest.raises(TraitError):
                node.dataset = [1, 2, 3]

    def test_session(self):
        """test session attribute and traitlet default """

        # hostname should be the same as the source, parsed by request
        node = PyDAP(source=self.source, datakey=self.datakey)
        assert node.hostname == "demo.opendap.org"

        # defaults to no auth required
        assert node.auth_required == False

        # session should be available
        assert node.session
        assert isinstance(node.session, requests.Session)

        # auth required
        del settings["username@test.org"]
        del settings["password@test.org"]

        node = PyDAP(source=self.source, datakey=self.datakey, hostname="test.org", auth_required=True)
        assert node.hostname == "test.org"

        # throw auth error
        with pytest.raises(ValueError):
            s = node.session

        node.set_credentials(username="user", password="pass")
        assert node.session
        assert isinstance(node.session, requests.Session)

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
