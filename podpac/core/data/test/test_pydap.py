import pydap
import pytest
import numpy as np
import traitlets as tl
import requests

from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.units import UnitsDataArray
from podpac.core import authentication
from podpac.core.data.pydap_source import PyDAP
from podpac import settings


class MockPyDAP(PyDAP):
    """mock pydap data source"""

    source = "http://demo.opendap.org"
    data_key = "key"
    data = np.random.rand(11, 11)

    def get_coordinates(self):
        return Coordinates([clinspace(-25, 25, 11), clinspace(-25, 25, 11)], dims=["lat", "lon"])

    def _open_url(self):
        base = pydap.model.BaseType(name="key", data=self.data)
        dataset = pydap.model.DatasetType(name="dataset")
        dataset["key"] = base
        return dataset


class TestPyDAP(object):
    """test pydap datasource"""

    source = "http://demo.opendap.org"
    data_key = "key"

    def test_init(self):
        node = PyDAP(source="mysource", data_key="key")

    def test_coordinates_not_implemented(self):
        node = PyDAP(source="mysource", data_key="key")
        with pytest.raises(NotImplementedError):
            node.coordinates

    def test_keys(self):
        """test return of dataset keys"""

        node = MockPyDAP()
        keys = node.keys
        assert "key" in keys

    def test_session(self):
        """test session attribute and traitlet default"""

        # hostname should be the same as the source, parsed by request
        node = PyDAP(source=self.source, data_key=self.data_key)
        assert node.hostname == "demo.opendap.org"

        # defaults to no auth required
        assert node.auth_required == False

        # session should be available
        assert node.session
        assert isinstance(node.session, requests.Session)

        # auth required
        with settings:
            if "username@test.org" in settings:
                del settings["username@test.org"]

            if "password@test.org" in settings:
                del settings["password@test.org"]

            node = PyDAP(source=self.source, data_key=self.data_key, hostname="test.org", auth_required=True)
            assert node.hostname == "test.org"

            # throw auth error
            with pytest.raises(ValueError):
                s = node.session

            node.set_credentials(username="user", password="pass")
            assert node.session
            assert isinstance(node.session, requests.Session)

    def test_dataset(self):
        node = MockPyDAP()
        assert isinstance(node.dataset, pydap.model.DatasetType)

    def test_url_error(self):
        node = PyDAP(source="mysource")
        with pytest.raises(Exception):
            node.dataset

    def test_get_data(self):
        """test get_data function of pydap"""
        node = MockPyDAP()
        output = node.eval(node.coordinates)
        np.testing.assert_array_equal(output.values, node.data)
