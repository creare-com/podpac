import pydap
import pytest
import numpy as np
import traitlets as tl

from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.units import UnitsDataArray
from podpac.core import authentication
from podpac.core.data.pydap_source import PyDAP


class MockPyDAP(PyDAP):
    """mock pydap data source """

    source = tl.Unicode("http://demo.opendap.org")
    data_key = tl.Unicode("key")
    data = np.random.rand(11, 11)

    @tl.default("native_coordinates")
    def _default_native_coordinates(self):
        return Coordinates([clinspace(-25, 25, 11), clinspace(-25, 25, 11)], dims=["lat", "lon"])

    def _open_url(self):
        base = pydap.model.BaseType(name="key", data=self.data)
        dataset = pydap.model.DatasetType(name="dataset")
        dataset["key"] = base
        return dataset


class MockAuthSession(authentication.Session):
    session = None

    def get(self, s):
        if s == "403.dds":
            raise Exception

        self.session = s


class MockPyDAPAuth(MockPyDAP):
    """mock pydap data source """

    def _open_url(self):
        if "%s.dds" % self.source != self.auth_session.session:
            raise Exception
        return super(MockPyDAPAuth, self)._open_url()


class TestPyDAP(object):
    """test pydap datasource"""

    def test_init(self):
        node = PyDAP(source="mysource", data_key="key")

    def test_native_coordinates_not_implemented(self):
        node = PyDAP(source="mysource", data_key="key")
        with pytest.raises(NotImplementedError):
            node.native_coordinates

    def test_auth_session(self):
        # default to none if no username and password
        node = PyDAP(source="mysource", data_key="key")
        assert node.auth_session is None

        # mocked
        node = MockPyDAPAuth(username="username", password="password", auth_class=MockAuthSession)
        assert node.auth_session is not None

        # error
        node = MockPyDAPAuth(source="403", username="username", password="password", auth_class=MockAuthSession)
        assert node.auth_session is None

    def test_keys(self):
        """test return of dataset keys"""

        node = MockPyDAP()
        keys = node.keys
        assert "key" in keys

    def test_dataset(self):
        node = MockPyDAP()
        assert isinstance(node.dataset, pydap.model.DatasetType)

        # using a manual auth_session here to cover dataset auth session case
        node = MockPyDAPAuth(username="username", password="password", auth_session=MockAuthSession())
        assert isinstance(node.dataset, pydap.model.DatasetType)

    def test_url_error(self):
        node = PyDAP(source="mysource")
        with pytest.raises(RuntimeError):
            node.dataset

    def test_get_data(self):
        """test get_data function of pydap"""
        node = MockPyDAP()
        output = node.eval(node.native_coordinates)
        np.testing.assert_array_equal(output.values, node.data)
