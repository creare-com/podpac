import pytest
import requests
import traitlets as tl
import s3fs
from numpy.testing import assert_equal
from unittest.mock import patch, PropertyMock
import logging

from podpac import settings, Node
from podpac.core.authentication import (
    RequestsSessionMixin,
    NASAURSSessionMixin,
    S3Mixin,
    set_credentials,
    _SessionWithHeaderRedirection,
)

_USERNAME_TEST_COM = "username@test.com"
_PASSWORD_TEST_COM = "password@test.com"
_TEST_COM = "test.com"

_log = logging.getLogger(__name__)


class TestAuthentication(object):
    def test_set_credentials(self):
        with settings:
            if _USERNAME_TEST_COM in settings:
                del settings[_USERNAME_TEST_COM]

            if _PASSWORD_TEST_COM in settings:
                del settings[_PASSWORD_TEST_COM]

            # require hostname
            with pytest.raises(ValueError):
                set_credentials(None, uname="test", password="test")

            with pytest.raises(ValueError):
                set_credentials("", uname="test", password="test")

            # make sure these are empty at first
            assert not settings[_USERNAME_TEST_COM]
            assert not settings[_PASSWORD_TEST_COM]

            # test input/getpass
            # TODO: how do you test this?

            # set both username and pw
            set_credentials(hostname=_TEST_COM, uname="testuser", password="testpass")
            assert settings[_USERNAME_TEST_COM] == "testuser"
            assert settings[_PASSWORD_TEST_COM] == "testpass"

            # set username only
            set_credentials(hostname=_TEST_COM, uname="testuser2")
            assert settings[_USERNAME_TEST_COM] == "testuser2"
            assert settings[_PASSWORD_TEST_COM] == "testpass"

            # set pw only
            set_credentials(hostname=_TEST_COM, password="testpass3")
            assert settings[_USERNAME_TEST_COM] == "testuser2"
            assert settings[_PASSWORD_TEST_COM] == "testpass3"

            # don't do anything if neither is provided, but the settings exist
            set_credentials(hostname=_TEST_COM)
            assert settings[_USERNAME_TEST_COM] == "testuser2"
            assert settings[_PASSWORD_TEST_COM] == "testpass3"


# dummy class mixing in RequestsSession with hostname
class SomeNodeWithHostname(RequestsSessionMixin):
    hostname = "myurl.org"


class SomeNode(RequestsSessionMixin):
    pass


class TestRequestsSessionMixin(object):
    def test_hostname(self):
        node = SomeNode(hostname="someurl.org")
        assert node.hostname == "someurl.org"

        # use class that implements
        node = SomeNodeWithHostname()
        assert node.hostname == "myurl.org"

    def test_property_value_errors(self):
        node = SomeNode(hostname="propertyerrors.com")

        with pytest.raises(ValueError, match="set_credentials"):
            _ = node.username

        with pytest.raises(ValueError, match="set_credentials"):
            _ = node.password

    def test_set_credentials(self):
        with settings:
            node = SomeNode(hostname="setcredentials.com")
            node.set_credentials(username="testuser", password="testpass")
            assert settings["username@setcredentials.com"] == "testuser"
            assert settings["password@setcredentials.com"] == "testpass"

    def test_property_values(self):
        with settings:
            node = SomeNode(hostname="propertyvalues.com")
            node.set_credentials(username="testuser2", password="testpass2")

            assert_equal(node.username, "testuser2")
            assert_equal(node.password, "testpass2")

    def test_session(self):
        with settings:
            node = SomeNode(hostname="session.net")
            node.set_credentials(username="testuser", password="testpass")

            assert node.session
            assert node.session.auth == ("testuser", "testpass")
            assert isinstance(node.session, requests.Session)

    def test_auth_required(self):
        with settings:
            with pytest.raises(tl.TraitError):
                SomeNode(hostname="auth.com", auth_required="true")

            # no auth
            node = SomeNode(hostname="auth.com")
            assert node.session
            assert isinstance(node.session, requests.Session)
            with pytest.raises(AttributeError):
                node.auth

            # auth required
            if "username@auth2.com" in settings:
                del settings["username@auth2.com"]

            if "password@auth2.com" in settings:
                del settings["password@auth2.com"]

            node = SomeNode(hostname="auth2.com", auth_required=True)
            with pytest.raises(ValueError):
                s = node.session
                _log.debug(s)

            node.set_credentials(username="testuser", password="testpass")
            assert node.session
            assert isinstance(node.session, requests.Session)


class TestSessionWithHeaderRedirection(object):
    AUTH = "TEST"
    AUTH_HEADER = {"Authorization": AUTH}

    def _prepared_request(self, url: str, headers: dict | None = None) -> requests.PreparedRequest:
        """Build a request from the URL and headers.

        Parameters
        ----------
        url : str
            The request URL.
        headers : dict | None
            Request headers, by default None

        Returns
        -------
        requests.PreparedRequest
            The request prepared for testing.
        """
        return requests.Request(method="GET", url=url, headers=headers or {}).prepare()

    def _response(self, url: str) -> requests.Response:
        """Build a response for the URL.

        Parameters
        ----------
        url : str
            The request URL.

        Returns
        -------
        requests.Response
            The request response.
        """
        response = requests.Response()
        response.request = self._prepared_request(url)
        return response

    def test_noop_without_authorization_header(self) -> None:
        """No Authorization header means there is nothing for rebuild_auth to strip."""
        session = _SessionWithHeaderRedirection()
        prepared = self._prepared_request("https://data.example.com/file.nc")
        session.rebuild_auth(prepared, self._response("https://data.example.com/other"))
        assert "Authorization" not in prepared.headers

    def test_keeps_header_when_redirecting_to_urs(self) -> None:
        """Redirecting to the URS auth host keeps the Authorization header."""
        session = _SessionWithHeaderRedirection()
        prepared = self._prepared_request("https://urs.earthdata.nasa.gov/oauth/authorize", headers=self.AUTH_HEADER)
        session.rebuild_auth(prepared, self._response("https://data.example.com/file.nc"))
        assert prepared.headers["Authorization"] == self.AUTH

    def test_keeps_header_when_redirecting_from_urs(self) -> None:
        """Redirecting away from the URS auth host keeps the Authorization header."""
        session = _SessionWithHeaderRedirection()
        prepared = self._prepared_request("https://data.example.com/file.nc", headers=self.AUTH_HEADER)
        session.rebuild_auth(prepared, self._response("https://urs.earthdata.nasa.gov/oauth/authorize"))
        assert prepared.headers["Authorization"] == self.AUTH

    def test_keeps_header_for_same_host_redirect(self) -> None:
        """A same-host redirect keeps the Authorization header regardless of URS."""
        session = _SessionWithHeaderRedirection()
        prepared = self._prepared_request("https://data.example.com/file2.nc", headers=self.AUTH_HEADER)
        session.rebuild_auth(prepared, self._response("https://data.example.com/file.nc"))
        assert prepared.headers["Authorization"] == self.AUTH

    def test_strips_header_for_unrelated_host_redirect(self) -> None:
        """A cross-host redirect unrelated to URS strips the Authorization header."""
        session = _SessionWithHeaderRedirection()
        prepared = self._prepared_request("https://other-host.example.com/file.nc", headers=self.AUTH_HEADER)
        session.rebuild_auth(prepared, self._response("https://data.example.com/file.nc"))
        assert "Authorization" not in prepared.headers


class TestNASAURSSessionMixin(object):
    def test_session(self) -> None:
        """Test NASAURSSessionMixin session with authetication."""
        node = NASAURSSessionMixin()
        with (
            patch.object(NASAURSSessionMixin, "username", new_callable=PropertyMock, return_value="testuser"),
            patch.object(NASAURSSessionMixin, "password", new_callable=PropertyMock, return_value="testpass"),
        ):
            assert isinstance(node.session, _SessionWithHeaderRedirection)
            assert node.session.auth == ("testuser", "testpass")

    def test_auth_required_traitlet(self) -> None:
        """Test auth_required for the the session mixin."""
        node_auth_required = NASAURSSessionMixin()
        node_no_auth_required = NASAURSSessionMixin(auth_required=False)
        with patch.object(NASAURSSessionMixin, "username", new_callable=PropertyMock, side_effect=ValueError):
            with pytest.raises(ValueError):
                node_auth_required.session
            assert isinstance(node_no_auth_required.session, _SessionWithHeaderRedirection)


class TestS3Mixin(object):
    class S3Node(S3Mixin, Node):
        pass

    def test_anon(self):
        node = self.S3Node(anon=True)
        assert isinstance(node.s3, s3fs.S3FileSystem)

    @pytest.mark.aws
    def test_auth(self):
        node = self.S3Node()
        assert isinstance(node.s3, s3fs.S3FileSystem)
