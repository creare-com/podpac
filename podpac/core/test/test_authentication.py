import pytest

import requests
import traitlets as tl

from podpac.core import authentication
from podpac import settings


class TestAuthentication(object):
    def test_set_credentials(self):

        del settings["username@test.com"]
        del settings["password@test.com"]

        # require hostname
        with pytest.raises(TypeError):
            authentication.set_credentials()

        with pytest.raises(ValueError):
            authentication.set_credentials(None, username="test", password="test")

        with pytest.raises(ValueError):
            authentication.set_credentials("", username="test", password="test")

        # make sure these are empty at first
        assert not settings["username@test.com"]
        assert not settings["password@test.com"]

        # test input/getpass
        # TODO: how do you test this?

        # set both username and password
        authentication.set_credentials(hostname="test.com", username="testuser", password="testpass")
        assert settings["username@test.com"] == "testuser"
        assert settings["password@test.com"] == "testpass"

        # set username only
        authentication.set_credentials(hostname="test.com", username="testuser2")
        assert settings["username@test.com"] == "testuser2"
        assert settings["password@test.com"] == "testpass"

        # set password only
        authentication.set_credentials(hostname="test.com", password="testpass3")
        assert settings["username@test.com"] == "testuser2"
        assert settings["password@test.com"] == "testpass3"

        # don't do anything if neither is provided, but the settings exist
        authentication.set_credentials(hostname="test.com")
        assert settings["username@test.com"] == "testuser2"
        assert settings["password@test.com"] == "testpass3"


# dummy class mixing in RequestsSession with hostname
class SomeNodeWithHostname(authentication.RequestsSessionMixin):
    hostname = "myurl.org"


class SomeNode(authentication.RequestsSessionMixin):
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
            u = node.username

        with pytest.raises(ValueError, match="set_credentials"):
            p = node.password

    def test_set_credentials(self):
        node = SomeNode(hostname="setcredentials.com")

        node.set_credentials(username="testuser", password="testpass")
        assert settings["username@setcredentials.com"] == "testuser"
        assert settings["password@setcredentials.com"] == "testpass"

    def test_property_values(self):
        node = SomeNode(hostname="propertyvalues.com")
        node.set_credentials(username="testuser2", password="testpass2")

        assert node.username == "testuser2"
        assert node.password == "testpass2"

    def test_session(self):
        node = SomeNode(hostname="session.net")
        node.set_credentials(username="testuser", password="testpass")

        assert node.session
        assert node.session.auth == ("testuser", "testpass")
        assert isinstance(node.session, requests.Session)

    def test_auth_required(self):
        with pytest.raises(tl.TraitError):
            node = SomeNode(hostname="auth.com", auth_required="true")

        # no auth
        node = SomeNode(hostname="auth.com")
        assert node.session
        assert isinstance(node.session, requests.Session)
        with pytest.raises(AttributeError):
            node.auth

        # auth required
        del settings["username@auth2.com"]
        del settings["password@auth2.com"]

        node = SomeNode(hostname="auth2.com", auth_required=True)
        with pytest.raises(ValueError):
            s = node.session
            print(s)

        node.set_credentials(username="testuser", password="testpass")
        assert node.session
        assert isinstance(node.session, requests.Session)
