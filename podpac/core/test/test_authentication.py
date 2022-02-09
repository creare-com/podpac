import pytest
import requests
import traitlets as tl
import s3fs
from numpy.testing import assert_equal

from podpac import settings, Node
from podpac.core.authentication import RequestsSessionMixin, S3Mixin, set_credentials


class TestAuthentication(object):
    def test_set_credentials(self):

        with settings:
            if "username@test.com" in settings:
                del settings["username@test.com"]

            if "password@test.com" in settings:
                del settings["password@test.com"]

            # require hostname
            with pytest.raises(TypeError):
                set_credentials()

            with pytest.raises(ValueError):
                set_credentials(None, uname="test", password="test")

            with pytest.raises(ValueError):
                set_credentials("", uname="test", password="test")

            # make sure these are empty at first
            assert not settings["username@test.com"]
            assert not settings["password@test.com"]

            # test input/getpass
            # TODO: how do you test this?

            # set both username and pw
            set_credentials(hostname="test.com", uname="testuser", password="testpass")
            assert settings["username@test.com"] == "testuser"
            assert settings["password@test.com"] == "testpass"

            # set username only
            set_credentials(hostname="test.com", uname="testuser2")
            assert settings["username@test.com"] == "testuser2"
            assert settings["password@test.com"] == "testpass"

            # set pw only
            set_credentials(hostname="test.com", password="testpass3")
            assert settings["username@test.com"] == "testuser2"
            assert settings["password@test.com"] == "testpass3"

            # don't do anything if neither is provided, but the settings exist
            set_credentials(hostname="test.com")
            assert settings["username@test.com"] == "testuser2"
            assert settings["password@test.com"] == "testpass3"


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
            u = node.username

        with pytest.raises(ValueError, match="set_credentials"):
            p = node.password

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
                node = SomeNode(hostname="auth.com", auth_required="true")

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
                print(s)

            node.set_credentials(username="testuser", password="testpass")
            assert node.session
            assert isinstance(node.session, requests.Session)


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
