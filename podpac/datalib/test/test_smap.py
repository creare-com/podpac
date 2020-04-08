import pytest

import requests
import traitlets as tl

import podpac.datalib
from podpac.datalib import smap

from podpac import settings

# dummy class mixing in custom Earthdata requests Session
class SomeSmapNode(smap.SMAPSessionMixin):
    pass


class TestSMAPSessionMixin(object):
    url = "urs.earthdata.nasa.gov"

    def test_hostname(self):
        node = SomeSmapNode()
        assert node.hostname == self.url

    def test_auth_required(self):
        # make sure auth is deleted from setttings, if it was already there

        # auth required
        with settings:
            if "username@urs.earthdata.nasa.gov" in settings:
                del settings["username@urs.earthdata.nasa.gov"]

            if "password@urs.earthdata.nasa.gov" in settings:
                del settings["password@urs.earthdata.nasa.gov"]

            node = SomeSmapNode()

            # throw auth error
            with pytest.raises(ValueError, match="username"):
                node.session

            node.set_credentials(username="testuser", password="testpass")

            assert node.session
            assert node.session.auth == ("testuser", "testpass")
            assert isinstance(node.session, requests.Session)

    def test_set_credentials(self):
        with settings:
            node = SomeSmapNode()
            node.set_credentials(username="testuser", password="testpass")

            assert settings["username@{}".format(self.url)] == "testuser"
            assert settings["password@{}".format(self.url)] == "testpass"

    def test_session(self):
        with settings:

            node = SomeSmapNode()
            node.set_credentials(username="testuser", password="testpass")

            assert node.session
            assert node.session.auth == ("testuser", "testpass")
            assert isinstance(node.session, requests.Session)

    def test_earth_data_session_rebuild_auth(self):
        class Dum(object):
            pass

        with settings:
            node = SomeSmapNode()
            node.set_credentials(username="testuser", password="testpass")

            prepared_request = Dum()
            prepared_request.headers = {"Authorization": 0}
            prepared_request.url = "https://example.com"

            response = Dum()
            response.request = Dum()
            response.request.url = "https://example2.com"

            node.session.rebuild_auth(prepared_request, response)
