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
        del settings["username@urs.earthdata.nasa.gov"]
        del settings["password@urs.earthdata.nasa.gov"]

        node = SomeSmapNode()

        with pytest.raises(ValueError, match="username"):
            s = node.session

        node.set_credentials(username="testuser", password="testpass")

        assert node.session
        assert node.session.auth == ("testuser", "testpass")
        assert isinstance(node.session, requests.Session)

    def test_set_credentials(self):
        node = SomeSmapNode()
        node.set_credentials(username="testuser", password="testpass")

        assert settings["username@{}".format(self.url)] == "testuser"
        assert settings["password@{}".format(self.url)] == "testpass"

    def test_session(self):
        node = SomeSmapNode()
        node.set_credentials(username="testuser", password="testpass")

        assert node.session
        assert node.session.auth == ("testuser", "testpass")
        assert isinstance(node.session, requests.Session)

    def test_earth_data_session_rebuild_auth(self):
        node = SomeSmapNode()
        node.set_credentials(username="testuser", password="testpass")

        class Dum(object):
            pass

        prepared_request = Dum()
        prepared_request.headers = {"Authorization": 0}
        prepared_request.url = "https://example.com"

        response = Dum()
        response.request = Dum()
        response.request.url = "https://example2.com"

        node.session.rebuild_auth(prepared_request, response)
