from __future__ import division, unicode_literals, print_function, absolute_import

import pytest
import sys
from io import StringIO

import podpac.core.authentication as auth
from podpac import settings


class TestAuthentication(object):
    def test_required(self):
        with pytest.raises(ValueError):
            sess = auth.Session(hostname=None, username="test", password="test")

        with pytest.raises(ValueError):
            sess = auth.Session(hostname="host", username=None, password="test")

        with pytest.raises(ValueError):
            sess = auth.Session(hostname="host", username="test", password=None)

    def test_store_credentials(self):
        sess = auth.Session(hostname="host", username="username", password="password", store_credentials=True)

        assert settings["username@host"] == "username"
        assert settings["password@host"] == "password"

    def test_load_from_credentials(self):
        settings["username@host"] == "username"
        settings["password@host"] == "password"

        sess = auth.Session(hostname="host")

        assert sess.username == "username"
        assert sess.password == "password"

    def test_earth_data_session_rebuild_auth(self):
        eds = auth.EarthDataSession(username="test", password="test")

        class Dum(object):
            pass

        prepared_request = Dum()
        prepared_request.headers = {"Authorization": 0}
        prepared_request.url = "https://example.com"

        response = Dum()
        response.request = Dum()
        response.request.url = "https://example2.com"

        eds.rebuild_auth(prepared_request, response)
