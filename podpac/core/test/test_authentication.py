from __future__ import division, unicode_literals, print_function, absolute_import

import pytest
import sys
from io import StringIO

import podpac.core.authentication as auth

class TestAuthentication(object):
    def test_earth_data_session_update(self):
        eds = auth.EarthDataSession()
        eds.update_login('testuser', 'testpassword')
        eds = auth.EarthDataSession()
        assert(eds.auth == ('testuser', 'testpassword'))
               
    def test_earth_data_session_update_input(self):
        eds = auth.EarthDataSession()
        auth.input = lambda x: 'testuser2'
        auth.getpass.getpass = lambda: 'testpass2'
        eds.update_login()
        eds = auth.EarthDataSession()
        assert(eds.auth == ('testuser2', 'testpass2'))
        
    def test_earth_data_session_rebuild_auth(self):
        eds = auth.EarthDataSession() 
        class Dum(object):
            pass
        
        prepared_request = Dum()
        prepared_request.headers = {'Authorization': 0}
        prepared_request.url = 'https://example.com'
        
        response = Dum()
        response.request = Dum()
        response.request.url = 'https://example2.com'
        
        eds.rebuild_auth(prepared_request, response)