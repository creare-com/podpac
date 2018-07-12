from __future__ import division, unicode_literals, print_function, absolute_import

import pytest
import sys
from io import StringIO

from podpac.core.authentication import *

class TestAuthentication(object):
    def test_earth_data_session_update(self):
        eds = EarthDataSession()
        eds.update_login('testuser', 'testpassword')
        eds = EarthDataSession()
        assert(eds.auth == ('testuser', 'testpassword'))
               
    def test_earth_data_session_update_input(self):
        eds = EarthDataSession()
        sys.stdin = StringIO('testuser2\ntestpass2\n')
        eds.update_login()
        eds = EarthDataSession()
        assert(eds.auth == ('testuser2', 'testpass2'))
        
    def test_earth_data_session_rebuild_auth(self):
        eds = EarthDataSession() 
        class Dum(object):
            pass
        
        prepared_request = Dum()
        prepared_request.headers = {'Authorization': 0}
        prepared_request.url = 'https://example.com'
        
        response = Dum()
        response.request = Dum()
        response.request.url = 'https://example2.com'
        
        eds.rebuild_auth(prepared_request, response)