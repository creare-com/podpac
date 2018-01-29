from __future__ import division, unicode_literals, print_function, absolute_import


import sys

# python 2/3 compatibility
if sys.version_info.major < 3:
    input = raw_input 
else:
    from builtins import input

# Optional PODPAC dependency
try:
    import requests
except:
    class Dum(object):
        def __init__(self, *args, **kwargs):
            pass
    requests = Dum()
    requests.Session = Dum()

# Internal dependencies
from podpac.core import utils

# overriding requests.Session.rebuild_auth to mantain headers when redirected
class SessionWithHeaderRedirection(requests.Session):
    AUTH_HOST = ''
    def __init__(self, username=None, password=None):
        super(SessionWithHeaderRedirection, self).__init__()
        if username is None:
            username = utils.load_setting('username@' + self.AUTH_HOST)
        if password is None:
            password = utils.load_setting('password@' + self.AUTH_HOST)
        self.auth = (username, password)

    # Overrides from the library to keep headers when redirected to or from
    # the NASA auth host.
    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url

        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            if (original_parsed.hostname != redirect_parsed.hostname) and \
                    redirect_parsed.hostname != self.AUTH_HOST and \
                    original_parsed.hostname != self.AUTH_HOST:
                del headers['Authorization']

        return
    
    def update_login(self, username=None, password=None):
        print ("Updating login information for", self.AUTH_HOST)
        if username is None:
            username = input("username: ")
            utils.save_setting('username@' + self.AUTH_HOST, username)
        if password is None:
            password = input("password: ")
            utils.save_setting('password@' + self.AUTH_HOST, password)
        
        self.auth = (username, password)
    
class EarthDataSession(SessionWithHeaderRedirection):
    AUTH_HOST = 'urs.earthdata.nasa.gov'
    
        
    