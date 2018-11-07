"""
Authentication Summary
"""


from __future__ import division, unicode_literals, print_function, absolute_import


import sys
import getpass

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
    requests.Session = Dum

# Internal dependencies
from podpac.core import utils

class SessionWithHeaderRedirection(requests.Session):
    """
    Modified from: https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+Python
    overriding requests.Session.rebuild_auth to maintain headers when redirected
    
    Attributes
    ----------
    auth : tuple
        (username, password) string in plain text
    AUTH_HOST : str
        Host address (eg. http://example.com) that gets authenticated
    """

    AUTH_HOST = ''
    hostname_regex = None

    def __init__(self, username=None, password=None, hostname_regex=None):
        '''
        Parameters
        ------------
        username: str, optional
            Username used for authentication. Loaded from podpac settings file using username@`self.AUTH_HOST` as the key.
        password: str
            Password used for authentication. Loaded from podpac settings file using password@`self.AUTH_HOST` as the key.
        '''
        super(SessionWithHeaderRedirection, self).__init__()
        if username is None:
            username = utils.load_setting('username@' + self.AUTH_HOST)
        if password is None:
            password = utils.load_setting('password@' + self.AUTH_HOST)
        self.hostname_regex = hostname_regex
        self.auth = (username, password)

    
    def rebuild_auth(self, prepared_request, response):
        """
        Overrides from the library to keep headers when redirected to or from
        the NASA auth host.
        
        Parameters
        ----------
        prepared_request : TYPE
            Description
        response : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        headers = prepared_request.headers
        url = prepared_request.url

        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            if ((original_parsed.hostname != redirect_parsed.hostname) or ((self.hostname_regex is not None)
                        and not (self.hostname_regex.match(redirect_parsed.hostname)))
                    ) and redirect_parsed.hostname != self.AUTH_HOST and \
                    original_parsed.hostname != self.AUTH_HOST:
                del headers['Authorization']

        return
    
    def update_login(self, username=None, password=None):
        """Summary
        
        Parameters
        ----------
        username : None, optional
            Description
        password : None, optional
            Description
        """
        print("Updating login information for", self.AUTH_HOST)
        if username is None:
            username = input("Username: ")
        utils.save_setting('username@' + self.AUTH_HOST, username)
        if password is None:
            password = getpass.getpass()
        utils.save_setting('password@' + self.AUTH_HOST, password)
        
        self.auth = (username, password)
    
class EarthDataSession(SessionWithHeaderRedirection):
    """Summary
    
    Attributes
    ----------
    AUTH_HOST : str
        Description
    """
    
    AUTH_HOST = 'urs.earthdata.nasa.gov'
    
        
    
