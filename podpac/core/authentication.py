"""
PODPAC Authentication
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
from podpac.core.settings import settings

class SessionWithHeaderRedirection(requests.Session):
    """
    Modified from: https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+Python
    overriding requests.Session.rebuild_auth to maintain headers when redirected
    
    Attributes
    ----------
    auth : tuple
        (username, password) string in plain text
    hostname : str
        Host address (eg. http://example.com) that gets authenticated
    hostname_regex : str
        Regex used to match redirected hostname if different from :attr:`self.hostname`
    password : str
        Password used for authentication.
        Loaded from podpac settings file using password@:attr:`self.hostname` as the key.
    username : str
        Username used for authentication.
        Loaded from podpac settings file using username@:attr:`self.hostname` as the key.
    """

    hostname = ''
    hostname_regex = None
    username = None
    password = None
    auth = tuple()

    def __init__(self, username=None, password=None, hostname_regex=None):

        super(SessionWithHeaderRedirection, self).__init__()
        
        if username is None:
            username = settings['username@' + self.hostname]
        
        if password is None:
            password = settings['password@' + self.hostname]
        
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
        None

        """
        headers = prepared_request.headers
        url = prepared_request.url

        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            # if redirected hostname is different than original and the self hostname
            # see if the redirected hostname can be found in hostname_regex
            if (original_parsed.hostname != redirect_parsed.hostname) \
                    and redirect_parsed.hostname != self.hostname and \
                    original_parsed.hostname != self.hostname:
                if self.hostname_regex is not None and self.hostname_regex.match(redirect_parsed.hostname):
                    pass
                else:
                    del headers['Authorization']

        return
    
    def update_login(self, username=None, password=None):
        """Summary
        
        Parameters
        ----------
        username : str, optional
            Username input
        password : str, optional
            Password input
        """
        print("Updating login information for: ", self.hostname)
        
        if username is None:
            username = input("Username: ")
        
        settings['username@' + self.hostname] = username
        
        if password is None:
            password = getpass.getpass()

        settings['password@' + self.hostname] = password
        
        self.auth = (username, password)


class EarthDataSession(SessionWithHeaderRedirection):
    """Nasa EarthData Authentication Session
    """
    
    hostname = 'urs.earthdata.nasa.gov'
    
        
    
