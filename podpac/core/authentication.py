"""
PODPAC Authentication
"""


from __future__ import division, unicode_literals, print_function, absolute_import


import sys
import getpass
import re

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

from podpac.core import utils
from podpac.core.settings import settings


class Session(requests.Session):
    """Base Class for authentication in PODPAC
    
    Attributes
    ----------
    auth : tuple
        (username, password) string in plain text
    hostname : str
        Host address (eg. http://example.com) that gets authenticated.
        By default, this is set to 'urs.earthdata.nasa.gov'
    password : str
        Password used for authentication.
        Loaded from podpac settings file using password@:attr:`hostname` as the key.
    username : str
        Username used for authentication.
        Loaded from podpac settings file using username@:attr:`hostname` as the key.
    """

    def __init__(self, hostname='', username=None, password=None):

        # requests __init__
        super(Session, self).__init__()

        self.hostname = hostname
        self.username = username
        self.password = password

        # load username/password from settings
        if self.username is None:
            self.username = settings['username@' + self.hostname]
        
        if self.password is None:
            self.password = settings['password@' + self.hostname]
        
        self.auth = (self.username, self.password)


class EarthDataSession(Session):
    """
    Modified from: https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+Python
    overriding requests.Session.rebuild_auth to maintain headers when redirected
    
    Attributes
    ----------
    product_url : str
        Url to NSIDC product OpenDAP server
    product_url_regex : str
        Regex used to match redirected hostname if different from :attr:`self.hostname`
    """

    # make sure attributes are persistent across all EarthDataSession classes
    hostname = None
    username = None
    password = None
    auth = tuple()

    def __init__(self, product_url='', **kwargs):

        # override hostname with earthdata url
        kwargs['hostname'] = 'urs.earthdata.nasa.gov'

        # Session init
        super(EarthDataSession, self).__init__(**kwargs)
        
        # store product_url
        self.product_url = product_url
        
        # parse product_url for hostname
        product_url_hostname = requests.utils.urlparse(self.product_url).hostname

        # make all numbers in product_url_hostname wildcards
        self.product_url_regex = re.compile(re.sub(r'\d', r'\\d', product_url_hostname)) \
                              if product_url_hostname is not None else None


    def rebuild_auth(self, prepared_request, response):
        """
        Overrides from the library to keep headers when redirected to or from
        the NASA auth host.
        
        Parameters
        ----------
        prepared_request : requests.Request
            Description
        response : requests.Response
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

            # delete Authorization headers if original and redirect do not match
            # is not in product_url_regex
            if (original_parsed.hostname != redirect_parsed.hostname) \
                    and redirect_parsed.hostname != self.hostname and \
                    original_parsed.hostname != self.hostname:

                # if redirect matches product_url_regex, then allow the headers to stay
                if self.product_url_regex is not None and self.product_url_regex.match(redirect_parsed.hostname):
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
