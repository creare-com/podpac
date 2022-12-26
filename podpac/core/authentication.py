"""
PODPAC Authentication
"""


import getpass
import logging

import requests
import traitlets as tl
from lazy_import import lazy_module, lazy_function

from podpac.core.settings import settings
from podpac.core.utils import cached_property

# Optional dependencies
pydap_setup_session = lazy_function("pydap.cas.urs.setup_session")

_log = logging.getLogger(__name__)


def set_credentials(hostname, uname=None, password=None):
    """Set authentication credentials for a remote URL in the :class:`podpac.settings`.

    Parameters
    ----------
    hostname : str
        Hostname for `uname` and `password`.
    uname : str, optional
        Username to store in settings for `hostname`.
        If no username is provided and the username does not already exist in the settings,
        the user will be prompted to enter one.
    password : str, optional
        Password to store in settings for `hostname`
        If no password is provided and the password does not already exist in the settings,
        the user will be prompted to enter one.
    """

    if hostname is None or hostname == "":
        raise ValueError("`hostname` must be defined")

    # see whats stored in settings already
    u_settings = settings.get("username@{}".format(hostname))
    p_settings = settings.get("password@{}".format(hostname))

    # get username from 1. function input 2. settings 3. python input()
    u = uname or u_settings or getpass.getpass("Username: ")
    p = password or p_settings or getpass.getpass()

    # set values in settings
    settings["username@{}".format(hostname)] = u
    settings["password@{}".format(hostname)] = p

    _log.debug("Set credentials for hostname {}".format(hostname))


class RequestsSessionMixin(tl.HasTraits):
    hostname = tl.Unicode(allow_none=False)
    auth_required = tl.Bool(default_value=False)

    @property
    def username(self):
        """Returns username stored in settings for accessing `self.hostname`.
        The username is stored under key `username@<hostname>`

        Returns
        -------
        str
            username stored in settings for accessing `self.hostname`

        Raises
        ------
        ValueError
            Raises a ValueError if not username is stored in settings for `self.hostname`
        """
        key = "username@{}".format(self.hostname)
        username = settings.get(key)
        if not username:
            raise ValueError(
                "No username found for hostname '{0}'. Use `{1}.set_credentials(username='<username>', password='<password>') to store credentials for this host".format(
                    self.hostname, self.__class__.__name__
                )
            )

        return username

    @property
    def password(self):
        """Returns password stored in settings for accessing `self.hostname`.
        The password is stored under key `password@<hostname>`

        Returns
        -------
        str
            password stored in settings for accessing `self.hostname`

        Raises
        ------
        ValueError
            Raises a ValueError if not password is stored in settings for `self.hostname`
        """
        key = "password@{}".format(self.hostname)
        password = settings.get(key)
        if not password:
            raise ValueError(
                "No password found for hostname {0}. Use `{1}.set_credentials(username='<username>', password='<password>') to store credentials for this host".format(
                    self.hostname, self.__class__.__name__
                )
            )

        return password

    @cached_property
    def session(self):
        """Requests Session object for making calls to remote `self.hostname`
        See https://2.python-requests.org/en/master/api/#sessionapi

        Returns
        -------
        :class:requests.Session
            Requests Session class with `auth` attribute defined
        """
        return self._create_session()

    def set_credentials(self, username=None, password=None):
        """Shortcut to :func:`podpac.authentication.set_crendentials` using class member :attr:`self.hostname` for the hostname

        Parameters
        ----------
        username : str, optional
            Username to store in settings for `self.hostname`.
            If no username is provided and the username does not already exist in the settings,
            the user will be prompted to enter one.
        password : str, optional
            Password to store in settings for `self.hostname`
            If no password is provided and the password does not already exist in the settings,
            the user will be prompted to enter one.
        """
        return set_credentials(self.hostname, uname=username, password=password)

    def _create_session(self):
        """Creates a :class:`requests.Session` with username and password defined

        Returns
        -------
        :class:`requests.Session`
        """
        s = requests.Session()

        try:
            s.auth = (self.username, self.password)
        except ValueError as e:
            if self.auth_required:
                raise e
            else:
                _log.warning("No auth provided for session")

        return s


class NASAURSSessionMixin(RequestsSessionMixin):
    check_url = tl.Unicode()
    hostname = tl.Unicode(default_value="urs.earthdata.nasa.gov")
    auth_required = tl.Bool(True)

    def _create_session(self):
        """Creates an authenticated :class:`requests.Session` with username and password defined

        Returns
        -------
        :class:`requests.Session`

        Notes
        -----
        The session is authenticated against the user-provided self.check_url
        """

        try:
            s = pydap_setup_session(self.username, self.password, check_url=self.check_url)
        except ValueError as e:
            if self.auth_required:
                raise e
            else:
                _log.warning("No auth provided for session")

        return s


class S3Mixin(tl.HasTraits):
    """Mixin to add S3 credentials and access to a Node."""

    anon = tl.Bool(False).tag(attr=True)
    aws_access_key_id = tl.Unicode(allow_none=True)
    aws_secret_access_key = tl.Unicode(allow_none=True)
    aws_region_name = tl.Unicode(allow_none=True)
    aws_client_kwargs = tl.Dict()
    config_kwargs = tl.Dict()
    aws_requester_pays = tl.Bool(False)

    @tl.default("aws_access_key_id")
    def _get_access_key_id(self):
        return settings["AWS_ACCESS_KEY_ID"]

    @tl.default("aws_secret_access_key")
    def _get_secret_access_key(self):
        return settings["AWS_SECRET_ACCESS_KEY"]

    @tl.default("aws_region_name")
    def _get_region_name(self):
        return settings["AWS_REGION_NAME"]

    @tl.default("aws_requester_pays")
    def _get_requester_pays(self):
        return settings["AWS_REQUESTER_PAYS"]

    @cached_property
    def s3(self):
        # this has to be done here for multithreading to work
        s3fs = lazy_module("s3fs")

        if self.anon:
            return s3fs.S3FileSystem(anon=True, client_kwargs=self.aws_client_kwargs)
        else:
            return s3fs.S3FileSystem(
                key=self.aws_access_key_id,
                secret=self.aws_secret_access_key,
                client_kwargs=self.aws_client_kwargs,
                config_kwargs=self.config_kwargs,
                requester_pays=self.aws_requester_pays,
            )
