"""
PyDap DataSource
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import logging

import numpy as np
import traitlets as tl

# Helper utility for optional imports
from lazy_import import lazy_module, lazy_class

# Internal dependencies
from podpac.core import authentication
from podpac.core.utils import common_doc, cached_property
from podpac.core.data.datasource import COMMON_DATA_DOC, DataSource

# Optional dependencies
pydap = lazy_module("pydap")
lazy_module("pydap.client")
lazy_module("pydap.model")


_logger = logging.getLogger(__name__)


@common_doc(COMMON_DATA_DOC)
class PyDAP(DataSource):
    """Create a DataSource from an OpenDAP server feed.
    
    Attributes
    ----------
    auth_class : :class:`podpac.authentication.Session`
        :class:`requests.Session` derived class providing authentication credentials.
        When username and password are provided, an auth_session is created using this class.
    auth_session : :class:`podpac.authentication.Session`
        Instance of the auth_class. This is created if username and password is supplied, but this object can also be
        supplied directly
    data_key : str
        Pydap 'key' for the data to be retrieved from the server. Datasource may have multiple keys, so this key
        determines which variable is returned from the source.
    dataset : pydap.model.DatasetType
        The open pydap dataset. This is provided for troubleshooting.
    native_coordinates : Coordinates
        {native_coordinates}
    password : str, optional
        Password used for authenticating against OpenDAP server. WARNING: this is stored as plain-text, provide
        auth_session instead if you have security concerns.
    source : str
        URL of the OpenDAP server.
    username : str, optional
        Username used for authenticating against OpenDAP server. WARNING: this is stored as plain-text, provide
        auth_session instead if you have security concerns.
    """

    source = tl.Unicode().tag(attr=True)
    data_key = tl.Unicode().tag(attr=True)

    _repr_keys = ["source"]

    # auth, to be replaced
    auth_class = tl.Type(default_value=authentication.Session)
    auth_session = tl.Instance(authentication.Session, allow_none=True)
    username = tl.Unicode(default_value=None, allow_none=True)
    password = tl.Unicode(default_value=None, allow_none=True)

    @tl.default("auth_session")
    def _auth_session_default(self):

        # requires username and password
        if not self.username or not self.password:
            return None

        # instantiate and check url
        try:
            session = self.auth_class(username=self.username, password=self.password)
            session.get(self.source + ".dds")
        except:
            # TODO: catch a 403 error
            return None

        return session

    @common_doc(COMMON_DATA_DOC)
    @tl.default("native_coordinates")
    def _default_native_coordinates(self):
        """{get_native_coordinates}
        
        Raises
        ------
        NotImplementedError
            PyDAP cannot create coordinates. A child class must implement this method.
        """
        raise NotImplementedError("PyDAP cannot create coordinates. A child class must implement this method.")

    @cached_property
    def dataset(self):
        # auth session
        # if self.auth_session:
        try:
            return self._open_url()
        except Exception:
            # TODO handle a 403 error
            # TODO: Check Url (probably inefficient...)
            try:
                self.auth_session.get(self.source + ".dds")
                return self._open_url()
            except Exception:
                # TODO: handle 403 error
                _logger.exception("Error opening PyDap url '%s'" % self.source)
                raise RuntimeError("Could not open PyDap url '%s'.\nCheck login credentials." % self.source)

    def _open_url(self):
        return pydap.client.open_url(self.source, session=self.auth_session)

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        data = self.dataset[self.data_key][tuple(coordinates_index)]
        # PyDAP 3.2.1 gives a numpy array for the above, whereas 3.2.2 needs the .data attribute to get a numpy array
        if not isinstance(data, np.ndarray) and hasattr(data, "data"):
            data = data.data
        d = self.create_output_array(coordinates, data=data.reshape(coordinates.shape))
        return d

    @cached_property
    def keys(self):
        """The list of available keys from the OpenDAP dataset.
        
        Returns
        -------
        List
            The list of available keys from the OpenDAP dataset. Any of these keys can be set as self.data_key
        """
        return self.dataset.keys()
