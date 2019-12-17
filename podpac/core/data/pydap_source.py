"""
PyDap DataSource
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np
import traitlets as tl

# Helper utility for optional imports
from lazy_import import lazy_module, lazy_class

# Internal dependencies
from podpac.core import authentication
from podpac.core.utils import common_doc
from podpac.core.data.datasource import COMMON_DATA_DOC, DataSource

# Optional dependencies
pydap = lazy_module("pydap")
lazy_module("pydap.client")
lazy_module("pydap.model")


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
    datakey : str
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

    source = tl.Unicode().tag(readonly=True)
    dataset = tl.Instance("pydap.model.DatasetType").tag(readonly=True)

    # node attrs
    datakey = tl.Unicode().tag(attr=True)

    # optional inputs
    auth_class = tl.Type(authentication.Session)
    auth_session = tl.Instance(authentication.Session, allow_none=True)
    username = tl.Unicode(default_value=None, allow_none=True)
    password = tl.Unicode(default_value=None, allow_none=True)

    @tl.default("auth_session")
    def _auth_session_default(self):

        # requires username and password
        if not self.username or not self.password:
            return None

        # requires auth_class
        # TODO: default auth_class?
        if not self.auth_class:
            return None

        # instantiate and check utl
        try:
            session = self.auth_class(username=self.username, password=self.password)
            session.get(self.source + ".dds")
        except:
            # TODO: catch a 403 error
            return None

        return session

    @tl.default("dataset")
    def _open_dataset(self):
        """Summary
        
        Parameters
        ----------
        source : str, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        """

        # auth session
        # if self.auth_session:
        try:
            dataset = self._open_url()
        except Exception:
            # TODO handle a 403 error
            # TODO: Check Url (probably inefficient...)
            try:
                self.auth_session.get(self.source + ".dds")
                dataset = self._open_url()
            except Exception:
                # TODO: handle 403 error
                print("Warning, dataset could not be opened. Check login credentials.")
                dataset = None

        return dataset

    def _open_url(self):
        return pydap.client.open_url(self.source, session=self.auth_session)

    @common_doc(COMMON_DATA_DOC)
    def get_native_coordinates(self):
        """{get_native_coordinates}
        
        Raises
        ------
        NotImplementedError
            DAP has no mechanism for creating coordinates automatically, so this is left up to child classes.
        """
        raise NotImplementedError(
            "DAP has no mechanism for creating coordinates"
            + ", so this is left up to child class "
            + "implementations."
        )

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        data = self.dataset[self.datakey][tuple(coordinates_index)]
        # PyDAP 3.2.1 gives a numpy array for the above, whereas 3.2.2 needs the .data attribute to get a numpy array
        if not isinstance(data, np.ndarray) and hasattr(data, "data"):
            data = data.data
        d = self.create_output_array(coordinates, data=data.reshape(coordinates.shape))
        return d

    @property
    def keys(self):
        """The list of available keys from the OpenDAP dataset.
        
        Returns
        -------
        List
            The list of available keys from the OpenDAP dataset. Any of these keys can be set as self.datakey
        """
        return self.dataset.keys()
