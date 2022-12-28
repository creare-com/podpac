"""
PyDap DataSource
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import logging
import time

import numpy as np
import traitlets as tl
import requests
from webob.exc import HTTPError

# Helper utility for optional imports
from lazy_import import lazy_module, lazy_class

# Internal dependencies
from podpac.core import authentication
from podpac.core.utils import common_doc, cached_property
from podpac.core.data.datasource import COMMON_DATA_DOC, DataSource
from podpac.core.interpolation.interpolation import InterpolationMixin


# Optional dependencies
pydap = lazy_module("pydap")
lazy_module("pydap.client")
lazy_module("pydap.model")


_logger = logging.getLogger(__name__)


@common_doc(COMMON_DATA_DOC)
class PyDAPRaw(authentication.RequestsSessionMixin, DataSource):
    """Create a DataSource from an OpenDAP server feed.

    Attributes
    ----------
    data_key : str
        Pydap 'key' for the data to be retrieved from the server. Datasource may have multiple keys, so this key
        determines which variable is returned from the source.
    dataset : pydap.model.DatasetType
        The open pydap dataset. This is provided for troubleshooting.
    coordinates : :class:`podpac.Coordinates`
        {coordinates}
    source : str
        URL of the OpenDAP server.

    See Also
    --------
    PyDAP : Interpolated OpenDAP datasource for general use.
    """

    source = tl.Unicode().tag(attr=True, required=True)
    data_key = tl.Unicode().tag(attr=True, required=True)
    server_throttle_sleep_time = tl.Float(
        default_value=0.001, help="Some server have a throttling time for requests per period. "
    ).tag(attr=True)
    server_throttle_retries = tl.Int(default_value=100, help="Number of retries for a throttled server.").tag(attr=True)

    # list of attribute names, used by __repr__ and __str__ to display minimal info about the node
    _repr_keys = ["source"]
    coordinate_index_type = "slice"

    # hostname for RequestsSession is source. Try parsing off netloc
    @tl.default("hostname")
    def _hostname(self):
        try:
            return requests.utils.urlparse(self.source).netloc
        except:
            return self.source

    @common_doc(COMMON_DATA_DOC)
    def get_coordinates(self):
        """{get_coordinates}

        Raises
        ------
        NotImplementedError
            PyDAP cannot create coordinates. A child class must implement this method.
        """
        raise NotImplementedError("PyDAP cannot create coordinates. A child class must implement this method.")

    @cached_property
    def dataset(self):
        # auth session
        try:
            return self._open_url()
        except HTTPError as e:
            # I need the 500 because pydap re-raises HTTPError wihout setting the code
            if not (e.code != 400 or e.code != 300 or e.code != 500):
                raise e
            # Check Url (probably inefficient..., but worth a try to get authenticated)
            try:
                self.session.get(self.source + ".dds")
                return self._open_url()
            except HTTPError as e:
                if e.code != 400:
                    raise e
                _logger.exception("Error opening PyDap url '%s'" % self.source)
                raise HTTPError("Could not open PyDap url '%s'.\nCheck login credentials." % self.source)

    def _open_url(self):
        return pydap.client.open_url(self.source, session=self.session)

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}"""
        data = None
        count = self.server_throttle_retries
        while data is None:
            count -= 1
            try:
                data = self.dataset[self.data_key][tuple(coordinates_index)]
            except HTTPError as e:
                if e.code == 500 and str(e).startswith("503") and count > 0:  # Service temporarily unavailable
                    time.sleep(self.server_throttle_sleep_time)
                    continue
                raise e
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


class PyDAP(InterpolationMixin, PyDAPRaw):
    """OpenDAP datasource with interpolation."""

    pass
