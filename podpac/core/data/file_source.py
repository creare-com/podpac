"""
Datasources from files
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import sys

if sys.version_info.major == 2:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen

from io import BytesIO
import logging

import traitlets as tl
import xarray as xr

from lazy_import import lazy_module, lazy_class

boto3 = lazy_module("boto3")
s3fs = lazy_module("s3fs")
requests = lazy_module("requests")

from podpac.core.utils import common_doc, cached_property
from podpac.core.cache.utils import expiration_timestamp
from podpac.core.coordinates import Coordinates
from podpac.core.authentication import S3Mixin
from podpac.core.data.datasource import COMMON_DATA_DOC, DataSource

# TODO common doc
_logger = logging.getLogger(__name__)


class BaseFileSource(DataSource):
    """
    Base class for data sources loaded from file.

    Attributes
    ----------
    source : str
        Path to the data source.
    coordinates : :class:`podpac.Coordinates`
        {coordinates}
    dataset : Any
        dataset object
    """

    source = tl.Unicode().tag(attr=True, required=True)

    # list of attribute names, used by __repr__ and __str__ to display minimal info about the node
    _repr_keys = ["source"]

    @tl.default("source")
    def _default_source(self):
        raise ValueError("%s 'source' required" % self.__class__.__name__)

    # -------------------------------------------------------------------------
    # public api properties and methods
    # -------------------------------------------------------------------------

    @property
    def dataset(self):
        raise NotImplementedError()

    def close_dataset(self):
        """Close opened resources. Subclasses should implement if appropriate."""
        pass


class LoadFileMixin(S3Mixin):
    """
    Mixin to load and cache files using various transport protocols.

    Attributes
    ----------
    cache_dataset : bool
        Default is False. Whether to cache the dataset after loading (as an optimization).
    """

    cache_dataset = tl.Bool(False)
    dataset_expires = tl.Any()
    _file = None

    @tl.validate("dataset_expires")
    def _validate_dataset_expires(self, d):
        expiration_timestamp(d["value"])
        return d["value"]

    @cached_property
    def _dataset_caching_node(self):
        # stub node containing only the source node attr
        return BaseFileSource(source=self.source, cache_ctrl=self.cache_ctrl)

    @cached_property
    def dataset(self):
        # get from the cache
        # use the _dataset_caching_node "stub" here because the only node attr we care about is the source
        if self.cache_dataset and self._dataset_caching_node.has_cache(key="dataset"):
            data = self._dataset_caching_node.get_cache(key="dataset")
            self._file = BytesIO(data)
            return self._open(self._file, cache=False)

        # otherwise, open the file (and cache it if desired)
        if self.source.startswith("s3://"):
            _logger.info("Loading AWS resource: %s" % self.source)
            self._file = self.s3.open(self.source, "rb")
        elif self.source.startswith("http://") or self.source.startswith("https://"):
            _logger.info("Downloading: %s" % self.source)
            response = requests.get(self.source)
            self._file = BytesIO(response.content)
        elif self.source.startswith("ftp://"):
            _logger.info("Downloading: %s" % self.source)
            addinfourl = urlopen(self.source)
            self._file = BytesIO(addinfourl.read())
        elif self.source.startswith("file://"):
            addinfourl = urlopen(self.source)
            self._file = BytesIO(addinfourl.read())
        else:
            self._file = open(self.source, "rb")

        return self._open(self._file)

    def _open(self, f, cache=True):
        if self.cache_dataset and cache:
            self._dataset_caching_node.put_cache(f.read(), key="dataset", expires=self.dataset_expires)
            f.seek(0)
        return self.open_dataset(f)

    def open_dataset(self, f):
        """TODO"""
        raise NotImplementedError()

    def close_dataset(self):
        if self._file is not None:
            self._file.close()


@common_doc(COMMON_DATA_DOC)
class FileKeysMixin(tl.HasTraits):
    """
    Mixin to specify data and coordinates dimensions keys.

    Attributes
    ----------
    lat_key : str
        latitude key, default 'lat'
    lon_key : str
        longitude key, default 'lon'
    time_key : str
        time key, default 'time'
    alt_key : str
        altitude key, default 'alt'
    data_key : str, list
        data key, or list of data keys for multiple-output nodes
    crs : str
        Coordinate reference system of the coordinates.
    cf_time : bool
        decode CF datetimes
    cf_units : str
        units, when decoding CF datetimes
    cf_calendar : str
        calendar, when decoding CF datetimes
    """

    # Other dims?
    data_key = tl.Union([tl.Unicode(), tl.List(trait=tl.Unicode())]).tag(attr=True)
    lat_key = tl.Unicode(default_value="lat").tag(attr=True)
    lon_key = tl.Unicode(default_value="lon").tag(attr=True)
    time_key = tl.Unicode(default_value="time").tag(attr=True)
    alt_key = tl.Unicode(default_value="alt").tag(attr=True)
    crs = tl.Unicode(allow_none=True, default_value=None).tag(attr=True)
    cf_time = tl.Bool(default_value=False).tag(attr=True)
    cf_units = tl.Unicode(allow_none=True, default_value=None).tag(attr=True)
    cf_calendar = tl.Unicode(allow_none=True, default_value=None).tag(attr=True)
    skip_validation = tl.Bool(False).tag(attr=True)

    @property
    def _repr_keys(self):
        """list of attribute names, used by __repr__ and __str__ to display minimal info about the node"""
        keys = ["source"]
        if len(self.available_data_keys) > 1 and not isinstance(self.data_key, list):
            keys.append("data_key")
        return keys

    @tl.default("data_key")
    def _default_data_key(self):
        if len(self.available_data_keys) == 1:
            return self.available_data_keys[0]
        else:
            return self.available_data_keys

    @tl.validate("data_key")
    def _validate_data_key(self, d):
        keys = d["value"]
        if self.skip_validation:
            return keys
        if not isinstance(keys, list):
            keys = [d["value"]]
        for key in keys:
            if key not in self.available_data_keys:
                raise ValueError("Invalid data_key '%s', available keys are %s" % (key, self.available_data_keys))
        return d["value"]

    @tl.default("outputs")
    def _default_outputs(self):
        if not isinstance(self.data_key, list):
            return None
        else:
            return self.data_key

    @tl.validate("outputs")
    def _validate_outputs(self, d):
        value = d["value"]
        if self.skip_validation:
            return value
        if not isinstance(self.data_key, list):
            if value is not None:
                raise TypeError("outputs must be None for single-output nodes")
        else:
            if value is None:
                raise TypeError("outputs and data_key mismatch (outputs=None, data_key=%s)" % self.data_key)
            if len(value) != len(self.data_key):
                raise ValueError("outputs and data_key size mismatch (%d != %d)" % (len(value), len(self.data_key)))
        return value

    # -------------------------------------------------------------------------
    # public api properties and methods
    # -------------------------------------------------------------------------

    @property
    def keys(self):
        raise NotImplementedError

    @property
    def dims(self):
        raise NotImplementedError

    @cached_property
    def available_data_keys(self):
        """available data keys"""
        dim_keys = [self.lat_key, self.lon_key, self.alt_key, self.time_key]
        keys = [key for key in self.keys if key not in dim_keys]
        if len(keys) == 0:
            raise ValueError("No data keys found in '%s'" % self.source)
        return keys

    def _lookup_key(self, dim):
        lookup = {"lat": self.lat_key, "lon": self.lon_key, "alt": self.alt_key, "time": self.time_key}
        return lookup[dim]

    @common_doc(COMMON_DATA_DOC)
    def get_coordinates(self):
        """{get_coordinates}"""

        cs = [self.dataset[self._lookup_key(dim)] for dim in self.dims]
        if self.cf_time and "time" in self.dims:
            time_ind = self.dims.index("time")
            cs[time_ind] = xr.coding.times.decode_cf_datetime(cs[time_ind], self.cf_units, self.cf_calendar)
        return Coordinates(cs, dims=self.dims, crs=self.crs)
