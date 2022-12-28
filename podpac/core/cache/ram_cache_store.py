from __future__ import division, print_function, absolute_import

import threading
import copy
import warnings
import os
import time

import psutil

from podpac.core.settings import settings
from podpac.core.cache.utils import CacheException, CacheWildCard, expiration_timestamp
from podpac.core.cache.cache_store import CacheStore

_thread_local = threading.local()


class RamCacheStore(CacheStore):
    """
    RAM CacheStore.

    Notes
    -----
     * the cache is thread-safe, but not yet accessible across separate processes
     * there is not yet a max RAM usage setting or a removal policy.
    """

    cache_mode = "ram"
    cache_modes = set(["ram", "all"])
    _limit_setting = "RAM_CACHE_MAX_BYTES"

    def __init__(self, max_size=None, use_settings_limit=True):
        """Summary

        Raises
        ------
        CacheException
            Description

        Parameters
        ----------
        max_size : None, optional
            Maximum allowed size of the cache store in bytes. Defaults to podpac 'S3_CACHE_MAX_BYTES' setting, or no limit if this setting does not exist.
        use_settings_limit : bool, optional
            Use podpac settings to determine cache limits if True, this will also cause subsequent runtime changes to podpac settings module to effect the limit on this cache. Default is True.
        """
        if not settings["RAM_CACHE_ENABLED"]:
            raise CacheException("RAM cache is disabled in the podpac settings.")

        super(CacheStore, self).__init__()

    def _get_full_key(self, node, key, coordinates):
        return (node.json, key, coordinates.json if coordinates is not None else None)

    @property
    def size(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss  # this is actually the total size of the process

    def put(self, node, data, item, coordinates=None, expires=None, update=True):
        """Cache data for specified node.

        Parameters
        ------------
        node : Node
            node requesting storage.
        data : any
            Data to cache
        item : str
            Cached object item, e.g. 'output'.
        coordinates : :class:`podpac.Coordinates`, optional
            Coordinates for which cached object should be retrieved, for coordinate-dependent data such as evaluation output
        expires : float, datetime, timedelta
            Expiration date. If a timedelta is supplied, the expiration date will be calculated from the current time.
        update : bool
            If True existing data in cache will be updated with `data`, If False, error will be thrown if attempting put something into the cache with the same node, key, coordinates of an existing entry.
        """

        if not hasattr(_thread_local, "cache"):
            _thread_local.cache = {}

        full_key = self._get_full_key(node, item, coordinates)

        if not update and self.has(node, item, coordinates):
            raise CacheException("Cache entry already exists. Use update=True to overwrite.")

        self.rem(node, item, coordinates)

        # check size
        if self.max_size is not None:
            if self.size > self.max_size:
                # cleanup and check again
                self.cleanup()

            if self.size > self.max_size:
                # TODO removal policy (using create time, last access, etc)
                warnings.warn(
                    "Warning: Process is using more RAM than the specified limit in settings.RAM_CACHE_MAX_BYTES. "
                    "No longer caching. Consider increasing this limit or try clearing the cache "
                    "(e.g. podpac.utils.clear_cache(mode='RAM') to clear ALL cached results in RAM)",
                    UserWarning,
                )
                return False

        # store
        entry = {"data": data, "created": time.time(), "accessed": None, "expires": expiration_timestamp(expires)}

        _thread_local.cache[full_key] = entry

    def get(self, node, item, coordinates=None):
        """Get cached data for this node.

        Parameters
        ------------
        node : Node
            node requesting storage.
        item : str
            Cached object item, e.g. 'output'.
        coordinates : :class:`podpac.Coordinates`, optional
            Coordinates for which cached object should be retrieved, for coordinate-dependent data such as evaluation output

        Returns
        -------
        data : any
            The cached data.

        Raises
        -------
        CacheException
            If the data is not in the cache, or is expired.
        """

        if not hasattr(_thread_local, "cache"):
            _thread_local.cache = {}

        full_key = self._get_full_key(node, item, coordinates)

        if full_key not in _thread_local.cache:
            raise CacheException("Cache miss. Requested data not found.")

        if self._expired(full_key):
            raise CacheException("Cache miss. Requested data expired.")

        self._set_metadata(full_key, "accessed", time.time())
        return copy.deepcopy(_thread_local.cache[full_key]["data"])

    def has(self, node, item, coordinates=None):
        """Check for cached data for this node

        Parameters
        ------------
        node : Node
            node requesting storage.
        item : str
            Cached object item, e.g. 'output'.
        coordinates: Coordinate, optional
            Coordinates for which cached object should be checked

        Returns
        -------
        has_cache : bool
             True if there as a cached object for this node for the given key and coordinates.
        """

        if not hasattr(_thread_local, "cache"):
            _thread_local.cache = {}

        full_key = self._get_full_key(node, item, coordinates)
        return full_key in _thread_local.cache and not self._expired(full_key)

    def rem(self, node, item=CacheWildCard(), coordinates=CacheWildCard()):
        """Delete cached data for this node.

        Parameters
        ------------
        node : Node
            node requesting storage.
        key : str, optional
            Delete only cached objects with this key.
        coordinates : :class:`podpac.Coordinates`
            Delete only cached objects for these coordinates.
        """

        if not hasattr(_thread_local, "cache"):
            _thread_local.cache = {}

        node_key = node.json

        if not isinstance(coordinates, CacheWildCard):
            coordinates_key = coordinates.json if coordinates is not None else None

        # loop through keys looking for matches
        rem_keys = []
        for nk, k, ck in _thread_local.cache.keys():
            if nk != node_key:
                continue
            if not isinstance(item, CacheWildCard) and k != item:
                continue
            if not isinstance(coordinates, CacheWildCard) and ck != coordinates_key:
                continue

            rem_keys.append((nk, k, ck))

        for k in rem_keys:
            del _thread_local.cache[k]

    def clear(self):
        """Remove all entries from the cache."""

        if hasattr(_thread_local, "cache"):
            _thread_local.cache.clear()

    def cleanup(self):
        """Remove all expired entries."""
        for full_key, entry in list(_thread_local.cache.items()):
            if entry["expires"] is not None and time.time() >= entry["expires"]:
                del _thread_local.cache[full_key]

    # -------------------------------------------------------------------------
    # helper methods
    # -------------------------------------------------------------------------

    def _get_metadata(self, full_key, key):
        return _thread_local.cache[full_key].get(key)

    def _set_metadata(self, full_key, key, value):
        _thread_local.cache[full_key][key] = value

    def _expired(self, full_key):
        """Check if the given entry is expired. Expired entries are removed."""

        expires = self._get_metadata(full_key, "expires")

        if expires is not None and time.time() >= expires:
            del _thread_local.cache[full_key]
            return True

        return False
