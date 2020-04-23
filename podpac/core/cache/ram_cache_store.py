from __future__ import division, print_function, absolute_import

import threading
import copy
import warnings
import os
import psutil

from podpac.core.settings import settings
from podpac.core.cache.utils import CacheException, CacheWildCard
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

    def put(self, node, data, key, coordinates=None, update=True):
        """Cache data for specified node.
        
        Parameters
        ------------
        node : Node
            node requesting storage.
        data : any
            Data to cache
        key : str
            Cached object key, e.g. 'output'.
        coordinates : :class:`podpac.Coordinates`, optional
            Coordinates for which cached object should be retrieved, for coordinate-dependent data such as evaluation output
        update : bool
            If True existing data in cache will be updated with `data`, If False, error will be thrown if attempting put something into the cache with the same node, key, coordinates of an existing entry.
        """

        if not hasattr(_thread_local, "cache"):
            _thread_local.cache = {}

        full_key = self._get_full_key(node, key, coordinates)

        if not update and full_key in _thread_local.cache:
            raise CacheException("Cache entry already exists. Use update=True to overwrite.")

        self.rem(node, key, coordinates)

        if self.max_size is not None and self.size >= self.max_size:
            #     # TODO removal policy
            warnings.warn(
                "Warning: Process is using more RAM than the specified limit in settings.RAM_CACHE_MAX_BYTES. No longer caching. Consider increasing this limit or try clearing the cache (e.g. podpac.utils.clear_cache(mode='RAM') to clear ALL cached results in RAM)",
                UserWarning,
            )
            return False

        # TODO include insert date, last retrieval date, and/or # retrievals for use in a removal policy
        _thread_local.cache[full_key] = data

    def get(self, node, key, coordinates=None):
        """Get cached data for this node.
        
        Parameters
        ------------
        node : Node
            node requesting storage.
        key : str
            Cached object key, e.g. 'output'.
        coordinates : :class:`podpac.Coordinates`, optional
            Coordinates for which cached object should be retrieved, for coordinate-dependent data such as evaluation output
            
        Returns
        -------
        data : any
            The cached data.
        
        Raises
        -------
        CacheError
            If the data is not in the cache.
        """

        if not hasattr(_thread_local, "cache"):
            _thread_local.cache = {}

        full_key = self._get_full_key(node, key, coordinates)

        if full_key not in _thread_local.cache:
            raise CacheException("Cache miss. Requested data not found.")

        return copy.deepcopy(_thread_local.cache[full_key])

    def has(self, node, key, coordinates=None):
        """Check for cached data for this node
        
        Parameters
        ------------
        node : Node
            node requesting storage.
        key : str
            Cached object key, e.g. 'output'.
        coordinates: Coordinate, optional
            Coordinates for which cached object should be checked
        
        Returns
        -------
        has_cache : bool
             True if there as a cached object for this node for the given key and coordinates.
        """

        if not hasattr(_thread_local, "cache"):
            _thread_local.cache = {}

        full_key = self._get_full_key(node, key, coordinates)
        return full_key in _thread_local.cache

    def rem(self, node, key=CacheWildCard(), coordinates=CacheWildCard()):
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
            if not isinstance(key, CacheWildCard) and k != key:
                continue
            if not isinstance(coordinates, CacheWildCard) and ck != coordinates_key:
                continue

            rem_keys.append((nk, k, ck))

        for k in rem_keys:
            del _thread_local.cache[k]

    def clear(self):
        _thread_local.cache = {}
