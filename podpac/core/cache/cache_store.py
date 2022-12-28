"""
Defines the common interface for a cache store.
"""

from __future__ import division, print_function, absolute_import

from podpac.core.settings import settings


class CacheStore(object):
    """Abstract parent class for classes representing actual data stores (e.g. RAM, local disk, network storage).
    Includes implementation of common hashing operations and call signature for required abstract methods:
    put(), get(), rem(), has()
    """

    cache_modes = []
    _limit_setting = None

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def max_size(self):
        return settings.get(self._limit_setting)

    @property
    def size(self):
        """Return size of cache store in bytes"""

        raise NotImplementedError

    def put(self, node, data, item, coordinates=None, expires=None, update=True):
        """Cache data for specified node.

        Parameters
        -----------
        node : Node
            node requesting storage.
        data : any
            Data to cache
        item : str
            Cached object item or key, e.g. 'output'.
        coordinates : :class:`podpac.Coordinates`, optional
            Coordinates for which cached object should be retrieved, for coordinate-dependent data such as evaluation output
        expires : float, datetime, timedelta
            Expiration date. If a timedelta is supplied, the expiration date will be calculated from the current time.
        update : bool
            If True existing data in cache will be updated with `data`, If False, error will be thrown if attempting put something into the cache with the same node, key, coordinates of an existing entry.
        """
        raise NotImplementedError

    def get(self, node, item, coordinates=None):
        """Get cached data for this node.

        Parameters
        ------------
        node : Node
            node requesting storage.
        item : str
            Cached object item or key, e.g. 'output'.
        coordinates : :class:`podpac.Coordinates`, optional
            Coordinates for which cached object should be retrieved, for coordinate-dependent data such as evaluation output

        Returns
        -------
        data : any
            The cached data.

        Raises
        -------
        CacheError
            If the data is not in the cache or is expired.
        """
        raise NotImplementedError

    def rem(self, node=None, item=None, coordinates=None):
        """Delete cached data for this node.

        Parameters
        ------------
        node : Node
            node requesting storage.
        item : str, optional
            Delete only cached objects with this item name or key.
        coordinates : :class:`podpac.Coordinates`
            Delete only cached objects for these coordinates.
        """
        raise NotImplementedError

    def has(self, node, item, coordinates=None):
        """Check for cached data for this node

        Parameters
        ------------
        node : Node
            node requesting storage.
        item : str
            Cached object item or key, e.g. 'output'.
        coordinates: Coordinate, optional
            Coordinates for which cached object should be checked

        Returns
        -------
        has_cache : bool
             True if there as a valid cached object for this node for the given key and coordinates.
        """
        raise NotImplementedError

    def clear(self, node):
        """
        Clear all cached data.
        """
        raise NotImplementedError

    def cleanup(self):
        """
        Cache housekeeping, e.g. remove expired entries.
        """
        raise NotImplementedError
