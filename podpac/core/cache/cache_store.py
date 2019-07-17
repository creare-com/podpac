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

    def put(self, node, data, key, coordinates=None, update=False):
        """Cache data for specified node.
        
        Parameters
        ------------
        node : Node
            node requesting storage.
        data : any
            Data to cache
        key : str
            Cached object key, e.g. 'output'.
        coordinates : Coordinates, optional
            Coordinates for which cached object should be retrieved, for coordinate-dependent data such as evaluation output
        update : bool
            If True existing data in cache will be updated with `data`, If False, error will be thrown if attempting put something into the cache with the same node, key, coordinates of an existing entry.
        """
        raise NotImplementedError

    def get(self, node, key, coordinates=None):
        """Get cached data for this node.
        
        Parameters
        ------------
        node : Node
            node requesting storage.
        key : str
            Cached object key, e.g. 'output'.
        coordinates : Coordinates, optional
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
        raise NotImplementedError

    def rem(self, node=None, key=None, coordinates=None):
        """Delete cached data for this node.
        
        Parameters
        ------------
        node : Node
            node requesting storage.
        key : str, optional
            Delete only cached objects with this key.
        coordinates : Coordinates
            Delete only cached objects for these coordinates.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def clear(self, node):
        """
        Clear all cached data.
        """
        raise NotImplementedError
