from __future__ import division, print_function, absolute_import

import six

import podpac
from podpac.core.settings import settings

from podpac.core.cache.utils import CacheWildCard, CacheException
from podpac.core.cache.ram_cache_store import RamCacheStore
from podpac.core.cache.disk_cache_store import DiskCacheStore
from podpac.core.cache.s3_cache_store import S3CacheStore


def get_default_cache_ctrl():
    """
    Get the default CacheCtrl according to the settings.

    Returns
    -------
    ctrl : CacheCtrl or None
        Default CachCtrl
    """

    if settings.get("DEFAULT_CACHE") is None:  # missing or None
        return CacheCtrl([])

    return make_cache_ctrl(settings["DEFAULT_CACHE"])


def make_cache_ctrl(stores):
    """
    Make a cache_ctrl from a list of cache store types.

    Arguments
    ---------
    stores : str or list
        cache store or stores, e.g. 'ram' or ['ram', 'disk'].

    Returns
    -------
    ctrl : CacheCtrl
        CachCtrl using the specified cache stores
    """

    if isinstance(stores, str):
        stores = [stores]

    cache_stores = []
    for elem in stores:
        if elem == "ram":
            cache_stores.append(RamCacheStore())
        elif elem == "disk":
            cache_stores.append(DiskCacheStore())
        elif elem == "s3":
            cache_stores.append(S3CacheStore())
        else:
            raise ValueError("Unknown cache store type '%s'" % elem)

    return CacheCtrl(cache_stores)


class CacheCtrl(object):

    """Objects of this class are used to manage multiple CacheStore objects of different types
    (e.g. RAM, local disk, s3) and serve as the interface to the caching module.
    """

    def __init__(self, cache_stores=[]):
        """Initialize a CacheCtrl object with a list of CacheStore objects.
        Care should be taken to provide the cache_stores list in the order that
        they should be interogated. CacheStore objects with faster access times 
        (e.g. RAM) should appear before others (e.g. local disk, or s3).
        
        Parameters
        ----------
        cache_stores : list, optional
            list of CacheStore objects to manage, in the order that they should be interogated.
        """
        self._cache_stores = cache_stores
        self._cache_mode = None

    def _get_cache_stores(self, mode):
        if mode is None:
            mode = self._cache_mode
            if mode is None:
                mode = "all"

        return [c for c in self._cache_stores if mode in c.cache_modes]

    def put(self, node, data, key, coordinates=None, mode=None, update=False):
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
        mode : str
            determines what types of the `CacheStore` are affected: 'ram','disk','network','all'. Defaults to `node._cache_mode` or 'all'. Overriden by `self._cache_mode` if `self._cache_mode` is not `None`.
        update : bool
            If True existing data in cache will be updated with `data`, If False, error will be thrown if attempting put something into the cache with the same node, key, coordinates of an existing entry.
        """

        if not isinstance(node, podpac.Node):
            raise TypeError("node must of type 'Node', not '%s'" % type(Node))

        if not isinstance(key, six.string_types):
            raise TypeError("key must be a string type, not '%s'" % (type(key)))

        if not isinstance(coordinates, podpac.Coordinates) and coordinates is not None:
            raise TypeError("coordinates must be of type 'Coordinates', not '%s'" % type(coordinates))

        if not isinstance(mode, six.string_types) and mode is not None:
            raise TypeError("mode must be of type 'str', not '%s'" % type(mode))

        if key == "*":
            raise ValueError("key cannot be '*'")

        for c in self._get_cache_stores(mode):
            c.put(node=node, data=data, key=key, coordinates=coordinates, update=update)

    def get(self, node, key, coordinates=None, mode=None):
        """Get cached data for this node.
        
        Parameters
        ------------
        node : Node
            node requesting storage.
        key : str
            Cached object key, e.g. 'output'.
        coordinates : Coordinates, optional
            Coordinates for which cached object should be retrieved, for coordinate-dependent data such as evaluation output
        mode : str
            determines what types of the `CacheStore` are affected: 'ram','disk','network','all'. Defaults to `node._cache_mode` or 'all'. Overriden by `self._cache_mode` if `self._cache_mode` is not `None`.
            
        Returns
        -------
        data : any
            The cached data.
        
        Raises
        -------
        CacheError
            If the data is not in the cache.
        """

        if not isinstance(node, podpac.Node):
            raise TypeError("node must of type 'Node', not '%s'" % type(Node))

        if not isinstance(key, six.string_types):
            raise TypeError("key must be a string type, not '%s'" % (type(key)))

        if not isinstance(coordinates, podpac.Coordinates) and coordinates is not None:
            raise TypeError("coordinates must be of type 'Coordinates', not '%s'" % type(coordinates))

        if not isinstance(mode, six.string_types) and mode is not None:
            raise TypeError("mode must be of type 'str', not '%s'" % type(mode))

        if key == "*":
            raise ValueError("key cannot be '*'")

        for c in self._get_cache_stores(mode):
            if c.has(node=node, key=key, coordinates=coordinates):
                return c.get(node=node, key=key, coordinates=coordinates)
        raise CacheException("Requested data is not in any cache stores.")

    def has(self, node, key, coordinates=None, mode=None):
        """Check for cached data for this node
        
        Parameters
        ------------
        node : Node
            node requesting storage.
        key : str
            Cached object key, e.g. 'output'.
        coordinates: Coordinate, optional
            Coordinates for which cached object should be checked
        mode : str
            determines what types of the `CacheStore` are affected: 'ram','disk','network','all'. Defaults to `node._cache_mode` or 'all'. Overriden by `self._cache_mode` if `self._cache_mode` is not `None`.
        
        Returns
        -------
        has_cache : bool
             True if there as a cached object for this node for the given key and coordinates.
        """

        if not isinstance(node, podpac.Node):
            raise TypeError("node must of type 'Node', not '%s'" % type(Node))

        if not isinstance(key, six.string_types):
            raise TypeError("key must be a string type, not '%s'" % (type(key)))

        if not isinstance(coordinates, podpac.Coordinates) and coordinates is not None:
            raise TypeError("coordinates must be of type 'Coordinates', not '%s'" % type(coordinates))

        if not isinstance(mode, six.string_types) and mode is not None:
            raise TypeError("mode must be of type 'str', not '%s'" % type(mode))

        if key == "*":
            raise ValueError("key cannot be '*'")

        for c in self._get_cache_stores(mode):
            if c.has(node=node, key=key, coordinates=coordinates):
                return True

        return False

    def rem(self, node, key, coordinates=None, mode=None):
        """Delete cached data for this node.
        
        Parameters
        ----------
        node : Node, str
            node requesting storage.
        key : str
            Delete only cached objects with this key. Use `'*'` to match all keys.
        coordinates : Coordinates, str
            Delete only cached objects for these coordinates. Use `'*'` to match all coordinates.
        mode : str
            determines what types of the `CacheStore` are affected: 'ram','disk','network','all'. Defaults to `node._cache_mode` or 'all'. Overriden by `self._cache_mode` if `self._cache_mode` is not `None`.
        """

        if not isinstance(node, podpac.Node):
            raise TypeError("node must of type 'Node', not '%s'" % type(podpac.Node))

        if not isinstance(key, six.string_types):
            raise TypeError("key must be a string type, not '%s'" % (type(key)))

        if not isinstance(coordinates, podpac.Coordinates) and coordinates is not None and coordinates != "*":
            raise TypeError("coordinates must be '*' or of type 'Coordinates' not '%s'" % type(coordinates))

        if not isinstance(mode, six.string_types) and mode is not None:
            raise TypeError("mode must be of type 'str', not '%s'" % type(mode))

        if key == "*":
            key = CacheWildCard()

        if coordinates == "*":
            coordinates = CacheWildCard()

        for c in self._get_cache_stores(mode):
            c.rem(node=node, key=key, coordinates=coordinates)

    def clear(self, mode=None):
        """
        Clear all cached data.

        Parameters
        ------------
        mode : str
            determines what types of the `CacheStore` are affected: 'ram','disk','network','all'. Defaults to `node._cache_mode` or 'all'. Overriden by `self._cache_mode` if `self._cache_mode` is not `None`.
        """

        if not isinstance(mode, six.string_types) and mode is not None:
            raise TypeError("mode must be of type 'str', not '%s'" % type(mode))

        for c in self._get_cache_stores(mode):
            c.clear()
