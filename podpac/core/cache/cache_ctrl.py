from __future__ import division, print_function, absolute_import

import six

import podpac
from podpac.core.settings import settings
from podpac.core.cache.utils import CacheWildCard, CacheException
from podpac.core.cache.ram_cache_store import RamCacheStore
from podpac.core.cache.disk_cache_store import DiskCacheStore
from podpac.core.cache.s3_cache_store import S3CacheStore


_CACHE_STORES = {"ram": RamCacheStore, "disk": DiskCacheStore, "s3": S3CacheStore}

_CACHE_NAMES = {RamCacheStore: "ram", DiskCacheStore: "disk", S3CacheStore: "s3"}

_CACHE_MODES = ["ram", "disk", "network", "all"]


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


def make_cache_ctrl(names):
    """
    Make a cache_ctrl from a list of cache store types.

    Arguments
    ---------
    names : str or list
        cache name or names, e.g. 'ram' or ['ram', 'disk'].

    Returns
    -------
    ctrl : CacheCtrl
        CachCtrl using the specified cache names
    """

    if isinstance(names, six.string_types):
        names = [names]

    for name in names:
        if name not in _CACHE_STORES:
            raise ValueError("Unknown cache store type '%s', options are %s" % (name, list(_CACHE_STORES)))

    return CacheCtrl([_CACHE_STORES[name]() for name in names])


def clear_cache(mode="all"):
    """
    Clear the entire default cache_ctrl.

    Arguments
    ---------
    mode : str
        determines what types of the `CacheStore` are affected. Options: 'ram', 'disk', 'network', 'all'. Default 'all'.
    """

    cache_ctrl = get_default_cache_ctrl()
    cache_ctrl.clear(mode=mode)


def cache_cleanup():
    cache_ctrl = get_default_cache_ctrl()
    cache_ctrl.cleanup()


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
            list of CacheStore objects to manage, in the order that they should be interrogated.
        """

        self._cache_stores = cache_stores

    def __repr__(self):
        return "CacheCtrl(cache_stores=%s)" % self.cache_stores

    @property
    def cache_stores(self):
        return [_CACHE_NAMES[store.__class__] for store in self._cache_stores]

    def _get_cache_stores_by_mode(self, mode="all"):
        return [c for c in self._cache_stores if mode in c.cache_modes]

    def put(self, node, data, item, coordinates=None, expires=None, mode="all", update=True):
        """Cache data for specified node.

        Parameters
        ------------
        node : Node
            node requesting storage.
        data : any
            Data to cache
        item : str
            Cached object item or key, e.g. 'output'.
        coordinates : :class:`podpac.Coordinates`, optional
            Coordinates for which cached object should be retrieved, for coordinate-dependent data such as evaluation output
        mode : str
            determines what types of the `CacheStore` are affected. Options: 'ram', 'disk', 'network', 'all'. Default 'all'.
        expires : float, datetime, timedelta
            Expiration date. If a timedelta is supplied, the expiration date will be calculated from the current time.
        update : bool
            If True existing data in cache will be updated with `data`, If False, error will be thrown if attempting put something into the cache with the same node, key, coordinates of an existing entry.
        """

        if not isinstance(node, podpac.Node):
            raise TypeError("Invalid node (must be of type Node, not '%s')" % type(node))

        if not isinstance(item, six.string_types):
            raise TypeError("Invalid item (must be a string, not '%s')" % (type(item)))

        if not isinstance(coordinates, podpac.Coordinates) and coordinates is not None:
            raise TypeError("Invalid coordinates (must be of type 'Coordinates', not '%s')" % type(coordinates))

        if mode not in _CACHE_MODES:
            raise ValueError("Invalid mode (must be one of %s, not '%s')" % (_CACHE_MODES, mode))

        if item == "*":
            raise ValueError("Invalid item ('*' is reserved)")

        for c in self._get_cache_stores_by_mode(mode):
            c.put(node=node, data=data, item=item, coordinates=coordinates, expires=expires, update=update)

    def get(self, node, item, coordinates=None, mode="all"):
        """Get cached data for this node.

        Parameters
        ------------
        node : Node
            node requesting storage.
        item : str
            Cached object item or key, e.g. 'output'.
        coordinates : :class:`podpac.Coordinates`, optional
            Coordinates for which cached object should be retrieved, for coordinate-dependent data such as evaluation output
        mode : str
            determines what types of the `CacheStore` are affected. Options: 'ram', 'disk', 'network', 'all'. Default 'all'.

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
            raise TypeError("Invalid node (must be of type Node, not '%s')" % type(node))

        if not isinstance(item, six.string_types):
            raise TypeError("Invalid item (must be a string, not '%s')" % (type(item)))

        if not isinstance(coordinates, podpac.Coordinates) and coordinates is not None:
            raise TypeError("Invalid coordinates (must be of type 'Coordinates', not '%s')" % type(coordinates))

        if mode not in _CACHE_MODES:
            raise ValueError("Invalid mode (must be one of %s, not '%s')" % (_CACHE_MODES, mode))

        if item == "*":
            raise ValueError("Invalid item ('*' is reserved)")

        for c in self._get_cache_stores_by_mode(mode):
            if c.has(node=node, item=item, coordinates=coordinates):
                return c.get(node=node, item=item, coordinates=coordinates)
        raise CacheException("Requested data is not in any cache stores.")

    def has(self, node, item, coordinates=None, mode="all"):
        """Check for cached data for this node

        Parameters
        ------------
        node : Node
            node requesting storage.
        item : str
            Cached object item or key, e.g. 'output'.
        coordinates: Coordinate, optional
            Coordinates for which cached object should be checked
        mode : str
            determines what types of the `CacheStore` are affected. Options: 'ram', 'disk', 'network', 'all'. Default 'all'.

        Returns
        -------
        has_cache : bool
             True if there as a cached object for this node for the given key and coordinates.
        """

        if not isinstance(node, podpac.Node):
            raise TypeError("Invalid node (must be of type Node, not '%s')" % type(node))

        if not isinstance(item, six.string_types):
            raise TypeError("Invalid item (must be a string, not '%s')" % (type(item)))

        if not isinstance(coordinates, podpac.Coordinates) and coordinates is not None:
            raise TypeError("Invalid coordinates (must be of type 'Coordinates', not '%s')" % type(coordinates))

        if mode not in _CACHE_MODES:
            raise ValueError("Invalid mode (must be one of %s, not '%s')" % (_CACHE_MODES, mode))

        if item == "*":
            raise ValueError("Invalid item ('*' is reserved)")

        for c in self._get_cache_stores_by_mode(mode):
            if c.has(node=node, item=item, coordinates=coordinates):
                return True

        return False

    def rem(self, node, item, coordinates=None, mode="all"):
        """Delete cached data for this node.

        Parameters
        ----------
        node : Node, str
            node requesting storage.
        item : str
            Delete only cached objects with this item/key. Use `'*'` to match all keys.
        coordinates : :class:`podpac.Coordinates`, str
            Delete only cached objects for these coordinates. Use `'*'` to match all coordinates.
        mode : str
            determines what types of the `CacheStore` are affected. Options: 'ram', 'disk', 'network', 'all'. Default 'all'.
        """

        if not isinstance(node, podpac.Node):
            raise TypeError("Invalid node (must be of type Node, not '%s')" % type(node))

        if not isinstance(item, six.string_types):
            raise TypeError("Invalid item (must be a string, not '%s')" % (type(item)))

        if not isinstance(coordinates, podpac.Coordinates) and coordinates is not None and coordinates != "*":
            raise TypeError("Invalid coordinates (must be '*' or of type 'Coordinates', not '%s')" % type(coordinates))

        if mode not in _CACHE_MODES:
            raise ValueError("Invalid mode (must be one of %s, not '%s')" % (_CACHE_MODES, mode))

        if item == "*":
            item = CacheWildCard()

        if coordinates == "*":
            coordinates = CacheWildCard()

        for c in self._get_cache_stores_by_mode(mode):
            c.rem(node=node, item=item, coordinates=coordinates)

    def clear(self, mode="all"):
        """
        Clear all cached data.

        Parameters
        ------------
        mode : str
            determines what types of the `CacheStore` are affected. Options: 'ram', 'disk', 'network', 'all'. Default 'all'.
        """

        if mode not in _CACHE_MODES:
            raise ValueError("Invalid mode (must be one of %s, not '%s')" % (_CACHE_MODES, mode))

        for c in self._get_cache_stores_by_mode(mode):
            c.clear()

    def cleanup(self):
        """
        Cleanup all cache stores.

        Removes expired cache entries, orphaned metadata, empty directories, etc.
        """

        for c in self._cache_stores:
            c.cleanup()
