"""

"""

from __future__ import division, print_function, absolute_import

import os
from glob import glob

try:
    import cPickle  # Python 2.7
except:
    import _pickle as cPickle

import podpac.settings

class CacheException(Exception):
    """Summary
    """
    pass


class CacheCtrl(object):

    def __init__(self, cache_stores=[]):
        self._cache_stores = cache_stores

    def put(self, node, data, key, coordinates=None, mode=None, update=False):
        '''Cache data for specified node.
        
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
        '''
        raise NotImplementedError

    def get(self, node, key, coordinates=None, mode=None):
        '''Get cached data for this node.
        
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
        '''
        raise NotImplementedError

    def rem(self, node=None, key=None, coordinates=None, mode=None):
        '''Delete cached data for this node.
        
        Parameters
        ------------
        node : Node
            node requesting storage.
        key : str, optional
            Delete only cached objects with this key.
        coordinates : Coordinates
            Delete only cached objects for these coordinates.
        mode : str
            determines what types of the `CacheStore` are affected: 'ram','disk','network','all'. Defaults to `node._cache_mode` or 'all'. Overriden by `self._cache_mode` if `self._cache_mode` is not `None`.
        '''
        raise NotImplementedError

    def has(self, node, key, coordinates=None, mode=None):
        '''Check for cached data for this node
        
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
        '''
        raise NotImplementedError


class CacheStore(object):

    def get_hash_val(self, obj):
        return hash(obj)

    def hash_node(self, node):
        hashable_repr = node.json
        return self.get_hash_val(hashable_repr)

    def hash_coordinates(self, coordinates):
        hashable_repr = coordinates.json
        return self.get_hash_val(hashable_repr)

    def hash_key(self, key):
        hashable_repr = str(repr(key))

    def put(self, node, data, key, coordinates=None, update=False):
        '''Cache data for specified node.
        
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
        '''
        raise NotImplementedError

    def get(self, node, key, coordinates=None):
        '''Get cached data for this node.
        
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
        '''
        raise NotImplementedError

    def rem(self, node=None, key=None, coordinates=None):
        '''Delete cached data for this node.
        
        Parameters
        ------------
        node : Node
            node requesting storage.
        key : str, optional
            Delete only cached objects with this key.
        coordinates : Coordinates
            Delete only cached objects for these coordinates.
        '''
        raise NotImplementedError

    def has(self, node, key, coordinates=None):
        '''Check for cached data for this node
        
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
        '''
        raise NotImplementedError

class CacheListing(object):

    def __init__(self, node_def, key, coordinate_def, data):
        self.node_def = node_def
        self.key = key
        self.coordinate_def = coordinate_def
        self.data = data

    def __eq__(self, other):
        return self.node_def == other.node_def and 
          self.key == other.key and 
          self.coordinate_def == other.coordinate_def

class CachePickleContainer(object):

    def __init__(self, listings=[]):
        self.listings = listings

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            cPickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(path, 'rb') as f:
            return cPickle.load(f)

    def put(self, listing):
        self.listings.append(listing)

    def get(self, listing):
        for l in self.listings:
            if l == listing:
                return l
        raise CacheException("Could not find requested listing.")

    def has(self, listing):
        for l in self.listings:
            if l == listing:
                return True
        return False

    def rem(self, listing):
        for i,l in enumerate(self.listings):
            if l == listing:
                self.listings.pop(i)


class DiskCacheStore(CacheStore):

    def __init__(self, root_cache_dir_path=None, storage_format='pickle'):
        if root_cache_dir_path is None:
            root_cache_dir_path = podpac.settings.CACHE_DIR
        self._root_dir_path = root_cache_dir_path
        if storage_format == 'pickle':
            self._extension = 'pkl'
        else:
            raise NotImplementedError
        self._storage_format = storage_format

    def cache_dir(self, node):
        basedir = self._root_dir_path
        subdir = str(node.__class__)[8:-2].split('.')
        dirs = [basedir] + subdir
        return os.path.join(*dirs)

    def cache_filename(self, node, key, coordinates):
        pre = str(node.base_ref).replace('/', '_').replace('\\', '_').replace(':', '_')
        self.cleanse_filename_str(pre)
        nKeY = 'nKeY%s'.format(self.hash_node(node))
        kKeY = 'kKeY%s'.format(self.hash_key(key))
        cKeY = 'cKeY%s'.format(self.hash_coordinates(coordinates))
        filename = '_'.join([pre, nKeY, kKeY, cKeY])
        filename = filename + '.' + self._extension
        return filename

    def cache_glob(self, node, key, coordinates):
        pre = '*'
        nKeY = 'nKeY%s'.format(self.hash_node(node))
        kKeY = 'kKeY%s'.format(self.hash_key(key))
        cKeY = 'cKeY%s'.format(self.hash_coordinates(coordinates))
        filename = '_'.join([pre, nKeY, kKeY, cKeY])
        filename = filename + '.' + self._extension
        return os.path.join(self.cache_dir(node), filename)

    def cache_path(self, node, key, coordinates):
        return os.path.join(self.cache_dir(node), self.cache_filename(node, key, coordinates))

    def cleanse_filename_str(self, s):
        s = s.replace('/', '_').replace('\\', '_').replace(':', '_')
        s = s.replace('nKeY': 'xxxx').replace('kKeY': 'xxxx').replace('cKeY': 'xxxx')
        return s

    def put(self, node, data, key, coordinates=None, update=False):
        if not update and self.has(node, key, coordinates):
            raise CacheException("Existing cache entry. Call put() with `update` argument set to True if you wish to overwrite.")


    def get(self, node, key, coordinates=None):
        raise NotImplementedError

    def rem(self, node=None, key=None, coordinates=None):
        raise NotImplementedError

    def has(self, node, key, coordinates=None):
        node_def = node.json()
        if coordinates is None:
            coordinate_def = None
        else:
            coordinate_def = coordinates.json()
        listing = CacheListing(node_def=node_def, key=key, coordinate_def=coordinate_def)
        paths = glob(self.cache_glob(node, key, coordinates))
        for p in paths:
            c = CachePickleContainer.load(p)
            if c.has(listing):
                return True
        return False
        
