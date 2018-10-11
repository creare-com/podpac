"""

"""

from __future__ import division, print_function, absolute_import

import podpac.settings

class CacheCtrl(object):

    def __init__(self, cache_stores=[]):
        self._cache_stores = cache_stores

    def put(self, node, data, key, coordinates=None, mode=None):
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

    def put(self, node, data, key, coordinates=None):
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

    def get_hash_val(self, obj):
        return hash(obj)

    def hash_node(self, node):
        hashable_repr = node.definition(type='json')
        return self.get_hash_val(hashable_repr)

    def hash_coordinates(self, coordinates):
        hashable_repr = str(repr(coordinates))
        return self.get_hash_val(hashable_repr)

    def hash_key(self, key):
        hashable_repr = str(repr(key))

    def cache_dir(self, node):
        basedir = self._root_dir_path
        subdir = str(node.__class__)[8:-2].split('.')
        dirs = [basedir] + subdir
        return os.path.join(*dirs)

    def cache_filename(self, node, key, coordinates):
        pre = str(node.source).replace('/', '_').replace('\\', '_').replace(':', '_')
        self.cleanse_filename_str(pre)
        nKeY = 'nKeY%s'.format(self.hash_node(node))
        kKeY = 'kKeY%s'.format(self.hash_key(key))
        cKeY = 'cKeY%s'.format(self.hash_coordinates(coordinates))
        filename = '_'.join([pre, nKeY, kKeY, cKeY])
        filename = filename + '.' + self._extension
        return filename

    def cleanse_filename_str(self, s):
        s = s.replace('/', '_').replace('\\', '_').replace(':', '_')
        s = s.replace('nKeY': 'xxxx').replace('kKeY': 'xxxx').replace('cKeY': 'xxxx')
        return s

    def put(self, node, data, key, coordinates=None):
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
