"""

"""

from __future__ import division, print_function, absolute_import

import os
from glob import glob
import shutil
from hashlib import md5 as hash_alg

try:
    import cPickle  # Python 2.7
except:
    import _pickle as cPickle

import podpac.settings

_cache_types = {'ram','disk','network','all'}

class CacheException(Exception):
    """Summary
    """
    pass


class CacheCtrl(object):

    def __init__(self, cache_stores=[]):
        self._cache_stores = cache_stores
        self._cache_mode = None

    def _determine_mode(self, mode):
        if mode is None:
            mode = self._cache_mode
            if mode is None:
                mode = 'all'
        return mode

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
        mode = self._determine_mode(mode)
        for c in self._cache_stores:
            if c.cache_modes_matches(set([mode])):
                c.put(node=node, data=data, key=key, coordinates=coordinates, update=update)
        

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
        mode = self._determine_mode(mode)
        for c in self._cache_stores:
            if c.cache_modes_matches(set([mode])):
                if c.has(node=node, key=key, coordinates=coordinates):
                    return c.get(node=node, key=key, coordinates=coordinates)
        raise CacheException("Requested data is not in any cache stores.")

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
        mode = self._determine_mode(mode)
        for c in self._cache_stores:
            if c.cache_modes_matches(set([mode])):
                c.rem(node=node, key=key, coordinates=coordinates)

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
        mode = self._determine_mode(mode)
        for c in self._cache_stores:
            if c.cache_modes_matches(set([mode])):
                if c.has(node=node, key=key, coordinates=coordinates):
                    return True
        return False


class CacheStore(object):

    def get_hash_val(self, obj):
        return hash_alg(obj).hexdigest()

    def hash_node(self, node):
        hashable_repr = 'None' if node is None else node.hash
        return hashable_repr 

    def hash_coordinates(self, coordinates):
        hashable_repr = 'None' if coordinates is None else coordinates.hash
        return hashable_repr 

    def hash_key(self, key):
        #hashable_repr = str(repr(key)).encode('utf-8')
        #return self.get_hash_val(hashable_repr)
        return key

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

    def __init__(self, node, key, coordinates, data=None):
        self._node_def = node.definition
        self.key = key
        self.coordinate_def = None if coordinates is None else coordinates.json
        self.data = data

    @property
    def node_def(self):
        return cPickle.dumps(self._node_def)
    
    def __eq__(self, other):
        return self.node_def == other.node_def and \
               self.key == other.key and \
               self.coordinate_def == other.coordinate_def

class CachePickleContainer(object):

    def __init__(self, listings=[]):
        self.listings = listings

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            cPickle.dump(self, f)

    @staticmethod
    def load(path):
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

    @property
    def empty(self):
        if len(self.listings) == 0:
            return True
        return False


class DiskCacheStore(CacheStore):

    def __init__(self, root_cache_dir_path=None, storage_format='pickle'):
        self._cache_modes = set(['disk','all'])
        if root_cache_dir_path is None:
            root_cache_dir_path = podpac.settings.CACHE_DIR
        self._root_dir_path = root_cache_dir_path
        if storage_format == 'pickle':
            self._extension = 'pkl'
        else:
            raise NotImplementedError
        self._storage_format = storage_format

    def cache_modes_matches(self, modes):
        if len(self._cache_modes.intersection(modes)) > 0:
            return True
        return False

    def make_cache_dir(self, node):
        cache_dir = self.cache_dir(node)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def cache_dir(self, node):
        basedir = self._root_dir_path
        subdir = str(node.__class__)[8:-2].split('.')
        dirs = [basedir] + subdir
        return self.cleanse_filename_str(os.path.join(*dirs))

    def cache_filename(self, node, key, coordinates):
        pre = str(node.base_ref).replace('/', '_').replace('\\', '_').replace(':', '_')
        self.cleanse_filename_str(pre)
        nKeY = 'nKeY{}'.format(self.hash_node(node))
        kKeY = 'kKeY{}'.format(self.hash_key(key))
        cKeY = 'cKeY{}'.format(self.hash_coordinates(coordinates))
        filename = '_'.join([pre, nKeY, kKeY, cKeY])
        filename = filename + '.' + self._extension
        return filename

    def cache_glob(self, node, key, coordinates):
        pre = '*'
        nKeY = 'nKeY{}'.format(self.hash_node(node))
        kKeY = 'kKeY*' if key == '*' else 'kKeY{}'.format(self.cleanse_filename_str(self.hash_key(key)))
        cKeY = 'cKeY*' if coordinates == '*' else 'cKeY{}'.format(self.hash_coordinates(coordinates))
        filename = '_'.join([pre, nKeY, kKeY, cKeY])
        filename = filename + '.' + self._extension
        return os.path.join(self.cache_dir(node), filename)

    def cache_path(self, node, key, coordinates):
        return os.path.join(self.cache_dir(node), self.cache_filename(node, key, coordinates))

    def cleanse_filename_str(self, s):
        s = s.replace('/', '_').replace('\\', '_').replace(':', '_').replace('<', '_').replace('_', '')
        s = s.replace('nKeY', 'xxxx').replace('kKeY', 'xxxx').replace('cKeY', 'xxxx')
        return s

    def put(self, node, data, key, coordinates=None, update=False):
        
        self.make_cache_dir(node)
        listing = CacheListing(node=node, key=key, coordinates=coordinates, data=data)
        if self.has(node, key, coordinates): # a little inefficient but will do for now
            if not update:
                raise CacheException("Existing cache entry. Call put() with `update` argument set to True if you wish to overwrite.")
            else:
                paths = glob(self.cache_glob(node, key, coordinates))
                for p in paths:
                    c = CachePickleContainer.load(p)
                    if c.has(listing):
                        c.rem(listing)
                        c.put(listing)
                        c.save(p)
                        return True
                raise CacheException("Data is cached, but unable to find for update.")
        # listing does not exist in cache
        path = self.cache_path(node, key, coordinates)
        # if file for listing already exists, listing needs to be added to file
        if os.path.exists(path):
            c = CachePickleContainer.load(path)
            c.put(listing)
            c.save(path)
        # if file for listing does not already exist, we need to create a new container, add the listing, and save to file
        else:
            CachePickleContainer(listings=[listing]).save(path)
        return True

    def get(self, node, key, coordinates=None):
        listing = CacheListing(node=node, key=key, coordinates=coordinates)
        paths = glob(self.cache_glob(node, key, coordinates))
        for p in paths:
            c = CachePickleContainer.load(p)
            if c.has(listing):
                data = c.get(listing).data
                if data is None:
                     CacheException("Stored data is None.")
                return data
        raise CacheException("Cache miss. Requested data not found.")

    def rem(self, node=None, key=None, coordinates=None):
        # need to handle cases for removing all entries for a node
        # for future update
        if node is None:
            # clear the entire cache store
            shutil.rmtree(self._root_dir_path)
            return True
        removed_something = False
        if key is None:
            # clear all files for data cached for `node`
            # and delete its cache subdirectory if it is empty
            paths = glob(self.cache_glob(node, key='*', coordinates='*'))
            for p in paths:
                os.remove(p)
                removed_something = True
            try:
                os.rmdir(self.cache_dir(node=node))
            except Exception as e:
                pass
            return removed_something
        listing = CacheListing(node=node, key=key, coordinates=coordinates)
        paths = glob(self.cache_glob(node, key, coordinates))
        for p in paths:
            c = CachePickleContainer.load(p)
            if c.has(listing):
                c.rem(listing)
                removed_something = True
                if c.empty:
                    os.remove(p)
                else:
                    c.save(p)
        return removed_something
        

    def has(self, node, key, coordinates=None):
        listing = CacheListing(node=node, key=key, coordinates=coordinates)
        paths = glob(self.cache_glob(node, key, coordinates))
        for p in paths:
            c = CachePickleContainer.load(p)
            if c.has(listing):
                return True
        return False


