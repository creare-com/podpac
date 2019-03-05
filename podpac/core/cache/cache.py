"""

"""

from __future__ import division, print_function, absolute_import

import os
import threading
from glob import glob
import shutil
from hashlib import md5 as hash_alg
import six
import fnmatch
from lazy_import import lazy_module
try:
    import cPickle  # Python 2.7
except:
    import _pickle as cPickle

boto3 = lazy_module('boto3')

from podpac.core.settings import settings

_cache_types = {'ram','disk','network','all','s3'}

class CacheException(Exception):
    """Summary
    """
    pass

class CacheWildCard(object):

    """Used to represent wildcard matches for inputs to remove operations (`rem`)
    that can match multiple items in the cache.
    """
    
    def __eq__(self, other):
        return True

def validate_inputs(node=None, key=None, coordinates=None, mode=None):
    """Used for validating the type of common cache inputs.
    Will throw an exception if any input is not the correct type.
    `None` is allowed for all inputs
    
    Parameters
    ----------
    node : None, optional
        podpac.core.node.Node
    key : None, optional
        str
    coordinates : None, optional
        podpac.core.coordinates.coordinates.Coordinates
    mode : None, optional
        str
    
    Returns
    -------
    TYPE bool
        Returns true if all specified inputs are of the correct type
    
    Raises
    ------
    CacheException
        Raises exception if any specified input is not of the correct type
    """
    from podpac.core.node import Node
    from podpac.core.coordinates.coordinates import Coordinates
    if not (node is None or isinstance(node, Node) or isinstance(node, CacheWildCard)):
        raise CacheException('`node` should either be an instance of `podpac.core.node.Node` or `None`.')
    if not (key is None or isinstance(key, six.string_types) or isinstance(key, CacheWildCard)):
        raise CacheException('`key` should either be an instance of string or `None`.')
    if not (coordinates is None or isinstance(coordinates, Coordinates) or isinstance(coordinates, CacheWildCard)):
        raise CacheException('`coordinates` should either be an instance of `podpac.core.coordinates.coordinates.Coordinates` or `None`.')
    if not (mode is None or isinstance(mode, six.string_types)):
        raise CacheException('`mode` should either be an instance of string or `None`.')
    return True

def get_default_cache_ctrl():
    """
    Get the default CacheCtrl according to the settings.

    Returns
    -------
    ctrl : CacheCtrl
        Default CachCtrl
    """

    if settings.get('DEFAULT_CACHE') is None:
        return CacheCtrl()

    cache_stores = []
    for elem in settings['DEFAULT_CACHE']:
        if elem == 'ram':
            cache_stores.append(RamCacheStore())
        elif elem == 'disk':
            cache_stores.append(DiskCacheStore())
        elif elem == 's3':
            cache_stores.append(S3CacheStore())
        else:
            raise ValueError("Unknown cache store type '%s'" % elem)
    
    return CachCtrl(cache_stores)

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
        validate_inputs(node=node, key=key, coordinates=coordinates, mode=mode)
        assert node is not None, "`node` can not be `None`"
        assert key is not None, "`key` can not be `None`"
        assert not isinstance(node, CacheWildCard)
        assert not isinstance(key, CacheWildCard)
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
        validate_inputs(node=node, key=key, coordinates=coordinates, mode=mode)
        assert node is not None, "`node` can not be `None`"
        assert key is not None, "`key` can not be `None`"
        assert not isinstance(node, CacheWildCard)
        assert not isinstance(key, CacheWildCard)
        mode = self._determine_mode(mode)
        for c in self._cache_stores:
            if c.cache_modes_matches(set([mode])):
                if c.has(node=node, key=key, coordinates=coordinates):
                    return c.get(node=node, key=key, coordinates=coordinates)
        raise CacheException("Requested data is not in any cache stores.")

    def rem(self, node, key, coordinates=None, mode=None):
        '''Delete cached data for this node.
        
        Parameters
        ----------
        node : Node, str
            node requesting storage. Use `'*'` to match all nodes.
        key : str
            Delete only cached objects with this key. Use `'*'` to match all keys.
        coordinates : Coordinates, str
            Delete only cached objects for these coordinates. Use `'*'` to match all coordinates.
        mode : str
            determines what types of the `CacheStore` are affected: 'ram','disk','network','all'. Defaults to `node._cache_mode` or 'all'. Overriden by `self._cache_mode` if `self._cache_mode` is not `None`.
        '''
        if isinstance(node, six.string_types) and node == '*': 
            node = CacheWildCard()
        if isinstance(coordinates, six.string_types) and coordinates == '*': 
            coordinates = CacheWildCard()
        validate_inputs(node=node, key=key, coordinates=coordinates, mode=mode)
        assert node is not None, "`node` can not be `None`"
        assert key is not None, "`key` can not be `None`"
        if key == '*':
            key = CacheWildCard()
        else:
            key = key.replace('*','_')
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
        validate_inputs(node=node, key=key, coordinates=coordinates, mode=mode)
        assert node is not None, "`node` can not be `None`"
        assert key is not None, "`key` can not be `None`"
        assert not isinstance(node, CacheWildCard)
        assert not isinstance(key, CacheWildCard)
        mode = self._determine_mode(mode)
        for c in self._cache_stores:
            if c.cache_modes_matches(set([mode])):
                if c.has(node=node, key=key, coordinates=coordinates):
                    return True
        return False


class CacheStore(object):

    """Abstract parent class for classes representing actual data stores (e.g. RAM, local disk, network storage).
    Includes implementation of common hashing operations and call signature for required abstract methods: 
    put(), get(), rem(), has()
    """
    
    cache_modes = []

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

    def cache_modes_matches(self, modes):
        """Returns True if this CacheStore matches any caching modes in `modes`
        
        Parameters
        ----------
        modes : List, Set
            collection of cache modes: subset of ['ram','disk','all']
        
        Returns
        -------
        TYPE : bool
            Returns True if this CacheStore matches any specified modes
        """
        if len(self.cache_modes.intersection(modes)) > 0:
            return True
        return False

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

    """Container for a single cached object. 
    Includes information pertinent to retrieval of cached objects 
    (e.g. representations of the node, key, and coordinages) and the actual cached data object.
    
    Attributes
    ----------
    coordinate_def : str
        JSON representation of coordinates of cached object
    data : any
        actual cached data object
    key : str
        key for retrieval
    """
    
    def __init__(self, node, key, coordinates, data=None):
        """
        
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
        """
        self._node_def = node.definition
        self.key = key
        self.coordinate_def = None if coordinates is None else coordinates.json
        self.data = data

    @property
    def node_def(self):
        """Returns serialized representation of node definition for comparison purposes
        
        Returns
        -------
        bytes object
            Serialized representation of node definition for comparison purposes
        """
        return cPickle.dumps(self._node_def)
    
    def __eq__(self, other):
        """Compares two CacheListing objects to determine appropriate behaviour for common cache operations
        (e.g. put(), get(), rem(), has())
        
        Parameters
        ----------
        other : CacheListing
            Cache Listing object to compare to.
        
        Returns
        -------
        boolean
            True, if the other listing is equivalent to this listing
        """
        return self.node_def == other.node_def and \
               self.key == other.key and \
               self.coordinate_def == other.coordinate_def

class CachePickleContainer(object):

    """Container for multiple cache listings that are different but whose signature (node, coordinates, key) hash are the same. Used for serializing the CacheListing objects to disk using pickle format.
    
    Attributes
    ----------
    listings : list
        list of CacheListing objects that share the same signature hashes, but are actually different
    """
    
    def __init__(self, listings=[]):
        """
        
        Parameters
        ----------
        listings : list, optional
            CacheListing objects that share the same signature hashes, but are actually different
        """
        self.listings = listings

    def serialize(self):
        """Convert this object to bytes so that it can be saved to local file, s3 object, etc.
        """
        return cPickle.dumps(self)

    @staticmethod
    def deserialize(s):
        """Convert bytes to instance of CachePickleContainer. Used in conjunction with serialize.
        
        Parameters
        ----------
        s : str
            bytes representing object
        """
        return cPickle.loads(s)


    def save(self, filepath):
        """Save this object to disk using pickle format.
        This function can be used instead of serialize() to directly save to local disk.
        
        Parameters
        ----------
        filepath : str
            path to file where this object should be saved
        """
        with open(filepath, 'wb') as f:
            cPickle.dump(self, f)

    @staticmethod
    def load(path):
        """Load this object from disk
        This function can be used instead of deserialize() to directly load from local disk.
        
        Parameters
        ----------
        path : str
            path to file where this object should be loaded from
        
        Returns
        -------
        CachePickleContainer
            container stored at path
        """
        with open(path, 'rb') as f:
            return cPickle.load(f)

    def put(self, listing):
        """Add CacheListing object to this CachePickleContainer
        
        Parameters
        ----------
        listing : CacheListing
            cache listing to add
        """
        self.listings.append(listing)

    def get(self, listing):
        """Check for CacheListing object from this CachePickleContainer
        
        Parameters
        ----------
        listing : CacheListing
            listing with signature for lookup.
        
        Returns
        -------
        CacheListing
            object in this CachePickleContainer that has the same signature as `listing`
        
        Raises
        ------
        CacheException
            If no objects match signature of `liasting`
        """
        for l in self.listings:
            if l == listing:
                return l
        raise CacheException("Could not find requested listing.")

    def has(self, listing):
        """Retrieve CacheListing object from this CachePickleContainer
        
        Parameters
        ----------
        listing : listing with signature for lookup.
        
        Returns
        -------
        boolean
            True, if there is a CacheListing in this CachePickleContainer with the same signature as `listing`
        """
        for l in self.listings:
            if l == listing:
                return True
        return False

    def rem(self, listing):
        """Removes CacheListing objecta from this CachePickleContainer
        
        Parameters
        ----------
        listing : CacheListing
            cache listing with same signature as objects that should be removed
        """
        for i,l in enumerate(self.listings):
            if l == listing:
                self.listings.pop(i)

    @property
    def empty(self):
        """Query for whether this CachePickleContainer is empty
        
        Returns
        -------
        boolean
            True if this CachePickleContainer is empty
        """
        if len(self.listings) == 0:
            return True
        return False


class FileCacheStore(CacheStore):

    """Abstract class with functionality common to persistent CacheStore objects (e.g. local disk, s3) that store things using multiple paths (filepaths or object paths)
    """
    
    cache_modes = ['all']
    _CacheContainerClass = CachePickleContainer

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def make_cache_dir(self, node):
        raise NotImplementedError

    def save_container(self, container, path):
        raise NotImplementedError

    def load_container(self, path):
        raise NotImplementedError

    def _path_join(self, parts):
        raise NotImplementedError

    def delete_file(self, path):
        raise NotImplementedError

    def file_exists(self, path):
        raise NotImplementedError

    def cache_dir(self, node):
        """subdirectory for caching data for `node`
        
        Parameters
        ----------
        node : podpac.core.node.Node
            Description
        
        Returns
        -------
        TYPE : str
            subdirectory path
        """
        basedir = self._root_dir_path
        subdir = str(node.__class__)[8:-2].split('.')
        dirs = [basedir] + subdir
        return (self._path_join(dirs)).replace('<', '_').replace('>', '_')

    def cache_filename(self, node, key, coordinates):
        """Filename for storing cached data for specified node,key,coordinates
        
        Parameters
        ----------
        node : podpac.core.node.Node
            Description
        key : str
            Description
        coordinates : podpac.core.coordinates.coordinates.Coordinates
            Description
        
        Returns
        -------
        TYPE : str
            filename (but not containing directory)
        """
        pre = self.cleanse_filename_str(str(node.base_ref))
        self.cleanse_filename_str(pre)
        nKeY = 'nKeY{}'.format(self.hash_node(node))
        kKeY = 'kKeY{}'.format(self.hash_key(key))
        cKeY = 'cKeY{}'.format(self.hash_coordinates(coordinates))
        filename = '_'.join([pre, nKeY, kKeY, cKeY])
        filename = filename + '.' + self._extension
        return filename

    def cache_glob(self, node, key, coordinates):
        raise NotImplementedError

    def cache_path(self, node, key, coordinates):
        """Filepath for storing cached data for specified node,key,coordinates
        
        Parameters
        ----------
        node : podpac.core.node.Node
            Description
        key : str
            Description
        coordinates : podpac.core.coordinates.coordinates.Coordinates
            Description
        
        Returns
        -------
        TYPE : str
            filename (including containing directory)
        """
        return self._path_join([self.cache_dir(node), self.cache_filename(node, key, coordinates)])

    def cleanse_filename_str(self, s):
        """Remove/replace characters from string `s` that could could interfere with proper functioning of cache if used to construct cache filenames.
        
        Parameters
        ----------
        s : str
            Description
        
        Returns
        -------
        TYPE : str
            Description
        """
        s = s.replace('/', '_').replace('\\', '_').replace(':', '_').replace('<', '_').replace('>', '_')
        s = s.replace('nKeY', 'xxxx').replace('kKeY', 'xxxx').replace('cKeY', 'xxxx')
        return s

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
        self.make_cache_dir(node)
        listing = CacheListing(node=node, key=key, coordinates=coordinates, data=data)
        if self.has(node, key, coordinates): # a little inefficient but will do for now
            if not update:
                raise CacheException("Existing cache entry. Call put() with `update` argument set to True if you wish to overwrite.")
            else:
                paths = self.cache_glob(node, key, coordinates)
                for p in paths:
                    c = self.load_container(p)
                    if c.has(listing):
                        c.rem(listing)
                        c.put(listing)
                        self.save_container(c,p)
                        return True
                raise CacheException("Data is cached, but unable to find for update.")
        # listing does not exist in cache
        path = self.cache_path(node, key, coordinates)
        # if file for listing already exists, listing needs to be added to file
        if self.file_exists(path):
            c = self.load_container(path)
            c.put(listing)
            self.save_container(c,path)
        # if file for listing does not already exist, we need to create a new container, add the listing, and save to file
        else:
            self.save_new_container(listings=[listing], path=path)
        return True

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
        listing = CacheListing(node=node, key=key, coordinates=coordinates)
        paths = self.cache_glob(node, key, coordinates)
        for p in paths:
            c = self.load_container(p)
            if c.has(listing):
                data = c.get(listing).data
                if data is None:
                    CacheException("Stored data is None.")
                return data
        raise CacheException("Cache miss. Requested data not found.")

    def clear_entire_cache_store(self):
        raise NotImplementedError

    def dir_is_empty(self, directory):
        raise NotImplementedError

    def rem_dir(self, directory):
        raise NotImplementedError

    def rem(self, node=CacheWildCard(), key=CacheWildCard(), coordinates=CacheWildCard()):
        '''Delete cached data for this node.
        
        Parameters
        ------------
        node : Node, CacheWildCard
            node requesting storage. If `node` is a `CacheWildCard` then everything in the cache will be deleted.
        key : str, CacheWildCard, optional
            Delete only cached objects with this key, or any key if `key` is a CacheWildCard.
        coordinates : Coordinates, CacheWildCard, None, optional
            Delete only cached objects for these coordinates, or any coordinates if `coordinates` is a CacheWildCard. `None` specifically indicates entries that do not have coordinates.
        '''
        if isinstance(node, CacheWildCard):
            # clear the entire cache store
            return self.clear_entire_cache_store()
        removed_something = False
        if isinstance(key, CacheWildCard) or isinstance(coordinates, CacheWildCard):
            # clear all files for data cached for `node`
            # and delete its cache subdirectory if it is empty
            paths = self.cache_glob(node, key=key, coordinates=coordinates)
            for p in paths:
                self.delete_file(p)
                removed_something = True
            cache_dir = self.cache_dir(node=node)
            if self.dir_is_empty(cache_dir):
                self.rem_dir(cache_dir)
            return removed_something
        listing = CacheListing(node=node, key=key, coordinates=coordinates)
        paths = self.cache_glob(node, key, coordinates)
        for p in paths:
            #import pdb
            #pdb.set_trace()
            c = self.load_container(p)
            if c.has(listing):
                c.rem(listing)
                removed_something = True
                if c.empty:
                   self.delete_file(p)
                else:
                    self.save_container(c,p)
        return removed_something
        

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
        listing = CacheListing(node=node, key=key, coordinates=coordinates)
        paths = self.cache_glob(node, key, coordinates)
        for p in paths:
            c = self.load_container(p)
            if c.has(listing):
                return True
        return False

class DiskCacheStore(FileCacheStore):

    cache_modes = set(['disk','all'])

    def __init__(self, root_cache_dir_path=None, storage_format='pickle'):
        """Initialize a cache that uses a folder on a local disk file system.
        
        Parameters
        ----------
        root_cache_dir_path : None, optional
            Root directory for the files managed by this cache. `None` indicates to use the folder specified in the global podpac settings. Should be a fully specified valid path.
        storage_format : str, optional
            Indicates the file format for storage. Defaults to 'pickle' which is currently the only supported format.
        
        Raises
        ------
        NotImplementedError
            If unsupported `storage_format` is specified
        """

        # set cache dir
        if root_cache_dir_path is not None:
            self._root_dir_path = root_cache_dir_path
        elif os.path.isabs(settings['DISK_CACHE_DIR']):
            self._root_dir_path = settings['DISK_CACHE_DIR']
        else:
            self._root_dir_path = self._path_join([settings['ROOT_PATH'], settings['DISK_CACHE_DIR']])

        # make directory if it doesn't already exist
        os.makedirs(self._root_dir_path, exist_ok=True)

        # set extension
        if storage_format == 'pickle':
            self._extension = 'pkl'
            self._CacheContainerClass = CachePickleContainer
        else:
            raise NotImplementedError
        self._storage_format = storage_format

    def save_container(self, container, path):
        container.save(path)

    def save_new_container(self, listings, path):
        self.save_container(self._CacheContainerClass(listings=listings),path)

    def load_container(self, path):
        return self._CacheContainerClass.load(path)

    def _path_join(self, parts):
        return os.path.join(*parts)

    def delete_file(self, path):
        os.remove(path)

    def file_exists(self, path):
        return os.path.exists(path)

    def make_cache_dir(self, node):
        """Create subdirectory for caching data for `node`
        
        Parameters
        ----------
        node : podpac.core.node.Node
            Description
        """
        cache_dir = self.cache_dir(node)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def cache_glob(self, node, key, coordinates):
        """Fileglob to match files that could be storing cached data for specified node,key,coordinates
        
        Parameters
        ----------
        node : podpac.core.node.Node
        key : str, CacheWildCard
            CacheWildCard indicates to match any key
        coordinates : podpac.core.coordinates.coordinates.Coordinates, CacheWildCard, None
            CacheWildCard indicates to match any coordinates
        
        Returns
        -------
        TYPE : str
            Fileglob of existing paths that match the request
        """
        pre = '*'
        nKeY = 'nKeY{}'.format(self.hash_node(node))
        kKeY = 'kKeY*' if isinstance(key, CacheWildCard) else 'kKeY{}'.format(self.cleanse_filename_str(self.hash_key(key)))
        cKeY = 'cKeY*' if isinstance(coordinates, CacheWildCard) else 'cKeY{}'.format(self.hash_coordinates(coordinates))
        filename = '_'.join([pre, nKeY, kKeY, cKeY])
        filename = filename + '.' + self._extension
        return glob(self._path_join([self.cache_dir(node), filename]))

    def clear_entire_cache_store(self):
        shutil.rmtree(self._root_dir_path)
        return True

    def dir_is_empty(self, directory):
        return os.path.exists(directory) and os.path.isdir(directory) and not os.listdir(directory)

    def rem_dir(self, directory):
        os.rmdir(directory)

class S3CacheStore(FileCacheStore):

    cache_modes = set(['s3','all'])
    _delim = '/'

    def __init__(self, root_cache_dir_path=None, storage_format='pickle', 
                 s3_bucket=None, aws_region_name=None, aws_access_key_id=None, aws_secret_access_key=None):
        """Initialize a cache that uses a folder on a local disk file system.
        
        Parameters
        ----------
        root_cache_dir_path : None, optional
            Root directory for the files managed by this cache. `None` indicates to use the folder specified in the global podpac settings. Should be the common "root" s3 prefix that you want to have all cached objects stored in. Do not store any objects in the bucket that share this prefix as they may be deleted by the cache.
        storage_format : str, optional
            Indicates the file format for storage. Defaults to 'pickle' which is currently the only supported format.
        s3_bucket : str, optional
            bucket name, overides settings
        aws_region_name : str, optional
            e.g. 'us-west-1', 'us-west-2','us-east-1'
        aws_access_key_id : str, optional
            overides podpac settings if both `aws_access_key_id` and `aws_secret_access_key` are specified
        aws_secret_access_key : str, optional
            overides podpac settings if both `aws_access_key_id` and `aws_secret_access_key` are specified
        
        Raises
        ------
        CacheException
            Description
        e
            Description
        NotImplementedError
            If unsupported `storage_format` is specified
        """
        self._cache_modes = set(['s3','all'])
        if root_cache_dir_path is None:
            root_cache_dir_path = settings['S3_CACHE_DIR']
        self._root_dir_path = root_cache_dir_path
        if storage_format == 'pickle':
            self._extension = 'pkl'
            self._CacheContainerClass = CachePickleContainer
        else:
            raise NotImplementedError
        self._storage_format = storage_format
        if s3_bucket is None:
            s3_bucket = settings['S3_BUCKET_NAME']
        if aws_access_key_id is None or aws_secret_access_key is None: 
             aws_access_key_id = settings['AWS_ACCESS_KEY_ID']
             aws_secret_access_key = settings['AWS_SECRET_ACCESS_KEY']
        if aws_region_name is None:
            aws_region_name = settings['AWS_REGION_NAME']
        aws_session = boto3.session.Session(region_name=aws_region_name)
        self._s3_client = aws_session.client('s3', 
                                             #config= boto3.session.Config(signature_version='s3v4'),
                                             aws_access_key_id=aws_access_key_id,
                                             aws_secret_access_key=aws_secret_access_key)
        self._s3_bucket = s3_bucket
        try:
            self._s3_client.head_bucket(Bucket=self._s3_bucket)
        except Exception as e:
            raise e

    def save_container(self, container, path):
        s = container.serialize()
        # note s needs to be b'bytes' or file below
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.put_object
        response = self._s3_client.put_object(Bucket=self._s3_bucket, Body=s, Key=path)

    def save_new_container(self, listings, path):
        self.save_container(self._CacheContainerClass(listings=listings),path)

    def load_container(self, path):
        response = self._s3_client.get_object(Bucket=self._s3_bucket, Key=path)
        s = response['Body'].read()
        return self._CacheContainerClass.deserialize(s)

    def _path_join(self, parts):
        return self._delim.join(parts)

    def delete_file(self, path):
        self._s3_client.delete_object(Bucket=self._s3_bucket, Key=path)

    def file_exists(self, path):
        response = self._s3_client.list_objects_v2(Bucket=self._s3_bucket, Prefix=path)
        obj_count = response['KeyCount']
        return obj_count == 1 and response['Contents'][0]['Key'] == path

    def make_cache_dir(self, node):
        """Create subdirectory for caching data for `node`
        
        Parameters
        ----------
        node : podpac.core.node.Node
            Description
        """
        # Place holder. Does not need to do anything for S3 as the prefix is just part of the object name.
        # note: I believe AWS uses prefixes to decide how to partition objects in a bucket which could affect performance.
        pass

    def cache_glob(self, node, key, coordinates):
        """Fileglob to match files that could be storing cached data for specified node,key,coordinates
        
        Parameters
        ----------
        node : podpac.core.node.Node
        key : str, CacheWildCard
            CacheWildCard indicates to match any key
        coordinates : podpac.core.coordinates.coordinates.Coordinates, CacheWildCard, None
            CacheWildCard indicates to match any coordinates
        
        Returns
        -------
        TYPE : str
            Fileglob of existing paths that match the request
        """
        delim = self._delim
        prefix = self.cache_dir(node)
        prefix = prefix if prefix.endswith(delim) else prefix + delim
        response = self._s3_client.list_objects_v2(Bucket=self._s3_bucket, Prefix=prefix, Delimiter=delim)

        if response['KeyCount'] > 0:
            obj_names = [o['Key'].replace(prefix,'') for o in response['Contents']]
        else:
            obj_names = []

        pre = '*'
        nKeY = 'nKeY{}'.format(self.hash_node(node))
        kKeY = 'kKeY*' if isinstance(key, CacheWildCard) else 'kKeY{}'.format(self.cleanse_filename_str(self.hash_key(key)))
        cKeY = 'cKeY*' if isinstance(coordinates, CacheWildCard) else 'cKeY{}'.format(self.hash_coordinates(coordinates))
        pat = '_'.join([pre, nKeY, kKeY, cKeY])
        pat = pat + '.' + self._extension

        obj_names = fnmatch.filter(obj_names, pat)

        paths = [delim.join([self.cache_dir(node), filename]) for filename in obj_names]
        return paths

    def clear_entire_cache_store(self):
        prefix = self._root_dir_path
        paginator = self._s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self._s3_bucket, Prefix=prefix)

        to_delete = dict(Objects=[])
        for item in pages.search('Contents'):
            if item:
                to_delete['Objects'].append(dict(Key=item['Key']))
            if len(to_delete['Objects']) >= 1000:
                self._s3_client.delete_objects(Bucket=self._s3_bucket, Delete=to_delete)
                to_delete = dict(Objects=[])

        if len(to_delete['Objects']):
            self._s3_client.delete_objects(Bucket=self._s3_bucket, Delete=to_delete)

        return True

    def dir_is_empty(self, directory):
        if not directory.endswith(self._delim):
            directory += self._delim
        response = self._s3_client.list_objects_v2(Bucket=self._s3_bucket, Prefix=directory, MaxKeys=2)
#TODO throw an error if key count is zero as this indicates `directory` is not an existing directory.
        return response['KeyCount'] == 1

    def rem_dir(self, directory):
        # s3 can have "empty" directories
        # should check if directory is empty and the delete
        # delete_objects could be used if recursive=True is specified to this function.
        # NOTE: This can remove the object representing the prefix without deleting other objects with the prefix.
        #       This is because s3 is really a key/value store. This function should maybe be changed to throw an
        #       error if the prefix is not "empty" or should delete all objects with the prefix. The former would 
        #       be more consistent with the os function used in the DiskCacheStore.
        # ToDo: 1) examine boto3 response, 2) handle object versioning (as it stands this will apply to the "null version")
        #      https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.delete_object
        if not directory.endswith(self._delim):
            directory += self._delim
        self._s3_client.delete_object(Bucket=self._s3_bucket, Key=directory)


_thread_local = threading.local()
_thread_local.cache = {}

class RamCacheStore(CacheStore):
    """
    RAM CacheStore.

    Notes
    -----
     * the cache is thread-safe, but not yet accessible across separate processes
     * there is not yet a max RAM usage setting or a removal policy.
    """

    cache_modes = set(['ram', 'all'])

    def _get_key(self, obj):
        # return obj.hash if obj else None
        # using the json directly means no hash collisions
        return obj.json if obj else None

    def _get_full_key(self, node, coordinates, key):
        return (self._get_key(node), self._get_key(coordinates), key)

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
        
        full_key = self._get_full_key(node, coordinates, key)
        
        if full_key in _thread_local.cache:
            if not update:
                raise CacheException("Cache entry already exists. Use update=True to overwrite.")

        # TODO
        # elif current_size + data_size > max_size:
        #     # TODO removal policy
        #     raise CacheException("RAM cache full. Remove some old entries and try again.")

        # TODO include insert date, last retrieval date, and/or # retrievals for use in a removal policy
        _thread_local.cache[full_key] = data

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

        full_key = self._get_full_key(node, coordinates, key)
        
        if full_key not in _thread_local.cache:
            raise CacheException("Cache miss. Requested data not found.")
        
        return _thread_local.cache[full_key]

    def rem(self, node=CacheWildCard(), key=CacheWildCard(), coordinates=CacheWildCard()):
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
        
        # shortcut
        if isinstance(node, CacheWildCard) and isinstance(coordinates, CacheWildCard) and isinstance(key, CacheWildCard):
            _thread_local.cache = {}
            return

        # loop through keys looking for matches
        if not isinstance(node, CacheWildCard):
            node_key = self._get_key(node)

        if not isinstance(coordinates, CacheWildCard):
            coordinates_key = self._get_key(coordinates)

        rem_keys = []
        for nk, ck, k in _thread_local.cache.keys():
            if not isinstance(node, CacheWildCard) and nk != node_key:
                continue
            if not isinstance(coordinates, CacheWildCard) and ck != coordinates_key:
                continue
            if not isinstance(key, CacheWildCard) and k != key:
                continue

            rem_keys.append((nk, ck, k))

        for k in rem_keys:
            del _thread_local.cache[k]

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
        
        full_key = self._get_full_key(node, coordinates, key)
        return full_key in _thread_local.cache
