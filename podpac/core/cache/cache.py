"""

"""

from __future__ import division, print_function, absolute_import

import os
import threading
import copy
from glob import glob
import shutil
import json
from hashlib import md5 as hash_alg
try:
    import cPickle  # Python 2.7
except:
    import _pickle as cPickle

import six
import fnmatch
from lazy_import import lazy_module
import psutil
import xarray as xr

import warnings

boto3 = lazy_module('boto3')

import podpac
from podpac.core.settings import settings

_cache_types = {'ram','disk','network','all','s3'}

def is_json_serializable(obj):
    try:
        json.dumps(obj)
    except:
        return False
    else:
        return True

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
    ctrl : CacheCtrl or None
        Default CachCtrl
    """

    if settings.get('DEFAULT_CACHE') is None: # missing or None
        return CacheCtrl([])

    return make_cache_ctrl(settings['DEFAULT_CACHE'])

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
        if elem == 'ram':
            cache_stores.append(RamCacheStore())
        elif elem == 'disk':
            cache_stores.append(DiskCacheStore())
        elif elem == 's3':
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

    def size(self):
        """Return size of cache store in bytes
        """
        raise NotImplementedError


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

class FileCacheStore(CacheStore):
    """Base class with functionality common to persistent CacheStore objects (e.g. local disk, s3) that store things using multiple paths (filepaths or object paths)
    """
    
    cache_mode = ''
    cache_modes = ['all']
    limit_setting = ''

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def make_cache_dir(self, node):
        raise NotImplementedError

    def _save(self, path, s):
        raise NotImplementedError

    def _load(self, path):
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
        return self._path_join(dirs).replace('<', '_').replace('>', '_')

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
        nKeY = 'nKeY{}'.format(self.hash_node(node))
        kKeY = 'kKeY{}'.format(self.hash_key(key))
        cKeY = 'cKeY{}'.format(self.hash_coordinates(coordinates))
        filename = '_'.join([pre, nKeY, kKeY, cKeY])
        return filename

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
        
        
        # check for existing entry
        if self.has(node, key, coordinates):
            if not update:
                raise CacheException("Existing cache entry. Use `update=True` to overwrite.")
            self.rem(node, key, coordinates)

        # serialize
        rootpath = self.cache_path(node, key, coordinates)
        
        if isinstance(data, (xr.DataArray, xr.DataArray)):
            path = rootpath + '.netcdf'
            s = data.to_netcdf()
        elif isinstance(data, podpac.Coordinates):
            path = rootpath + '.coords'
            s = data.json
        elif is_json_serializable(data):
            path = rootpath + '.json'
            s = json.dumps(data)
        else:
            warnings.warn("Object of type '%s' is not json serializable; caching object to file using pickle, which "
                          "may not be compatible with other Python versions or podpac versions.")
            path = rootpath + '.pkl'
            s = cPickle.dumps(data)

        # check size
        if self.max_size is not None and self.size + len(s) > self.max_size:
            # TODO removal policy
            warnings.warn("Warning: {cache_mode} cache is full. No longer caching. Consider increasing the limit in "
                          "settings.{cache_limit_setting} or try clearing the cache (e.g. node.rem_cache(key='*', "
                          "mode='{cache_mode}', all_cache=True) to clear ALL cached results in {cache_mode} cache)".format(
                            cache_mode=self.cache_mode, cache_limit_setting=self.limit_setting), RuntimeWarning)
            return False

        # save
        self.make_cache_dir(node)
        self._save(path, s)
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

        path = self.find(node, key, coordinates)

        if path is None:
            raise CacheException("Cache miss. Requested data not found.")

        # read
        s = self._load(path)
        
        # deserialize
        if path.endswith('.netcdf'):
            data = xr.open_dataset(s)
        elif path.endswith('.coords'):
            data = Coordinates.from_json(s)
        elif path.endswith('.json'):
            data = json.loads(s)
        elif path.endswith('.pkl'):
            data = cPickle.loads(s)
        else:
            raise CacheException("Unexpected file type '%s'" % os.path.basename(path))

        # TODO should we allow None?
        if data is None:
            raise CacheException("Stored data is None.")

        return data

    def find(self, node, key, coordinates=None):
        rootpath = self.cache_path(node, key, coordinates)
        paths = glob(rootpath + '.*')
        if len(paths) == 0:
            return None
        elif len(paths) > 1:
            return CacheException("Too many files matching pattern root '%s'" % rootpath)
        else:
            return paths[0]

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

        # clear the entire cache store
        if isinstance(node, CacheWildCard):
            return self.clear_entire_cache_store()

        # otherwise, remove matching cache entries for the given node, and remove empty directories
        else:
            paths = self.cache_glob(node, key=key, coordinates=coordinates)
            for path in paths:
                self.delete_file(path)

            cache_dir = self.cache_dir(node=node)
            if self.dir_is_empty(cache_dir):
                self.rem_dir(cache_dir)

            return len(paths) > 0

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
        
        path = self.find(node, key, coordinates)
        return path is not None

class DiskCacheStore(FileCacheStore):

    cache_mode = 'disk'
    cache_modes = set(['disk','all'])
    limit_setting = 'DISK_CACHE_MAX_BYTES'

    def __init__(self, root_cache_dir_path=None, max_size=None, use_settings_limit=True):
        """Initialize a cache that uses a folder on a local disk file system.
        
        Parameters
        ----------
        root_cache_dir_path : None, optional
            Root directory for the files managed by this cache. `None` indicates to use the folder specified in the global podpac settings. Should be a fully specified valid path.
        max_size : None, optional
            Maximum allowed size of the cache store in bytes. Defaults to podpac 'DISK_CACHE_MAX_BYTES' setting, or no limit if this setting does not exist.
        use_settings_limit : bool, optional
            Use podpac settings to determine cache limits if True, this will also cause subsequent runtime changes to podpac settings module to effect the limit on this cache. Default is True.
        
        Raises
        ------
        CacheException
            Description
        """

        if not settings['DISK_CACHE_ENABLED']:
            raise CacheException("Disk cache is disabled in the podpac settings.")

        # set cache dir
        if root_cache_dir_path is not None:
            self._root_dir_path = root_cache_dir_path
        elif os.path.isabs(settings['DISK_CACHE_DIR']):
            self._root_dir_path = settings['DISK_CACHE_DIR']
        else:
            self._root_dir_path = self._path_join([settings['ROOT_PATH'], settings['DISK_CACHE_DIR']])

        self._use_settings_limit = use_settings_limit
        if max_size is not None:
            self._max_size = max_size
        elif self._use_settings_limit and self.limit_setting in settings and settings[self.limit_setting]:
            self._max_size = settings[self.limit_setting]
        else:
            self._max_size = None

        # make directory if it doesn't already exist
        os.makedirs(self._root_dir_path, exist_ok=True)

    def _save(self, path, s):
        with open(path, 'wb') as f:
            f.write(s)

    def _load(self, path):
        with open(path, 'rb') as f:
            return f.read()

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
        filename = filename + '.*'
        return glob(self._path_join([self.cache_dir(node), filename]))

    def clear_entire_cache_store(self):
        shutil.rmtree(self._root_dir_path, ignore_errors=True)
        return True

    def dir_is_empty(self, directory):
        return os.path.exists(directory) and os.path.isdir(directory) and not os.listdir(directory)

    def rem_dir(self, directory):
        os.rmdir(directory)

    @property
    def max_size(self):
        if self._use_settings_limit and self.limit_setting and self.limit_setting in settings:
            return settings[self.limit_setting]
        return self._max_size

    @property
    def size(self):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self._root_dir_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size

class S3CacheStore(FileCacheStore): # pragma: no cover

    cache_mode = 's3'
    limit_setting = 'S3_CACHE_MAX_BYTES'
    cache_modes = set(['s3','all'])
    _delim = '/'

    def __init__(self, root_cache_dir_path=None, max_size=None, use_settings_limit=True,
                 s3_bucket=None, aws_region_name=None, aws_access_key_id=None, aws_secret_access_key=None):
        """Initialize a cache that uses a folder on a local disk file system.
        
        Parameters
        ----------
        root_cache_dir_path : None, optional
            Root directory for the files managed by this cache. `None` indicates to use the folder specified in the global podpac settings. Should be the common "root" s3 prefix that you want to have all cached objects stored in. Do not store any objects in the bucket that share this prefix as they may be deleted by the cache.
        max_size : None, optional
            Maximum allowed size of the cache store in bytes. Defaults to podpac 'S3_CACHE_MAX_BYTES' setting, or no limit if this setting does not exist.
        use_settings_limit : bool, optional
            Use podpac settings to determine cache limits if True, this will also cause subsequent runtime changes to podpac settings module to effect the limit on this cache. Default is True.
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
        """
        
        if not settings['S3_CACHE_ENABLED']:
            raise CacheException("S3 cache is disabled in the podpac settings.")

        self._cache_modes = set(['s3','all'])
        if root_cache_dir_path is None:
            root_cache_dir_path = settings['S3_CACHE_DIR']
        self._root_dir_path = root_cache_dir_path
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

        self._use_settings_limit = use_settings_limit
        if max_size is not None:
            self._max_size = max_size
        elif self._use_settings_limit and self.limit_setting in settings and settings[self.limit_setting]:
            self._max_size = settings[self.limit_setting]
        else:
            self._max_size = None

        try:
            self._s3_client.head_bucket(Bucket=self._s3_bucket)
        except Exception as e:
            raise e

    def _save(self, path, s):
        # note s needs to be b'bytes' or file below
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.put_object
        self._s3_client.put_object(Bucket=self._s3_bucket, Body=s, Key=path)

    def _load(self, path):
        response = self._s3_client.get_object(Bucket=self._s3_bucket, Key=path)
        return response['Body'].read()

    def _path_join(self, parts):
        return self._delim.join(parts)

    def delete_file(self, path):
        self._s3_client.delete_object(Bucket=self._s3_bucket, Key=path)

    def file_exists(self, path):
        response = self._s3_client.list_objects_v2(Bucket=self._s3_bucket, Prefix=path)
        obj_count = response['KeyCount']
        return obj_count == 1 and response['Contents'][0]['Key'] == path

    @property
    def max_size(self):
        if self._use_settings_limit and self.limit_setting and self.limit_setting in settings:
            return settings[self.limit_setting]
        return self._max_size

    @property
    def size(self):
        paginator = self._s3_client.get_paginator('list_objects')
        operation_parameters = {'Bucket': self._s3_bucket,
                                'Prefix': self._root_dir_path}
        page_iterator = paginator.paginate(**operation_parameters)
        total_size = 0
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    total_size += obj['Size']
        return total_size

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
            CacheWildCard indicates to macth any coordinates
        
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
        pat = pat + '.*'

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

class RamCacheStore(CacheStore):
    """
    RAM CacheStore.

    Notes
    -----
     * the cache is thread-safe, but not yet accessible across separate processes
     * there is not yet a max RAM usage setting or a removal policy.
    """

    cache_mode = 'ram'
    cache_modes = set(['ram', 'all'])
    limit_setting = 'RAM_CACHE_MAX_BYTES'

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
        if not settings['RAM_CACHE_ENABLED']:
            raise CacheException("RAM cache is disabled in the podpac settings.")

        self._use_settings_limit = use_settings_limit
        if max_size is not None:
            self._max_size = max_size
        elif self._use_settings_limit and self.limit_setting in settings and settings[self.limit_setting]:
            self._max_size = settings[self.limit_setting]
        else:
            self._max_size = None

        super(CacheStore, self).__init__()

    def _get_key(self, obj):
        # return obj.hash if obj else None
        # using the json directly means no hash collisions
        return obj.json if obj else None

    def _get_full_key(self, node, coordinates, key):
        return (self._get_key(node), self._get_key(coordinates), key)

    @property
    def max_size(self):
        if self._use_settings_limit and self.limit_setting and self.limit_setting in settings:
            return settings[self.limit_setting]
        return self._max_size

    @property
    def size(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss # this is actually the total size of the process

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
        
        if not hasattr(_thread_local, 'cache'):
            _thread_local.cache = {}

        full_key = self._get_full_key(node, coordinates, key)
        
        if full_key in _thread_local.cache:
            if not update:
                raise CacheException("Cache entry already exists. Use update=True to overwrite.")

        if self.max_size is not None and self.size >= self.max_size:
        #     # TODO removal policy
            warnings.warn("Warning: Process is using more RAM than the specified limit in settings.RAM_CACHE_MAX_BYTES. No longer caching. Consider increasing this limit or try clearing the cache (e.g. node.rem_cache(key='*', mode='RAM', all_cache=True) to clear ALL cached results in RAM)", RuntimeWarning)
            return False

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

        if not hasattr(_thread_local, 'cache'):
            _thread_local.cache = {}

        full_key = self._get_full_key(node, coordinates, key)
        
        if full_key not in _thread_local.cache:
            raise CacheException("Cache miss. Requested data not found.")
        
        return copy.deepcopy(_thread_local.cache[full_key])

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
        
        if not hasattr(_thread_local, 'cache'):
            _thread_local.cache = {}

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
        
        if not hasattr(_thread_local, 'cache'):
            _thread_local.cache = {}

        full_key = self._get_full_key(node, coordinates, key)
        return full_key in _thread_local.cache
