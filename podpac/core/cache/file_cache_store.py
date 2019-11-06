from __future__ import division, print_function, absolute_import

import json
import io
import warnings
import re
import hashlib

try:
    import cPickle as pickle  # python 2
except:
    import pickle

import numpy as np
import xarray as xr

import podpac
from podpac.core.settings import settings
from podpac.core.utils import is_json_serializable
from podpac.core.cache.utils import CacheException, CacheWildCard
from podpac.core.cache.cache_store import CacheStore


def _hash_string(s):
    return hashlib.md5(s.encode()).hexdigest()


class FileCacheStore(CacheStore):
    """Abstract class with functionality common to persistent CacheStore objects (e.g. local disk, s3) that store things using multiple paths (filepaths or object paths)
    """

    cache_mode = ""
    cache_modes = ["all"]

    # -----------------------------------------------------------------------------------------------------------------
    # public cache API methods
    # -----------------------------------------------------------------------------------------------------------------

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

        # check for existing entry
        if self.has(node, key, coordinates):
            if not update:
                raise CacheException("Cache entry already exists. Use `update=True` to overwrite.")
            self.rem(node, key, coordinates)

        # serialize
        path_root = self._path_join(self._get_node_dir(node), self._get_filename(node, key, coordinates))

        if isinstance(data, podpac.core.units.UnitsDataArray):
            path = path_root + ".uda.nc"
            s = data.to_netcdf()
        elif isinstance(data, xr.DataArray):
            path = path_root + ".xrda.nc"
            s = data.to_netcdf()
        elif isinstance(data, xr.Dataset):
            path = path_root + ".xrds.nc"
            s = data.to_netcdf()
        elif isinstance(data, np.ndarray):
            path = path_root + ".npy"
            with io.BytesIO() as f:
                np.save(f, data)
                s = f.getvalue()
        elif isinstance(data, podpac.Coordinates):
            path = path_root + ".coords.json"
            s = data.json.encode()
        elif isinstance(data, podpac.Node):
            path = path_root + ".node.json"
            s = data.json.encode()
        elif is_json_serializable(data):
            path = path_root + ".json"
            s = json.dumps(data).encode()
        else:
            warnings.warn(
                "Object of type '%s' is not json serializable; caching object to file using pickle, which "
                "may not be compatible with other Python versions or podpac versions."
            )
            path = path_root + ".pkl"
            s = pickle.dumps(data)

        # check size
        if self.max_size is not None and self.size + len(s) > self.max_size:
            # TODO removal policy
            warnings.warn(
                "Warning: {cache_mode} cache is full. No longer caching. Consider increasing the limit in "
                "settings.{cache_limit_setting} or try clearing the cache (e.g. podpac.utils.clear_cache(, "
                "mode='{cache_mode}') to clear ALL cached results in {cache_mode} cache)".format(
                    cache_mode=self.cache_mode, cache_limit_setting=self._limit_setting
                ),
                UserWarning,
            )
            return False

        # save
        self._make_node_dir(node)
        self._save(path, s)
        return True

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

        path = self.find(node, key, coordinates)
        if path is None:
            raise CacheException("Cache miss. Requested data not found.")

        # read
        s = self._load(path)

        # deserialize
        if path.endswith("uda.nc"):
            x = xr.open_dataarray(s)
            data = podpac.core.units.UnitsDataArray(x)
        elif path.endswith(".xrda.nc"):
            data = xr.open_dataarray(s)
        elif path.endswith(".xrds.nc"):
            data = xr.open_dataset(s)
        elif path.endswith(".npy"):
            with io.BytesIO(s) as b:
                data = np.load(b)
        elif path.endswith(".coords.json"):
            data = podpac.Coordinates.from_json(s.decode())
        elif path.endswith(".node.json"):
            data = podpac.Node.from_json(s.decode())
        elif path.endswith(".json"):
            data = json.loads(s.decode())
        elif path.endswith(".pkl"):
            data = pickle.loads(s)
        else:
            raise RuntimeError("Unexpected cached file type '%s'" % self._basename(path))

        return data

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

        path = self.find(node, key, coordinates)
        return path is not None

    def rem(self, node, key=CacheWildCard(), coordinates=CacheWildCard()):
        """Delete cached data for this node.
        
        Parameters
        ------------
        node : Node
            node requesting storage
        key : str, CacheWildCard, optional
            Delete cached objects with this key, or any key if `key` is a CacheWildCard.
        coordinates : Coordinates, CacheWildCard, None, optional
            Delete only cached objects for these coordinates, or any coordinates if `coordinates` is a CacheWildCard. `None` specifically indicates entries that do not have coordinates.
        """

        # delete matching cached objects
        for path in self.search(node, key=key, coordinates=coordinates):
            self._remove(path)

        # remove empty node directories
        node_dir = self._get_node_dir(node=node)
        if self._is_empty(node_dir):
            self._rmdir(node_dir)

    def clear(self):
        """
        Clear all cached data.
        """

        self._rmtree(self._root_dir_path)

    # -----------------------------------------------------------------------------------------------------------------
    # helper methods
    # -----------------------------------------------------------------------------------------------------------------

    def search(self, node, key=CacheWildCard(), coordinates=CacheWildCard()):
        """
        Search for matching cached objects.
        """
        NotImplementedError

    def find(self, node, key, coordinates=None):
        """
        Find the path for a specific cached object.
        """

        paths = self.search(node, key=key, coordinates=coordinates)

        if len(paths) == 0:
            return None
        elif len(paths) == 1:
            return paths[0]
        elif len(paths) > 1:
            return RuntimeError("Too many cached files matching '%s'" % rootpath)

    def _get_node_dir(self, node):
        fullclass = str(node.__class__)[8:-2]
        subdirs = fullclass.split(".")
        return self._path_join(self._root_dir_path, *subdirs)

    def _get_filename(self, node, key, coordinates):
        prefix = self._sanitize("%s-%s" % (node.base_ref, key))
        filename = "%s_%s_%s_%s" % (prefix, node.hash, _hash_string(key), coordinates.hash if coordinates else "None")
        return filename

    def _match_filename(self, node, key, coordinates):
        match_prefix = "*"
        match_node = node.hash

        if isinstance(key, CacheWildCard):
            match_key = "*"
        else:
            match_key = _hash_string(key)

        if isinstance(coordinates, CacheWildCard):
            match_coordinates = "*"
        elif coordinates is None:
            match_coordinates = "None"
        else:
            match_coordinates = coordinates.hash

        match_filename = "%s_%s_%s_%s.*" % (match_prefix, match_node, match_key, match_coordinates)
        return match_filename

    def _sanitize(self, s):
        return re.sub("[_:<>/\\\\*]+", "-", s)  # replaces _:<>/\*

    # -----------------------------------------------------------------------------------------------------------------
    # file storage abstraction
    # -----------------------------------------------------------------------------------------------------------------

    def _save(self, path, s):
        raise NotImplementedError

    def _load(self, path):
        raise NotImplementedError

    def _path_join(self, path, *paths):
        raise NotImplementedError

    def _basename(self, path):
        raise NotImplementedError

    def _remove(self, path):
        raise NotImplementedError

    def _exists(self, path):
        raise NotImplementedError

    def _is_empty(self, directory):
        raise NotImplementedError

    def _rmdir(self, directory):
        raise NotImplementedError

    def _make_node_dir(self, node):
        raise NotImplementedError
