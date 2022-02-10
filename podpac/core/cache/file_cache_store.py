from __future__ import division, print_function, absolute_import

import json
import io
import warnings
import re
import hashlib
import time

try:
    import cPickle as pickle  # python 2
except:
    import pickle

import numpy as np
import xarray as xr

import podpac
from podpac.core.settings import settings
from podpac.core.utils import is_json_serializable
from podpac.core.cache.utils import CacheException, CacheWildCard, expiration_timestamp
from podpac.core.cache.cache_store import CacheStore
from podpac.core.utils import hash_alg


def _hash_string(s):
    return hash_alg(s.encode()).hexdigest()


class FileCacheStore(CacheStore):
    """Abstract class with functionality common to persistent CacheStore objects (e.g. local disk, s3) that store things using multiple paths (filepaths or object paths)"""

    cache_mode = ""
    cache_modes = ["all"]

    _root_dir_path = None  # should be set by children

    # -----------------------------------------------------------------------------------------------------------------
    # public cache API methods
    # -----------------------------------------------------------------------------------------------------------------

    def has(self, node, item, coordinates=None):
        """Check for valid cached data for this node.

        Parameters
        ------------
        node : Node
            node requesting storage.
        item : str
            Cached object key, e.g. 'output'.
        coordinates: Coordinate, optional
            Coordinates for which cached object should be checked

        Returns
        -------
        has_cache : bool
             True if there is a valid cached object for this node for the given key and coordinates.
        """

        path = self.find(node, item, coordinates)
        return path is not None and not self._expired(path)

    def put(self, node, data, item, coordinates=None, expires=None, update=True):
        """Cache data for specified node.

        Parameters
        ------------
        node : Node
            node requesting storage.
        data : any
            Data to cache
        item : str
            Cached object item, e.g. 'output'.
        coordinates : :class:`podpac.Coordinates`, optional
            Coordinates for which cached object should be retrieved, for coordinate-dependent data such as evaluation output
        expires : float, datetime, timedelta
            Expiration date. If a timedelta is supplied, the expiration date will be calculated from the current time.
        update : bool
            If True existing data in cache will be updated with `data`, If False, error will be thrown if attempting put something into the cache with the same node, key, coordinates of an existing entry.
        """

        # check for valid existing entry (expired entries are automatically ignored and overwritten)
        if self.has(node, item, coordinates):
            if not update:
                raise CacheException("Cache entry already exists. Use `update=True` to overwrite.")
            else:
                self._remove(self.find(node, item, coordinates))

        # serialize
        root = self._get_filename(node, item, coordinates)

        if isinstance(data, podpac.core.units.UnitsDataArray):
            ext = "uda.nc"
            s = data.to_netcdf()
        elif isinstance(data, xr.DataArray):
            ext = "xrda.nc"
            s = data.to_netcdf()
        elif isinstance(data, xr.Dataset):
            ext = "xrds.nc"
            s = data.to_netcdf()
        elif isinstance(data, np.ndarray):
            ext = "npy"
            with io.BytesIO() as f:
                np.save(f, data)
                s = f.getvalue()
        elif isinstance(data, podpac.Coordinates):
            ext = "coords.json"
            s = data.json.encode()
        elif isinstance(data, podpac.Node):
            ext = "node.json"
            s = data.json.encode()
        elif is_json_serializable(data):
            ext = "json"
            s = json.dumps(data).encode()
        else:
            warnings.warn(
                "Object of type '%s' is not json serializable; caching object to file using pickle, which "
                "may not be compatible with other Python versions or podpac versions." % type(data)
            )
            ext = "pkl"
            s = pickle.dumps(data)

        # check size
        if self.max_size is not None:
            new_size = self.size + len(s)

            if new_size > self.max_size:
                # cleanup and check again
                self.cleanup()

            if new_size > self.max_size:
                # TODO removal policy (using create time, last access, etc)
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
        node_dir = self._get_node_dir(node)
        path = self._path_join(node_dir, "%s.%s" % (root, ext))

        metadata = {"created": time.time(), "accessed": None, "expires": expiration_timestamp(expires)}

        self._make_dir(node_dir)
        self._save(path, s, metadata=metadata)

        return True

    def get(self, node, item, coordinates=None):
        """Get cached data for this node.

        Parameters
        ------------
        node : Node
            node requesting storage.
        item : str
            Cached object item, e.g. 'output'.
        coordinates : :class:`podpac.Coordinates`, optional
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

        path = self.find(node, item, coordinates)
        if path is None:
            raise CacheException("Cache miss. Requested data not found.")

        # get metadata
        if self._expired(path):
            raise CacheException("Cache miss. Requested data expired.")

        # read
        s = self._load(path)
        self._set_metadata(path, "accessed", time.time())

        # deserialize
        if path.endswith(".uda.nc"):
            x = xr.open_dataarray(s)
            data = podpac.core.units.UnitsDataArray(x)._pp_deserialize()
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

    def rem(self, node, item=CacheWildCard(), coordinates=CacheWildCard()):
        """Delete cached data for this node.

        Parameters
        ------------
        node : Node
            node requesting storage
        item : str, CacheWildCard, optional
            Delete cached objects item, or any item if `item` is a CacheWildCard.
        coordinates : :class:`podpac.Coordinates`, CacheWildCard, None, optional
            Delete only cached objects for these coordinates, or any coordinates if `coordinates` is a CacheWildCard. `None` specifically indicates entries that do not have coordinates.
        """

        # delete matching cached objects
        for path in self.search(node, item=item, coordinates=coordinates):
            self._remove(path)

        # remove empty node directories
        if not self.search(node):
            path = self._get_node_dir(node=node)
            while self._is_empty(path):
                self._rmtree(path)
                path = self._dirname(path)

    def clear(self):
        """
        Clear all cached data.
        """

        self._rmtree(self._root_dir_path)

    # -----------------------------------------------------------------------------------------------------------------
    # helper methods
    # -----------------------------------------------------------------------------------------------------------------

    def search(self, node, item=CacheWildCard(), coordinates=CacheWildCard()):
        """
        Search for matching cached objects.
        """
        raise NotImplementedError

    def find(self, node, item, coordinates=None):
        """
        Find the path for a specific cached object.
        """

        paths = self.search(node, item=item, coordinates=coordinates)

        if len(paths) == 0:
            return None
        elif len(paths) == 1:
            return paths[0]
        elif len(paths) > 1:
            return RuntimeError("Too many cached files matching '%s'" % self._root_dir_path)

    def _get_node_dir(self, node):
        fullclass = str(node.__class__)[8:-2]
        subdirs = fullclass.split(".")
        return self._path_join(self._root_dir_path, *subdirs)

    def _get_filename(self, node, key, coordinates):
        prefix = self._sanitize("%s-%s" % (node.base_ref, key))
        filename = "%s_%s_%s_%s" % (prefix, node.hash, _hash_string(key), coordinates.hash if coordinates else "None")
        return filename

    def _get_filename_pattern(self, node, key, coordinates):
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

    def _expired(self, path):
        expires = self._get_metadata(path, "expires")
        if expires is not None and time.time() >= expires:
            self._remove(path)
            return True
        return False

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

    def _make_dir(self, path):
        raise NotImplementedError

    def _dirname(self, path):
        raise NotImplementedError

    def _get_metadata(self, path, key):
        raise NotImplementedError

    def _set_metadata(self, path, key, value):
        raise NotImplementedError
