from __future__ import division, print_function, absolute_import

import os
import glob
import shutil

import podpac
from podpac.core.settings import settings
from podpac.core.cache.utils import CacheException, CacheWildCard
from podpac.core.cache.file_cache_store import FileCacheStore


class DiskCacheStore(FileCacheStore):
    """Cache that uses a folder on a local disk file system."""

    cache_mode = "disk"
    cache_modes = set(["disk", "all"])
    _limit_setting = "DISK_CACHE_MAX_BYTES"

    def __init__(self):
        """Initialize a cache that uses a folder on a local disk file system."""

        if not settings["DISK_CACHE_ENABLED"]:
            raise CacheException("Disk cache is disabled in the podpac settings.")

        if os.path.isabs(settings["DISK_CACHE_DIR"]):
            self._root_dir_path = settings["DISK_CACHE_DIR"]
        else:
            self._root_dir_path = os.path.join(
                settings["ROOT_PATH"], settings["DISK_CACHE_DIR"]
            )

    # -----------------------------------------------------------------------------------------------------------------
    # public cache API
    # -----------------------------------------------------------------------------------------------------------------

    @property
    def size(self):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self._root_dir_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size

    # -----------------------------------------------------------------------------------------------------------------
    # helper methods
    # -----------------------------------------------------------------------------------------------------------------

    def search(self, node, key=CacheWildCard(), coordinates=CacheWildCard()):
        match_path = self._path_join(
            self._get_node_dir(node), self._match_filename(node, key, coordinates)
        )
        return glob.glob(match_path)

    # -----------------------------------------------------------------------------------------------------------------
    # file storage abstraction
    # -----------------------------------------------------------------------------------------------------------------

    def _save(self, path, s):
        with open(path, "wb") as f:
            f.write(s)

    def _load(self, path):
        with open(path, "rb") as f:
            return f.read()

    def _path_join(self, path, *paths):
        return os.path.join(path, *paths)

    def _basename(self, path):
        return os.path.basename(path)

    def _remove(self, path):
        os.remove(path)

    def _exists(self, path):
        return os.path.exists(path)

    def _is_empty(self, directory):
        return (
            os.path.exists(directory)
            and os.path.isdir(directory)
            and not os.listdir(directory)
        )

    def _rmdir(self, directory):
        os.rmdir(directory)

    def _rmtree(self, path, ignore_errors=False):
        shutil.rmtree(self._root_dir_path, ignore_errors=True)

    def _make_node_dir(self, node):
        node_dir = self._get_node_dir(node)
        if not os.path.exists(node_dir):
            os.makedirs(node_dir)
