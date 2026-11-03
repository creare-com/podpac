from __future__ import division, print_function, absolute_import

import os
import glob
import shutil
import json
import fnmatch
import logging
import threading
from filelock import ReadWriteLock

from podpac.core.settings import settings
from podpac.core.cache.utils import CacheException, CacheWildCard
from podpac.core.cache.file_cache_store import FileCacheStore

logger = logging.getLogger(__name__)

_DOT_META = "%s.meta"
_ERROR_READING_METADATA_FILE = "Error reading metadata file: '%s'"

_thread_local = threading.local()


class DiskCacheStore(FileCacheStore):
    """Cache that uses a folder on a local disk file system."""

    cache_mode = "disk"
    cache_modes = set(["disk", "all"])
    _limit_setting = "DISK_CACHE_MAX_BYTES"

    def __init__(self):
        """Initialize a cache that uses a folder on a local disk file system."""

        if not settings["ENABLE_CACHE"]:
            raise CacheException("Cache is disabled in the podpac settings.")

        self._root_dir_path = settings.cache_path

    # -----------------------------------------------------------------------------------------------------------------
    # public cache API
    # -----------------------------------------------------------------------------------------------------------------

    @property
    def _read_lock(self):
        """Shared lock for reads, held concurrently by any number of threads and processes.

        Returns
        -------
        contextlib.AbstractContextManager
            The read lock context manager.
        """
        return self._rw_lock.read_lock()

    @property
    def _write_lock(self):
        """Exclusive lock for writes, blocks all other readers and writers.

        Returns
        -------
        contextlib.AbstractContextManager
            The write lock context manager.
        """
        return self._rw_lock.write_lock()

    @property
    def _rw_lock(self):
        """A read-write lock for file operations.

        Locked state is tracked per instance of the lock so each thread needs to hold its own instance separately.
        """
        if not hasattr(_thread_local, "locks"):
            _thread_local.locks = {}
        locks = _thread_local.locks
        path = f"{os.path.normpath(self._root_dir_path)}.lock"

        # forked children inherit thread-locals, use process id to avoid using a parent lock
        key = (os.getpid(), path)
        if key not in locks:
            locks[key] = ReadWriteLock(path, is_singleton=False)
        return locks[key]

    # -----------------------------------------------------------------------------------------------------------------
    # helper methods
    # -----------------------------------------------------------------------------------------------------------------

    def _search(self, node, item=CacheWildCard(), coordinates=CacheWildCard()):
        pattern = self._path_join(self._get_node_dir(node), self._get_filename_pattern(node, item, coordinates))
        return [path for path in glob.glob(pattern) if not path.endswith(".meta")]

    def _size(self):
        total_size = 0
        for dirpath, _dirnames, filenames in os.walk(self._root_dir_path):
            for f in filenames:
                total_size += os.path.getsize(os.path.join(dirpath, f))
        return total_size

    # -----------------------------------------------------------------------------------------------------------------
    # file storage abstraction
    # -----------------------------------------------------------------------------------------------------------------

    def _save(self, path, s, metadata=None):
        with open(path, "wb") as f:
            f.write(s)

        if metadata:
            metadata_path = _DOT_META % path
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

    def _load(self, path):
        with open(path, "rb") as f:
            return f.read()

    def _path_join(self, path, *paths):
        return os.path.join(path, *paths)

    def _basename(self, path):
        return os.path.basename(path)

    def _remove(self, path):
        os.remove(path)
        if os.path.exists(_DOT_META % path):
            os.remove(_DOT_META % path)

    def _exists(self, path):
        return os.path.exists(path)

    def _is_empty(self, directory):
        return os.path.exists(directory) and os.path.isdir(directory) and not os.listdir(directory)

    def _rmdir(self, directory):
        os.rmdir(directory)

    def _rmtree(self, path, ignore_errors=False):
        shutil.rmtree(path, ignore_errors=True)

    def _make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def _dirname(self, path):
        return os.path.dirname(path)

    def _get_metadata(self, path, key):
        metadata_path = _DOT_META % path
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except IOError:
            # missing, permissions
            logger.exception(_ERROR_READING_METADATA_FILE % metadata_path)
            return None
        except ValueError:
            # invalid json
            logger.exception(_ERROR_READING_METADATA_FILE % metadata_path)
            return None

        return metadata.get(key)

    def _set_metadata(self, path, key, value):
        metadata_path = _DOT_META % path

        # read existing
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except IOError:
            # missing, permissions
            logger.exception(_ERROR_READING_METADATA_FILE % metadata_path)
            metadata = {}
        except ValueError:
            # invalid json
            logger.exception(_ERROR_READING_METADATA_FILE % metadata_path)
            metadata = {}

        # write
        metadata[key] = value
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    def _cleanup(self):
        for root, _dirnames, filenames in os.walk(self._root_dir_path):
            for filename in fnmatch.filter(filenames, "*.meta"):
                metadata_path = os.path.join(root, filename)
                path = os.path.join(root, filename[:-5])  # strip .meta
                if not os.path.exists(path):  # orphaned
                    os.remove(metadata_path)
                elif self._expired(path):
                    self._remove(path)

        # remove empty directories
        for root, dirnames, _filenames in os.walk(self._root_dir_path):
            for dirname in dirnames:
                path = os.path.join(root, dirname)
                if not os.path.exists(path):
                    continue
                if not [f for r, d, fs in os.walk(path) for f in fs]:
                    shutil.rmtree(path)
