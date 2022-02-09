from __future__ import division, print_function, absolute_import

import os
import glob
import shutil
import json
import fnmatch
import logging

import podpac
from podpac.core.settings import settings
from podpac.core.cache.utils import CacheException, CacheWildCard
from podpac.core.cache.file_cache_store import FileCacheStore


logger = logging.getLogger(__name__)


class DiskCacheStore(FileCacheStore):
    """Cache that uses a folder on a local disk file system."""

    cache_mode = "disk"
    cache_modes = set(["disk", "all"])
    _limit_setting = "DISK_CACHE_MAX_BYTES"

    def __init__(self):
        """Initialize a cache that uses a folder on a local disk file system."""

        if not settings["DISK_CACHE_ENABLED"]:
            raise CacheException("Disk cache is disabled in the podpac settings.")

        self._root_dir_path = settings.cache_path

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

    def search(self, node, item=CacheWildCard(), coordinates=CacheWildCard()):
        pattern = self._path_join(self._get_node_dir(node), self._get_filename_pattern(node, item, coordinates))
        return [path for path in glob.glob(pattern) if not path.endswith(".meta")]

    # -----------------------------------------------------------------------------------------------------------------
    # file storage abstraction
    # -----------------------------------------------------------------------------------------------------------------

    def _save(self, path, s, metadata=None):
        with open(path, "wb") as f:
            f.write(s)

        if metadata:
            metadata_path = "%s.meta" % path
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
        if os.path.exists("%s.meta" % path):
            os.remove("%s.meta" % path)

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
        metadata_path = "%s.meta" % path
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except IOError:
            # missing, permissions
            logger.exception("Error reading metadata file: '%s'" % metadata_path)
            return None
        except ValueError:
            # invalid json
            logger.exception("Error reading metadata file: '%s'" % metadata_path)
            return None

        return metadata.get(key)

    def _set_metadata(self, path, key, value):
        metadata_path = "%s.meta" % path

        # read existing
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except IOError:
            # missing, permissions
            logger.exception("Error reading metadata file: '%s'" % metadata_path)
            metadata = {}
        except ValueError:
            # invalid json
            logger.exception("Error reading metadata file: '%s'" % metadata_path)
            metadata = {}

        # write
        metadata[key] = value
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    def cleanup(self):
        """
        Remove expired entries and orphaned metadata.
        """

        for root, dirnames, filenames in os.walk(self._root_dir_path):
            for filename in fnmatch.filter(filenames, "*.meta"):
                metadata_path = os.path.join(root, filename)
                path = os.path.join(root, filename[:-5])  # strip .meta
                if not os.path.exists(path):  # orphaned
                    os.remove(metadata_path)
                elif self._expired(path):
                    # _expired removes the entry automatically
                    pass

        # remove empty directories
        for root, dirnames, filenames in os.walk(self._root_dir_path):
            for dirname in dirnames:
                path = os.path.join(root, dirname)
                if not os.path.exists(path):
                    continue
                if not [f for r, d, fs in os.walk(path) for f in fs]:
                    shutil.rmtree(path)
