import os
import shutil
import copy
import tempfile
import time
import datetime
import threading
from typing import Any
from unittest.mock import DEFAULT, patch

import filelock
import pytest
import xarray as xr
import numpy as np

import podpac
from podpac.core.cache.utils import CacheException
from podpac.core.cache.ram_cache_store import RamCacheStore
from podpac.core.cache.disk_cache_store import DiskCacheStore, _thread_local
from podpac.core.cache.s3_cache_store import S3CacheStore

COORDS1 = podpac.Coordinates([[0, 1, 2], [10, 20, 30, 40], ["2018-01-01", "2018-01-02"]], dims=["lat", "lon", "time"])
COORDS2 = podpac.Coordinates([[0, 1, 2], [10, 20, 30]], dims=["lat", "lon"])
NODE1 = podpac.data.Array(source=np.ones(COORDS1.shape), coordinates=COORDS1)
NODE2 = podpac.algorithm.Arange()
CACHE_MISS = "Cache miss. Requested data expired"


class BaseCacheStoreTests(object):
    Store = None
    enabled_setting = None
    limit_setting = None

    def setup_method(self):
        self.settings_orig = copy.deepcopy(podpac.settings)

    def teardown_method(self):
        for key in podpac.settings:
            podpac.settings[key] = self.settings_orig[key]

    def test_init(self):
        _ = self.Store()

    def test_disabled(self):  # this is the only test that passes for S3 because I don't have a bucket set up
        podpac.settings[self.enabled_setting] = False
        with pytest.raises(CacheException, match="Cache is disabled"):
            _ = self.Store()

    def test_put_has_get(self):
        store = self.Store()

        store.put(NODE1, 10, "mykey1")
        store.put(NODE1, 20, "mykey2")
        store.put(NODE1, 30, "mykeyA", COORDS1)
        store.put(NODE1, 40, "mykeyB", COORDS1)
        store.put(NODE1, 50, "mykeyA", COORDS2)
        store.put(NODE2, 110, "mykey1")
        store.put(NODE2, 120, "mykeyA", COORDS1)

        assert store.has(NODE1, "mykey1") is True
        assert store.has(NODE1, "mykey2") is True
        assert store.has(NODE1, "mykeyA", COORDS1) is True
        assert store.has(NODE1, "mykeyB", COORDS1) is True
        assert store.has(NODE1, "mykeyA", COORDS2) is True
        assert store.has(NODE2, "mykey1") is True
        assert store.has(NODE2, "mykeyA", COORDS1) is True

        assert store.get(NODE1, "mykey1") == 10
        assert store.get(NODE1, "mykey2") == 20
        assert store.get(NODE1, "mykeyA", COORDS1) == 30
        assert store.get(NODE1, "mykeyB", COORDS1) == 40
        assert store.get(NODE1, "mykeyA", COORDS2) == 50
        assert store.get(NODE2, "mykey1") == 110
        assert store.get(NODE2, "mykeyA", COORDS1) == 120

    def test_has_empty(self):
        store = self.Store()
        assert store.has(NODE1, "mykey") is False

    def test_get_empty(self):
        store = self.Store()
        with pytest.raises(CacheException, match="Cache miss"):
            store.get(NODE1, "mykey")

    def test_rem_empty(self):
        store = self.Store()
        store.rem(NODE1, "mykey")

    def test_update(self):
        store = self.Store()

        store.put(NODE1, 10, "mykey1")
        assert store.get(NODE1, "mykey1") == 10

        # raise exception and do not change
        with pytest.raises(CacheException, match="Cache entry already exists."):
            store.put(NODE1, 10, "mykey1", update=False)
        assert store.get(NODE1, "mykey1") == 10

        # update
        store.put(NODE1, 20, "mykey1")
        assert store.get(NODE1, "mykey1") == 20

    def test_get_put_none(self):
        store = self.Store()
        store.put(NODE1, None, "mykey")
        assert store.get(NODE1, "mykey") is None

    def test_rem_object(self):
        store = self.Store()

        store.put(NODE1, 10, "mykey1")
        store.put(NODE1, 20, "mykey2")
        store.put(NODE1, 30, "mykeyA", COORDS1)
        store.put(NODE1, 40, "mykeyB", COORDS1)
        store.put(NODE1, 50, "mykeyA", COORDS2)
        store.put(NODE2, 110, "mykey1")
        store.put(NODE2, 120, "mykeyA", COORDS1)

        store.rem(NODE1, item="mykey1")
        store.rem(NODE1, item="mykeyA", coordinates=COORDS1)

        assert store.has(NODE1, "mykey1") is False
        assert store.has(NODE1, "mykey2") is True
        assert store.has(NODE1, "mykeyA", COORDS1) is False
        assert store.has(NODE1, "mykeyB", COORDS1) is True
        assert store.has(NODE1, "mykeyA", COORDS2) is True
        assert store.has(NODE2, "mykey1") is True
        assert store.has(NODE2, "mykeyA", COORDS1) is True

    def test_rem_key(self):
        store = self.Store()

        store.put(NODE1, 10, "mykey1")
        store.put(NODE1, 20, "mykey2")
        store.put(NODE1, 30, "mykeyA", COORDS1)
        store.put(NODE1, 40, "mykeyB", COORDS1)
        store.put(NODE1, 50, "mykeyA", COORDS2)
        store.put(NODE2, 110, "mykey1")
        store.put(NODE2, 120, "mykeyA", COORDS1)

        store.rem(NODE1, item="mykey1")
        store.rem(NODE1, item="mykeyA")

        assert store.has(NODE1, "mykey1") is False
        assert store.has(NODE1, "mykey2") is True
        assert store.has(NODE1, "mykeyA", COORDS1) is False
        assert store.has(NODE1, "mykeyB", COORDS1) is True
        assert store.has(NODE1, "mykeyA", COORDS2) is False
        assert store.has(NODE2, "mykey1") is True
        assert store.has(NODE2, "mykeyA", COORDS1) is True

    def test_rem_coordinates(self):
        store = self.Store()

        store.put(NODE1, 10, "mykey1")
        store.put(NODE1, 20, "mykey2")
        store.put(NODE1, 30, "mykeyA", COORDS1)
        store.put(NODE1, 40, "mykeyB", COORDS1)
        store.put(NODE1, 50, "mykeyA", COORDS2)
        store.put(NODE2, 110, "mykey1")
        store.put(NODE2, 120, "mykeyA", COORDS1)

        store.rem(NODE1, coordinates=COORDS1)

        assert store.has(NODE1, "mykey1") is True
        assert store.has(NODE1, "mykey2") is True
        assert store.has(NODE1, "mykeyA", COORDS1) is False
        assert store.has(NODE1, "mykeyB", COORDS1) is False
        assert store.has(NODE1, "mykeyA", COORDS2) is True
        assert store.has(NODE2, "mykey1") is True
        assert store.has(NODE2, "mykeyA", COORDS1) is True

    def test_rem_node(self):
        store = self.Store()

        store.put(NODE1, 10, "mykey1")
        store.put(NODE1, 20, "mykey2")
        store.put(NODE1, 30, "mykeyA", COORDS1)
        store.put(NODE1, 40, "mykeyB", COORDS1)
        store.put(NODE1, 50, "mykeyA", COORDS2)
        store.put(NODE2, 110, "mykey1")
        store.put(NODE2, 120, "mykeyA", COORDS1)

        store.rem(NODE1)

        assert store.has(NODE1, "mykey1") is False
        assert store.has(NODE1, "mykey2") is False
        assert store.has(NODE1, "mykeyA", COORDS1) is False
        assert store.has(NODE1, "mykeyB", COORDS1) is False
        assert store.has(NODE1, "mykeyA", COORDS2) is False
        assert store.has(NODE2, "mykey1") is True
        assert store.has(NODE2, "mykeyA", COORDS1) is True

    def test_clear(self):
        store = self.Store()

        store.put(NODE1, 10, "mykey1")
        store.put(NODE1, 20, "mykey2")
        store.put(NODE1, 30, "mykeyA", COORDS1)
        store.put(NODE1, 40, "mykeyB", COORDS1)
        store.put(NODE1, 50, "mykeyA", COORDS2)
        store.put(NODE2, 110, "mykey1")
        store.put(NODE2, 120, "mykeyA", COORDS1)

        store.clear()

        assert store.has(NODE1, "mykey1") is False
        assert store.has(NODE1, "mykey2") is False
        assert store.has(NODE1, "mykeyA", COORDS1) is False
        assert store.has(NODE1, "mykeyB", COORDS1) is False
        assert store.has(NODE1, "mykeyA", COORDS2) is False
        assert store.has(NODE2, "mykey1") is False
        assert store.has(NODE2, "mykeyA", COORDS1) is False

    def test_max_size(self):
        store = self.Store()
        assert store.max_size == podpac.settings[self.limit_setting]

        podpac.settings[self.limit_setting] = 1000
        assert store.max_size == 1000

    def test_limit(self):
        podpac.settings[self.limit_setting] = 10
        store = self.Store()

        store.put(NODE1, "11111111", "mykey1")

        with pytest.warns(UserWarning, match="Warning: .* cache is full"):
            store.put(NODE1, "11111111", "mykey2")

    def test_expiration(self):
        store = self.Store()

        # timestamp
        expires = time.time() + 100  # in 100 seconds
        store.put(NODE1, 10, "mykey1", expires=expires)
        assert store.has(NODE1, "mykey1") is True

        expires = time.time() - 100  # 100 seconds ago
        store.put(NODE1, 10, "mykey2", expires=expires)
        assert store.has(NODE1, "mykey2") is False

        # datetime
        expires = datetime.datetime.now() + datetime.timedelta(1)  # in 1 day
        store.put(NODE1, 10, "mykey3", expires=expires)
        assert store.has(NODE1, "mykey3") is True

        expires = datetime.datetime.now() - datetime.timedelta(1)  # 1 day ago
        store.put(NODE1, 10, "mykey4", expires=expires)
        assert store.has(NODE1, "mykey4") is False

        # timedelta
        expires = datetime.timedelta(1)  # in 1 day
        store.put(NODE1, 10, "mykey5", expires=expires)
        assert store.has(NODE1, "mykey5") is True

        expires = -datetime.timedelta(1)  # 1 day ago
        store.put(NODE1, 10, "mykey6", expires=expires)
        assert store.has(NODE1, "mykey6") is False

        # string datetime
        expires = "3000-01-01"  # in the year 3000
        store.put(NODE1, 10, "mykey7", expires=expires)
        assert store.has(NODE1, "mykey7") is True

        expires = "1000-01-01"  # in the year 1000
        store.put(NODE1, 10, "mykey8", expires=expires)
        assert store.has(NODE1, "mykey8") is False

        # string timedelta
        expires = "1,D"  # in 1 day
        store.put(NODE1, 10, "mykey9", expires=expires)
        assert store.has(NODE1, "mykey9") is True

        expires = "-1,D"  # 1 day ago
        store.put(NODE1, 10, "mykey10", expires=expires)
        assert store.has(NODE1, "mykey10") is False

        # None
        expires = None  # never (default)
        store.put(NODE1, 10, "mykey11", expires=None)
        assert store.has(NODE1, "mykey11") is True

    def test_expiration_put(self):
        store = self.Store()

        # exception putting data that is not expired when update=False
        expires = time.time() + 100  # in 100 seconds
        store.put(NODE1, 10, "mykey1", expires=expires)
        with pytest.raises(CacheException, match="Cache entry already exists"):
            store.put(NODE1, 10, "mykey1", update=False)
        store.put(NODE1, 10, "mykey1")

        # no exception putting data that is expired even when update=False
        expires = time.time() - 100  # 100 seconds ago
        store.put(NODE1, 10, "mykey2", expires=expires)
        store.put(NODE1, 10, "mykey2", update=False)

    def test_expiration_get(self):
        store = self.Store()

        # getting unexpired data
        store.put(NODE1, 10, "mykey1", expires=time.time() + 100)
        assert store.get(NODE1, "mykey1") == 10

        # exception getting expired data
        store.put(NODE1, 10, "mykey2", expires=time.time() - 100)
        with pytest.raises(CacheException, match=CACHE_MISS):
            store.get(NODE1, "mykey2")

    def test_clean_basic(self):
        store = self.Store()
        store.put(NODE1, 10, "mykey1", expires=time.time())
        store.cleanup()


class FileCacheStoreTests(BaseCacheStoreTests):
    def test_cache_units_data_array(self):
        store = self.Store()

        data = podpac.core.units.UnitsDataArray([1, 2, 3], attrs={"units": "m"})
        store.put(NODE1, data, "mykey")
        cached = store.get(NODE1, "mykey")
        assert isinstance(cached, podpac.core.units.UnitsDataArray)
        xr.testing.assert_identical(cached, data)  # assert_identical checks attributes as wel

    def test_cache_xarray(self):
        store = self.Store()

        # data array
        data = xr.DataArray([1, 2, 3])
        store.put(NODE1, data, "mykey")
        cached = store.get(NODE1, "mykey")
        xr.testing.assert_identical(cached, data)  # assert_identical checks attributes as wel

        # dataset
        data = xr.Dataset({"a": [1, 2, 3]})
        store.put(NODE1, data, "mykey2")
        cached = store.get(NODE1, "mykey2")
        xr.testing.assert_identical(cached, data)  # assert_identical checks attributes as wel

    def test_cache_podpac(self):
        store = self.Store()

        # coords
        store.put(NODE1, COORDS1, "mykey")
        cached = store.get(NODE1, "mykey")
        assert cached == COORDS1

        # node
        store.put(NODE1, NODE2, "mykey2")
        cached = store.get(NODE1, "mykey2")
        assert cached.json == NODE2.json

    def test_cache_numpy(self):
        store = self.Store()

        data = np.array([1, 2, 3])
        store.put(NODE1, data, "mykey")
        cached = store.get(NODE1, "mykey")
        np.testing.assert_equal(cached, data)

    def test_pkl_fallback(self):
        store = self.Store()

        data = [xr.DataArray([1, 2, 3]), np.array([1, 2, 3])]
        with pytest.warns(UserWarning, match="caching object to file using pickle"):
            store.put(NODE1, data, "mykey")
        cached = store.get(NODE1, "mykey")
        xr.testing.assert_equal(cached[0], data[0])
        np.testing.assert_equal(cached[1], data[1])


class TestRamCacheStore(BaseCacheStoreTests):
    Store = RamCacheStore
    enabled_setting = "ENABLE_CACHE"
    limit_setting = "RAM_CACHE_MAX_BYTES"

    def setup_method(self):
        super(TestRamCacheStore, self).setup_method()

        from podpac.core.cache.ram_cache_store import _thread_local

        if hasattr(_thread_local, "cache"):
            del _thread_local.cache

    def teardown_method(self):
        super(TestRamCacheStore, self).teardown_method()

        from podpac.core.cache.ram_cache_store import _thread_local

        if hasattr(_thread_local, "cache"):
            del _thread_local.cache

    @pytest.mark.skip(reason="not testable")
    def test_size(self):
        # Prevent from calling into the parent's version of this method
        pass

    @pytest.mark.skip(reason="not testable")
    def test_limit(self):
        # Prevent from calling into the parent's version of this method
        super(TestRamCacheStore, self).test_size()

    def test_cleanup(self):
        from podpac.core.cache.ram_cache_store import _thread_local

        store = self.Store()
        store.put(NODE1, 10, "mykey1", expires=time.time() + 100)
        store.put(NODE1, 10, "mykey2", expires=time.time() - 100)
        assert len(_thread_local.cache) == 2

        store.cleanup()
        assert len(_thread_local.cache) == 1

    def test_has_keeps_expired(self):
        """Test expired data is not removed by the read-only `has` method."""
        from podpac.core.cache.ram_cache_store import _thread_local

        store = self.Store()
        store.put(NODE1, 10, "mykey1", expires=time.time() + 100)
        store.put(NODE1, 10, "mykey2", expires=time.time() - 100)
        assert len(_thread_local.cache) == 2

        assert store.has(NODE1, "mykey1") is True
        assert store.has(NODE1, "mykey2") is False

        assert len(_thread_local.cache) == 2

    def test_get_keeps_expired(self):
        """Test expired data is not removed by the read-only `get` method."""
        from podpac.core.cache.ram_cache_store import _thread_local

        store = self.Store()
        store.put(NODE1, 10, "mykey1", expires=time.time() + 100)
        store.put(NODE1, 10, "mykey2", expires=time.time() - 100)
        assert len(_thread_local.cache) == 2

        assert store.get(NODE1, "mykey1") == 10
        with pytest.raises(CacheException, match=CACHE_MISS):
            store.get(NODE1, "mykey2")

        assert len(_thread_local.cache) == 2


class TestDiskCacheStore(FileCacheStoreTests):
    Store = DiskCacheStore
    enabled_setting = "ENABLE_CACHE"
    limit_setting = "DISK_CACHE_MAX_BYTES"

    def setup_method(self):
        super(TestDiskCacheStore, self).setup_method()

        self.test_cache_dir = tempfile.mkdtemp(prefix="podpac-test-")
        podpac.settings["DISK_CACHE_DIR"] = self.test_cache_dir

    def teardown_method(self):
        super(TestDiskCacheStore, self).teardown_method()

        if hasattr(_thread_local, "locks"):
            for lock in _thread_local.locks.values():
                lock.close()
            del _thread_local.locks

        shutil.rmtree(self.test_cache_dir, ignore_errors=True)
        lock_path = self.get_lock_path(self.test_cache_dir)
        if os.path.exists(lock_path):
            os.remove(lock_path)

    def get_lock_path(self, cache_dir: str) -> str:
        """Get a lock file path based on a cache directory path."""
        return f"{os.path.normpath(cache_dir)}.lock"

    def get_probe(self) -> filelock.ReadWriteLock:
        """Get a separate lock on the store lock file to probe the lock state."""
        return filelock.ReadWriteLock(self.get_lock_path(self.test_cache_dir), is_singleton=False)

    def _assert_read_locked(self, *args: Any, **kwargs: Any) -> Any:
        """Side effect asserting a read lock is held, `DEFAULT` returns the normal value of the function."""
        probe = self.get_probe()
        with probe.read_lock(blocking=False):
            pass
        with pytest.raises(filelock.Timeout):
            probe.acquire_write(blocking=False)
        probe.close()
        return DEFAULT

    def _assert_write_locked(self, *args: Any, **kwargs: Any) -> Any:
        """Side effect asserting a write lock is held, `DEFAULT` returns the normal value of the function."""
        probe = self.get_probe()
        with pytest.raises(filelock.Timeout):
            probe.acquire_read(blocking=False)
        probe.close()
        return DEFAULT

    def _append_lock(self, locks: list):
        """Append this thread's store lock to `locks`."""
        locks.append(self.Store()._rw_lock)

    def _assert_unlocked(self, *args: Any, **kwargs: Any) -> Any:
        """Side effect asserting no lock is held, `DEFAULT` returns the normal value of the function."""
        probe = self.get_probe()
        with probe.write_lock(blocking=False):
            pass
        probe.close()
        return DEFAULT

    def test_cache_dir(self):
        with podpac.settings:
            # absolute path
            podpac.settings["DISK_CACHE_DIR"] = self.test_cache_dir
            expected = self.test_cache_dir
            store = DiskCacheStore()
            store.put(NODE1, 10, "mykey1")
            assert store.find(NODE1, "mykey1").startswith(expected)
            store.clear()

            # relative path
            podpac.settings["DISK_CACHE_DIR"] = "_testcache_"
            expected = os.path.join(
                os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".config", "podpac")),
                "_testcache_",
            )
            store = DiskCacheStore()
            store.clear()
            store.put(NODE1, 10, "mykey1")
            assert store.find(NODE1, "mykey1").startswith(expected)
            store.clear()
            os.remove(self.get_lock_path(expected))

    def test_rem_node_dir(self):
        store = self.Store()

        store.put(NODE1, 10, "mykey1")
        store.put(NODE1, 10, "mykey2")
        store.put(NODE2, 10, "mykey1")

        assert store._exists(store._get_node_dir(NODE1))
        assert store._exists(store._get_node_dir(NODE2))

        store.rem(NODE1, "mykey1")
        assert store._exists(store._get_node_dir(NODE1))
        assert store._exists(store._get_node_dir(NODE2))

        store.rem(NODE1, "mykey2")
        assert not store._exists(store._get_node_dir(NODE1))
        assert store._exists(store._get_node_dir(NODE2))

        store.rem(NODE2)
        assert not store._exists(store._get_node_dir(NODE1))
        assert not store._exists(store._get_node_dir(NODE2))

    def test_size(self):
        store = self.Store()
        assert store.size == 0

        store.put(NODE1, 10, "mykey1")
        store.put(NODE1, np.array([0, 1, 2]), "mykey2")

        p1 = store.find(NODE1, "mykey1", None)
        p2 = store.find(NODE1, "mykey2", None)
        expected_size = (
            os.path.getsize(p1)
            + os.path.getsize("%s.meta" % p1)
            + os.path.getsize(p2)
            + os.path.getsize("%s.meta" % p2)
        )
        assert store.size == expected_size

    def test_cleanup(self):
        store = self.Store()
        store.put(NODE1, 10, "mykey1", expires=time.time() + 100)
        store.put(NODE1, 10, "mykey2", expires=time.time() - 100)
        assert len(store.search(NODE1)) == 2

        store.cleanup()
        assert len(store.search(NODE1)) == 1

        store = self.Store()
        store.put(NODE1, 10, "mykey1", expires=time.time() - 100)
        store.put(NODE1, 10, "mykey2", expires=time.time() - 100)
        assert len(store.search(NODE1)) == 2

        store.cleanup()
        assert len(store.search(NODE1)) == 0
        assert not store._exists(store._get_node_dir(NODE1))  # empty node directories are removed

    def test_has_keeps_expired(self):
        """Test expired data is not removed by the read-only `has` method."""
        store = self.Store()
        store.put(NODE1, 10, "mykey1", expires=time.time() + 100)
        store.put(NODE1, 10, "mykey2", expires=time.time() - 100)
        assert len(store.search(NODE1)) == 2

        assert store.has(NODE1, "mykey1") is True
        assert store.has(NODE1, "mykey2") is False

        assert len(store.search(NODE1)) == 2

    def test_get_keeps_expired(self):
        """Test expired data is not removed by the read-only `get` method."""
        store = self.Store()
        store.put(NODE1, 10, "mykey1", expires=time.time() + 100)
        store.put(NODE1, 10, "mykey2", expires=time.time() - 100)
        assert len(store.search(NODE1)) == 2

        assert store.get(NODE1, "mykey1") == 10
        with pytest.raises(CacheException, match=CACHE_MISS):
            store.get(NODE1, "mykey2")

        assert len(store.search(NODE1)) == 2

    @pytest.mark.parametrize(
        "method, args, patched",
        [
            ("get", (NODE1, "mykey1"), "_load"),
            ("has", (NODE1, "mykey1"), "_find"),
            ("find", (NODE1, "mykey1"), "_search"),
            ("search", (NODE1,), "_search"),
        ],
    )
    def test_public_method_read_locked(self, method: str, args: tuple, patched: str):
        """Test each read method holds the shared lock while calling `patched` and releases it after."""
        store = self.Store()
        store.put(NODE1, 10, "mykey1")
        with patch.object(store, patched, wraps=getattr(store, patched), side_effect=self._assert_read_locked) as mock:
            getattr(store, method)(*args)
        assert mock.called
        self._assert_unlocked()

    @pytest.mark.parametrize(
        "method, args, patched",
        [
            ("put", (NODE1, 10, "mykey1"), "_save"),
            ("rem", (NODE1, "mykey1"), "_remove"),
            ("clear", (), "_rmtree"),
            ("cleanup", (), "_expired"),
        ],
    )
    def test_public_method_write_locked(self, method: str, args: tuple, patched: str):
        """Test each mutating method holds the exclusive lock while calling `patched` and releases it after."""
        store = self.Store()
        store.put(NODE1, 10, "mykey1")
        with patch.object(store, patched, wraps=getattr(store, patched), side_effect=self._assert_write_locked) as mock:
            getattr(store, method)(*args)
        assert mock.called
        self._assert_unlocked()

    def test_expired_removed_by_put_under_write_lock(self):
        """Test an expired entry is removed by `put` method under the write lock."""
        store = self.Store()
        store.put(NODE1, 10, "mykey1", expires=time.time() - 100)
        with patch.object(store, "_remove", wraps=store._remove, side_effect=self._assert_write_locked) as remove:
            store.put(NODE1, 20, "mykey1", update=False)
        assert remove.called
        assert store.get(NODE1, "mykey1") == 20
        self._assert_unlocked()

    def test_get_deserializes_outside_lock(self):
        """Test deserialization happens outside the lock to avoid unneeded blocking."""
        store = self.Store()
        store.put(NODE1, np.array([0, 1, 2]), "mykey1")
        with patch.object(np, "load", wraps=np.load, side_effect=self._assert_unlocked) as load:
            store.get(NODE1, "mykey1")
        assert load.called
        self._assert_unlocked()

    def test_size_read_locked(self):
        """Test size holds the read lock and releases it."""
        store = self.Store()
        with patch.object(os, "walk", wraps=os.walk, side_effect=self._assert_read_locked) as walk:
            _ = store.size
        assert walk.called
        self._assert_unlocked()

    def test_lock_file_persists_cache_clear(self):
        """Test lock file persists when cache is cleared."""
        store = self.Store()

        store.clear()
        assert os.path.exists(self.get_lock_path(self.test_cache_dir))

    def test_lock_shared_between_instances(self):
        """Test store instances share one lock within a thread."""
        assert self.Store()._rw_lock is self.Store()._rw_lock

    def test_lock_per_thread(self):
        """Test each thread gets its a different lock instance."""
        locks = []
        thread = threading.Thread(target=self._append_lock, args=(locks,))
        thread.start()
        thread.join()
        assert locks[0] is not self.Store()._rw_lock

    def test_lock_reentrant(self):
        """Test the lock is re-entrant with nested calls of the same mode."""
        store = self.Store()
        store.put(NODE1, 10, "mykey1")
        with store._read_lock:
            assert store.get(NODE1, "mykey1") == 10
        with store._write_lock:
            store.put(NODE1, 10, "mykey1")

    def test_read_lock_allows_reads(self):
        """Test a read lock held elsewhere does not block reads."""
        store = self.Store()
        store.put(NODE1, 10, "mykey1")
        with self.get_probe().read_lock(blocking=False):
            assert store.get(NODE1, "mykey1") == 10

    def test_read_lock_blocks_writes(self):
        """Test a shared lock held elsewhere blocks writes until it is released."""
        store = self.Store()
        with patch.object(store._rw_lock, "blocking", False):
            with self.get_probe().read_lock(blocking=False):
                with pytest.raises(filelock.Timeout):
                    store.put(NODE1, 10, "mykey1")
            store.put(NODE1, 10, "mykey1")
        assert store.get(NODE1, "mykey1") == 10

    def test_write_lock_blocks_other_lock(self):
        """Test an exclusive lock held by the store blocks other readers."""
        store = self.Store()
        with store._write_lock:
            with pytest.raises(filelock.Timeout):
                self.get_probe().acquire_read(blocking=False)

    def test_find_multiple_raises(self):
        """Test find raises when more than one cached file matches."""
        store = self.Store()
        with patch.object(store, "_search", return_value=["mykey1.json", "mykey1.pkl"]):
            with pytest.raises(RuntimeError, match="Too many cached files"):
                store.find(NODE1, "mykey1")

    def test_put_cleanup_frees_space(self):
        """Test put stores the entry when cleanup frees enough space."""
        store = self.Store()
        store.put(NODE1, 10, "mykey1", expires=time.time() - 100)
        podpac.settings[self.limit_setting] = store.size
        assert store.put(NODE1, 10, "mykey2") is True
        assert store.get(NODE1, "mykey2") == 10
        assert len(store.search(NODE1)) == 1


@pytest.mark.aws
@pytest.mark.skip("Unreachable source")
class TestS3CacheStore(FileCacheStoreTests):
    Store = S3CacheStore
    enabled_setting = "ENABLE_CACHE"
    limit_setting = "S3_CACHE_MAX_BYTES"
    test_cache_dir = "tmp_cache"

    def setup_method(self):
        super(TestS3CacheStore, self).setup_method()

        podpac.settings["S3_CACHE_DIR"] = self.test_cache_dir

    def teardown_method(self):
        try:
            store = S3CacheStore()
            store._rmtree(self.test_cache_dir)
        except (OSError, CacheException):
            pass

        super(TestS3CacheStore, self).teardown_method()

    def test_size(self):
        store = self.Store()
        assert store.size == 0

        store.put(NODE1, 10, "mykey1")
        store.put(NODE1, np.array([0, 1, 2]), "mykey2")
        assert store.size == 142
