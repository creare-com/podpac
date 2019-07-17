import numpy as np
import os
import shutil
import copy

import pytest
import xarray as xr

import podpac

from podpac.core.cache.utils import CacheException
from podpac.core.cache.ram_cache_store import RamCacheStore
from podpac.core.cache.disk_cache_store import DiskCacheStore
from podpac.core.cache.s3_cache_store import S3CacheStore

COORDS1 = podpac.Coordinates([[0, 1, 2], [10, 20, 30, 40], ["2018-01-01", "2018-01-02"]], dims=["lat", "lon", "time"])
COORDS2 = podpac.Coordinates([[0, 1, 2], [10, 20, 30]], dims=["lat", "lon"])
NODE1 = podpac.data.Array(source=np.ones(COORDS1.shape), source_coordinates=COORDS1)
NODE2 = podpac.algorithm.Arange()


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
        store = self.Store()

    def test_disabled(self):
        podpac.settings[self.enabled_setting] = False
        with pytest.raises(CacheException, match="cache is disabled"):
            store = self.Store()

    def test_put_has_get(self):
        store = self.Store()

        store.put(NODE1, 10, "mykey1")
        store.put(NODE1, 20, "mykey2")
        store.put(NODE1, 30, "mykeyA", COORDS1)
        store.put(NODE1, 40, "mykeyB", COORDS1)
        store.put(NODE1, 50, "mykeyA", COORDS2)
        store.put(NODE2, 110, "mykey1")
        store.put(NODE2, 120, "mykeyA", COORDS1)

        assert store.has(NODE1, "mykey1")
        assert store.has(NODE1, "mykey2")
        assert store.has(NODE1, "mykeyA", COORDS1)
        assert store.has(NODE1, "mykeyB", COORDS1)
        assert store.has(NODE1, "mykeyA", COORDS2)
        assert store.has(NODE2, "mykey1")
        assert store.has(NODE2, "mykeyA", COORDS1)

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
            store.put(NODE1, 10, "mykey1")
        assert store.get(NODE1, "mykey1") == 10

        # update
        store.put(NODE1, 20, "mykey1", update=True)
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

        store.rem(NODE1, key="mykey1")
        store.rem(NODE1, key="mykeyA", coordinates=COORDS1)

        store.has(NODE1, "mykey1") is False
        store.has(NODE1, "mykey2") is True
        store.has(NODE1, "mykeyA", COORDS1) is False
        store.has(NODE1, "mykeyB", COORDS1) is True
        store.has(NODE1, "mykeyA", COORDS2) is True
        store.has(NODE2, "mykey1") is True
        store.has(NODE2, "mykeyA", COORDS1) is True

    def test_rem_key(self):
        store = self.Store()

        store.put(NODE1, 10, "mykey1")
        store.put(NODE1, 20, "mykey2")
        store.put(NODE1, 30, "mykeyA", COORDS1)
        store.put(NODE1, 40, "mykeyB", COORDS1)
        store.put(NODE1, 50, "mykeyA", COORDS2)
        store.put(NODE2, 110, "mykey1")
        store.put(NODE2, 120, "mykeyA", COORDS1)

        store.rem(NODE1, key="mykey1")
        store.rem(NODE1, key="mykeyA")

        store.has(NODE1, "mykey1") is False
        store.has(NODE1, "mykey2") is True
        store.has(NODE1, "mykeyA", COORDS1) is False
        store.has(NODE1, "mykeyB", COORDS1) is True
        store.has(NODE1, "mykeyA", COORDS2) is False
        store.has(NODE2, "mykey1") is True
        store.has(NODE2, "mykeyA", COORDS1) is True

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

        store.has(NODE1, "mykey1") is True
        store.has(NODE1, "mykey2") is True
        store.has(NODE1, "mykeyA", COORDS1) is False
        store.has(NODE1, "mykeyB", COORDS1) is False
        store.has(NODE1, "mykeyA", COORDS2) is True
        store.has(NODE2, "mykey1") is True
        store.has(NODE2, "mykeyA", COORDS1) is True

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

        store.has(NODE1, "mykey1") is False
        store.has(NODE1, "mykey2") is False
        store.has(NODE1, "mykeyA", COORDS1) is False
        store.has(NODE1, "mykeyB", COORDS1) is False
        store.has(NODE1, "mykeyA", COORDS2) is False
        store.has(NODE2, "mykey1") is True
        store.has(NODE2, "mykeyA", COORDS1) is True

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

        store.has(NODE1, "mykey1") is False
        store.has(NODE1, "mykey2") is False
        store.has(NODE1, "mykeyA", COORDS1) is False
        store.has(NODE1, "mykeyB", COORDS1) is False
        store.has(NODE1, "mykeyA", COORDS2) is False
        store.has(NODE2, "mykey1") is False
        store.has(NODE2, "mykeyA", COORDS1) is False

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
    enabled_setting = "RAM_CACHE_ENABLED"
    limit_setting = "RAM_CACHE_MAX_BYTES"

    def setup_method(self):
        super(TestRamCacheStore, self).setup_method()

        from podpac.core.cache.ram_cache_store import _thread_local

        if hasattr(_thread_local, "cache"):
            delattr(_thread_local, "cache")

    def teardown_method(self):
        super(TestRamCacheStore, self).teardown_method()

        from podpac.core.cache.ram_cache_store import _thread_local

        if hasattr(_thread_local, "cache"):
            delattr(_thread_local, "cache")

    @pytest.mark.skip(reason="not testable")
    def test_size(self):
        pass

    @pytest.mark.skip(reason="not testable")
    def test_limit(self):
        super(TestRamCacheStore, self).test_size()


class TestDiskCacheStore(FileCacheStoreTests):
    Store = DiskCacheStore
    enabled_setting = "DISK_CACHE_ENABLED"
    limit_setting = "DISK_CACHE_MAX_BYTES"
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "tmp_cache"))

    def setup_method(self):
        super(TestDiskCacheStore, self).setup_method()

        podpac.settings["DISK_CACHE_DIR"] = self.cache_dir
        assert not os.path.exists(self.cache_dir)

    def teardown_method(self):
        super(TestDiskCacheStore, self).teardown_method()

        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_cache_dir(self):
        # absolute path
        podpac.settings["DISK_CACHE_DIR"] = self.cache_dir
        expected = self.cache_dir
        store = DiskCacheStore()
        store.put(NODE1, 10, "mykey1")
        assert store.find(NODE1, "mykey1").startswith(expected)
        store.clear()

        # relative path
        podpac.settings["DISK_CACHE_DIR"] = "_testcache_"
        expected = os.path.join(podpac.core.settings.DEFAULT_SETTINGS["ROOT_PATH"], "_testcache_")
        store = DiskCacheStore()
        store.clear()
        store.put(NODE1, 10, "mykey1")
        assert store.find(NODE1, "mykey1").startswith(expected)
        store.clear()

    def test_size(self):
        store = self.Store()
        assert store.size == 0

        store.put(NODE1, 10, "mykey1")
        store.put(NODE1, np.array([0, 1, 2]), "mykey2")

        p1 = store.find(NODE1, "mykey1", None)
        p2 = store.find(NODE1, "mykey2", None)
        expected_size = os.path.getsize(p1) + os.path.getsize(p2)
        assert store.size == expected_size


@pytest.mark.aws
class TestS3CacheStore(FileCacheStoreTests):
    Store = S3CacheStore
    enabled_setting = "S3_CACHE_ENABLED"
    limit_setting = "S3_CACHE_MAX_BYTES"
    cache_dir = "tmp_cache"

    def setup_method(self):
        super(TestS3CacheStore, self).setup_method()

        podpac.settings["S3_CACHE_DIR"] = self.cache_dir

    def teardown_method(self):
        try:
            store = S3CacheStore()
            store._rmtree(self.cache_dir)
        except:
            pass

        super(TestS3CacheStore, self).teardown_method()

    def test_size(self):
        store = self.Store()
        assert store.size == 0

        store.put(NODE1, 10, "mykey1")
        store.put(NODE1, np.array([0, 1, 2]), "mykey2")
        assert store.size == 154
