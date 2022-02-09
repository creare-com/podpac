import copy
import tempfile
import shutil

import pytest

import podpac
from podpac.core.cache.utils import CacheException
from podpac.core.cache.ram_cache_store import RamCacheStore
from podpac.core.cache.disk_cache_store import DiskCacheStore
from podpac.core.cache.cache_ctrl import CacheCtrl
from podpac.core.cache.cache_ctrl import get_default_cache_ctrl, make_cache_ctrl, clear_cache, cache_cleanup


class CacheCtrlTestNode(podpac.Node):
    pass


NODE = CacheCtrlTestNode()


class TestCacheCtrl(object):
    def setup_method(self):
        # store the current settings
        self.settings_orig = copy.deepcopy(podpac.settings)

        # delete the ram cache
        from podpac.core.cache.ram_cache_store import _thread_local

        if hasattr(_thread_local, "cache"):
            delattr(_thread_local, "cache")

        # use an fresh temporary disk cache
        self.test_cache_dir = tempfile.mkdtemp(prefix="podpac-test-")
        podpac.settings["DISK_CACHE_DIR"] = self.test_cache_dir

    def teardown_method(self):
        # delete the ram cache
        from podpac.core.cache.ram_cache_store import _thread_local

        if hasattr(_thread_local, "cache"):
            delattr(_thread_local, "cache")

        # delete the disk cache
        shutil.rmtree(self.test_cache_dir, ignore_errors=True)

        # reset the settings
        for key in podpac.settings:
            podpac.settings[key] = self.settings_orig[key]

    def test_init_default(self):
        ctrl = CacheCtrl()
        assert len(ctrl._cache_stores) == 0
        assert ctrl.cache_stores == []
        repr(ctrl)

    def test_init_list(self):
        ctrl = CacheCtrl(cache_stores=[])
        assert len(ctrl._cache_stores) == 0
        assert ctrl.cache_stores == []
        repr(ctrl)

        ctrl = CacheCtrl(cache_stores=[RamCacheStore()])
        assert len(ctrl._cache_stores) == 1
        assert isinstance(ctrl._cache_stores[0], RamCacheStore)
        assert ctrl.cache_stores == ["ram"]
        repr(ctrl)

        ctrl = CacheCtrl(cache_stores=[RamCacheStore(), DiskCacheStore()])
        assert len(ctrl._cache_stores) == 2
        assert isinstance(ctrl._cache_stores[0], RamCacheStore)
        assert isinstance(ctrl._cache_stores[1], DiskCacheStore)
        assert ctrl.cache_stores == ["ram", "disk"]
        repr(ctrl)

    def test_put_has_get(self):
        ctrl = CacheCtrl(cache_stores=[RamCacheStore(), DiskCacheStore()])

        # has False
        assert not ctrl._cache_stores[0].has(NODE, "key")
        assert not ctrl._cache_stores[1].has(NODE, "key")
        assert not ctrl.has(NODE, "key")

        # put
        ctrl.put(NODE, 10, "key")

        # has True
        assert ctrl._cache_stores[0].has(NODE, "key")
        assert ctrl._cache_stores[1].has(NODE, "key")
        assert ctrl.has(NODE, "key")

        # get value
        assert ctrl._cache_stores[0].get(NODE, "key") == 10
        assert ctrl._cache_stores[1].get(NODE, "key") == 10
        assert ctrl.get(NODE, "key") == 10

    def test_partial_has_get(self):
        ctrl = CacheCtrl(cache_stores=[RamCacheStore(), DiskCacheStore()])

        # has False
        assert not ctrl._cache_stores[0].has(NODE, "key")
        assert not ctrl._cache_stores[1].has(NODE, "key")
        assert not ctrl.has(NODE, "key")

        # put only in disk
        ctrl._cache_stores[1].put(NODE, 10, "key")

        # has
        assert not ctrl._cache_stores[0].has(NODE, "key")
        assert ctrl._cache_stores[1].has(NODE, "key")
        assert ctrl.has(NODE, "key")

        # get
        with pytest.raises(CacheException, match="Cache miss"):
            ctrl._cache_stores[0].get(NODE, "key")
        assert ctrl._cache_stores[1].get(NODE, "key") == 10
        assert ctrl.get(NODE, "key") == 10

    def test_get_cache_miss(self):
        ctrl = CacheCtrl(cache_stores=[RamCacheStore(), DiskCacheStore()])

        with pytest.raises(CacheException, match="Requested data is not in any cache stores"):
            ctrl.get(NODE, "key")

    def test_put_rem(self):
        ctrl = CacheCtrl(cache_stores=[RamCacheStore(), DiskCacheStore()])

        # put and check has
        ctrl.put(NODE, 10, "key")
        assert ctrl.has(NODE, "key")

        # rem other and check has
        ctrl.rem(NODE, "other")
        assert ctrl.has(NODE, "key")

        # rem and check has
        ctrl.rem(NODE, "key")
        assert not ctrl.has(NODE, "key")

    def test_rem_wildcard_key(self):
        ctrl = CacheCtrl(cache_stores=[RamCacheStore(), DiskCacheStore()])

        # put and check has
        ctrl.put(NODE, 10, "key")
        assert ctrl.has(NODE, "key")

        # rem other and check has
        ctrl.rem(NODE, item="*")
        assert not ctrl.has(NODE, "key")

    def test_rem_wildcard_coordinates(self):
        pass

    def test_put_clear(self):
        ctrl = CacheCtrl(cache_stores=[RamCacheStore(), DiskCacheStore()])

        # put and check has
        ctrl.put(NODE, 10, "key")
        assert ctrl.has(NODE, "key")

        # clear and check has
        ctrl.clear()

        # check has
        assert not ctrl.has(NODE, "key")

    def test_put_has_mode(self):
        ctrl = CacheCtrl(cache_stores=[RamCacheStore(), DiskCacheStore()])

        # put disk and check has
        assert not ctrl.has(NODE, "key")

        ctrl.put(NODE, 10, "key", mode="disk")
        assert not ctrl._cache_stores[0].has(NODE, "key")
        assert not ctrl.has(NODE, "key", mode="ram")
        assert ctrl._cache_stores[1].has(NODE, "key")
        assert ctrl.has(NODE, "key", mode="disk")
        assert ctrl.has(NODE, "key")

        # put ram and check has
        ctrl.clear()
        assert not ctrl.has(NODE, "key")

        ctrl.put(NODE, 10, "key", mode="ram")
        assert ctrl._cache_stores[0].has(NODE, "key")
        assert ctrl.has(NODE, "key", mode="ram")
        assert not ctrl._cache_stores[1].has(NODE, "key")
        assert not ctrl.has(NODE, "key", mode="disk")
        assert ctrl.has(NODE, "key")

    def test_put_has_expires(self):
        ctrl = CacheCtrl(cache_stores=[RamCacheStore(), DiskCacheStore()])

        ctrl.put(NODE, 10, "key1", expires="1,D")
        ctrl.put(NODE, 10, "key2", expires="-1,D")
        assert ctrl.has(NODE, "key1")
        assert not ctrl.has(NODE, "key2")

    def test_put_get_expires(self):
        ctrl = CacheCtrl(cache_stores=[RamCacheStore(), DiskCacheStore()])

        ctrl.put(NODE, 10, "key1", expires="1,D")
        ctrl.put(NODE, 10, "key2", expires="-1,D")
        assert ctrl.get(NODE, "key1") == 10
        with pytest.raises(CacheException, match="Requested data is not in any cache stores"):
            ctrl.get(NODE, "key2")

    def test_cleanup(self):
        from podpac.core.cache.ram_cache_store import _thread_local

        ctrl = CacheCtrl(cache_stores=[RamCacheStore(), DiskCacheStore()])

        ctrl.put(NODE, 10, "key1", expires="1,D")
        ctrl.put(NODE, 10, "key2", expires="-1,D")

        # 2 entries (even though one is expired)
        assert len(_thread_local.cache) == 2
        assert len(ctrl._cache_stores[1].search(NODE)) == 2

        ctrl.cleanup()

        # only 1 entry (the expired entry has been removed)
        assert len(_thread_local.cache) == 1
        assert len(ctrl._cache_stores[1].search(NODE)) == 1

    def test_invalid_node(self):
        ctrl = CacheCtrl(cache_stores=[RamCacheStore(), DiskCacheStore()])

        # type
        with pytest.raises(TypeError, match="Invalid node"):
            ctrl.put("node", 10, "key")

        with pytest.raises(TypeError, match="Invalid node"):
            ctrl.get("node", "key")

        with pytest.raises(TypeError, match="Invalid node"):
            ctrl.has("node", "key")

        with pytest.raises(TypeError, match="Invalid node"):
            ctrl.rem("node", "key")

    def test_invalid_key(self):
        ctrl = CacheCtrl(cache_stores=[RamCacheStore(), DiskCacheStore()])

        # type
        with pytest.raises(TypeError, match="Invalid item"):
            ctrl.put(NODE, 10, 10)

        with pytest.raises(TypeError, match="Invalid item"):
            ctrl.get(NODE, 10)

        with pytest.raises(TypeError, match="Invalid item"):
            ctrl.has(NODE, 10)

        with pytest.raises(TypeError, match="Invalid item"):
            ctrl.rem(NODE, 10)

        # wildcard
        with pytest.raises(ValueError, match="Invalid item"):
            ctrl.put(NODE, 10, "*")

        with pytest.raises(ValueError, match="Invalid item"):
            ctrl.get(NODE, "*")

        with pytest.raises(ValueError, match="Invalid item"):
            ctrl.has(NODE, "*")

        # allowed
        ctrl.rem(NODE, "*")

    def test_invalid_coordinates(self):
        ctrl = CacheCtrl(cache_stores=[RamCacheStore(), DiskCacheStore()])

        # type
        with pytest.raises(TypeError, match="Invalid coordinates"):
            ctrl.put(NODE, 10, "key", coordinates="coords")

        with pytest.raises(TypeError, match="Invalid coordinates"):
            ctrl.get(NODE, "key", coordinates="coords")

        with pytest.raises(TypeError, match="Invalid coordinates"):
            ctrl.has(NODE, "key", coordinates="coords")

        with pytest.raises(TypeError, match="Invalid coordinates"):
            ctrl.rem(NODE, "key", coordinates="coords")

    def test_invalid_mode(self):
        ctrl = CacheCtrl(cache_stores=[RamCacheStore(), DiskCacheStore()])

        with pytest.raises(ValueError, match="Invalid mode"):
            ctrl.put(NODE, 10, "key", mode="other")

        with pytest.raises(ValueError, match="Invalid mode"):
            ctrl.get(NODE, "key", mode="other")

        with pytest.raises(ValueError, match="Invalid mode"):
            ctrl.has(NODE, "key", mode="other")

        with pytest.raises(ValueError, match="Invalid mode"):
            ctrl.rem(NODE, "key", mode="other")

        with pytest.raises(ValueError, match="Invalid mode"):
            ctrl.clear(mode="other")


def test_get_default_cache_ctrl():
    with podpac.settings:
        podpac.settings["DEFAULT_CACHE"] = []
        ctrl = get_default_cache_ctrl()
        assert isinstance(ctrl, CacheCtrl)
        assert ctrl._cache_stores == []

        podpac.settings["DEFAULT_CACHE"] = ["ram"]
        ctrl = get_default_cache_ctrl()
        assert isinstance(ctrl, CacheCtrl)
        assert len(ctrl._cache_stores) == 1
        assert isinstance(ctrl._cache_stores[0], RamCacheStore)


class TestMakeCacheCtrl(object):
    def test_str(self):
        ctrl = make_cache_ctrl("ram")
        assert isinstance(ctrl, CacheCtrl)
        assert len(ctrl._cache_stores) == 1
        assert isinstance(ctrl._cache_stores[0], RamCacheStore)

        ctrl = make_cache_ctrl("disk")
        assert len(ctrl._cache_stores) == 1
        assert isinstance(ctrl._cache_stores[0], DiskCacheStore)

    def test_list(self):
        ctrl = make_cache_ctrl(["ram", "disk"])
        assert len(ctrl._cache_stores) == 2
        assert isinstance(ctrl._cache_stores[0], RamCacheStore)
        assert isinstance(ctrl._cache_stores[1], DiskCacheStore)

        ctrl = make_cache_ctrl(["ram", "disk"])
        assert len(ctrl._cache_stores) == 2
        assert isinstance(ctrl._cache_stores[0], RamCacheStore)
        assert isinstance(ctrl._cache_stores[1], DiskCacheStore)

    def test_invalid(self):
        with pytest.raises(ValueError, match="Unknown cache store type"):
            ctrl = make_cache_ctrl("other")

        with pytest.raises(ValueError, match="Unknown cache store type"):
            ctrl = make_cache_ctrl(["other"])


def test_clear_cache():
    with podpac.settings:
        # make a default cache
        podpac.settings["DEFAULT_CACHE"] = ["ram"]

        # fill the default cache
        node = podpac.algorithm.Arange()
        node.put_cache(0, "mykey")
        assert node.has_cache("mykey")

        clear_cache()

        assert not node.has_cache("mykey")


def test_cache_cleanup():
    with podpac.settings:
        # make a default cache
        podpac.settings["DEFAULT_CACHE"] = ["ram"]

        cache_cleanup()
