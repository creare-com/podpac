
import numpy as np
import os
import shutil
import copy

import pytest
import xarray as xr

import podpac

from podpac.core.cache.cache import DiskCacheStore
from podpac.core.cache.cache import CacheException

COORDS1 = podpac.Coordinates([[0, 1, 2], [10, 20, 30, 40], ['2018-01-01', '2018-01-02']], dims=['lat','lon','time'])
COORDS2 = podpac.Coordinates([[0, 1, 2], [10, 20, 30], ['2018-01-01', '2018-01-02']], dims=['lat','lon','time'])
NODE1 = podpac.data.Array(source=np.ones(COORDS1.shape), source_coordinates=COORDS1)
NODE2 = podpac.algorithm.Arange()

class TestDiskCacheStore(object):
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tmp_cache'))

    def setup_method(self):
        self.settings_orig = copy.deepcopy(podpac.settings)

        podpac.settings['DISK_CACHE_DIR'] = self.cache_dir
        assert not os.path.exists(self.cache_dir)

    def teardown_method(self):
        for key in podpac.settings:
            podpac.settings[key] = self.settings_orig[key]
        
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_init(self):
        store = DiskCacheStore()
        
    def test_disabled(self):
        podpac.settings['DISK_CACHE_ENABLED'] = False
        with pytest.raises(CacheException, match="Disk cache is disabled"):
            store = DiskCacheStore()

    def test_cache_dir(self):
        # absolute path
        store = DiskCacheStore()
        assert store.root_dir_path == self.cache_dir

        # relative path
        podpac.settings['DISK_CACHE_DIR'] = 'test'
        store = DiskCacheStore()
        assert store.root_dir_path == os.path.join(podpac.settings['ROOT_PATH'], 'test')

    def test_empty(self):
        store = DiskCacheStore()

        assert store.has(NODE1, 'mykey') is False
        assert store.has(NODE1, 'mykey', COORDS1) is False

        with pytest.raises(CacheException, match="Cache miss"):
            store.get(NODE1, 'mykey')
        with pytest.raises(CacheException, match="Cache miss"):
            store.get(NODE1, 'mykey', COORDS1)

        store.rem(NODE1, 'mykey')
        store.rem(NODE1, 'mykey', COORDS1)

    def test_cache(self):
        store = DiskCacheStore()

        store.put(NODE1, 10, 'mykey1')
        store.put(NODE1, 20, 'mykey2')
        store.put(NODE1, 30, 'mykeyA', COORDS1)
        store.put(NODE1, 40, 'mykeyB', COORDS1)
        store.put(NODE1, 50, 'mykeyA', COORDS2)
        store.put(NODE2, 110, 'mykey1')
        store.put(NODE2, 120, 'mykeyA', COORDS1)

        assert store.has(NODE1, 'mykey1')
        assert store.has(NODE1, 'mykey2')
        assert store.has(NODE1, 'mykeyA', COORDS1)
        assert store.has(NODE1, 'mykeyB', COORDS1)
        assert store.has(NODE1, 'mykeyA', COORDS2)
        assert store.has(NODE2, 'mykey1')
        assert store.has(NODE2, 'mykeyA', COORDS1)

        assert store.get(NODE1, 'mykey1') == 10
        assert store.get(NODE1, 'mykey2') == 20
        assert store.get(NODE1, 'mykeyA', COORDS1) == 30
        assert store.get(NODE1, 'mykeyB', COORDS1) == 40
        assert store.get(NODE1, 'mykeyA', COORDS2) == 50
        assert store.get(NODE2, 'mykey1') == 110
        assert store.get(NODE2, 'mykeyA', COORDS1) == 120

    def test_update(self):
        store = DiskCacheStore()

        store.put(NODE1, 10, 'mykey1')
        assert store.get(NODE1, 'mykey1') == 10

        # raise exception and do not change
        with pytest.raises(CacheException, match='Existing cache entry'):
            store.put(NODE1, 10, 'mykey1')
        assert store.get(NODE1, 'mykey1') == 10

        # update
        store.put(NODE1, 20, 'mykey1', update=True)
        assert store.get(NODE1, 'mykey1') == 20

    def test_rem_object(self):
        store = DiskCacheStore()

        store.put(NODE1, 10, 'mykey1')
        store.put(NODE1, 20, 'mykey2')
        store.put(NODE1, 30, 'mykeyA', COORDS1)
        store.put(NODE1, 40, 'mykeyB', COORDS1)
        store.put(NODE1, 50, 'mykeyA', COORDS2)
        store.put(NODE2, 110, 'mykey1')
        store.put(NODE2, 120, 'mykeyA', COORDS1)

        store.rem(NODE1, key='mykey1')
        store.rem(NODE1, key='mykeyA', coordinates=COORDS1)
        
        store.has(NODE1, 'mykey1') is False
        store.has(NODE1, 'mykey2') is True
        store.has(NODE1, 'mykeyA', COORDS1) is False
        store.has(NODE1, 'mykeyB', COORDS1) is True
        store.has(NODE1, 'mykeyA', COORDS2) is True
        store.has(NODE2, 'mykey1') is True
        store.has(NODE2, 'mykeyA', COORDS1) is True

    def test_rem_key(self):
        store = DiskCacheStore()

        store.put(NODE1, 10, 'mykey1')
        store.put(NODE1, 20, 'mykey2')
        store.put(NODE1, 30, 'mykeyA', COORDS1)
        store.put(NODE1, 40, 'mykeyB', COORDS1)
        store.put(NODE1, 50, 'mykeyA', COORDS2)
        store.put(NODE2, 110, 'mykey1')
        store.put(NODE2, 120, 'mykeyA', COORDS1)

        store.rem(NODE1, key='mykey1')
        store.rem(NODE1, key='mykeyA')
        
        store.has(NODE1, 'mykey1') is False
        store.has(NODE1, 'mykey2') is True
        store.has(NODE1, 'mykeyA', COORDS1) is False
        store.has(NODE1, 'mykeyB', COORDS1) is True
        store.has(NODE1, 'mykeyA', COORDS2) is False
        store.has(NODE2, 'mykey1') is True
        store.has(NODE2, 'mykeyA', COORDS1) is True

    def test_rem_coordinates(self):
        store = DiskCacheStore()

        store.put(NODE1, 10, 'mykey1')
        store.put(NODE1, 20, 'mykey2')
        store.put(NODE1, 30, 'mykeyA', COORDS1)
        store.put(NODE1, 40, 'mykeyB', COORDS1)
        store.put(NODE1, 50, 'mykeyA', COORDS2)
        store.put(NODE2, 110, 'mykey1')
        store.put(NODE2, 120, 'mykeyA', COORDS1)

        store.rem(NODE1, coordinates=COORDS1)
        
        store.has(NODE1, 'mykey1') is True
        store.has(NODE1, 'mykey2') is True
        store.has(NODE1, 'mykeyA', COORDS1) is False
        store.has(NODE1, 'mykeyB', COORDS1) is False
        store.has(NODE1, 'mykeyA', COORDS2) is True
        store.has(NODE2, 'mykey1') is True
        store.has(NODE2, 'mykeyA', COORDS1) is True

    def test_rem_node(self):
        store = DiskCacheStore()

        store.put(NODE1, 10, 'mykey1')
        store.put(NODE1, 20, 'mykey2')
        store.put(NODE1, 30, 'mykeyA', COORDS1)
        store.put(NODE1, 40, 'mykeyB', COORDS1)
        store.put(NODE1, 50, 'mykeyA', COORDS2)
        store.put(NODE2, 110, 'mykey1')
        store.put(NODE2, 120, 'mykeyA', COORDS1)

        store.rem(NODE1)
        
        store.has(NODE1, 'mykey1') is False
        store.has(NODE1, 'mykey2') is False
        store.has(NODE1, 'mykeyA', COORDS1) is False
        store.has(NODE1, 'mykeyB', COORDS1) is False
        store.has(NODE1, 'mykeyA', COORDS2) is False
        store.has(NODE2, 'mykey1') is True
        store.has(NODE2, 'mykeyA', COORDS1) is True

    def test_rem_all(self):
        store = DiskCacheStore()

        store.put(NODE1, 10, 'mykey1')
        store.put(NODE1, 20, 'mykey2')
        store.put(NODE1, 30, 'mykeyA', COORDS1)
        store.put(NODE1, 40, 'mykeyB', COORDS1)
        store.put(NODE1, 50, 'mykeyA', COORDS2)
        store.put(NODE2, 110, 'mykey1')
        store.put(NODE2, 120, 'mykeyA', COORDS1)

        store.rem()
        
        store.has(NODE1, 'mykey1') is False
        store.has(NODE1, 'mykey2') is False
        store.has(NODE1, 'mykeyA', COORDS1) is False
        store.has(NODE1, 'mykeyB', COORDS1) is False
        store.has(NODE1, 'mykeyA', COORDS2) is False
        store.has(NODE2, 'mykey1') is False
        store.has(NODE2, 'mykeyA', COORDS1) is False

    def test_cache_units_data_array(self):
        store = DiskCacheStore()

        data = podpac.core.units.UnitsDataArray([1, 2, 3], attrs={'units': 'm'})
        store.put(NODE1, data, 'mykey')
        cached = store.get(NODE1, 'mykey')
        assert isinstance(cached, podpac.core.units.UnitsDataArray)
        xr.testing.assert_identical(cached, data) # assert_identical checks attributes as wel

    def test_cache_xarray(self):
        store = DiskCacheStore()

        # data array
        data = xr.DataArray([1, 2, 3])
        store.put(NODE1, data, 'mykey')
        cached = store.get(NODE1, 'mykey')
        xr.testing.assert_identical(cached, data) # assert_identical checks attributes as wel

        # dataset
        data = xr.Dataset({'a': [1, 2, 3]})
        store.put(NODE1, data, 'mykey2')
        cached = store.get(NODE1, 'mykey2')
        xr.testing.assert_identical(cached, data) # assert_identical checks attributes as wel

    def test_cache_podpac(self):
        store = DiskCacheStore()

        # coords
        store.put(NODE1, COORDS1, 'mykey')
        cached = store.get(NODE1, 'mykey')
        assert cached == COORDS1

        # node
        store.put(NODE1, NODE2, 'mykey2')
        cached = store.get(NODE1, 'mykey2')
        assert cached.json == NODE2.json

    def test_cache_numpy(self):
        store = DiskCacheStore()

        data = np.array([1, 2, 3])
        store.put(NODE1, data, 'mykey')
        cached = store.get(NODE1, 'mykey')
        np.testing.assert_equal(cached, data)

    def test_pkl_fallback(self):
        store = DiskCacheStore()

        data = [xr.DataArray([1, 2, 3]), np.array([1, 2, 3])]
        with pytest.warns(UserWarning, match="caching object to file using pickle"):
            store.put(NODE1, data, 'mykey')
        cached = store.get(NODE1, 'mykey')
        xr.testing.assert_equal(cached[0], data[0])
        np.testing.assert_equal(cached[1], data[1])

    def test_size(self):
        store = DiskCacheStore()
        assert store.size == 0

        store.put(NODE1, 10, 'mykey1')
        store.put(NODE1, np.array([0, 1, 2]), 'mykey2')

        expected_size = os.path.getsize(store.find(NODE1, 'mykey1')) + os.path.getsize(store.find(NODE1, 'mykey2'))
        assert store.size == expected_size

    def test_max_size(self):
        store = DiskCacheStore()
        assert store.max_size == podpac.settings['DISK_CACHE_MAX_BYTES']

        podpac.settings['DISK_CACHE_MAX_BYTES'] = 1000
        assert store.max_size == 1000

    def test_limit(self):
        podpac.settings['DISK_CACHE_MAX_BYTES'] = 10
        store = DiskCacheStore()

        store.put(NODE1, '11111111', 'mykey1')
        
        with pytest.warns(UserWarning, match="Warning: disk cache is full"):
            store.put(NODE1, '11111111', 'mykey2')