
import pytest

import podpac

from podpac.core.cache.utils import CacheException
from podpac.core.cache.ram_cache_store import RamCacheStore
from podpac.core.cache.disk_cache_store import DiskCacheStore
from podpac.core.cache.cache_ctrl import CacheCtrl
from podpac.core.cache.cache_ctrl import get_default_cache_ctrl, make_cache_ctrl, clear_cache

class TestCacheCtrl(object):
    def test_init(self):
    	pass

def test_get_default_cache_ctrl():
	ctrl = get_default_cache_ctrl()

	assert isinstance(ctrl, CacheCtrl)
	assert ctrl._cache_stores == []

	podpac.settings['DEFAULT_CACHE'] = ['ram']
	ctrl = get_default_cache_ctrl()
	assert isinstance(ctrl, CacheCtrl)
	assert len(ctrl._cache_stores) == 1
	assert isinstance(ctrl._cache_stores[0], RamCacheStore)
	
	podpac.settings['DEFAULT_CACHE'] = []

def test_make_cache_ctrl():
	ctrl = make_cache_ctrl('ram')
	assert isinstance(ctrl, CacheCtrl)
	assert len(ctrl._cache_stores) == 1
	assert isinstance(ctrl._cache_stores[0], RamCacheStore)

	ctrl = make_cache_ctrl('disk')
	assert len(ctrl._cache_stores) == 1
	assert isinstance(ctrl._cache_stores[0], DiskCacheStore)

	ctrl = make_cache_ctrl(['ram', 'disk'])
	assert len(ctrl._cache_stores) == 2
	assert isinstance(ctrl._cache_stores[0], RamCacheStore)
	assert isinstance(ctrl._cache_stores[1], DiskCacheStore)
	
	with pytest.raises(ValueError, match="Unknown cache store type"):
		ctrl = make_cache_ctrl('other')

def test_clear_cache():
	# make a default cache
	podpac.settings['DEFAULT_CACHE'] = ['ram']
	
	# fill the default cache
	node = podpac.algorithm.Arange()
	node.put_cache(0, 'mykey')
	assert node.has_cache('mykey')

	clear_cache()

	assert not node.has_cache('mykey')
	
	# reset default cache
	podpac.settings['DEFAULT_CACHE'] = []