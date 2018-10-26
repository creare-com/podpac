import pytest
import numpy as np
import os

from numpy.testing import assert_equal

from podpac.core.cache.cache import CacheException
from podpac.core.cache.cache import CacheCtrl
from podpac.core.cache.cache import CacheStore
from podpac.core.cache.cache import DiskCacheStore

from podpac.core.data.types import Array
from podpac.core.coordinates.coordinates import Coordinates

root_disk_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tmp_cache'))

def make_cache_ctrl():
    store = DiskCacheStore(root_cache_dir_path=root_disk_cache_dir)
    ctrl = CacheCtrl(cache_stores=[store])
    return ctrl

cache = make_cache_ctrl()

def test_cache_array_datasource_output():
    lat = [0, 1, 2]
    lon = [10, 20, 30, 40]
    dates = ['2018-01-01', '2018-01-02']
    native_coordinates = Coordinates([lat,lon,dates],['lat','lon','time'])
    source = np.zeros(native_coordinates.shape)
    array_data_source = Array(source=source, native_coordinates=native_coordinates)
    output = array_data_source.eval(native_coordinates)
    cache.put(node=array_data_source, data=output, key='output', coordinates=native_coordinates, mode='all', update=False)
    cached_output = cache.get(node=array_data_source, key='output', coordinates=native_coordinates, mode='all')
    assert (cached_output == output).all()
    cache.rem() # clear the cache stores
