import pytest
import numpy as np
import os

from numpy.testing import assert_equal

from podpac.core.cache.cache import CacheException
from podpac.core.cache.cache import CacheCtrl
from podpac.core.cache.cache import CacheStore
from podpac.core.cache.cache import DiskCacheStore
from podpac.core.cache.cache import CacheException

from podpac.core.data.types import Array
from podpac.core.coordinates.coordinates import Coordinates

root_disk_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tmp_cache'))

def make_cache_ctrl():
    store = DiskCacheStore(root_cache_dir_path=root_disk_cache_dir)
    ctrl = CacheCtrl(cache_stores=[store])
    return ctrl

cache = make_cache_ctrl()

def test_put_and_get_array_datasource_output():
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

def test_put_and_get_with_different_instances_of_same_key_objects_array_datasource_output():
    lat = [0, 1, 2]
    lon = [10, 20, 30, 40]
    dates = ['2018-01-01', '2018-01-02']
    
    # create data source node and coordinates for put operation
    native_coordinates_put = Coordinates([lat,lon,dates],['lat','lon','time'])
    source_put = np.zeros(native_coordinates_put.shape)
    array_data_source_put = Array(source=source_put, native_coordinates=native_coordinates_put)
    output = array_data_source_put.eval(native_coordinates_put)
    
    cache.put(node=array_data_source_put, data=output, key='output', coordinates=native_coordinates_put, mode='all', update=False)
    
    # create equivalent (but new objects) data source node and coordinates for get operation
    native_coordinates_get = Coordinates([lat,lon,dates],['lat','lon','time'])
    source_get = np.zeros(native_coordinates_get.shape)
    array_data_source_get = Array(source=source_get, native_coordinates=native_coordinates_get)    
    
    cached_output = cache.get(node=array_data_source_get, key='output', coordinates=native_coordinates_get, mode='all')
    
    assert (cached_output == output).all()
    cache.rem() # clear the cache stores

def test_put_and_update_array_datasource_numpy_array():
    lat = [0, 1, 2]
    lon = [10, 20, 30, 40]
    dates = ['2018-01-01', '2018-01-02']
    native_coordinates = Coordinates([lat,lon,dates],['lat','lon','time'])
    source = np.zeros(native_coordinates.shape)
    array_data_source = Array(source=source, native_coordinates=native_coordinates)
    put_data = np.zeros(native_coordinates.shape)
    cache.put(node=array_data_source, data=put_data, key='key', coordinates=native_coordinates, mode='all', update=False)
    cached_data = cache.get(node=array_data_source, key='key', coordinates=native_coordinates, mode='all')
    assert (cached_data == put_data).all()
    update_data = np.ones(native_coordinates.shape)
    assert (update_data != put_data).any()
    with pytest.raises(CacheException):
        cache.put(node=array_data_source, data=update_data, key='key', coordinates=native_coordinates, mode='all', update=False)
    cache.put(node=array_data_source, data=update_data, key='key', coordinates=native_coordinates, mode='all', update=True)
    cached_data = cache.get(node=array_data_source, key='key', coordinates=native_coordinates, mode='all')
    assert (cached_data == update_data).all()
    cache.rem() # clear the cache stores

def test_put_and_remove_array_datasource_numpy_array():
    lat = [0, 1, 2]
    lon = [10, 20, 30, 40]
    dates = ['2018-01-01', '2018-01-02']
    native_coordinates = Coordinates([lat,lon,dates],['lat','lon','time'])
    source = np.zeros(native_coordinates.shape)
    array_data_source = Array(source=source, native_coordinates=native_coordinates)
    put_data = np.zeros(native_coordinates.shape)
    cache.put(node=array_data_source, data=put_data, key='key', coordinates=native_coordinates, mode='all', update=False)
    cached_data = cache.get(node=array_data_source, key='key', coordinates=native_coordinates, mode='all')
    assert (cached_data == put_data).all()
    cache.rem(node=array_data_source, key='key', coordinates=native_coordinates, mode='all')
    assert not cache.has(node=array_data_source, key='key', coordinates=native_coordinates, mode='all')
    with pytest.raises(CacheException):
        cache.get(node=array_data_source, key='key', coordinates=native_coordinates, mode='all')
    cache.rem() # clear the cache stores

