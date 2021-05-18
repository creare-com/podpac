# Cache

This document describes the caching methodology used in PODPAC, and how to control it. PODPAC uses a central cache shared by all nodes. Retrieval from the cache is based on the node's definition (`node.json`), the coordinates, and a key. 

Each node has a **Cache Control** (`cache_ctrl`) defined by default, and the **Cache Control** may contain multiple **Cache Stores** (.e.g 'disk', 'ram').



## Caching Outputs

By default, PODPAC caches evaluated node outputs to memory (RAM). When a node is evaluated with the same coordinates, the output is retrieved from the cache.

The following example demonstrates that the output was retrieved from the cache on teh second evaluation:

```python
[.] import podpac
[.] import podpac.datalib
[.] coords = podpac.Coordinates([podpac.clinspace(40, 39, 16),
                                 podpac.clinspace(-100, -90, 16),
                                 '2015-01-01T00', ['lat', 'lon', 'time']])
[.] smap = podpac.datalib.smap.SMAP()
[.] o = smap1.eval(coords)
[.] smap._from_cache
False
[.] o = smap1.eval(coords)
[.] smap._from_cache
True
```

Importantly, different instances of the same node share a cache. The following example demonstrates that a different instance of a node will retrieve output from the cache as well:

```python
[.] smap2 = podpac.datalib.smap.SMAP()
[.] o = smap2.eval(coords)
[.] smap2._from_cache
True
```

### Configure Output Caching

Automatic caching of outputs can be controlled globally and in individual nodes. For example, to globally disable caching outputs:

```python
podpac.settings["CACHE_OUTPUT_DEFAULT"] = False
```

To disable output caching for a particular node:

```python
smap = podpac.datalib.smap.SMAP(cache_output=False)
```

## Disk Cache

In addition to caching to memory (RAM), PODPAC provides a disk cache that persists across processes. For example, when the disk cache is used, a script that evaluates a node can be run multiple times and will retrieve node outputs from the disk cache on subsequent runs.

Each node has a `cache_ctrl` that specifies which cache stores to use, in priority order. For example, to use the RAM cache and the disk cache:

```python
smap = podpac.datalib.smap.SMAP(cache_ctrl=['ram', 'disk'])
```

The default cache control can be set globally in the settings:

```python
podpac.settings["DEFAULT_CACHE"] = ['ram', 'disk']
```

### Configure Disk Caching

The disk cache directory can be set using the `DISK_CACHE_DIR` setting.

## S3 Cache

PODPAC also provides caching to the cloud using AWS S3. Configure the S3 bucket and cache subdirectory using the `S3_BUCKET_NAME` and `S3_CACHE_DIR` settings.

## Clearing the Cache

To clear the entire cache use:

```python
podpac.utils.clear_cache()
```

To clear the cache for a particular node: 

```python
smap.clear_cache()
```

You can also clear a particular cache store, for example clear the disk cache leaving the RAM cache in place:

```python
# node
smap.clear_cache('disk')

# entire cache
podpac.utils.clear_cache('disk')
```

## Cache Limits

PODPAC provides a limit for each cache store in the podpac settings.

```
RAM_CACHE_MAX_BYTES
DISK_CACHE_MAX_BYTES
S3_CACHE_MAX_BYTES
```

When a cache store is full, new entries are ignored cached.


## Advanced Usage

### Caching Other Objects

Nodes can cache other data and objects using a cache key and, optionally, coordinates. The following example caches and retrieves data using the key `my_data`.

```python
[.] smap.put_cache(10, 'my_data')
[.] smap.get_cache('my_data')
10
```

In general, the node cache can be managed using the `Node.put_cache`, `Node.get_cache`, `Node.has_cache`, and `Node.rem_cache` methods.


### Cache Expiration

Cached entries can optionally have an expiration date, after which the entry is considered invalid and automatically removed.

To specify an expiration date

```python
# specific datetime
node.put_cache(10, 'my_data', expires='2021-01-01T12:00:00')

# timedelta, in 12 hours
node.put_cache(10, 'my_data', expires='12,h')
```

### Cached Node Properties

PODPAC provides a `cached_property` decorator that enhances the builtin `property` decorator.

By default, the `cached_property` stores the value as a private attribute in the object. To use the PODPAC cache so that the property persists across objects or processes according to the node node `cache_ctrl`:

```python
class MyNode(podpac.Node):
    @podpac.cached_property(use_cache_ctrl=True)
    def my_cached_property(self):
        return 10
```

### Updating Existing Entries

By default, a existing cache entries will be overwritten with new data.

```python
[.] smap.put_cache(10, 'my_data')
[.] smap.put_cache(20, 'my_data')
[.] smap.get_cache('my_data')
20
```

To prevent overwriting existing cache entries, use `overwrite=False`:

```python
[.] smap.put_cache(100, 'my_data', overwrite=False)
podpac.core.node.NodeException: Cached data already exists for key 'my_data' and coordinates None
```