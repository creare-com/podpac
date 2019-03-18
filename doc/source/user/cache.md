# Cache

This document describes the caching methodology used in PODPAC, and how to control it. PODPAC uses a central cache shared by all nodes. Retrieval from the cache is based on the node's definition (`node.json`), the coordinates, and a key. 

Each node has a **Cache Control** (`cache_ctrl`) defined by default, and the **Cache Control** may contain multiple **Cache Stores** (.e.g 'disk', 'ram'). A **Cache Store** may also have a specific **Cache Container**. 



## Default Cache

By default, every node caches their outputs to memory (RAM). These settings can be controlled using `podpac.settings`.

**Settings and their Defaults:**

* DEFAULT_CACHE : list
    * Defines a default list of cache stores in priority order. Defaults to `['ram']`. Can include ['ram', 'disk', 's3'].
    * This can be over-written on an individual node by specifying `cache_ctrl` when creating the node. E.g. `node = podpac.Node(cache_ctrl=['disk'])`
    * Authors of nodes may require certain caches always be available. For example, the `podpac.datalib.smap.SMAPDateFolder` node always requires a 'disk' cache, and will add it. 
* DISK_CACHE_DIR : str
    * Subdirectory to use for the disk cache. Defaults to ``'cache'`` in the podpac root directory.
* S3_CACHE_DIR : str
    * Subdirectory to use for S3 cache (within the specified S3 bucket). Defaults to ``'cache'``.
* CACHE_OUTPUT_DEFAULT : bool
    * Automatically cache node outputs to the default cache store(s). Outputs for nodes with `cache_output=False` will not be cached. Defaults to ``True``.
* RAM_CACHE_ENABLED: bool
    * Enable caching to RAM. Note that if disabled, some nodes may fail. Defaults to ``True``.
* DISK_CACHE_ENABLED: bool
    * Enable caching to disk. Note that if disabled, some nodes may fail. Defaults to ``True``.
* S3_CACHE_ENABLED: bool
    * Enable caching to S3. Note that if disabled, some nodes may fail. Defaults to ``True``.

## Examples

To globally disable automatic caching of outputs use:
```python
import podpac
podpac.settings["CACHE_OUTPUT_DEFAULT"] = False
podpac.settings.save()
```

To overwrite this behavior for a particular node (i.e. making sure outputs are cached) use:
```python
smap = podpac.datalib.smap.SMAP(cache_output=True)
```

Different instances of the same node share a cache. For example:
```python
>>> coords = podpac.Coordinates([podpac.clinspace(40, 39, 16),
                                 podpac.clinspace(-100, -90, 16),
                                 '2015-01-01T00', ['lat', 'lon', 'time']])
>>> smap1 = podpac.datalib.smap.SMAP()
>>> o = smap1.eval(coords)
>>> smap1._from_cache
False
>>> del smap1
>>> smap2 = podpac.datalib.smap.SMAP()
>>> o = smap2.eval(coords)
>>> smap2._from_cache
True
```
