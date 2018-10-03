# Requirements
* Support storing results of calculations
    * This includes:
        * ? numpy arrays
        * xarrays
        * UnitsDataArray
        * ? DataSource
        * ? other serialized binary data
            * GeoTIFFs
* Cached queries are idempotent with respect to some set of parameters that define the query (location/datetime, datasource, paramters used to compute a prediction).
* ability to retrieve calculated data from a pipeline after execution in an interactive shell for debugging or analysis purposes
* ? Support retrieval of subsets of data. For example, previous results of a calculation over North America at some resolution are cached. Does cache support retrieval of just the results for California at the same resolution? Or, does interpolation handle this in conjunction with cache? Or, does interpolation only handle it when the resolution doesn't match. Or does caller request data using "key" of original cache and an additional "subset" parameter?
* Support different storage mediums
    * This includes:
        * RAM
        * local disk (HDD/SSD)
        * AWS s3
        * ? Databases (MongoDB supports geospatial queries and binary data)
        * ? HDFS file system
* Support resource limitations 
    * This includes:
        * Total number of bytes cached in RAM/disk/s3 across a process running podpac
        * Total number of bytes cached in RAM/disk/s3 across multiple processes on the same computer/server running podpac
        * Total number of bytes cached in RAM/disk/s3 across a cluster of servers running podpac
* Support prioritization of cached data under resource limitations
    * Bassed on:
        * size of data (bytes)
        * computational cost to reproduce (time to compute)
        * access patterns:
            * frequency of use
            * recency of use
* Support expiration of data (e.g. forecast data that is updated at some time interval)
* Support cache tiers:
    * respond from "better" tier when possible (e.g. RAM)
    * fall back on "worse" tier (e.g. local disk then s3 or some other networked storage)
* Support saving "providence" of data with the stored data
    * For archived data (local disk, s3, database, HDFS) this could be the json pipeline definition and should include:
        * version of podpac that was used to create it
        * timestamp when it was created
        * information about the root datasources (version if they have it)
        * computational graph definition
    * For in memory we may not want to be so robust but we may want include:
        * Timestamp when it was computed/stored (to support expiration)
        * Possibly information about the function call that created the data (for a cached property). This maybe could be a lambda function wrapping the original function with the original args/kwargs. But would have to be careful about args/kwargs that have state that may have changed. Could we maybe force these to be "static".

# Example Use cases
1. SMAP data retrieved from NASA servers while serving a request is stored on local disk for future requests.
2. Server that uses podpac to handle requests for data that involves complex computations using forecast data (e.g. weather predictions) as an input. Multiple processes are used to load-balance handling of requests. The same calculation should ideally not be performed by more than one of these processes. However, this could be subject to available RAM. In addition, results of calculations bassed on forecast data should be redone when updated forecast data becomes available. 
3. Server like above, but requests are handled by a "serverless" technology like AWS lambda. Intermediate results are cached in a network storage like AWS s3 (or sharded MongoDB cluster, or HDFS filesystem).
4. TODO: Add example usecases

# Specification
## User Interface
TODO: Add user interface specs

* `@cached_property(dependson, expiresin, key_spec)` 
* `get()`
* `put()`
* `delete()`

## Developer interface 
TODO: Add developer interface specs

# Implementation Notes
* Idempotence can be supported using key computed via some hashing protocol. Maybe the key is the hash of a dictionary representing parameter key/values and there is some specification on the keys like the time parameter is always "datetime".
* Keys used to retrieve data need to support a distributed system. Basically, cache system needs to know, or be able to figure out, what server to make a request to when data is cached over some kind of network storage system (multiple s3 buckets, sharded database, HDFS). 
    * This can be accomplished by:
        * reserving a portion of a retrieval key (hash) to specify the network resource
            * could be hard to change the "key" for a particular network resource after the fact.
            * likely need to support adding (maybe deleting) network resources over time.
        * Using a centralized lookup server
            * central server could be bottleneck
            * retrieval becomes at least two network bassed queries (one to central server, and then one for data).
* libraries to keep in mind:
    * [pathos](https://github.com/uqfoundation/pathos) Graph Execution Manager with generic `map` (think python map) and `pipe` (aka `apply`, not sure what this is maybe reduction) operations.
        * execution manager versions (implementations) include:
            * MPI
            * ssh (rpc over ssh tunnels)
            * multiprocessors
    * [klepto](https://github.com/uqfoundation/klepto) In-memory and archival cache. Works-with/used-by [pathos](https://github.com/uqfoundation/pathos) in conjunction with `dill` (serializer that extends pickle).
        * Supported caching algorithms: `lfu`,`lru`,`mru`,`rr`. Does not have the something that takes into account "compute time".
        * Supported archival systems: file, directory,sql-table,sql-database,directory of hdf5 files, single hdf5 file
        * Supported key calculations: raw-python objects (obj), hash of obj, str(obj), pickle of obj
    * [Dask](https://dask.pydata.org/en/latest/) Execution manager. Dataframe style computations.
    * [cachey](https://github.com/dask/cachey) Cache that works with Dask. In-memory only. Simple dictionary style interface. Uses formula to compute priority of data abssed on size, use, and time to compute. 
    * key/value memory caching servises from the database-driven website community:
        * [memcached](http://memcached.org/) ([wikipedia](https://en.wikipedia.org/wiki/Memcached))
        * [redis](https://redis.io/) ([wikipedia](https://en.wikipedia.org/wiki/Redis))
            * [geohash](https://en.wikipedia.org/wiki/Geohash)
