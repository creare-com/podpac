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

# Example Use cases
1. SMAP data retrieved from NASA servers while serving a request is stored on local disk for future requests.
2. Server that uses podpac to handle requests for data that involves complex computations using forecast data (e.g. weather predictions) as an input. Multiple processes are used to load-balance handling of requests. The same calculation should ideally not be performed by more than one of these processes. However, this could be subject to available RAM. In addition, results of calculations bassed on forecast data should be redone when updated forecast data becomes available. 
3. Server like above, but requests are handled by a "serverless" technology like AWS lambda. Intermediate results are cached in a network storage like AWS s3 (or sharded MongoDB cluster, or HDFS filesystem).
4. TODO: Add example usecases

# Specification
## User Interface
TODO: Add user interface specs

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

