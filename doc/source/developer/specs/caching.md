# Requirements
* Support storing results of calculations
    * This includes:
        * ? numpy arrays
        * xarrays
        * UnitsDataArray
        * ? DataSource
* Support different storage mediums
    * This includes:
        * RAM
        * local disk (HDD/SSD)
        * AWS s3
        * ? Databases (MongoDB supports geospatial queries and binary data)
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

# Example Use cases
TODO: Add example usecases

# Specification
## User Interface
TODO: Add user interface specs

## Developer interface 
TODO: Add developer interface specs
