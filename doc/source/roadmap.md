# Roadmap

This document records the roadmap for PODPAC development. Specific issues will be referenced for features / bugs. 

When features / bugs are identified, they will be placed in the `TODO` section. The features and bugs will be prioritized, and targeted for a release, by moving it under the release heading. 

## Versioning scheme

We use the following versioning format: 
`Major.minor.hotfix+hash`
* Major: 
    * For major releases > 1, the interface will remain backwards compatible
    * For the 0.x.x release, backwards compatibility is not guaranteed
* Minor:
    * Each minor release adds requested features, and fixes known bugs
* hotfix:
    * Hotfix releases fix high priority bugs
* +hash: 
    * During development, the git hash is appended to the end of the version
    * Allows a particular point in the development to be referenced
    * Tagged releases will not include this hash

## 0.3.0

### Features

* Add unit support to Coordinate
* Add automated geospatial reprojection to UniformCoordinates

### Bugs

## 0.2.0

### Features

* #7: Refactor interpolation mechanism for DataSource nodes
* Refactor implementation of pipeline node
    * Improve specification of output coordinates, dimensions, and shape
    * Handle cases where evaluated coordinates are expanded/modified by other nodes
* Refactor usage of traitlets

### Bugs

* Fix implementation of units in Nodes and UnitsDataArray
* #13: Generalize/fix convolution nodes
* #12: Add transpose to Coordinates

## 0.1.0

The community update -- to help promote community contributions

### Features

* Add doctrings to everything (#4, #9)
* Increase unit test coverage to > 90% (#3)
* Implement automated tests

### Bugs

## TODO

### Features

* Improve data caching
    * Specify max cache size
    * Cache expiration
* Integrate pipeline execution with parallel processing through AWS
    * Integrate with dask? 
* Tighter integration with AWS -- automatic lambda function creation and invocation
* #2: Test coordinate group
* Add support for 'None' endpoints/startpoints in UniformCoordinates objects (and potentially monotonic coordinates)

### Bugs

* #11: Nodes with default inputs breaks pipeline
* #10: Enable access to sub-attributes in Pipelines
* #8: Rethink how params are used and how they should behave
* #1: Custom nodes my silently output incomplete/incorrect pipeline definitions
