# Changelog


## 1.3.0

### Introduction

The purpose of this release was to make the software more robust and to improve the Drought Monitor Application.

### Features

* Algorithm arrays can now be multi-threaded. This allows an algorithm with multiple S3 data sources to fetch the data 
  in parallel before doing the computation, speeding up the process. See #343
* Improvements to AWS interface. See #336
* Added budgeting / billing capability to manage AWS resources. See #361
* Added GeoTIFF export / import capability. Lots of work with geotransforms in the Coordinates object. See #364.
* Nodes can now have multiple output channels. This support multispectral or multichannel data. See #348.

### Bug Fixes

* When intersecting time coordinate of different precision, no intersection would result. See #344
* Fixed `Array` datasource serialization 55fcf30

### Breaking Changes

* The H5PY, CSV, and Zarr nodes interfaces were unified as such, the following attributes have changed:
    * datakey --> data_key
    * latkey --> lat_key
    * lonkey --> lon_key
    * altkey --> alt_key
    * timekey --> time_key
    * keys --> available_keys
    * CSV.lat_col --> lat_key
    * CSV.lon_col --> lon_key
    * CSV.time_col --> time_key
    * CSV.alt_col --> alt_key
    

## 1.2.0

### Introduction

The purpose of this release was to develop a short course for AMS2020. A major feature of this release is automated
creation of the PODPAC Lambda function. As part of this we implemented a few more additional
features, and fixed a number of bugs. 

### Features

* Automated Lambda Function creation using PODPAC. See #326, #306
* Added a context manager for easy temporary settings. See #329
* Added generic algorithm module with `Mask` and `Generic` nodes. See #325, #323
    * Note, this required the new unsafe evaluation setting ce8dd68355c1635214644bb5295325918493da63 bbe251a893212ea42b8c41103330e6dad0e235fe
* Added styles to node serialization, enabling customization of WMS layers. See #317
* Made mode attr's read-only by default. See #315
* Updated the definition of advanced interpolation for nodes 05163b44ca7ab66729150991bf7a6995382e3c35

### Bug Fixes

* Corrected string comparison in AWS Lambda function 7dbaf3f7d79c740cb00a37580fed259d59472e55
* _first_init usage should have called super c02dc0387fe3de9223bfac6165d7bc60922982d4
* Made style definition consistent 279eab9c40abcd8e8fb49afe366d28058391cfc6
* Fixed DroughtCategory algorithm -- the upper limit was not correct 9b92bb70b9e9de96d68ed67c4f3b053f71052229
* Fixed failure on pre-commit hook installation for dev version of podpac aebe6b0e1db39c802e5613582774594c225c6ff1 #322
* Fixed to interpolation #320 f9ad4936a8df0ce9a46f12a7c71b8c50d2ef0701 f9ad4936a8df0ce9a46f12a7c71b8c50d2ef0701 fadf939db1f679594d22690da7a58c6554705440
* Fixed error due to missing quality flag in SMAP(EGI) node 26fcb169bd1b137c51d00aa0bc7ddd2f1ae686d8

### Breaking Changes

* The Pipeline Node is being deprecated, targeted for 2.0 release. Use Node.from_json instead.


## 0.2.3

- BUG: Fix `version.HOTFIX`
- DOC: Add a release procedure (RELEASE.md)


## 0.2.2

- DOC: Add user guide for Settings
- DOC: Improve Documentation For Nodes ([#57](https://github.com/creare-com/podpac/issues/57))
- DOC: Deploy documentation site to `https://podpac.org`
- TEST: Add coordinate testing ([#3](https://github.com/creare-com/podpac/issues/3))
- ENH: Cache evaluation results and check the cache before evaluating ([#125](https://github.com/creare-com/podpac/issues/125))
- ENH: Add datalib for TerrainTiles
- ENH: Add datalib for GFS
- ENH: AWS - Update Dockerfile for AWS Lambda to support podpac changes


## 0.2.1

- BUG: Fix unit tests with renamed decorator
- BUG: Fix file path sanitization
- ENH: Make `SMAP_BASE_URL` check lazy to avoid warning at import
- ENH: Remove deprecated caching methods from Node


## 0.2.0

Initial public release
