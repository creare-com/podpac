# Changelog

## 3.4.0 Point Probe Value Format for Enumerated Legends

### Features
* Adds the label next to the value for enumerated legends in the point prober.
  * Before: "value": 1.0
  * After: "value": "1 (Sand)"

## 3.3.1

### Hotfix
* Can now use `np.timedelta64` as a valid type for coordinates (merge issues on 3.3.0 accidentally removed this feature)

## 3.3.0

### Features
* Now supporting custom coordinate dimensions for downstream applications. Just use `podpac.utils.add_valid_dimension("my_dimension_name")` to register and start using your custom dimension name.
* Can now use `np.timedelta64` as a valid type for coordinates

### Maintenance
* Removed the "datalib" module, and made it its own package `podpacdatalib`.


## 3.2.1
### Bugfixes
* Fixed documentation build
* Fixed nearest neighbor interpolation bug where same pixel could
  give different values due to rounding ambiguity (0.5-->0, 1.5-->2)

## 3.2.0

### Features
* Added the "none" interpolator, which disables the Interpolation node of Mixin
* Enabled nearest-neighbor interpolation for N-D stacked coordinates (#488)
* Implemented Xarray interpolation for dependent coordinates (multi-dimensional stacked coordinates)
* Improved caching
  * Extra diemnsions are discarded when caching datasource and compositor outputs
  * Added CRS-agnostic caching
* Added utility function that generates a JSON spec of available PODPAC nodes. This spec can be used to make UI elements
* Added the `get_source_data` method for `Datasource` nodes to bypass interpolation (#485)
* Added the `get_bounds` method to `Node` to retrieve the coordinate bounds (#486)
* Added `Node` point prober features that will evaluate every node in a pipeline at a point and record its value. Useful for debugging complex pipelines.
* Added `AffineCoorindates` (#491) these are stacked spatial shaped coordinates that are parameterized by a `geotransform`. Replaces `RotatedCoordinates`.
* `OGC.WCS` `Node` can now handle multi-band geotiffs

### Bugfixes
* Fixed a number of floating point coordinates discrepancies
* `Rasterio` node will no longer attempt to use overviews when no overviews are present.
* Fixed `geotransform` shape transpose error
* Fixed numpy deprecation of using `list` indices
* Fixed regressions due to dependency updates



## 3.1.0

This release was in support of the GeoWATCH application. Bugs/features added were to support server deployment.

### Features
* Added `OGR` datasource node for reading shapefiles
* `Compositers.multithreading`: For some compositors, it's important to actually evaluate the nodes in serial for performance reasons, regardless of the global multithreading setting. Now compositors user settings['MULTITHREADING'] by default, but `OrderedCompositors` always set this to `False`. In either case it can be overwritten on a node-by-node basis.
* `RasterioSource.prefer_overview_closest`: when selecting overview levels, we can either select the coarsest overview smaller than the eval coordinates OR we can select the overview with the closest step size to the eval coordinates (this may be coarser than the eval coordinates). Setting this attr to `True` will select the closest overview instead of the closest higher resolution overview.
* Improved speed of evaluations by eliminating unneccessary CRS validations
* Added `decode_cf` attribute to `Dataset` data source node
* Default interpolation can now be specificief application-wide through the `podpac.settings["DEFAULT_INTERPOLATION"]` setting
* Added `MockWCSClient` to `ogc.py` for WCS endpoints that do not implement `get_coverage`. This make it easy to turn PODPAC into a lightweight WCS server, and then use a PODPAC WCS client.
* Added `prefer_overviews` and `prefer_overviews_closest` attributes to `Rasterio` data source node. These attributes allow users to pull from the overviews directly for coarse requests.
* Added the point prober. This allows users to probe the values of an algorithm pipeline at a point. See `Node.probe`
* Added the `from_name_params` method to `Node`, allowing nodes to be created from the node name + additional parameters.
* Renamed `set_unsafe_eval` to `allow_unrestricted_code_execution` for a more descriptive name.
* Improved specification of enumerated colormaps in the `Style`
* Enabled saving to a geotiff memory file to support WCS calls

### Bugfixes
* Fixed crs mismatch bug in `Reproject` node
* Fixed lat/lon ordering bug for different versions of WMS/WCS in `from_url` method of `Coordinates`
* Fixed bug in `Coordinates.transform` where `ArrayCoordinates` turned into `UniformCoordinates` for two CRS with linear mapping.
* Fixed bug in `DataSource` node where `get_data` returns coordinates that are different from the request (this happens in the case where raw data is returned)
* Fixed BBOX order specification error in `WCS` node, where different versions of WCS change the order of lat/lon. This is now handled correctly.
* Fixed a number of interpolation errors:
  * `InterpolationMixin` will no longer cache internal evaluations which lead to strange caching errors
  * Fixed selector bugs related to negative step sizes
  * Fixed nearest neighbor interpolation bugs related to negative step sizes
  * Fixed Selector uniform coordinates short-cut
* Fixed bug where `DataArray` attributes were dropped when doing basic math operations
* Fixed bug in `to_geotiff` export function (misplaced parenthesis)

## 3.0.0
Interpolation refactoring. Interpolation now lives as an Algorithm Node. As such,
interpolation can exist in any part of a pipeline, and even multiple times. As
part of this improvement, we also implemented "Selectors" which subselect data
based on the interpolation method specified BEFORE data is pulled from remote
servers.

Because this refactor changed the interface somewhat, we bumped the major version number.

The MAJOR change with the PODPAC functionality is that now some Nodes may return DIFFERENT (not interpolated) coordinates than the eval coordinates.

### Features
* Added `Interpolation` Node and `InterpolationMixin` to restore backwards compatibility with most nodes.
* Replace WCS node with a new version that uses owslib under the hood. Also added authentiation support.
* Added SoilGrids WCS data sources
* Added an "Xarray" interpolator, which uses `xarray`'s interpolation methods. This now allows linear project for time, for example.
* Interpolators will now throw warning if the user specifies an interpolation parameter which is not used.
* Improved interpolation documentation
* Added "Autozoom" functionality for TerrainTiles datasource
* Added `Compositor` nodes that combine multiple files/tiles of a single datasource BEFORE interpolation
* Removed SMAP PyDAP datalib -- it was always unstable whereas the EGI version usually works
* Improved Rasterio node -- it now read datasources directly using Rasterio instead of going through s3fs.

### Bugfixes
* Can now clear ram cache before cache is eliminated
* Fixed #303, UnitsDataArray deserialization
* Removed support for "numpy" return type in Algorithm nodes, since coordinates can now be altered in Algorithm Nodes
* Fixed styling and plugin information is being set 7aef43b5a
* Fixed some floating point rounding issues at tile edges 8ac834d4
* Fixed Coordinates.from_url to work correctly with different versions of OCG WMS call (and possible WCS calls, but the WCS documentation and my reference servers disagree...)

## 2.3.0
### Introduction

Adding subdataset support for hdf4 data sources (i.e. downloaded MODIS netcdf file), wrapping SoilScape data, and adding
expiration to cache.

This release also drops Python 3.5 support.

### Features
* Subdataset support in Rasterio Node, see #410
* Adding SoilScape data source, and disk cache expiration, see #419
* PyDAP node will now retry requests incase of server throttling, see 514dc5da3975fa1c6051f0da1691cf5ba4972bf8

### Bugfixes
* Added dimensions to `modis` and `cosmos` compositors
* Fixed version numbers in `smap_egi` datasource, and these are now looked up automatically
* Fixed a precision bug on selection with time coordinates

## 2.2.2
### Bug Fixes
* Fixed floating point errors on selection of data subset (short circuit optimization to avoid unnecessary interpolation)
* Fixed bug in cosmos_stations.latlon_from_label giving the wrong latlon for a label
* Fixing compositor to update interpolation of sources automatically (and deleting cached definitions).
    * Also making cached node definitions easier to remove -- no longer caching node.json, node.json_pretty and node.hash

## 2.2.0
### Introduction

Wrapping Landsat8, Sentinel2, and MODIS data and improving interpolation.

### Features
* Added `datalib.satutils` which wraps Landsat8 and Sentinel2 data
* Added `datalib.modis_pds` which wraps MODIS products ["MCD43A4.006", "MOD09GA.006", "MYD09GA.006", "MOD09GQ.006", "MYD09GQ.006"]
* Added settings['AWS_REQUESTER_PAYS'] and `authentication.S3Mixing.aws_requester_pays` attribute to support Sentinel2 data
* Added `issubset` method to Coordinates which allows users to test if a coordinate is a subset of another one
* Added environmental variables in Lambda function deployment allowing users to specify the location of additional
dependencies (`FUNCTION_DEPENDENCIES_KEY`) and settings (`SETTINGS`). This was in support the WMS service.
* Intake nodes can now filter inputs by additional data columns for .csv files / pandas dataframes by using the pandas
`query` method.
* Added documentation on `Interpolation` and `Wrapping Datasets`

### Bug Fixes
* Added `dims` attributes to `Compositor` nodes which indicates the dimensions that sources are expected to have. This
fixes a bug where `Nodes` throw and error if Coordinates contain extra dimensions when the `Compositor` sources are missing
those dimensions.
* `COSMOSStations` will no longer fail for sites with no data or one data point. These sites are now automatically filtered.
* Fixed `core.data.file_source` closing files prematurely due to using context managers
* Fixed heterogenous interpolation (where lat/lon uses a different interpolator than time, for example)
* `datalib.TerrainTiles` now accesses S3 anonymously by default. Interpolation specified at the compositor level are
also now passed down to the sources.

### Breaking changes
* Fixed `core.algorithm.signal.py` and in the process removed `SpatialConvolution` and `TemporalConvolutions`. Users now
have to label the dimensions of the kernel -- which prevents results from being modified if the eval coordinates are
transposed. This was a major bug in the `Convolution` node, and the new change obviates the need for the removed Nodes,
but it may break some pipelines.


## 2.1.0
### Introduction

Fixing some bugs associated with AWS evaluation and the drought-monitor application

### Features
* Added `algorithm.TransformTimeUnits` algorithm to be able to utilize time in terms of `dayofyear`
* Added `Coordinates.transform_time` method to transform YYYY-MM-DD time units to something like `dayofyear`

### Bug Fixes
* Removed `region_name` from `authentication.S3Mixin` because it is not a valid argument
* Made the `algorithm.Mask` node simpler to JSON serialize
* Made `interpolator` tolerant of string-based timedelta values
* Added the `crs` to `Node` outputs if it is not present -- this caused issues with the `union` call in `Algorithm.eval`
* Fixed `data.Zarr`'s `open_consolidated` call. It was erroneously passing the `r` argument.


## 2.0.0
### Introduction

### Features
* Added `MODIS` datasource `datalib.modis_pds`
* Added `datalib.weathercitizen` to retrieve weathercitizen data
* Added `datalib.cosmos_stations` to retrieve soil moisture data from the stationary COSMOS soil moisture network
* Added `algorithm.ResampleReduce`, which allows users to coarsen a dataset based on a reduce operation (such as mean, max, etc.).
* Added the `managers.parallel` submodule that enables parallel computation with PODPAC in a multi-threaded, multi-process, or multi-AWS-Lambda-function way
* Added the `managers.multi_process` submodule that enables PODPAC nodes to be run in another process.
* Added the `compositor.UniformTileCompositor` and `compositor.UniformTileMixin` to enable compositing of data sources BEFORE harmonization (so that interpolation can happen across data sources with the same coordinate systems)
* Added the `validate_crs` flag to `Coordinates` so that validation of the `crs` can be skipped if needed (this improved speed)
* Added `managers.aws.Lambda.eval_timeout` attribute so that Python will only wait a certain amount of time before returning from a invoke call.
* Added asynchronous evaluation of `managers.aws.Lambda` Nodes.  Note! This does not do error checking (e.g. exceeding AWS allocation f the number of allowed concurrent threads).
* Added `alglib.climatology`, which allows computation of Beta distribution fits to time series for each day of the year
* Added `core.algorithm.stats.DayOfYearWindow`, which is similar to `GroupReduce`, but also allows a window for the computation and only works for time
* `Coordinates` will now be automatically simplified when transformed from one `crs` to another. Previously, transformations always resulted in `DependentCoordinates`. This change adds the `.simply` method to some `Coordinates1d` classes.
* Improved performance of PODPAC AWS Lambda function by only downloading dependencies when the function is not already "hot". "Hot" functions retain the files in the `/tmp` directory of the Lambda function.
* `data.Zarr` will now use `zarr.open_consolidated` whenever the `file_mode='r'` for more efficient read operations on S3
* Added podpac's version to pipeline definitions

### Bug Fixes
* Fixed `algorithm.GroupReduce` to accept `dayofyear`, `weekofyear`, `season`, and `month`. It also now returns the time coordinate in one of these units.
* Implemented a circular dependency check to avoid infinite recursion and locking up due to cache accessing. This change also defined the `NodeDefinitionError` exception.
* Fixed the `UnitsDataArray.to_format` function's `zarr_part` format to work propertly with parallel computations
* Added the `[algorithm]` dependencies as part of the AWS Lambda function build -- previously the `numexpr` Python package was missing

### Breaking changes
* Renamed `native_coordinates` to `coordinates`
* Deprecated `Pipeline` nodes
* Removed `ctype` and `segment_length` from `Coordinates`. Now specify the `boundary` attribute for `DataSource` nodes instead.
* Fixed `datalib.smap_egi` module to work with `xarray` version `0.15`
* Modified PODPAC's root setting directory from `~/.podpac` to `~/.config/podpac`
* Changed default cache behaviour to overwrite by default (prevents certain cache lock situations)
* Removed `datalib.airmoss` -- it was no longer working!

### Maintenance
* Refactored the way PODPAC keeps track of `Node` definition. Most all of it is now handled by the base class, previously `DataSource`, `Algorithm`, and `Compositor` had to implement specialized functions.
* Refactored `datalib` nodes to prefer using the new `cached_property` decorator instead of `defaults` which were causing severe circular dependencies
* Refactored `DataSource` nodes that access files on S3 to use a common `Mixin`
* Refactored authentication to use more consistent approach across the library

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
