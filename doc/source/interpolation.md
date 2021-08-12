# Interpolation

## Description

PODPAC allows users to specify various different interpolation schemes for nodes with
increased granularity, and even lets users write their own interpolators. By default
PODPAC uses the `podpac.settings["DEFAULT_INTERPOLATION"] == "nearest"`, which may
be modified by users. Users who wish to see raw datasources can also use `"none"`
for the interpolator type.

Relevant example notebooks include:
* [Advanced Interpolation](https://github.com/creare-com/podpac-examples/blob/master/notebooks/4-advanced/interpolation.ipynb)
* [Basic Interpolation](https://github.com/creare-com/podpac-examples/blob/master/notebooks/2-combining-data/automatic-interpolation-and-regridding.ipynb)
* [Drought Monitor Data Access Harmonization Processing](https://github.com/creare-com/podpac-examples/blob/master/notebooks/examples/drought-monitor/03-data-access-harmonization-processing.ipynb)

## Examples
Consider a `DataSource` with `lat`, `lon`, `time` coordinates that we will instantiate as:
`node = DataSource(..., interpolation=interpolation)`

`interpolation` can be specified ...

### ...as a string

`interpolation='nearest'`
<<<<<<< HEAD
* **Descripition**: All dimensions are interpolated using nearest neighbor interpolation. This is the default, but available options can be found here: `podpac.core.interpolation.interpolation.INTERPOLATION_METHODS`. In particular, for no interpolation, use `interpolation="none"`. *NOTE* the `none` interpolator ONLY considers the bounds of any evaluated coordinates. This means the data is returned at FULL resolution (no striding or sub-selection).
* **Details**: PODPAC will automatically select appropriate interpolators based on the source coordinates and eval coordinates. Default interpolator orders can be found in `podpac.core.interpolation.interpolation.INTERPOLATION_METHODS_DICT`

### ...as a dictionary

```python
interpolation = {
    'method': 'nearest',
    'params': {    # Optional. Available parameters depend on the particular interpolator
        'spatial_tolerance': 1.1,
        'time_tolerance': np.timedelta64(1, 'D')
    },
    'interpolators': [ScipyGrid, NearestNeighbor]  # Optional. Available options are in podpac.core.interpolation.interpolation.INTERPOLATORS
}
```
* **Descripition**: All dimensions are interpolated using nearest neighbor interpolation, and the type of interpolators are tried in the order specified. For applicable interpolators, the specified parameters will be used.
* **Details**: PODPAC loops through the `interpolators` list, checking if the interpolator is able to interpolate between the evaluated and source coordinates. The first capable interpolator available will be used.

### ...as a list

```python
interpolation = [
    {
        'method': 'bilinear',
        'dims': ['lat', 'lon']
    },
    {
        'method': 'nearest',
        'dims': ['time']
    }
]
```

* **Descripition**: The dimensions listed in the `'dims'` list will used the specified method. These dictionaries can also specify the same field shown in the previous section.
* **Details**: PODPAC loops through the `interpolation` list, using the settings specified for each dimension independently.

**NOTE! Specifying the interpolation as a list also control the ORDER of interpolation.**
The first item in the list will be interpolated first. In this case, `lat`/`lon` will be bilinearly interpolated BEFORE `time` is nearest-neighbor interpolated.

## Interpolators
The list of available interpolators are as follows:
* `NoneInterpolator`: An interpolator that passes through the raw, source data at full resolution -- it does not do any interpolation. **Note**: This interpolator can be used for **some** of the dimension by specifying `interpolation` as a list.
* `NearestNeighbor`: A custom implementation based on `scipy.cKDtree`, which handles nearly any combination of source and destination coordinates
* `XarrayInterpolator`: A light-weight wrapper around `xarray`'s `DataArray.interp` method, which is itself a wrapper around `scipy` interpolation functions, but with a clean `xarray` interface
* `RasterioInterpolator`: A wrapper around `rasterio`'s interpolation/reprojection routines. Appropriate for grid-to-grid interpolation.
* `ScipyGrid`: An optimized implementation for `grid` sources that uses `scipy`'s `RegularGridInterpolator`, or `RectBivariateSplit` interpolator depending on the method.
* `ScipyPoint`: An implementation based on `scipy.KDtree` capable of `nearest` interpolation for `point` sources
* `NearestPreview`: An approximate nearest-neighbor interpolator useful for rapidly viewing large files

The default order for these interpolators can be found in `podpac.data.INTERPOLATORS`.

### NearestNeighbor
Since this is the most general of the interpolators, this section deals with the available parameters and settings for the `NearestNeighbor` interpolator.

#### Parameters
The following parameters can be set by specifying the interpolation as a dictionary or a list, as described above.

* `respect_bounds` : `bool`
  * Default is `True`. If `True`, any requested dimension OUTSIDE of the bounds will be interpolated as `nan`.
    Otherwise, any point outside the bounds will have the value of the nearest neighboring point
* `remove_nan` : `bool`
  * Default is `False`. If `True`, `nan`'s in the source dataset will NOT be interpolated. This can be used if a value for the function
    is needed at every point of the request. It is not helpful when computing statistics, where `nan` values will be explicitly
    ignored. In that case, if `remove_nan` is `True`, `nan` values will take on the values of neighbors, skewing the statistical result.
* `*_tolerance` : `float`, where `*` in ["spatial", "time", "alt"]
  * Default is `inf`. Maximum distance to the nearest coordinate to be interpolated.
    Corresponds to the unit of the `*` dimension.
* `*_scale` : `float`, where `*` in ["spatial", "time", "alt"]
  * Default is `1`. This only applies when the source has stacked dimensions with different units. The `*_scale`
    defines the factor that the coordinates will be scaled by (coordinates are divided by `*_scale`) to output
    a valid distance for the combined set of dimensions.
    For example, when "lat, lon, and alt" dimensions are stacked, ["lat", "lon"] are in degrees
    and "alt" is in feet, the `*_scale` parameters should be set so that
   `|| [dlat / spatial_scale, dlon / spatial_scale, dalt / alt_scale] ||` results in a reasonable distance.
* `use_selector` : `bool`
  * Default is `True`. If `True`, a subset of the coordinates will be selected BEFORE the data of a dataset is retrieved. This
    reduces the number of data retrievals needed for large datasets. In cases where `remove_nan` = `True`, the selector may select
    only `nan` points, in which case the interpolation fails to produce non-`nan` data. This usually happens when requesting a single
    point from a dataset that contains `nan`s. As such, in these cases set `use_selector` = `False` to get a non-`nan` value.

#### Advanced NearestNeighbor Interpolation Examples
* Only interpolate points that are within `1` of the source data lat/lon locations
```python
interpolation={"method": "nearest", "params": {"spatial_tolerance": 1}},
```
* When interpolating with mixed time/space, use `1` day as equivalent to `1` degree for determining the distance
```python
interpolation={
    "method": "nearest",
    "params": {
        "spatial_scale": 1,
        "time_scale": "1,D",
        "alt_scale": 10,
    }
}
```
* Remove nan values in the source datasource -- in some cases a `nan` may still be interpolated
```python
interpolation={
    "method": "nearest",
    "params": {
        "remove_nan": True,
    }
}
```
* Remove nan values in the source datasource in all cases, even for single point requests located directly at `nan`-values in the source.
```python
interpolation={
    "method": "nearest",
    "params": {
        "remove_nan": True,
        "use_selector": False,
    }
}
```
* Do nearest-neighbor extrapolation outside of the bounds of the source dataset
```python
interpolation={
    "method": "nearest",
    "params": {
        "respect_bounds": False,
    }
}
```
* Do nearest-neighbor interpolation of time with `nan` removal followed by spatial interpolation
```python
interpolation = [
    {
        "method": "nearest",
        "params": {
            "remove_nan": True,
        },
        "dims": ["time"]
    },
    {
        "method": "nearest",
        "dims": ["lat", "lon", "alt"]
    },
]
```
## Notes and Caveats
While the API is well developed, all conceivable functionality is not. For example, while we can interpolate gridded data to point data, point data to grid data interpolation is not as well supported, and there may be errors or unexpected results. Advanced users can develop their own interpolators, but this is not currently well-documented.

**Gotcha**: Parameters for a specific interpolator may be ignored if a different interpolator is automatically selected. These ignored parameters are logged as warnings.

