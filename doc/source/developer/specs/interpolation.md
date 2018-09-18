# Requirements

- decide and run interpolate before request to `get_data` to minimize data on the wire, if possible
    - handle points/segments at the edge of a data source boundary
    - handle out-of-extents cases
- support most used geospatial temporal interpolation methods
- require minimal configuration for generic types of interpolation
- configuration works in python or in pipeline equally
- handle stacked and unstacked coordinates in the same pipelines
- specify priority for interpolators
- Each interpolator should know what coordinates it can interpolate to/from
- Each interpolator should know how to select appropriate coordinates from the datasource
Multiple interpolators may be required for each request:
    - Time could use NN interpolation
    - lat/lon could use bilinear with a specified CRS/Projection
- The order of these multiple interpolators matters from an optimization perpsective
    - Consider the size of the dataset before/after interpolation
    - Consider the cost of the interpolation operation
- **TODO**: support custom interpolator classes?
- **TODO**: support `+` (`__add__`) for Interpolators?

# Example Use cases

- user requests a single value at a point between coordinates in a datasource
- user requests an array of coordinates from a dataset from a dataset with a different coordinate system
- user requests data at coordinates that overlap the extent of the native dataset
- user requests a different type of interpolation for lat/long and time

# Specification

## Constants

- `INTERPOLATORS`: List
- `INTERPOLATION_OPTIONS`: List
    - ['nearest', 'nearest_preview', 'bilinear', 'cubic', 'cubic_spline', 'lanczos', 'average', 'mode', 'gauss', 'max', 'min', 'med', 'q1', 'q3']
    - Only include the supported interpolation options

## Utility methods

- `get_interpolator(interpolator<str>)`: Return interpolator class given a string shortname

## Interpolator Class

#### Constants

#### Traits

- `method`: one of:
    - str: Enum(`INTERPOLATION_OPTIONS`)
    - Dict({`dim`: Enum(`interpolate_options`)})
    - For all dims or single dims.
- `tolerance`: tl.CFloat(np.inf)
    + optional tolerance for specifying when to exclude interpolated data
    + Units?

#### Private members

- `_requested_coordinates` = tl.Instance(Coordinates, allow_none=True)
- `_source_coordinates` = tl.Instance(Coordinates)
- `_source_coordinates_index` = tl.List()
- `_source_data` = tl.Instance(UnitsDataArray)

**Cost Optimization**

- `cost_func`: tl.CFloat(-1)
    + rough cost FLOPS/DOF to do interpolation
- `cost_setup`: tl.CFloat(-1)
    + rough cost FLOPS/DOF to set up the interpolator
#### Methods

- `interpolate`: run the interpolator
    + `Interpolator` raises a `NotImplemented` if child does not overide
- `to_pipeline()`: export interpolator class to pipeline

## Implementations

### NearestNeighbor

### Rasterio

### Scipy




## InterpolatorException

- custom exception for interpolation errors

## User Interface

TODO

## Developer interface

TODO

