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

## Utility methods

- `get_interpolator(interpolator<str>)`: Return interpolator class given a string shortname

## Interpolator Class

#### Constants

- `interpolate_options`:
    - Enum('nearest', 'nearest_preview', 'bilinear', 'cubic', 'cubic_spline', 'lanczos', 'average', 'mode', 'gauss', 'max', 'min', 'med', 'q1', 'q3'), 
    - Default: `nearest`
    - Only include the supported interpolation options

#### Traits

- `method`: one of:
    - str: Enum(`interpolate_options`)
    - Dict({`dim`: Enum(`interpolate_options`)})
    - For all dims or single dims.
- `tolerance`: 

**Cost Optimization**

- `cost_func`
- `cost_setup`
- 
#### Methods

- `interpolate`: run the interpolator
    + `Interpolator` raises a `NotImplemented` if child does not overide

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

