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

- `INTERPOLATION_METHODS`: dict of shortcut: InterpolationMethod class
- `INTERPOLATION_SHORTCUTS`: List
    - Only include the supported interpolation options
- `TOLERANCE_DEFAULTS`: dict of deafult tolerance for each dim
    + TODO: how to handle units?

## Utility methods





## InterpolationMethod Class

#### Constants

#### Traits


#### Private members
#### Methods

- `interpolate(source_coordinates, source_data, requested_coordinates, requested_data)`
    + `InterpolationMethod` raises a `NotImplemented` if child does not overide
- `select_coordinates(requested_coordinates, source_coordinates, source_coordinates_idx)`
    + `InterpolationMethod` raises a `NotImplemented` if child does not overide

#### Private Methods


## Interpolator Class

#### Constructor

- `__init__(definition, coordinates, default_method=, tolerance=)`:
    + `definition`: (InterpolationMethod, str, dict, list)
    + `source_coordinates`
    + `default_method`: (InterpolationMethod, str)

#### Members

#### Private members

- `_tolerance`: dict
    + optional tolerance for specifying when to exclude interpolated data
    + Units?
- `_definition`: dict { dim: InterpolationMethod }
- `_source_coordinates` = tl.Instance(Coordinates)
- `_requested_coordinates` = tl.Instance(Coordinates, allow_none=True)
- `_source_coordinates_index` = tl.List()
- `_source_data` = tl.Instance(UnitsDataArray) 

**Cost Optimization**

- `cost_func`: tl.CFloat(-1)
    + rough cost FLOPS/DOF to do interpolation
- `cost_setup`: tl.CFloat(-1)
    + rough cost FLOPS/DOF to set up the interpolator

#### Methods

- `interpolate(source_coordinates, source_data, requested_coordinates, requested_data)`: run the interpolator
- `select_coordinates(requested_coordinates, source_coordinates, source_coordinates_idx)`: interpolate child coordinates
- `to_pipeline()`: export interpolator class to pipeline
- `from_pipeline()`: create interpolator class from pipeline

#### Private Methods

- `_parse_interpolation_definition(definition, coordinates, default_method)`:
- `_parse_tolerance(tolerance, coordinates)`:
- `_parse_interpolation_methods(definition)`: 
    + variable input definition (str, InterpolationMethod) returns an InterpolationMethod
- `_validate_interpolation_method(method)`:
- `_set_interpolation_method(dim, definition)`:
    + set the InterpolationMethod to be associated iwth the current dimension
    + if `dim` is stacked, split it up and run `_set_interpolation_method` for each part independently
    + store a record that `dim` was stacked
- `_set_interpolation_tolerance(dim, tolerance)`:




## Implementations

### NearestNeighbor

### NearestPreview

- can select coordinates

### Rasterio

### Scipy


## InterpolatorException

- custom exception for interpolation errors

## User Interface

TODO

## Developer interface

TODO

