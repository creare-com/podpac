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
- user requests an array of coordinates from a dataset with a different coordinate system
- user requests data at coordinates that overlap the extent of the native dataset
- user requests a different type of interpolation for lat/long and time


# User Interface

**DataSource** usage (primary):

```python

# specify method
DataSource(... interpolation='nearest')

# specify dict of methods for each dimension
# value can be a method string, or a tuple which overrides the default method InterpolationMethods
DataSource(... interpolation={
        'lat': 'bilinear',
        'lon': 'bilinear',
        'time': ('nearest', [Nearest, Rasterio])
    })

# specify an interpolation class itself (useful when you need to override args to Interpolators)
DataSource(... interpolation=Interpolation() )
```

## `Interpolation`

Used to organize multiple interpolators across the dimensions of a DataSource

```python
# definition generally comes from DataSource `interpolation`
# **kwargs will get passed on to interpolators
Interpolation(definition, coordinates, **kwargs)

# simple string definition applies to all dimensions
# this string must be a member of INTERPOLATION_SHORTCUTS
Interpolation('nearest', coordinates)

# more complicated specify a tuple with a method name and the order of Interpolators to use this method with
# the method string in the tuple does not necessarily have to be a member of INTERPOLATION_SHORTCUTS
Interpolation( ('nearest', [Rasterio, Nearest]), coordinates)

# more complicated dict definition specifies interpolators for each dimension
Interpolation({
    ('lat', 'lon'): 'bilinear',
    }, coordinates)

# most complicated dict definition specifies tuple interpolators for dimensions
Interpolation({
    'lat': 'nearest',
    'lon': ('nearest', [Rasterio, ...])
    'time': ('nearest', [Nearest, ...])
    }, coordinates)

# most complicated dict definition specifies tuple interpolators for dimensions
Interpolation({
    ('lat', 'lon'): 'nearest',
    'time': ('nearest', [Nearest, ...])
    }, coordinates)


# can include kwargs that get passed on to Interpolator methods
Interpolation({
    'lat': 'nearest',
    'lon': ('nearest', [Rasterio, ...])
    'time': ('nearest', [Nearest, ...])
    }, coordinates, tolerance=1, arg='for interpolator')
```


## `Interpolator`

Create **Interpolator** classes that can be assigned in **Interpolation** definitions.

Examples: `NearestNeighbor`, `Rasterio`, `Scipy`

```python

class MyInterpolator(Interpolator):
    """ 
    class has traits 
    """

    # method = tl.Unicode() defined by Interpolator base class
    tolerance = tl.Int()
    kwarg = tl.Any()

    init(self):
        # act on inputs after traits have been set up
    
    validate(self, requested_coordinates, source_coordinates):
        # validate requested coordinates and source_coordinates can 
        # be interpolated with`self.method`
    
    select_coordinates(self, requested_coordinates, source_coordinates, source_coordinates_index):
        # down select coordinates (if valid) based on `self.method`
    
    interpolate(source_coordinates, source_data, requested_coordinates, output):
        # interpolate data (if valid) based on `self.method`
```






# Specification

## Constants

- `INTERPOLATION_METHODS`: dict of shortcut: InterpolationMethod class
- `INTERPOLATION_SHORTCUTS`: List
    - Only include the supported interpolation options
- `INTERPOLATION_DEFAULT`: 'nearest', interpolation method if none is specified to datasource or if dimensions are left out of the dict

## Utility methods

## `Interpolator` Abstract Class

This is a traits based class since users may be expected to define Interoplator subclasses

#### Traits

- `method`: tl.Unicode() - current interpolation method name


#### Methods

- `validate(requested_coordinates, source_coordinates)`
    + check to see if the Interpolator supports the requested/soruce coordinate pair for the current method
    + `InterpolationMethod` raises a `NotImplemented` if child does not overide
- `interpolate(source_coordinates, source_data, requested_coordinates, requested_data)`
    + `InterpolationMethod` raises a `NotImplemented` if child does not overide
- `select_coordinates(requested_coordinates, source_coordinates, source_coordinates_idx)`
    + `InterpolationMethod` raises a `NotImplemented` if child does not overide

### `NearestNeighbor`

### `NearestPreview`

- can select coordinates

### `Rasterio`

### `Scipy`



## `Interpolation` Class

#### Constructor

- `__init__(definition, coordinates, **kwargs)`:
    + `definition`: (str, dict, tuple)
    + `coordinates`
    + kwargs will get passed through the Interpolator classes

#### Members

#### Private members


- `_definition`: dict { dim: ('method', [Interpolator]) }


**Cost Optimization**

TODO: figure out where to implement optimization steps
- `cost_func`: tl.CFloat(-1)
    + rough cost FLOPS/DOF to do interpolation
- `cost_setup`: tl.CFloat(-1)
    + rough cost FLOPS/DOF to set up the interpolator

#### Methods

- `interpolate(source_coordinates, source_data, requested_coordinates, requested_data)`: run the interpolator
- `select_coordinates(requested_coordinates, source_coordinates, source_coordinates_idx)`: interpolate child coordinates
- `to_pipeline()`: export interpolator class to pipeline
- `from_pipeline()`: create interpolator class from pipeline


## InterpolatorException

- custom exception for interpolation errors





