# Requirements
* Support lat, lon, alt, and time dimensions
* Support arbitrary stacking of coordinates (eg. lat_lon lat_time, lat_lon_time, alt_time, etc.)
* Support dimensions for coordinates, but allow overwriting to ignore dimensions
* Support different coordinate reference systems for lat, lon dimensions
* Support arbitrary start, end, steps that get filled with the native coordinates when requested
* Support intersection of coodinates
* Support addition of coordinates
* Support dropping coordintes
* Support replacement of coordinates for certain dimensions
* Support sub-selection of coordinates based on index
* Support multiple types of calculated, and explicitly labelled coordinates
    * This includes:
        * Explicit lists of coordinates
        * Uniform or linspace coordinates
        * Rotated coordinates
        * Mapped coordinates (mapping based on i/j(/k/l) index of data array)

# Example Use cases
1. I want to create a regularly spaced set of (lat, lon) coordinates
2. I want to create an irregularly spaced set of (time) coordinates
3. I want to create a rotated lat,lon coordinate system
4. I want to create a regularly spaced line of (lat_lon) coordinates
5. I have a set of native_coordinates, and I want to sub-select based on a window of coordinates. The step size of the native_coordinates should be preserved
6. I have a set of native_coordinates, but I only want every 5th element in the time dimension
7. I have a set of native_coordinates, but I want to replace the coordinates of the time dimension with a different coordinate
8. I have a set of (lat_lon) stacked coordinates but I want a new set of coordinates that describe a box containing all of these (lat_lon) coordinates -- this should be at the resolution of the lat_lon coordinates
9. I want to create (lat,lon) coordinates in UTM-feet using zone T18 with the NAD87 CRS
10. I want the intersection of the (lat,lon) coordinates in UTM-feet using zone T18 with the NAD87 CRS with another coordinate system using CRS WGS84 in geodetic coordinates. 
11. I want to specify a single (lat, lon) coordinate represented as a single point. Intersections with another coordinate should only give results at the point. 
12. I want to specify a single (lat, lon) coordinate representing an area of a certiain size (dlat, dlon). Intersections with another coordinate will give resoluts over this area. 
13. TODO: 

# Specification

## General

 * Coordinate values are either `float` or `np.datetime64`
 * Coordinate deltas are either `float` or `np.timedelta64`
 * When input:
   - numerical values and deltas are cast to `float`
   - string values are parsed with `np.datetime64`
   - string deltas are split and then parsed by `np.timedelta64` (e.g. `'1,D'` -> `np.timedelta(1, 'D')`)

## Coordinates for One Dimension

### BaseCoordinates1d

`BaseCoordinates1d` is the base class for Coordinates1d and StackedCoordinates.

Common Attributes:
 - `name`
 - `dims`
 - `size`
 - `is_monotonic`
 - `is_uniform`

Common Methods:
 - `select(bounds, outer=False)`
 - `intersect(other, outer=False)`

Common Operators:
 - `len`
 - `[]`: supports integer, slice, index array, or boolean array

### Coordinates1d(BaseCoordinates1d)

Base class for a singe dimension of coordinates.

Common Traits:
- `name`: Enum('lat', 'lon', 'time', 'alt')
- `units`: Units
- `coord_ref_sys`: string, default: 'WGS84'
- `ctype`: Enum('point', 'left', 'rigth', 'midpoint'), default: 'midpoint'
- `extents`: array, shape (2,), optional

Common Properties
- `coordinates`: read-only array
- `properties`: dictionary of coordinate properties
- `dtype`: `np.datetime64` or `float`
- `size`: int
- `bounds`: read-only array, `[float, float]`. Coordinate values min and max.
- `area_bounds`: read-only array, `[float, float]`.
  - For point coordinates, this is just the `bounds`.
  - For segment coordinates, use `extents` when available, otherwise calculated depending on the ctype mode.
- `is_monotonic`: bool
- `is_descending`: bool
- `is_uniform`: bool

Common Methods
 - `select(bounds, outer=False)`: select coordinates within the given bounds
    - returns a new Coordinates1d
    - if `outer` is true, the coordinates just outside the bounds are returned
 - `intersect(other, outer=False)`: intersect these coordinates with other coordinates
    - returns a new Coordinates1d
    - if `other` is Coordinates1d, raises an exception if `other.name != self.name`
    - if `other` is StackedCoordinates or Coordinates, intersects with `other[self.name]`
 - `add`: add delta value to each coordinate value
    - returns a new Coordinates1d object by default, or can be modified in-place
 - `concat(other)`: concatenate additional coordinates
    - returns a new Coordinates1d object by default, or can be used in-place
    - raises an exception if `other.name != self.name`
    - *not sure we need this...*

Operators
 - `+`, `+=`: wraps add
 - `-`, `-=`: wraps add

### ArrayCoordinates1d(Coordinates1d)

A 1d array of coordinates.

Constructor:
 - `ArrayCoordinates1d()`: empty coordinates
 - `ArrayCoordinates1d(value)`: a singleton array with one coordinate
 - `ArrayCoordinates1d(values)`: an array of coordinates

Alternate Constructors:
 - `ArrayCoordinates1d.from_xarray(xcoords)`: Create coordinates from an xarray dimension (a named DataArray)
 
Traits:
 - `coords`: array

### UniformCoordinates1d(Coordinates1d)

Uniformly-spaced coordinates, parameterized by a start, stop, and step.

Constructor
 - `UniformCoordinates1d(start, stop, step)`
 - `UniformCoordinates1d(start, stop, step=step)`
 - `UniformCoordinates1d(start, stop, size=N)`

Alternate Constructors
 - `UniformCoordinates1d.from_tuple(items)`: items is either (start, stop, step) or (start, stop, size)

Traits:
 - start: float or datetime64
 - stop: float or datetime64
 - step: float or timedelta64

### StackedCoordinates(BaseCoordinates1d)

Coordinates for two or more physical dimensions that are indexed together (aligned, as opposed to defining a grid). This class should be considered an implementation detail that behaves like a tuple (is iterable) but also facilitates a common interface with Coordinates1d by mapping indexing and other methods to its Coordinates1d objects.

Properties
- `name`: dimension names joined by an underscore, e.g. `'lat_lon'`
- `dims`: tuple of dimension names
- `coordinates`: pandas.MultiIndex
- `size`: int
- `is_monotonic`: if all of its dimensions are monotonic
- `is_monotonic`: if all of its dimensions are uniform

Methods
 - `select(bounds)`: select coordinates within the given bounds
    - returns a new StackedCoordinates object
    - TODO: how are bounds defined? is this necessary
 - `intersect(other)`: intersect these coordinates with other coordinates
    - outer=False, intersection of intersect in each dimension
    - outer=True, union of intersection in each dimension
 - `concat(other)`: concatenate additional coordinates
    - returns a new StackedCoordinates object by default, or can be used in-place
    - raises an exception if `other.name != self.name`
    - *not sure we need this...*

## Convenience Functions

### crange

```
podpac.crange(0, 2.5, 0.5)
podpac.crange('2018-01-01', '2018-01-10', '2,D')
podpac.crange(np.datetime64('2018-01-01'), np.datetime64('2018-01-10'), np.timedelta64(2, 'D'))
```

 - Similar to np.arange, but
   - contains the stop value if it falls exactly on a step
   - supports time coordinates, either datetime64/timedelta64 or strings
 - Under the hood, this is implemented by mapping directly to `UniformCoordinates1d(start, stop, step)`

### clinspace

```
podpac.clinspace(0, 2.5, 5)
podpac.clinspace('2018-01-01', '2018-01-09', 5)
podpac.clinspace(np.datetime64('2018-01-01'), np.datetime64('2018-01-09'), 5)
podpac.clinspace([0, 1], [2.5, 20], 5)
podpac.clinspace([0, 1, '2018-01-01'], [2.5, 20, '2018-01-09'], 5)
```

 - Similar to np.linspace, but
   - supports time coordinates, either datetime64 or strings
   - supports stacked coordinates
 - Under the hood, this is implemented by mapping to `UniformCoordinates1d(start, stop, size=N)` and `StackedCoordinates`

### Shorthand/aliases

For ease-of-use, the following aliases will be available in the toplevel `podpac` package:
 
 - `_ca`: `ArrayCoordinates1d`
 - `_cu`: `UniformCoordinates1d`
  - `_stacked`: `StackedCoordinates`
 
16 shortcut functions *may* also be defined, e.g. `_ca_lat`, `_cu_time`, etc

So that the following single coordinates are equivalent

```
UniformCoordinates1d(0, 1, 0.1, name='lat')
podpac._cu(0, 1, 0.1, name='lat')
podpac._cu_lat(0, 1, 0.1)
podpac.crange(0, 1, 0.1, name='lat')

# these are also functionally equivalent to the above
ArrayCoordinates1d(np.arange(0, 1.1, 0.1), name='lat')
podpac._ca(np.arange(0, 1.1, 0.1), name='lat')
podpac._ca_lat(np.arange(0, 1.1, 0.1))
```

And the following stacked coordinates are equivalent

```
StackedCoordinates([
    UniformCoordinates1d(0, 1, size=100, name='lat'),
    UniformCoordinates1d(0, 1, size=100, name='lon'),
    UniformCoordinates1d('2018-01-01', '2018-01-10', size=10, name='time')])
    
podpac._stacked([
    podpac._cu(0, 1, size=10, name='lat'),
    podpac._cu(0, 1, size=10, name='lon'),
    podaac._cu('2018-01-01', '2018-01-10', size=10, name='time')])
    
podpac._stacked([
    podpac._cu_lat(0, 1, size=100),
    podpac._cu_lon(0, 1, size=100)),
    podpac._cu_time('2018-01-01', '2018-01-10', size=10)])
    
podpac.clinspace((0, 0, '2018-01-01'), (1, 1, '2018-01-10'), 10, dims=['lat', 'lon', 'time'])
```

## Multidemensional Coordinates

### Coordinate Creation

Coordinates are created from a list or dict containing BaseCoordinates1d objects (Coordinates1d or StackedCoordinates).

 - `Coordinates()`
 - `Coordinates([coords1d, coords1d])`
 - `Coordinates([StackedCoordinates([coords1d, coords1d]), coords1d])`
 - `Coordinates([(coords1d, coords1d), coords1d])`
 - `Coordinates([array1d, array1d], dims=['lat', 'lon'])`
 - `Coordinates([(array1d, array1d), array], dims=['lat_lon', 'time'])`
 - `Coordinates([array2d, array1d], dims=['lat_lon', 'time'])`

### Alternate Constructors

 - `Coordinates.from_xarray(xcoords)`: maps multi-dimensional xarray `DataArrayCoordinates` to podpac `Coordinates`
 - `Coordinates.grid(...)`
 - `Coordinates.points(...)`

### Traits

- `coord_ref_sys`: Unicode
- `default_distance_units`: Units
- `default_ctype`: Enum('left', 'right', 'midpoint', 'point')

### Properties

 - `dims`: tuple(str, str, ...)
 - `shape`: tuple(int, int, ...)
 - `ndim`: int
 - `size`: int
 - `udims`: tuple(str, str, ...), "unstacked"
 - `coords`: `xarray.core.DataArrayCoordinates`

### Methods

In general, methods will return a new Coordinates object by default, with an option to modify the Coordinates in-place.

 * `keys()`: return dims, stacked
 * `values()`: returns BaseCoordinates1d, stacked
 * `items()`: zips keys and values
 * `get(key, default=None)`: wraps [] with fallback
 * `add`: TODO
 * `concat(other)`: TODO
 * `intersect(other, outer=False)`: maps intersection to each dimension, returns new Coordinates object
 * `drop(dims)`: remove dimensions, stacked dimensions are removed together
 * `udrop(dims)`: remove dimensions, stacked dimensions can be removed individually
 * `transpose`: TODO
 * `iterchunks`: TODO

### Operators

 * `[dim]`: Get the BaseCoordinates1d object for the given dimension, stacked or unstacked
 * `[dim] = <BaseCoordinates1d>`: Set the coordinates for this dimension.
   - If the dimension is part of stacked dimensions, raises an exception (*we could change this to allow setting part of stacked coordinates and just validate that the size is the same*)
   - If the dimension is missing, raises an exception (*we could change this to add dimensions*)

TODO: `coords['lat_lon']['lat'] = ArrayCoordinates(...)` vs `coords[lat] = ArrayCoordinates(...)`

### Example

```
lat = ArrayCoordinates1d(np.arange(10))
lon = ArrayCoordinates1d(np.arange(10))
time = ArrayCoordinates1d(np.range(4))
lat_lon = StackedCoordinates(lat, lon)
coords = Coordinates([lat_lon, time])

coords.dims -> ('lat_lon', 'time')
coords.shape -> (10, 4)
coords.ndim -> 2
coords.size -> 40
coords.udims -> ('lat', 'lon', 'time')
coords.keys() -> ('lat_lon', 'time')
coords.values() -> (lat_lon, time)
coords.items() -> (('lat_lon', lat_lon), ('time', time))
coords['lat_lon'] -> lat_lon
coords['time'] -> time
coords['lat'] -> lat
coords['alt'] -> KeyError
coords.get('alt') -> None
len(coords) -> 2
coords.drop('time') -> Coordinates with only lat_lon
coords.drop('lat_lon') -> Coordinates with only time
coords.drop('alt') -> KeyError
coords.drop('lat') -> KeyError
coords.drop(['time', 'lat_lon']) -> empty Coordinates
coords.drop(['time', 'alt'], ignore_missing=True) -> Coordinates with only lat_lon
coords.udrop('lat') -> Coordinates with only time and lon
```

### Miscellaneous Examples


Some equivalent ways to copy:

```
coords_copy = Coordinates(other) # I'm not sure we need this one
coords_copy = Coordinates(other.coords1d)
coords_copy = Coordinates(other.coords1d.values())
```

Select specific dimensions

```
dims = ['lat', 'lon']
c1 = Coordinates([other.coords1d[dim] for dim in dims])
```

Downsample (even if some dimensions are stacked)

```
c2 = Coordinates([c[::10] for c in other.coords1d.values()])
```

Downsample only the time dimension (only works if time is not stacked)

```
d = other.coords1d.copy()
d['time'] = d['time'][::10]
c3 = Coordinates(d)
```

The safe way would would be:

```
d = other.coords1d.copy()
k = d.get('time')
d[k] = d[k][::10]
c3 = Coordinates(d)
```
