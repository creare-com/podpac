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

## Coordinates for One Dimension

 * Coordinate values are either `float` or `np.datetime64`
 * Coordinate deltas are either `float` or `np.timedelta64`
 * When input:
   - numerical values and deltas are cast to `float`
   - string values are parsed with `np.datetime64`
   - string deltas are split and then parsed by `np.timedelta64` (e.g. `'1,D'` -> `np.timedelta(1, 'D')`)

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
 - `[]`

### Coordinates1d(BaseCoordinates1d)

A 1d array of coordinates.

Constructor:
 - `Coordinates1d()`: empty coordinates
 - `Coordinates1d(value)`: a singleton array with one coordinate
 - `Coordinates1d(values)`: an array of coordinates

Alternate Constructors:
 - `Coordinates1d.from_xarray(xcoords)`: Create coordinates from an xarray dimension (a named DataArray)

Traits:
- `name`: Enum('lat', 'lon', 'time', 'alt')
- `units`: Units
- `coord_ref_sys`: string, default: 'WGS84'
- `ctype`: Enum('point', 'left', 'rigth', 'midpoint'), default: 'midpoint'
- `extents`: array, shape (2,), optional

Properties
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

Methods
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
 - `len`: size
 - `[]`: index the coordinate values
    - returns a new Coordinates1d object
    - supports an integer, a slice, an index array, or a boolean array
 - `+`, `+=`: either concat Coordinates1d or add value
 - `-`, `-=`: subtract value
 - `&`: intersect

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

### Shorthand

For ease-of-use, the following aliases will be available in the toplevel `podpac` package:
 
 - `_ca`: Coordinates1d
 - `_cu`: UniformCoordinates1d

16 shortcut functions *may* also be defined, e.g.

 - `_ca_lat`
 - `_cu_time`

So that the following are equivalent
```
UniformCoordinates1d(0, 1, 0.1, name='lat')
_cu(0, 1, 0.1, name='lat')
_cu_lat(0, 1, 0.1)
```

## Stacked Coordinates

### StackedCoordinates(BaseCoordinates1d)

Coordinates for two or more physical dimensions that are indexed together (aligned, as opposed to defining a grid). This class should be considered an implementation detail that behaves like a tuple (is iterable) but also facilitates a common interface with Coordinates1d by mapping indexing and other methods to its Coordinates1d objects.

Properties
- `name`: dimension names joined by an underscore, e.g. `'lat_lon'`
- `dims`: tuple of dimension names
- `coordinates`: read-only DataArray with MultiIndex
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

Operators
 - `len`: size
 - `[]`: index each of coordinates values
    - returns a new StackedCoordinates object
    - supports an integer, a slice, an index array, or a boolean array

### Helper Functions

The `stacked_linspace` helper function returns a uniformly spaced line as StackedCoordinates, so that the following examples are equivalent

```
stacked_linspace((0, 0, 20), (1, 1, 100), 100, names=('lat', 'lon', 'alt'))
```

```
lat = UniformCoordinates1d(0, 1, size=100), name='lat')
lon = UniformCoordinates1d(0, 1, size=100), name='lon')
alt = UniformCoordinates1d(20, 100, size=100), name=alt')
StackedCoordinates(lat, lon, alt)
```

### Shorthand

For ease-of-use, the following aliases will be defined in the toplevel podpac package:
 
 - `_stacked`: StackedCoordinates
 
So that the above can be rewritten:

```
_stacked(_cu(0, 1, 100, name='lat'), _cu(0, 1, 100, name='lon'), _cu(20, 100, 100, name='alt'))
_stacked(_culat(0, 1, 100), _culon(0, 1, 100)), _cualt(20, 100, 100))
```

## Convenience Functions

# crange

```
podpac.crange(0, 2.5, 0.5)
podpac.crange('2018-01-01', '2018-01-10', '2,D')
podpac.crange(np.datetime64('2018-01-01'), np.datetime64('2018-01-10'), np.timedelta64(2, 'D'))
```

 - Similar to np.arange, but
   - contains the stop value if it falls exactly on a step
   - supports time coordinates, either datetime64/timedelta64 or strings
 - Under the hood, this is implemented by mapping directly to `UniformCoordinates1d(start, stop, step)`

# clinspace

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

- `default_coord_ref_sys`: Unicode
- `default_distance_units`: Units
- `default_ctype`: Enum('left', 'right', 'midpoint', 'point')

### Properties

 - `ndim`: Int
 - `dims`: tuple(str, str, ...)
 - `shape`: tuple(int, int, ...)
 - `coords`: xarray.core.DataArrayCoordinates type (dictionary-like mapping dimension names to named DataArrays) 
 - TODO


For stacked dimensions, the dimension names are combined by an underscore, and the coordinates are implemented as a pandas MultiIndex:

```
coords.dims -> ('lat_lon', 'time')
coords.coords1d.keys() -> ('lat_lon', 'time')
coords.coords1d['lat_lon'] -> (UniformCoordinates1d, UniformCoordinates1d) # this is actually a StackedCoordinates object
coords.coords1d['lat'] -> KeyError
coords.coords['lat_lon'] -> DataArray with MultiIndex
coords.coords['lat'] -> DataArray # we get this for free due to the MultiIndex
```

### Methods

In general, methods will return a new Coordinates object by default, with an option to modify the Coordinates in-place.

 * `add`: TODO
 * `concat(other)`: TODO
 * `select(bounds)`: TODO
 * `intersect(other):` TODO
 * `drop_dims(dims)`: remove dimensions
   - `dims` may be a dimension name (str) or a list of dimension names
   - stacked dimensions can be removed together or individually
    
    ```
    # the folowing are equivalent
    coords.drop_dims('lat_lon')
    coords.drop_dims(['lat', 'lon'])
    
    # drop part of a stacked dimension
    coords.drop_dims('lat')
    ```

 * `add_dim`
 * `unstack`
 * `stack`

### Operators

 * `[]`: Get or set the Coordinates1d object for the given dimension; checks the size when setting coordinates for a stacked dimension

   ```
   coords.dims -> ('lat_lon', 'time')
   coords['lat'] -> UniformCoordinates1d
   coords['lat_lon'] -> KeyError
   coords['lat'] = UniformCoordinates1d(...) # modify the coordinates (or raise an exception for size mismatch)

 * `[dim] = <Coordinates1d>`: Set the coordinates for this dimension. If this dimension is stacked, the size is checked.


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
