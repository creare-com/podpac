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

### Coordinates1D(BaseCoordinates1D)

`Coordinates1D` is the base class for coordinates in a single dimension and defines the common interface.

Traits:
- `name`: Enum('lat', 'lon', 'time', 'alt'), required, *read-only*
- `units`: Units, *read-only*
- `coord_ref_sys`: Unicode, *read-only*
- `ctype`: Enum('segment', 'point'), default: 'segment' for numerical, 'point' for datetime, *read-only*
- `segment_position`: Float, default: 0.5 for 'segment' and None for 'point', *read-only*
- `extents`: shape (2,), optional, *read-only*

Properties
- `properties`: dictionary of coordinate properties
- `coordinates`: read-only array
- `dtype`: `np.datetime64` or `np.float64`
- `size`: Int
- `bounds`: read-only array, [Float, Float]. Coordinate values min and max.
- `area_bounds`: read-only array, [Float, Float]. For point coordinates, this is just the `bounds`. For segmentment coordinates, use `extents` when available, otherwise calculated from the segment position.
- `is_monotonic`: Boolean
- `is_descending`: Boolean

Methods
 - `select(bounds)`: select coordinates within the given bounds
 	- returns a new Coordinates1D object by default, or can be modified in-place
 - `intersect(other)`: intersect these coordinates with other coordinates
 	- returns a new Coordinates1D object by default, or can be modified in-place
 	- if `other` is Coordinates1D, raises an exception if `other.name != self.name`
    - if `other` is StackedCoordinates or Coordinates, intersects with `other[self.name]` or return empty coordinates if `other[self.name] does not exist *(TODO or raise exception?)*
 - `add`: add delta value to each coordinate value
    - returns a new Coordinates1D object by default, or can be modified in-place
 - `concat(other)`: concatenate additional coordinates
    - returns a new Coordinates1D object by default, or can be used in-place
    - raises an exception if `other.name != self.name`
 	- *not sure we need this...*

Operators
 - `len`: size
 - `[]`: index the coordinate values
    - returns a new Coordinates1D object
    - supports a single index, a slice, an index array, or a boolean array
 - `+`, `+=`: either concat Coordinates1D or add value
 - `-`, `-=`: subtract value
 - `&`: intersect

### ArrayCoordinates1D(Coordinates1D)

An arbitrary list of coordinates, where `values` is array-like. The `ctype` must be 'point'.

```
ArrayCoordinates1D(values, ...)
```

Notes:
 - `ctype` must be 'point'

### MonotonicCoordinates1D(ArrayCoordinates1D)

A sorted list of coordinates, where `values` is array-like and sorted (in either direction).

```
MonotonicCoordinates1D(values, ...)
```
 
Notes:
 - *TODO `area_bounds` calculation for segment coordinates without explicit extents*

### UniformCoordinates1D(Coordinates1D)

Uniformly-spaced coordinates, defined by a start, stop, and step/size.

```
UniformCoordinates1D(start, stop, step=None, size=None, ...)
```

Notes:
 - use `segment_position` and `step` for the `area_bounds` calculation for segment coordinates without explicit extents

### Shorthand

For ease-of-use, the following aliases will be defined:
 
 - `_ca`: ArrayCoordinates1D
 - `_cm`: MonotonicCoordinates1D
 - `_cu`: UniformCoordinates1D
 - `_cl`: LinspaceCoordinates1D

The named versions proposal was rejected (`_calat`, `_cutime`, ...)

Thus the following are equivalent

```
UniformCoordinates1D(0, 1, 0.1, name='lat')
_cu(0, 1, 0.1, name='lat')
_culat(0, 1, 0.1) # rejected
```

## Stacked Coordinates

### StackedCoordinates(BaseCoordinates1D)

Coordinates for two or more physical dimensions that are indexed together (aligned, as opposed to defining a grid). This class should be considered an implementation detail that behaves like a tuple (is iterable) but also facilitates a common interface with Coordinates1D by mapping indexing and other methods to its Coordinates1D objects.

Properties
- `name`: dimension names joined by an underscore, e.g. `'lat_lon'`
- `coordinates`: read-only DataArray or MultiIndex
- `size`: Int

Methods
 - `select(bounds)`: select coordinates within the given bounds
 	- returns a new StackedCoordinates object by default, or can be modified in-place
    - TODO: how are bounds defined?
 - `intersect(other)`: intersect these coordinates with other coordinates
    - TODO
 - `concat(other)`: concatenate additional coordinates
 	- returns a new StackedCoordinates object by default, or can be used in-place
 	- raises an exception if `other.name != self.name`
 	- *not sure we need this...*

Operators
 - `[]`: index each of coordinates values
    - returns a new StackedCoordinates object
    - supports a single index, a slice, an index array, or a boolean array

### Helper Functions

The `stacked_linspace` helper function returns a uniformly spaced line as StackedCoordinates, so that the following examples are equivalent

```
stacked_linspace((0, 0, 20), (1, 1, 100), 100, names=('lat', 'lon', 'alt'))
```

```
lat = UniformCoordinates1D(0, 1, size=100), name='lat')
lon = UniformCoordinates1D(0, 1, size=100), name='lon')
alt = UniformCoordinates1D(20, 100, size=100), name=alt')
StackedCoordinates(lat, lon, alt)
```

### Shorthand

For ease-of-use, the following aliases will be defined (on of these options):
 
 - `_stacked`: StackedCoordinates
 
So that the above can be rewritten:

```
_stacked(_cu(0, 1, 100, name='lat'), _cu(0, 1, 100, name='lon'), _cu(20, 100, 100, name='alt'))
_stacked(_culat(0, 1, 100), _culon(0, 1, 100)), _cualt(20, 100, 100)) # rejected
```
 
## Multidemensional Coordinates

### Coordinate Creation

Coordinates are created from a list or dict containing BaseCoordinates1D objects (Coordinates1D or StackedCoordinates).

```
Coordinates(values, **traits)
```

For example:

```
c1 = Coordinates([lat, lon, time], **traits)
c2 = Coordinates([StackedCoordinates(lat, lon), time], **traits)
```

When `values` is a dict, dimension names are mapped to the BaseCoordinates1D objects. *Note: in Python <= 3.5, an OrderedDict is required.*

```
c1 = Coordinates({'lat': lat, 'lon': lon, 'time': time}, **traits)
c2 = Coordinates({'lat_lon': Stacked(lat, lon), 'time': time}, **traits)
```

### Implicit Stacking

Explicit stacked dimension creation is recommended (`StackedCoordinates` or `stacked_linspace`) as it makes stacking more explicit, but we may wish to support stacking implicitly:

```
c1 = Coordinates([(lat, lon), time], **traits)
c2 = Coordinates({'lat_lon': (lat, lon), 'time': time}, **traits)
```

### Traits

- `coord_ref_sys`: Unicode
- `distance_units`: Units
- `segment_position`: Float
- `ctype`: Enum('segment', 'point')

*TODO: defaults, passing on to Coordinates1D, overriding with Coordinates1D*

### Properties

 * `ndim`: Int
 * `dims`: tuple(str, str, ...)
 * `shape`: tuple(int, int, ...)
 * `coords`: OrderedDict mapping dimension names to Coordinate1D or StackedCoordinates
 * `coordinates`: xarray Coordinates (dictionary-like mapping dimension names to named DataArrays) 

For stacked dimensions, the dimension names are combined by an underscore, and the coordinates are implemented as a pandas MultiIndex:

```
coords.dims -> ('lat_lon', 'time')
coords.coords.keys() -> ('lat_lon', 'time')
coords.coords['lat_lon'] -> (UniformCoordinates1D, UniformCoordinates1D) # this is actually a StackedCoordinates object
coords.coords['lat'] -> KeyError
coords.coordinates['lat_lon'] -> DataArray with MultiIndex
coords.coordinates['lat'] -> DataArray # we get this for free due to the MultiIndex
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

 * `[]`: Get or set the Coordinates1D object for the given dimension; checks the size when setting coordinates for a stacked dimension

   ```
   coords.dims -> ('lat_lon', 'time')
   coords['lat'] -> UniformCoordinates1D
   coords['lat_lon'] -> KeyError
   coords['lat'] = UniformCoordinates1D(...) # modify the coordinates (or raise an exception for size mismatch)

 * `[dim] = <Coordinates1D>`: Set the coordinates for this dimension. If this dimension is stacked, the size is checked.


### Miscellaneous Examples


Some equivalent ways to copy:

```
coords_copy = Coordinates(other) # I'm not sure we need this one
coords_copy = Coordinates(other.coords)
coords_copy = Coordinates(other.coords.values())
```

Select specific dimensions

```
dims = ['lat', 'lon']
c1 = Coordinates([other.coords[dim] for dim in dims])
```

Downsample (even if some dimensions are stacked)

```
c2 = Coordinates([c[::10] for c in other.coords.values()])
```

Downsample only the time dimension (only works if time is not stacked)

```
d = other.coords.copy()
d['time'] = d['time'][::10]
c3 = Coordinates(d)
```

The safe way would would be:

```
d = other.coords.copy()
k = d.get_dim('time')
d[k] = d[k][::10]
c3 = Coordinates(d)
```
