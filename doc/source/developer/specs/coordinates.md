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

 * Coordinate values are either float or np.datetime64
 * Coordinate deltas are either float or np.timedelta64
 * When input:
   - numerical values and deltas are cast to float
   - string values are parsed with np.datetime64
   - string deltas are split by `,` and then parsed by np.timedelta64 (e.g. `'1,D'` -> `np.timedelta(1, 'D')`)

### Coordinates1D

Traits:
- name: Enum('lat', 'lon', 'time', 'alt') *Required*
- units: Units
- coord_ref_sys: Unicode
- ctype: Enum('segment', 'point')
- segment_position: Float
- extents: array, [Float, Float]

Properties
- coordinates: read-only array
- dtype: `np.datetime64` or `np.float64`
- size: Int
- bounds: read-only array, [Float, Float]
- area_bounds: read-only array, [Float, Float]
- is_datetime
- is_monotonic
- is_descending
- rasterio_regularity
- scipy_regularity

Methods
 - select(bounds): returns a new Coordinates1D object within the given bounds
 - intersect(other1D): returns a new Coordinates1D object with the other1D.bounds 
 - add: add delta value to each coordinate value (returns a new Coordinates1D by default, can be used in-place)
 - concat: concatenate another Coordinates1D  (returns a new Coordinates1D by default, can be used in-place)

Operators
 - len: size
 - in: area_bounds[0] < value < area_bounds[1]
 - []: returns a new Coordinates1D object; supports a single index, a slice, an index array, and a boolean array
 - +, +=: concat Coordinates1D or add value
 - -, -=: subtract value
 - &: intersect

### ArrayCoordinates1D

An arbitrary list of coordinates, where `values` is array-like.

```
ArrayCoordinates1D(values, ...)
```

### MonotonicCoordinates1D

A sorted list of coordinates, where `values` is array-like and sorted (in either direction)

```
MonotonicCoordinates1D(values, ...)
```

### UniformCoordinates1D

Uniformly-spaced coordinates, defined by a start, stop, and step/size.

```
UniformCoordinates1D(start, stop, step=None, size=None, ...)
```

### Shorthand

For ease-of-use, the following aliases will be defined (one of these options):
 
 - `_ca`, `_cm`, `_cu`, `_cl`
 - `A1D`, `M1D`, `U1D`, `L1D`

Also, consider named versions, 16 in total (one of these options):

 - `A1DLat`, `U1DTime`
 - `ArrayLat1D`, `UniformTime1D`
 - `LatArray1D`, `TimeUniform1D`
 - `ArrayLat`, `UniformTime`
 - `LatArray`, `TimeUniform`
 - `ALat`, `UTime`
 - `_alat`, `_utime`

So that the following are equivalent

```
UniformCoordinates1D(0, 1, 0.1, name='lat')
U1D(0, 1, 0.1, name='lat')
ULat(0, 1, 0.1)
```

or possibly

```
UniformCoordinates1D(0, 1, 0.1, name='lat')
_cu(0, 1, 0.1, name='lat')
_ulat(0, 1, 0.1)
```

## Multidemensional Coordinates

### Coordinate Creation

Coordinates are created from a list or dict containing Coordinates1D objects.

```
Coordinates(values, **traits)
```

When using a list, each item is either Coordinates1D object for unstacked dimensions or a tuple of Coordinate1D objects for stacked dimensions.

```
lat = UniformCoordinate1D(0, 1, size=10, name='lat', ...)
lon = UniformCoordinate1D(0, 1, size=10, name='lon', ...)
time = ArrayCoordinate1D(['2018-01-01', '2018-01-02'], name='time', ...)
coords = Coordinates([lat, lon, time], **traits)
coords = Coordinates([(lat, lon), time], **traits) # stacked
```

When using a dict, dimension names are mapped to Coordinate1D objects for unstacked dimensions or a tuple of Coordinate1D objects for stacked dimensions.

```
lat = UniformCoordinate1D(0, 1, size=10, name='lat', ...)
lon = UniformCoordinate1D(0, 1, size=10, name='lon', ...)
time = ArrayCoordinate1D(['2018-01-01', '2018-01-02'], name='time', ...)
coords = Coordinates({'lat': lat, 'lon': lon, 'time': time}, **traits)
coords = Coordinates({'lat_lon': (lat, lon), 'time': time}, **traits) # stacked
```

*note: in Python <= 3.5, an OrderedDict is required.*

### Stacking functions

The `stack` function is recommended as it makes stacking more explicit.

```
coords = Coordinates([stack(lat, lon), time], **traits)
coords = Coordinates({'lat_lon': stack(lat, lon), 'time': time}, **traits)
```

The `stacked_linspace` helper function returns a tuple of UniformCoordinates1D, each with the given size

```
stacked_linspace((0, 0, 20), (1, 1, 100), 100, names=('lat', 'lon', 'alt'))
```

is the equivalent to

```
lat = UniformCoordinates1D(0, 1, size=100), name='lat')
lon = UniformCoordinates1D(0, 1, size=100), name='lon')
alt = UniformCoordinates1D(20, 100, size=100), name=alt')
stacked(lat, lon, alt)
```

Coordinate creation is based on [xarray.DataArray](http://xarray.pydata.org/en/stable/data-structures.html#dataarray).

```
coords = Coordinates(values, dims=dims, ctype='segment', segment_position=0.5, ...)
```

 * `values` - a list of coordinate values, one for each dimension. Each item is translated to the correct Coordinate1D type under the hood.
 * `dims` - a list of dimension labels, matching the length of `values`

Valid coordinate values:
 * a single value (number, string, or datetime object)
 * an array-like of numbers, strings, or datetime objects
 * a Coordinate1D object
 * a tuple of valid coordinate values (for stacked coordinates)

Valid dimension labels:
 * 'lat', 'lon', 'time', or 'alt'
 * two or more dimension labels joined by underscores, e.g. 'lat_lon' or 'time_lat_lon' (for stacked coordinates)

### Traits

 * coord_ref_sys
 * distance_units
 * segment_position
 * ctype

*TODO: defaults, passing on to Coordinates1D, overriding with Coordinates1D*

### Properties

 * ndim: Int
 * dims: tuple(str, str, ...)
 * shape - tuple(int, int, ...)
 * coords - OrderedDict, maps dimension names to Coordinate1D objects or tuples of Coordinate1D objects
 * coords1d - OrderedDict, maps dimension names to Coordinate1D objects (always unstacked)
 * coordinates - xarray Coordinates (dictionary mapping dimension names to named DataArrays)

For stacked dimensions, the names are combined by an underscore, and the coordinates are implemented as a pandas MultiIndex.

```
coords.dims -> ('lat_lon', 'time')
coords.coords.keys() -> ('lat_lon', 'time')
coords.coords['lat_lon'] -> (UniformCoordinates1D, UniformCoordinates1D)
coords.coords['lat'] -> KeyError
coords.coordinates['lat_lon'] -> DataArray with MultiIndex
coords.coordinates['lat'] -> DataArray # we get this for free due to the MultiIndex
coords.coords1d['lat'] -> UniformCoordinates1D
coords.coords1d['lat_lon'] -> KeyError
```

### Methods

 TODO

### Operators

 TODO


### Miscellaneous Examples


```
# Some equivalent ways to copy:
coords_copy = Coordinates(other) # I'm not sure we need this one
coords_copy = Coordinates(other.coords)
coords_copy = Coordinates(other.coords.values())
```


```
# Select specific dimensions
dims = ['lat', 'lon']
c1 = Coordinates([other.coords[dim] for dim in dims])

# downsample (note: in the current spec, this only works if there is no stacking)
c2 = Coordinates([c[::10] for c in other.coords.values()])

# downsample only the time dimension
d = other.coords.copy()
d['time'] = d['time'][::10]
c3 = Coordinates(d)
```