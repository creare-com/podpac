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

## User Interface

### Coordinate Creation

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

### Coordinate attributes

 * `ndim` - number of dimensions
 * `dims` - tuple of dimensions
 * `shape` - tuple of sizes for each dimension
 * `coords1d` - OrderedDict, maps dimensions to Coordinate1D objects (a tuple of Coordinate1D objects for stacked dimensions)
 * `coords` - DataArray.coords object, (maps dimensions to named DataArrays)

### Basic Examples:

A grid of lat, lon points:

```
coords = Coordinates([[1, 2, 3, 4, 5], [10, 20, 30, 40]], dims=['lat', 'lon'])
```

A uniform grid of lat, lon points:

```
coords = Coordinates([(1, 5, 5), (10, 40, 4)], dims=['lat', 'lon'])
```

Uniform coordinate values can be defined using a step instead of a size, which is particularly natural for time coordinates:

```
coords = Coordinates([('2018-01-01', '2018-01-10', '1,D')], dims=['time'])
```

When there is only one dimension, as above, you can pass the value directly (without a list):

```
coords = Coordinates(('2018-01-01', '2018-01-10', '1,D'), dims='time')
```

A single point with four dimenions:

```
coords = Coordinates([1, 1, 100, '2018-01-01'], dims=['lat', 'lon', 'time', 'alt'])
```

### Stacked Coordinates

Sets of coordinates can be "stacked" in a single dimension. In the resulting output DataArray, these dimensions are implemented as [MultiIndex coordinates](http://xarray.pydata.org/en/stable/data-structures.html#multiindex-coordinates), which themselves are `pandas.MultiIndex` objects.


A list of stacked lat, lon coordinates:

```
# TODO which of these two?
coords = Coordinates(([0, 1, 2, 3], [10, 20, 30, 40]), dims='lat_lon')
coords = Coordinates([[0, 10], [1, 20], [2, 30], [3, 40]], dims='lat_lon')
```

A uniformly-spaced line:

```
# TODO which of these two?
coords = Coordinates((0, 3, 4), (10, 40, 4)), dims='lat_lon')
coords = Coordinates(([0, 10], [3, 40], 4), dims='lat_lon')
```

Stacked and unstacked coordinates can be mixed:

```
lat_lon = [[0, 10], [1, 20], [2, 30], [3, 40]]
time = ('2018-01-01', '2018-01-10', '1,D')
# TODO which of these two?
coords = Coordinates([(lat, lon), time], dims=['lat_lon', 'time'])
coords = Coordinates([lat_lon, time], dims=('lat_lon', 'time'])
```

### Creating Coordinates from Existing Coordinates

The actual coordinates for each dimension can be accessed in the `.coords` attribute as named DataArrays. One way to copy existing coordinates or to create a new Coordinates object with only certain dimensions or certain coordinate values:

```
coords_copy = Coordinates(other.coords.values())
coords_new = Coordinates([other.coords['lat'], other.coords['lon'][10:20:2]])
```

The coordinate values are stored as Coordinate1D objects in the `coords1d` attribute. Coordinates can be created from these Coordinate1D objects directly, so another way to use existing coordinates:

```
coords_copy = Coordinates(other.coords1d.values())
coords_new = Coordinates([other.coords['lat'], other.coords['lon'][10:20:2]])
```

Using the values from `coords1d` is often preferable because it preserves the coordinate type (which can be more efficient) and metadata (e.g. ctype).

```
>>> a = Coordinates([other.coords1d['lat'], other.coords1d['lon'][10:20:2]])
>>> b = Coordinates([other.coords['lat'], other.coords['lon'][10:20:2]])

>>> other.coords1d['lat']
UniformCoordinate1D
>>> a.coords1d['lat']
UniformCoordinate1D
>>> b.coords1d['lat']
Coordinate1D # nonparameterized
```

Note that the `dims` argument has been omitted from the above statements. The DataArrays and Coordinate1D objects are named, and these names are used automatically as the dims.

### fromdict

Coordinates can also be created from the `coords` and `coords1d` attributes of existing Coordinates, where again the coords1d input may be preferred:

```
coords = Coordinates.fromdict(other.coords)
coords = Coordinates.fromdict(other.coords1d)
```

TODO: not sure if this needs to be supported:

```
d = OrderedDict([('lat', [1, 2, 3]), ('lon', [10, 20, 30, 40])])
coords = Coordinates.fromdict(d)
```

### Advanced usage

Of course, you can create DataArrays and Coordinate1D objects manually, as well:

```
lat = xr.DataArray([1, 2, 3], name='lat')
lon = xr.DataArray([10, 20, 30, 40], name='lon')
coords = Coordinates([lat, lon])
```

```
lat = Coordinate1D([1, 2, 3], name='lat')
lon = UniformCoordinate1D(10, 40, step=10, ctype='point', name='lon')
coords = Coordinates([lat, lon])
```

Defining Coordinate1D objects manually is particularly important if you need to use a different ctype or other metadata for a particular dimension:

```
>>> lat = Coordinate1D([1, 2, 3])
>>> lon = UniformCoordinate1D(10, 40, step=10, ctype='point')
>>> coords = Coordinates([lat, lon], dims=['lat', 'lon'])

>>> coords.ctype
'segment'
>>> coords.coords1d['lat'].ctype
'segment'
>>> coords.coords1d['lon'].ctype
'point'
```

### Coordinate modification

TODO: Add specification for how to and what happens when adding/intersecting/replacing/dropping/sub-selecting coordinates

## Developer interface 
### Coordinate1D interface
TODO: Add common attributes and function expected by any Coordinate1D class in case we want to develop new types of coordinates
