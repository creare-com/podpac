
# Coordinates

## Overview

Coordinates are used to: 
1. Evaluate [nodes](nodes.md) which retrieve and process data
2. Define the coordinates of [data sources](nodes.md#datasource)

Podpac Coordinates are modeled after the coords in [xarray](http://xarray.pydata.org/en/stable/data-structures.html),
with some additional restrictions and enhancements. Coordinates are created from a list of coordinate `values` and a corresponding list of `dims`:

```
podpac.Coordinates(values, dims=dims, ...)
```

Unlike xarray, podpac coordinate values are always either `float` or `np.datetime64`. For convenience, podpac
automatically converts datetime strings such as `'2018-01-01'` to `np.datetime64`. In addition, the allowed dimensions
are `'lat'`, `'lon'`, `'time'`, and `'alt'`.

## Coordinate Creation

### Unstacked Coordinates

Unstacked multidimensional coordinates form a grid of points. For example, the following Coordinates contain three dimensions and a total of 24 points.

```
>>> lat = [0, 1, 2]
>>> lon = [10, 20, 30, 40]
>>> time = ['2018-01-01', '2018-01-02']
>>> Coordinates([lat, lon], dims=['lat', 'lon'])
>>> Coordinates([lat, lon, time], dims=['lat', 'lon', 'time'])
```

You can also create coordinates with just one dimension the same way:

```
>>> Coordinates([time], dims=['time'])
```

### Stacked Coordinates

Coordinates from multiple dimensions can be stacked together in a list (rather than representing a grid).

For example, Coordinates with stacked latitude and longitude contain one point for each (lat, lon) pair. Note
that the name for this stacked dimension is 'lat_lon', using an underscore to combine the underlying dimensions.
The following example has a single stacked dimension and a total of 3 points.

```
lat = [0, 1, 2]
lon = [10, 20, 30]
c = Coordinates([np.stack([lat, lon]).T], dims=['lat_lon'])
>>> c
>>> c.coords[0]
```

Coordinates can combine stacked dimensions and unstacked dimensions. For example, in the following Coordinates the `(lat, lon)` values and the `time` values form a grid of 6 total points.

```
lat = [0, 1, 2]
lon = [10, 20, 30]
>>> time = ['2018-01-01', '2018-01-02']
c = Coordinates([np.stack([lat, lon]).T, time], dims=['lat_lon', 'time'])
>>> c
>>> c.coords[0]
```

### Uniformly-Spaced Coordinates

Podpac provides two convenience functions `crange` and `clinspace` for creating uniformly-spaced coordinates, similar to the `arange` and `linspace` functions provided by numpy.

**Coordinates Range**

`podpac.crange` creates uniformly-spaced coordinates from a start, stop, and step.

Unlike `np.arange`:
 * string inputs are supported for datetimes and timedeltas
 * the stop value will be included in the coordinates if it falls an exact number of steps from the start

```
>>> c = podpac.crange(0, 9, 2)
>>> c.coordinates
[0.0, 2.0, 4.0, 6.0, 8.0]
>>> c = podpac.crange(0, 10, 2)
>>> c.coordinates
[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
>>> c = podpac.crange('2018-01-01', '2018-3-01', '1,M')
>>> c.coordinates
 ```

**Coordinates Linspace**

`podpac.clinspace` creates uniformly-spaced coordinates from a start, stop, and size.

Unlike `np.linspace`:
 * string inputs are supported for datetimes
 * tuple inputs are supported for stacked coordinates

```
>>> c = podpac.clinspace(0, 10, 6)
>>> c.coordinates
[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
>>> c = podpac.clinspace('2018-01-01', '2018-3-01', 4)
>>> c.coordinates
>>> c = podpac.clinspace((0, 10), (1, 20), 3)
>>> c.coordinates
 ```

These functions wrap UniformCoordinates1d (see Advanced Usage), which is particularly useful for coordinates with an
extremely large number of points.

### Rotated Coordinates

TODO

### Coordinate Properties

TODO ctype, etc

### Alternate Constructors

Unstacked coordinates can also be created using the `Coordinates.grid` alternate constructor:

```
>>> Coordinates.grid(lat=[0, 1, 2], lon=[10, 20, 30, 40])
>>> Coordinates([[0, 1, 2], [10, 20, 30, 40]], dims=['lat', 'lon'])
```

Stacked coordinates can be created using the `Coordinates.points` alternate constructor:

```
>>> Coordinates.points(lat=[0, 1, 2], lon=[10, 20, 30])
>>> Coordinates([np.stack([0, 1, 2], [10, 20, 30, 40]).T], dims=['lat_lon'])
```

For convenience, a tuple can be used to generate uniformly-spaced coordinates. If the third item is an integer, it
is interpreted as a size, otherwise it is interpreted as a step. The following will all be equivalent:

```
>>> Coordinates.grid(lat=(0, 2, 3), lon=(10, 40, 4))
>>> Coordinates.grid(lat=clinspace(0, 2, 3), lon=clinspace(10, 40, 4))
>>> Coordinates.grid(lat=(0, 2, 1.0), lon=(10, 40, 10.0))
>>> Coordinates.grid(lat=crange(0, 2, 1), lon=crange(10, 40, 10))
```

Note that in Python 3.5 and below, the `order` argument is required to both `grid` and `points`

```
>>> Coordinates.grid(lat=[0, 1, 2], lon=[10, 20, 30], order=['lat', 'lon'])
>>> Coordinates.grid(lat=[0, 1, 2], lon=[10, 20, 30], order=['lat', 'lon'])
```

### Advanced Usage

TODO

```
lat = UniformCoordinates1d(0, 1, size=100, name='lat')
lon = UniformCoordinates1d(10, 20, size=100, name='lon')
time = ArrayCoordinates1d(['2018-01-01', '2018-02-03'], name='time')
Coordinates([StackedCoordinates([lat, lon]), time])
```

TODO mixed ctypes, etc...

## Coordinate API

TODO

Coordinates contain some useful properties relating to its dimensions and underlying coordinate values.

```
>>> c = Coordinates([lat, lon, time], dims=['lat', 'lon', 'time'])
>>> c.ndims
>>> c.dims
>>> c.shape
>>> c.size
```

Coordinates are dict-like. The `keys()`, `values()`, and `items()`

## Coordinates Groups

TODO
