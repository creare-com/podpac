
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
[.] from podpac import Coordinates
[.] lat = [0, 1, 2]
[.] lon = [10, 20, 30, 40]
[.] time = ['2018-01-01', '2018-01-02']
[.] Coordinates([lat, lon], dims=['lat', 'lon'])
Coordinates
    lat: ArrayCoordinates1d(lat): Bounds[0.0, 2.0], N[3]
    lon: ArrayCoordinates1d(lon): Bounds[10.0, 40.0], N[4]
[.] Coordinates([lat, lon, time], dims=['lat', 'lon', 'time'])
Coordinates
    lat: ArrayCoordinates1d(lat): Bounds[0.0, 2.0], N[3]
    lon: ArrayCoordinates1d(lon): Bounds[10.0, 40.0], N[4]
    time: ArrayCoordinates1d(time): Bounds[2018-01-01, 2018-01-02], N[2]
```

You can also create coordinates with just one dimension the same way:

```
>>> from podpac import Coordinates
>>> Coordinates([time], dims=['time'])
Coordinates
    time: ArrayCoordinates1d(time): Bounds[2018-01-01, 2018-01-02], N[2]
```

### Stacked Coordinates

Coordinates from multiple dimensions can be stacked together in a list (rather than representing a grid).

For example, Coordinates with stacked latitude and longitude contain one point for each (lat, lon) pair. Note
that the name for this stacked dimension is 'lat_lon', using an underscore to combine the underlying dimensions.
The following example has a single stacked dimension and a total of 3 points.

```
[.] from podpac import Coordinates
[.] lat = [0, 1, 2]
[.] lon = [10, 20, 30]
[.] c = Coordinates([[lat, lon]], dims=['lat_lon'])
[.] c
Coordinates
    lat_lon[lat]: ArrayCoordinates1d(lat): Bounds[0.0, 2.0], N[3]
    lat_lon[lon]: ArrayCoordinates1d(lon): Bounds[10.0, 30.0], N[3]
[.] c['lat_lon'].coordinates[0]
(0.0, 10.0)
```

Coordinates can combine stacked dimensions and unstacked dimensions. For example, in the following Coordinates the `(lat, lon)` values and the `time` values form a grid of 6 total points.

```
[.] from podpac import Coordinates
[.] lat = [0, 1, 2]
[.] lon = [10, 20, 30]
[.] time = ['2018-01-01', '2018-01-02']
[.] c = Coordinates([[lat, lon], time], dims=['lat_lon', 'time'])
Coordinates
    lat_lon[lat]: ArrayCoordinates1d(lat): Bounds[0.0, 2.0], N[3]
    lat_lon[lon]: ArrayCoordinates1d(lon): Bounds[10.0, 30.0], N[3]
    time: ArrayCoordinates1d(time): Bounds[2018-01-01, 2018-01-02], N[2]
[.] c['lat_lon'].coordinates[0]
(0.0, 10.0)
[.] c['time'].coordinates[0]
numpy.datetime64('2018-01-01')
```

### Uniformly-Spaced Coordinates

Podpac provides two convenience functions `crange` and `clinspace` for creating uniformly-spaced coordinates, similar to the `arange` and `linspace` functions provided by numpy.

**Coordinates Range**

`podpac.crange` creates uniformly-spaced coordinates from a start, stop, and step.

Unlike `np.arange`:
 * string inputs are supported for datetimes and timedeltas
 * the stop value will be included in the coordinates if it falls an exact number of steps from the start

```
>>> import podpac
>>> c = podpac.crange(0, 7, 2)
>>> c.coordinates
array([0., 2., 4., 6.])
>>> c = podpac.crange(0, 8, 2)
>>> c.coordinates
array([0., 2., 4., 6., 8.])
>>> c = podpac.crange('2018-01-01', '2018-03-01', '1,M')
>>> c.coordinates
array(['2018-01-01', '2018-02-01', '2018-03-01'], dtype='datetime64[D]')
 ```

**Coordinates Linspace**

`podpac.clinspace` creates uniformly-spaced coordinates from a start, stop, and size.

Unlike `np.linspace`:
 * string inputs are supported for datetimes
 * tuple inputs are supported for stacked coordinates

```
>>> import podpac
>>> c = podpac.clinspace(0, 8, 5)
>>> c.coordinates
array([0., 2., 4., 6., 8.])
>>> c = podpac.clinspace('2018-01-01', '2018-03-01', 3)
>>> c.coordinates
array(['2018-01-01', '2018-01-30', '2018-02-28'], dtype='datetime64[D]')
>>> c = podpac.clinspace((0, 10), (1, 20), 3)
>>> c.coordinates
MultiIndex(levels=[[0.0, 0.5, 1.0], [10.0, 15.0, 20.0]],
           labels=[[0, 1, 2], [0, 1, 2]])
```

These functions wrap UniformCoordinates1d (see Advanced Usage), which is particularly useful for coordinates with an
extremely large number of points.

### Rotated Coordinates

TODO

### Alternate Constructors

Unstacked coordinates can also be created using the `Coordinates.grid` alternate constructor:

```
>>> from podpac import Coordinates
>>> Coordinates.grid(lat=[0, 1, 2], lon=[10, 20, 30, 40])
Coordinates
    lat: ArrayCoordinates1d(lat): Bounds[0.0, 2.0], N[3]
    lon: ArrayCoordinates1d(lon): Bounds[10.0, 40.0], N[4]
```

Stacked coordinates can be created using the `Coordinates.points` alternate constructor:

```
>>> from podpac import Coordinates
>>> Coordinates.points(lat=[0, 1, 2], lon=[10, 20, 30])
Coordinates
    lat_lon[lat]: ArrayCoordinates1d(lat): Bounds[0.0, 2.0], N[3]
    lat_lon[lon]: ArrayCoordinates1d(lon): Bounds[10.0, 30.0], N[3]
```

For convenience, a tuple can be used to generate uniformly-spaced coordinates. If the third item is an integer, it
is interpreted as a size, otherwise it is interpreted as a step. The following will all be equivalent:

```
Coordinates.grid(lat=(0, 2, 3), lon=(10, 40, 4))
Coordinates.grid(lat=(0, 2, 1.0), lon=(10, 40, 10.0))
Coordinates.grid(lat=clinspace(0, 2, 3), lon=clinspace(10, 40, 4))
Coordinates.grid(lat=crange(0, 2, 1), lon=crange(10, 40, 10))
```

Note that in Python 3.5 and below, the `order` argument is required to both `grid` and `points`

```
Coordinates.grid(lat=[0, 1, 2], lon=[10, 20, 30], order=['lat', 'lon'])
Coordinates.points(lat=[0, 1, 2], lon=[10, 20, 30], order=['lat', 'lon'])
```

### Advanced Usage

TODO

```
from podpac.coordinates import UniformCoordinates1d, ArrayCoordinates1d, Coordinates, StackedCoordinates
>>> lat = UniformCoordinates1d(0, 1, size=100, name='lat')
>>> lon = UniformCoordinates1d(10, 20, size=100, name='lon')
>>> time = ArrayCoordinates1d(['2018-01-01', '2018-02-03'], name='time')
>>> Coordinates([StackedCoordinates([lat, lon]), time])
Coordinates
    lat_lon[lat]: UniformCoordinates1d(lat): Bounds[0.0, 1.0], N[100]
    lat_lon[lon]: UniformCoordinates1d(lon): Bounds[10.0, 20.0], N[100]
    time: ArrayCoordinates1d(time): Bounds[2018-01-01, 2018-02-03], N[2]
```

## Coordinate API

TODO

Coordinates contain some useful properties relating to its dimensions and underlying coordinate values.

```
>>> from podpac import Coordinates
>>> c = Coordinates([lat, lon, time], dims=['lat', 'lon', 'time'])
>>> c.ndims
>>> c.dims
>>> c.shape
>>> c.size
```

Coordinates are dict-like. The `keys()`, `values()`, and `items()`

## Coordinates Groups

TODO
