# Requirements

* Quick and simple to define arbitrary data source given podpac Coordinates and some kind of data
* Provide a base class for all other data types, including user defined data types

# Example Use cases

* I want to load tabular data from a local file and create a podpac data source quickly
* I want to create a data source class that provides access to a new server that serves GeoTIFF data
* I want to create a data source class the provides access to a new flat file data source
* I want to access my data on Backblaze B2 instead of S3 bucket
* I want to access a GeoServer which implements WCS
* I want to provide access to a new NASA dataset that is stored on S3 in a new data format

# Specification

## DataSource Class

`DataSource(Node)` is the base class from which all data other data sources are implemented. Extends the `Node` base class.

#### Traits

(See [Node documentation](https://creare-com.github.io/podpac-docs/user/api/podpac.core.node.html#podpac.core.node.Node) for nodes attributes)

- `source`: Any, required
    + The location of the source. Depending on the child node this can be a filepath, numpy array, or dictionary as a few examples.
- `interpolator`: 
    - `Interpolator()` - NearestNeighbor
    - Dict({`dim` (not stacked): `Interpolator()` NearestNeighbor})
    - string - Enum(`interpolate.INTERPOLATION_OPTIONS`)
    - Dict({`dim` (not stacked): string - Enum(`interpolate.INTERPOLATION_OPTIONS`)})
- `coordinate_index_type`: Enum('list','numpy','xarray','pandas'). By default this is `numpy`
- `nan_vals`: List
    + list of values from source data that should be interpreted as 'no data' or 'nans' (replaces `no_data_vals`)

#### Private Members

- `_interpolator` - interpolator chosen from `interpolator` and `interpolation`

*TODO* : the names of these memebers will be changed

- `_requested_coordinates` = tl.Instance(Coordinates, allow_none=True)
- `_requested_source_coordinates` = tl.Instance(Coordinates)
- `_requested_source_coordinates_index` = tl.List()
- `_requested_source_data` = tl.Instance(UnitsDataArray)

#### Properties

#### Contructor

- FUTURE: After implementing a limiter on the request size, implement:
    + Take one input (i.e. `evaluate`) that will automatically execute the datasource at the native_coordinates on contruction. This will allow a shortcut when you just want to load a simple data source for processing with other more complication data sources
- Choose a default interpolation option if neither interpolator or interpolation is defined
- Instantiate `_interpolator`  classes with data sources based on input to `interpolator`

#### Methods

- `eval(coordinates, output=None, method=None)`: Evaluate this node using the supplied coordinates
    + `self.requested_coordinates` gets set to the coordinates that are input
    + remove dims that don't exist in native coordinates
    + intersect the `self.requested_coordinates` with `self.native_coordinates` to create `self.requested_source_coordinates` and `self.requested_source_coordinates_index` to get requested via `get_data`.  DataSource `coordinate_index_type` informs `self.requested_source_coordinates_index` (Array[int], Array[boolean], List[slices])
    + interpolate requested coordinates `self.requested_source_coordinates` using `_interpolate_requested_coordinates()`.
    + `self.requested_source_coordinates` coordinates MUST exists exactly in the data source native coordinates.
    + run `_get_data` which runs the user defined `get_data()` and check/fix order of dims when UnitsDataArray or Xarray is returned from get_data. Otherwise create UnitsDataArray using values from get_data and requested_source_coordinates. This return from `_get_data()` sets `self.requested_source_data`
    + Run `_interpolate()`
    + Set `self.evaluated` to True
    + Output the user the UnitsDataArray passed back from interpolate
- `get_data(coordinates, coordinates_index)`:
    + Raise a `NotImplementedError`
    + `coordinates` and `coordinates_index` are guarenteed to exists in the datasource
    + return an UnitsDataArray, numpy array, or xarray of values. this will get turned into a UnitsDataArray aftwards using `self.requested_source_coordinates` even if the xarray passes back coordinates
        * Need to check/fix order of dims in UnitsDataArray and Xarray case
- `get_native_coordinates()`: return the native coordinates from the data source. By default, this should return `self.native_coordinates` if defined, otherwise raise a `NotImplementedError`
- `definition()`: Pipeline node definition for DataSource nodes.
    + Transport mechanism for going to the cloud
    + Leave as is

#### Private Methods

- `_interpolate_requested_coordinates()`: Use `self.requested_coordinates`, `self.native_coordinates`, `self.interpolate` to determine the requested coordinates interpolated into the source coordinates.
    + overwrites `self.requested_source_coordinates` (Coordinates) to interpolated coordinates that need to get requested from the data source via `get_data`. 
    + These coordinates MUST exists exactly in the data source native coordinates
    + Returns None
- `_interpolate()`: Use `self.interpolate` and call the appropriate functions in the `interpolate` module
    + Returns a UnitDataArray which becomes the output of the eval method

#### Operators

## User Interface

Simple datasource that doesn't need its own subclass

```python
class ArraySource(DataSource):
    source = tl.Instance(np.ndarray)

    def get_data(self, coordinates, coordinates_index):
        return self.source[coordinates_index]
```

Using this basic class

```python
source = np.random.rand(101, 101)
source_coordinates = coordinates(lat=(-25, 25, 101), lon=(-25, 25, 101), order=['lat', 'lon'])
node = ArraySource(source=source, native_coordinates=source_coordinates)
output = node.eval(node.native_coordinates)
```

FUTURE: automatically execute

```python
source = np.random.rand(101, 101)
source_coordinates = coordinates(lat=(-25, 25, 101), lon=(-25, 25, 101), order=['lat', 'lon'])
node = ArraySource(source=source, native_coordinates=source_coordinates)
output = node.eval()
```

More Complicated Source. 
This datasource gets new `native_coordinates` every time the source updates.

```python
class RasterioSource(DataSource):
    
    source = tl.Unicode(allow_none=False)  # specifies source MUST be a Unicode
    dataset = tl.Any(allow_none=True)
    band = tl.CInt(1).tag(attr=True)
    
    @tl.default('dataset')
    def open_dataset(self, source):
        return module.open(source)

    @tl.observe('source')
    def _update_dataset(self):
        self.dataset = self.open_dataset()
        self.native_coordinates = self.get_native_coordinates()
        
    def get_native_coordinates(self):
        dlon = self.dataset.width
        dlat = self.dataset.height
        left, bottom, right, top = self.dataset.bounds

        return podpac.Coordinate(lat=(top, bottom, dlat),
                                 lon=(left, right, dlon),
                                 order=['lat', 'lon'])

    def get_data(self, coordinates, coordinates_index):
        data = self.dataset.read(coordinates_index)
        return data
```

## Developer interface


TODO: Add developer interface specs
