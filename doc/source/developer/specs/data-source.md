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

Potentailly execute datasource @ native_coordinates

Traits:
- `source`: Any, required
    + The location of the source. Depending on the child node this can be a filepath, numpy array, or dictionary as a few examples.
- `interpolation`: Enum('nearest', 'nearest_preview', 'bilinear', 'cubic', 'cubic_spline', 'lanczos', 'average', 'mode', 'gauss', 'max', 'min', 'med', 'q1', 'q3'), Default: `nearest`
    + Type of interpolation
- `params`: Any
    + Includes `interpolation_param` and any other kwargs that are necessary for the DataSource
- pick one of (`nodata`, `nonevals`, `none_vals`, `nan`, `nanvals`, `nan_vals`): List
    + list of values from source data that should be interpreted as 'no data' or 'nans'

Properties

Contructor
- Take one input (i.e. `auto`) that will automatically execute the datasource at the native_coordinates on contruction. This will allow a shortcut when you just want to load a simple data source for processing with other more complication data sources

Methods
- `execute(coordinates, params=None, output=None, method=None)` (potentially rename to or provide alias as `request(...)`): Execute this node using the supplied coordinates
    + `self.requested_coordinates` gets set to the input coordinates
    + remove dims that don't exist in native coordinates
    + `_determine_source_coordinates()` translates the requested coordinates into coordinates to get requested via `get_data`
- `get_data(coordinates, coordinates_index=None)`:
    + by default, try returning the data source at the the coordinates_index. If this fails, raise a `NotImplementedError`
    + `coordinates` and `coordinates_index` are guarenteed to exists in the datasource as calculated by `_determine_source_coordinates`
    + potentially remove inputs and advice use of `self.requested_source_coordinates` and `self.requested_source_coordinates_index`
    + returns a UnitsDataArray of data subset at the source coordinates
- `get_native_coordinates()`: return the native coordinates from the data source. By default, this should return `self.native_coordinates` if defined, otherwise raise a `NotImplementedError`
- `definition()`: Pipeline node definition for DataSource nodes.

Private Methods
- `_determine_source_coordinates()`: Use `self.requested_coordinates`, `self.interpolate`, and `self.params` to determine the requested coordinates translated into the source coordinates. This could also be a non-class method that takes inputs `(requested_coordinates, native_coordinates, interpolate, params)`. Alternate names: `translate_requested_coordinates`, `_interpolate_coordinates`, `_intersect_coordinates`
    + sets `self.requested_source_coordinates` (Coordinates) to the coordinates that need to get requested from the data source via `get_data`. These coordinates MUST exists exactly in the data source native coordinates. They coordinates that get returned may be affected by the type of interpolate requested.
    + (optionally) sets `self.requested_source_coordinates_index` (Array[int])
    + If we use a non-class method, this would return `requested_source_coordinates` and `requested_source_coordinates_index`
- Potential: `_pre_get_data()`: method that gets called before `get_data()`
- Potential: `_post_get_data()`: method that gets called after `get_data()`
    + if the output from get_data is an numpy array, then try to create a UnitsDataArray from the numpy array, the source coordinates, and the source coordinates index.
    + Returns a UnitsDataArray for the data subset
- `_interpolate(data_subset)`: Use `self.interpolate` and call the appropriate functions in the `interpolate` module

Operators

## User Interface

Simple datasource that doesn't need its own subclass

```python
source = np.random.rand(101, 101)
source_coordinates = coordinates(lat=(-25, 25, 101), lon=(-25, 25, 101), order=['lat', 'lon'])
node = DataSource(source=source, native_coordinates=source_coordinates)
output = node.request(node.native_coordinates)
```
automatically execute:

```
source = np.random.rand(101, 101)
source_coordinates = coordinates(lat=(-25, 25, 101), lon=(-25, 25, 101), order=['lat', 'lon'])
ds = DataSource(source=source, native_coordinates=source_coordinates, execute=True)
```

Basic subclass datasource

```python
class MockDataSource(DataSource):
    """ Mock Data Source for testing """
    source = np.random.rand(101, 101)
    native_coordinates = Coordinates(lat=(-25, 25, 101), lon=(-25, 25, 101), order=['lat', 'lon'])
    interpolate = 'bilinear'

    def get_native_coordinates(self):
        """ see DataSource """
        return self.native_coordinates

    def get_data(self, coordinates, coordinates_index):
        """ see DataSource """
        s = coordinates_index
        d = self.initialize_coord_array(coordinates, 'data', fillval=self.source[s])
        return d
```

TODO: add more complicated implementations

## Developer interface 
TODO: Add developer interface specs
