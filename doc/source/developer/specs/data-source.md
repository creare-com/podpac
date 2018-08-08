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
- `interpolation`: `Interpolate()`  class (tbd)
    - Should include `interpolation_param` for all dims or single dims.  i.e. Enum('nearest', 'nearest_preview', 'bilinear', 'cubic', 'cubic_spline', 'lanczos', 'average', 'mode', 'gauss', 'max', 'min', 'med', 'q1', 'q3'), Default: `nearest`
- `coordinate_index_type`: Enum('list','numpy','xarray','pandas'). By default this is `numpy`
- `nan_vals`: List
    + list of values from source data that should be interpreted as 'no data' or 'nans' (replaces `no_data_vals`)

#### Properties

#### Contructor

- FUTURE: After implementing a govener on the request size, implement:
    + Take one input (i.e. `evaluate`) that will automatically execute the datasource at the native_coordinates on contruction. This will allow a shortcut when you just want to load a simple data source for processing with other more complication data sources

#### Methods

- `eval(coordinates, params=None, output=None, method=None)`: Evaluate this node using the supplied coordinates
    + `self.requested_coordinates` gets set to the coordinates that are input
    + remove dims that don't exist in native coordinates
    + `_determine_source_coordinates()` translates the `self.requested_coordinates` into `self.requested_source_coordinates`, `self.requested_source_coordinates_index` to get requested via `get_data`.
    + run `get_data`
    + Check/fix order of dims when UnitsDataArray or Xarray is returned from get_data. Otherwise create UnitsDataArray using values from get_data and requested_source_coordinates
    + Run `_interpolate()`
    + Output the user the UnitsDataArray passed back from interpolate
- `get_data(coordinates=self.requested_source_coordinates, coordinates_index=requested_source_coordinates_index)`:
    + Raise a `NotImplementedError`
    + `coordinates` and `coordinates_index` are guarenteed to exists in the datasource as calculated by `_determine_source_coordinates`
    + return an UnitsDataArray numpy array or xarray of values. this will get turned into a UnitsDataArray aftwards using `self.requested_source_coordinates` even if the xarray passes back coordinates
        * Need to check/fix order of dims in UnitsDataArray and Xarray case
- `get_native_coordinates()`: return the native coordinates from the data source. By default, this should return `self.native_coordinates` if defined, otherwise raise a `NotImplementedError`
- `definition()`: Pipeline node definition for DataSource nodes.
    + Transport mechanism for going to the cloud
    + Leave as is

#### Private Methods

- `_determine_source_coordinates()`: Use `self.requested_coordinates`, `self.native_coordinates`, `self.interpolate` to determine the requested coordinates translated into the source coordinates.
    + sets `self.requested_source_coordinates` (Coordinates) to the coordinates that need to get requested from the data source via `get_data`. These coordinates MUST exists exactly in the data source native coordinates. They coordinates that get returned may be affected by the type of interpolate() class used in request.
    + Based on `coordinate_index_type`, set `self.requested_source_coordinates_index` (Array[int], Array[boolean], List[slices])
    + Returns a UnitsDataArray for the data subset
- `_interpolate()`: Use `self.interpolate` and call the appropriate functions in the `interpolate` module
    + Returns a UnitDataArray which becomes the output of the eval method

#### Operators

## User Interface

Simple datasource that doesn't need its own subclass

```python
source = np.random.rand(101, 101)
source_coordinates = coordinates(lat=(-25, 25, 101), lon=(-25, 25, 101), order=['lat', 'lon'])
node = DataSource(source=source, native_coordinates=source_coordinates)
output = node.eval(node.native_coordinates)
```
automatically execute:

```
source = np.random.rand(101, 101)
source_coordinates = coordinates(lat=(-25, 25, 101), lon=(-25, 25, 101), order=['lat', 'lon'])
ds = DataSource(source=source, native_coordinates=source_coordinates, evaluate=True)
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

More Complicated Rasterio Source. 
This datasource gets new `native_coordinates` every time the source updates.

```python
class RasterioSource(DataSource):
    
    source = tl.Unicode(allow_none=False)  # specifies source MUST be a Unicode
    dataset = tl.Any(allow_none=True)
    band = tl.CInt(1).tag(attr=True)
    
    @tl.default('dataset')
    def open_dataset(self, source):
        return rasterio.open(source)
    
    def close_dataset(self):
        self.dataset.close()

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

    def get_data(self):
        data = self.initialize_coord_array(coordinates)
        slc = coordinates_index
        data.data.ravel()[:] = self.dataset.read(
            self.band, window=((slc[0].start, slc[0].stop),
                               (slc[1].start, slc[1].stop)),
            out_shape=tuple(coordinates.shape)
            ).ravel()
            
        return data
    
    @cached_property
    def band_count(self):
        """The number of bands
        
        Returns
        -------
        int
            The number of bands in the dataset
        """
        return self.dataset.count
    
    @cached_property
    def band_descriptions(self):
        """A description of each band contained in dataset.tags
        
        Returns
        -------
        OrderedDict
            Dictionary of band_number: band_description pairs. The band_description values are a dictionary, each 
            containing a number of keys -- depending on the metadata
        """
        bands = OrderedDict()
        for i in range(self.dataset.count):
            bands[i] = self.dataset.tags(i + 1)
        return bands

    @cached_property
    def band_keys(self):
        """An alternative view of band_descriptions based on the keys present in the metadata
        
        Returns
        -------
        dict
            Dictionary of metadata keys, where the values are the value of the key for each band. 
            For example, band_keys['TIME'] = ['2015', '2016', '2017'] for a dataset with three bands.
        """
        keys = {}
        for i in range(self.band_count):
            for k in self.band_descriptions[i].keys():
                keys[k] = None
        keys = keys.keys()
        band_keys = defaultdict(lambda: [])
        for k in keys:
            for i in range(self.band_count):
                band_keys[k].append(self.band_descriptions[i].get(k, None))
        return band_keys
    
    @tl.observe('source')
    def _clear_band_description(self, change):
        clear_cache(self, change, ['band_descriptions', 'band_count',
                                   'band_keys'])

    def get_band_numbers(self, key, value):
        """Return the bands that have a key equal to a specified value.
        
        Parameters
        ----------
        key : str / list
            Key present in the metadata of the band. Can be a single key, or a list of keys.
        value : str / list
            Value of the key that should be returned. Can be a single value, or a list of values
        
        Returns
        -------
        np.ndarray
            An array of band numbers that match the criteria
        """
        if (not hasattr(key, '__iter__') or isinstance(key, string_types))\
                and (not hasattr(value, '__iter__') or isinstance(value, string_types)):
            key = [key]
            value = [value]

        match = np.ones(self.band_count, bool)
        for k, v in zip(key, value):
            match = match & (np.array(self.band_keys[k]) == v)
        matches = np.where(match)[0] + 1

        return matches
```

## Developer interface


TODO: Add developer interface specs
