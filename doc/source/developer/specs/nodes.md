# Requirements
* Must provide an evaluation interface with consistent rules for outputs
* Must provide a way of instantiating outputs following these rules
* Provide methods to test if these rules are obeyed for child classes
* Provide a consistent interface/rules for passing runtime parameters
* Must provide an interface for finding available coordinates within a pipeline
* Must provide an interface for caching and retrieving cached results
* Must provide methods to inspect the complete pipeline and:
    * Ouput json formatted text used to reproduce it
    * Report inconsistencies or inabilities to produce pipelines
    * Evaluate pipelines at a point with indications of where data came from (which nodes are active within compositors)
* Must give the same output if evaluated from a newly instantiated node, or previously instantiated and evaluated node
* Must define public attributes that should be implemented/available to all nodes
* Must define public parameters that should be implemented/available to all nodes
* Potentially provide a mechanism to evaluate nodes in parallel or remotely on the cloud (This may be handled by an evaluation manager)
* Could provide a method to estimate cost of executing a job over given coordinates using AWS cloud
* Provide a standard methods for creators of child nodes to:
    * Specify defaults for attributes and parameters
    * Specify defaults for input nodes
    * initialze the node without overwriting __init__

# Example Use cases
## Typical
* The primary use case is essentially: A user evaluates nodes written by various authors with the same set of coordinates and the results can interact together using common mathematical operators (e.g. +-/*)
* A users wants to retrieve all of the available times for a complex node pipeline in the native coordinates of the underlying datasets
    * In this case, there may be multiple datasets spanning different times with different temporal resolutions
* After interactively creating a complex processing pipeline, a users wants to:
    * share their work using the .json format
    * evaluate the node for a larger region using cloud-resources
    * inspect outputs from various stages of the algorithm to debug/analyze the results
    
## Advanced
* Advanced users create new nodes to interface with custom data sources
    * May specify cachining behaviour for expensive to calculate quantities such as indexes
    * May specify settings that should be saved as part of the user's setting file
    * Should be able to test if node is properly implemented
* Advanced users create new nodes to implement custom algorithms (see Algorithm Node spec)
* Advanced users create new nodes to composite various nodes together following custom rules (see Compositor Node spec)
* Advanced users create new nodes to construct pipelines from custom json (see Pipeline Node spec)
* Advanced users inspect results from an evaluation by examining the cached data quantities (see caching spec)

# Specification
## User Interface
Starting from here for examples below: 
```python
import numpy as np
import traitlets as tl
from podpac.core.node import Node
from podpac.core.coordinate import Coordinate
from podpac.core.units import UnitsDataArray, ureg
node = Node()
```
### Methods
#### __init__(...)
* Any attributes set at this stage are constant for all executions

#### eval(coordinates, output=None, method=None)
```python
def eval(coordinates, output=None, method=None):
    '''
    Parameters
    -----------
    coordinates : podpac.Coordinate
        The set of coordinates requested by a user. The Node will be evaluated using these coordinates.
    output : podpac.UnitsDataArray, optional
        Default is None. Optional input array used to store the output data. When supplied, the node will not allocate its own memory for the output array. This array needs to have the correct dimensions and coordinates.
    method : str, optional
        Default is None. How the node will be evaluated: serial, parallel, on aws, locally, etc. Currently only local evaluation is supported.
        
    Returns
    --------
    podpac.core.units.UnitsDataArray
    
    Raises
    -------
    NotImplementedError: Base class raises this because the interface needs to be implemented by children.
    CoordinateError: If Node contains a dimension not present in the requsted coordinates
    '''
    pass
```

**Notes on what's returned: **
* This function should always return a `UnitsDataArray` object.
    ```python
    >>> node.native_coordinates = Coordinate(lat=(90, -90, -1.), lon=(-180, 180, 2.), order=['lat', 'lon'])
    >>> type(node.initialize_output_array())
    podpac.core.units.UnitsDataArray
    ```
* This `UnitsDataArray` may contain the following dimensions `['lat', 'lon', 'time', 'alt']`
* For cases with multiple outputs, it may additionally contain the field `band`
    * This is to supported datasets such as multi-spectral imagery
    ```python
    >>> grey = UnitsDataArray(np.ones((2, 1)), dims=['lat', 'lon'], coords=[[0, 1], [0]])
    >>> rgba = UnitsDataArray(np.ones((2, 1, 4)), dims=['lat', 'lon', 'band'],                                               coords=[[0, 1], [0], ['r', 'g', 'b', 'a']])
    >>> grey + rgba
    <xarray.UnitsDataArray (lat: 2, lon: 1, band: 4)>
    array([[[2., 2., 2., 2.]],

           [[2., 2., 2., 2.]]])
    Coordinates:
      * lat      (lat) int32 0 1
      * lon      (lon) int32 0
      * band    (band) <U1 'r' 'g' 'b' 'a'
    ```
    * This requres that all the bands have the same `units` 
    ```python
    >>> grey = UnitsDataArray(np.ones((2, 1)), dims=['lat', 'lon'],
                              coords=[[0, 1], [0]], attrs={'units': ureg.m})
    >>> rgba1 = UnitsDataArray(np.ones((2, 1, 4)), dims=['lat', 'lon', 'band'],
                              coords=[[0, 1], [0], ['r', 'g', 'b', 'a']],
                              attrs={'units': ureg.km})
    >>> grey + rgba1
    <xarray.UnitsDataArray (lat: 2, lon: 1, band: 4)>
    array([[[1001., 1001., 1001., 1001.]],

           [[1001., 1001., 1001., 1001.]]])
    Coordinates:
      * lat      (lat) int32 0 1
      * lon      (lon) int32 0
      * band    (band) <U1 'r' 'g' 'b' 'a'
    Attributes:
        units:    meter
    ```
    * [**FUTURE FEATURE**] We could support different units for differents bands as follows, but the dimensionality still has to be consistent, unless a specific band is selected 
    ```python
    >>> rgba2 = UnitsDataArray(np.ones((2, 1, 4)), dims=['lat', 'lon', 'band'],
                              coords=[[0, 1], [0], ['r', 'g', 'b', 'a']],
                              attrs={'units': {'r': ureg.m, 'g': ureg.ft,
                                              'b': ureg.km, 'a': ureg.mile}})

    >>> grey + rgba2
    <xarray.UnitsDataArray (lat: 2, lon: 1, band: 4)>
    array([[[2., 1.3048, 1001., 1610.344]],

           [[2., 1.3048, 1001., 1610.344]]])
    Coordinates:
      * lat      (lat) int32 0 1
      * lon      (lon) int32 0
      * band    (band) <U1 'r' 'g' 'b' 'a'
    Attributes:
        units:    {'r': <Unit('meter')>, 'g': <Unit('foot')>, 'b': <Unit('kilomet...
    ```
* Dimensions should be returned in the order of the requested `coordinates`
    * eg. if underlying dataset has `['lat', 'lon']` `coordinates`, but the request has `['lon', 'lat']` coordinates, the output should match `['lon', 'lat']`
    ```python
    >>> node.native_coordinates = Coordinate(lat=(90, -90, -1.), lon=(-180, 180, 2.), order=['lat', 'lon'])
    >>> node.evaluated_coordinates = Coordinate(lon=(-180, 180, 4.), lat=(90, -90, -2.), order=['lon', 'lat'])
    >>> node.initialize_output_array().dims
    ('lon', 'lat')
    ```
* If the underlying Node has unstacked dimensions not in the request, an exception is raised
    * eg. Node has `['lat', 'lon', 'time']` dimensions, but coordinates only have `['time']`
    ```python
    >>> node.native_coordinates = Coordinate(lat=(90, -90, -1.), lon=(-180, 180, 2.), order=['lat', 'lon'])
    >>> node.evaluated_coordinates = Coordinate(lon=(-180, 180, 4.),)
    >>> node.initialize_output_array()
    CoordinateError: 'Dimension "lat" not present in requested coordinates with dims ["lon"]'
    ```
    * Because some datasets may be huge, and without information about the subset, the safest behaviour is to throw an exception
* If the underlying Node has *stacked* dimensions not in the request, raise an exception if the Node native_coordinates contains duplicates for the requested dimensions, otherwise just drop the missing dimension from the coordinates.
  * eg. Node has `[lat_lon_time]` dimensions, but coordinates only have `['time'`], we can drop the lat and lon portion of the stacked coordinates as long as that doesn't result in duplicate times. Note that this doesn't change the dimensionality, but is required for correct xarray broadcasting.
* If the request has unstacked dimensions not in the Node, just return without those dimensions
    * eg. Node has `['lat', 'lon']` dimensions, but evaluated coordinates have `['time', 'lon', 'lat']` then `UnitsDataArray` will have dimensions `['lon', 'lat']`
    ```python
    >>> node.native_coordinates = Coordinate(lat=45, lon=0, order=['lat', 'lon'])
    >>> node.evaluated_coordinates = Coordinate(lat=45, lon=0, time='2018-01-01', order=['lat', 'lon', 'time'])
    >>> node.initialize_output_array()
    <xarray.UnitsDataArray (lat: 1, lon: 1)>
    array([[nan]])
    Coordinates:
      * lat      (lat) float64 45.0
      * lon      (lon) float64 0.0
    ```
* If the request has *stacked* dimensions not in the underlying Node, add the missing coordinates.
    * eg. Node has `[lat_lon]` dimensions, but coordinates have `['lat_lon_time'`], we need to add the time portion to the stacked coordinates for correct xarray broadcasting.

* [**???FUTURE FEATURE???**] If `isinstance(coordinates, GroupCoordinates)` then a `UnitsDataArray` with a single `group` dimension is return. Each `group` is indexed by the the coordinates group in `GroupCoordinates`, and contains a `UnitsDataArray` that matches the above rules.
    ```python
    >>> from podpac.core.coordinate import GroupCoordinate
    >>> node.native_coordinates = Coordinate(lat=(90, -90, -1.), lon=(-180, 180, 2.), order=['lat', 'lon'])
    >>> node.evaluated_coordinates = GroupCoordinate([Coordinate(lat=(90, 0, -1.), lon=(-180, 0, 2.), order=['lat', 'lon']), Coordinate(lat=(0, -90, -2.), lon=(0, 180, 4.), order=['lat', 'lon'])])
    >>> o = node.initialize_output_array()
    >>> o[0] + o; o + o  # This all works fine
    <xarray.UnitsDataArray (group: 2)>
    array([<xarray.UnitsDataArray (lat: 91, lon: 91)>
    array([[nan, nan, nan, ..., nan, nan, nan],
           ...,
           [nan, nan, nan, ..., nan, nan, nan]])
    Coordinates:
      * lat      (lat) float64 90.0 89.0 88.0 87.0 86.0 85.0 84.0 83.0 82.0 81.0 ...
      * lon      (lon) float64 -180.0 -178.0 -176.0 -174.0 -172.0 -170.0 -168.0 ...
    Attributes:
        layer_style:  <podpac.core.node.Style object at 0x0000022DEE17AB38>
        params:       {},
           <xarray.UnitsDataArray (lat: 46, lon: 46)>
    array([[nan, nan, nan, ..., nan, nan, nan],
           ...,
           [nan, nan, nan, ..., nan, nan, nan]])
    Coordinates:
      * lat      (lat) float64 0.0 -2.0 -4.0 -6.0 -8.0 -10.0 -12.0 -14.0 -16.0 ...
      * lon      (lon) float64 0.0 4.0 8.0 12.0 16.0 20.0 24.0 28.0 32.0 36.0 ...
    Attributes:
        layer_style:  <podpac.core.node.Style object at 0x0000022DEE17AB38>
        params:       {}], dtype=object)
    Coordinates:
      * group    (group) <U144 'Coordinate\n\tlat: UniformCoord: Bounds[-90.0, 0.0], N[46], ctype["segment"]\n\tlon: UniformCoord: Bounds[-180.0, 0.0], N[46], ctype["segment"]' ...    
    ```
    * This will break any node that uses `o.data`... 
    * Alternatively, we could disallow evaluation of `GroupCoordinates` -- raising an exception
        * This leaves it up to the user to loop through all the coordinates in `GroupCoordinates` and evaluate the node that way
        * Perhaps we can do that? In that case only the bottom node returns the `group` dataarray, and anything internal should be fine
        * This will break if an internal node uses a group coordinate as part of the pipeline
        * Safest is just to disallow this behaviour... 
    * Alternatively (Currently the favorite): WE could raise an exception, and provide a eval_group method that will do the looping for the user. This avoids any group coordinates evaluated within the pipeline, but still let's a user evaluate a group in a consistent manner. 
* The output will contain metadata on the: 
    * Units
    * Styling? 
    * Other metadata to track provenance? 
    
#### find_coordinates
```python
def find_coordinates(dims=None, bounds=None, number=None, sortby='size', stop_types=[Compositor, DataSource]):
    '''This is a helper function to get all the native coordinates within a pipeline, sorted by
       the largest extents, or highest resolution

    Parameters
    -----------
    dims : str/list, optional
        The dimension or set of dimensions for which the user wants to find the underlying dataset coordinates. If None, all available dimensions for underlying coordinates will be found. Stacked dimensions cannot be used (or are automatically unstacked)
    bounds : dict, optional
        Default is None, in which case search is not restricted to a bounding window. Bound within which to search for coordinates. Format should be {'dim': (start, end)}
    number : int, optional
        Default is None, in which case all coordinates found are returned. Otherwise, only the first `number` of coordinates are returned, based on the `sortby` specified. If number is <0, the the last `number` of coordinates are returned. 
    sortby : str, optional
        Default is 'size'. `sortby` should be in ['size', 'extents']. If 'size', the returned coordinates are sorted by the number of coordinates in the dimension, with the largest first. If 'extents', the returned coordinates are sorted by the largest geographical extent. In case of ties, the sorting looks at the other option to break the tie. 
    stop_type : list, optional
        List of node types where search should be stopped. By default, searches stop at DataSource and Compositor nodes. Remove the compositor node to search for native coordinates of individual files, but this may take a long time.
        
    Returns
    --------
    OrderedDict:
        Format is as follows: {dims[0]: {'node_address0': Coordinate1D,
                                         'node_address1': Coordinate1D, ...},
                               dims[1]: {'node_address2': Coordinate1D,
                                         'node_address0': Coordinate1D, ...}}
       where find_coordinates(dims)[dims[0]].items() are sorted as specified

    Notes
    ------
    The `native_coordinates` could be retrieved using (for example):
    >>> c = node.find_coordinates('time')
    >>> node_key = list(c['time'].keys())[0]
    >>> node[node_key].native_coordinates
    '''
    pass
```

#### public cache interface

```python
def get_cache(self, key, coordinates):
    '''Get cached data for this node.
    
    Parameters
    ------------
    key : str
        Cached object key, e.g. 'output'.
    coordinates : Coordinates
        Coordinates for which cached object should be retrieved
        
    Returns
    -------
    UnitsDataArray 
        The cached data from the requested coordinates/attrs
    
    Raises
    -------
    CacheError
        If the data is not in the cache and evaluate == False
    '''
```

```python
def del_cache(self, key, coordinates):
    '''Delete cached data for this node
    
    Parameters
    ------------
    key : str
        Cached object key, e.g. 'output'.
    coordinates : Coordinates
        Coordinates for which cached object should be retrieved
    '''
```


```python
def put_cache(self, key, coordinates, data):
    '''Cache data for this node
    
    Parameters
    ------------
    key : str
        Cached object key, e.g. 'output'.
    coordinates : Coordinates
        Coordinates for which cached object should be retrieved
    data : any
        Data to cache
    '''
```


```python
def has_cache(self, key, coordinates):
    '''Check for cached data for this node
    
    Parameters
    ------------
    key : str
        Cached object key, e.g. 'output'.
    coordinates: Coordinate 
        Coordinates for which cached object should be retrieved
    
    Returns
    -------
    has_cache : bool
         True if there as a cached object for this node for the given key and coordinates.
    '''
```

### definition
```python
def definition(self, type='dict'):
    ''' Returns the pipeline node definition in the desired format
    
    Parameters
    -----------
    type: str, optional
        Default is 'dict', which returns a dictionary definition of the pipeline. 
        'json': returns a json-formatted text string of the pipeline
        'rich': returns a javascript widget <-- should this be 'widget' instead? 
        'node': Returns a PipelineNode instance of this object
    
    Returns
    --------
    various
        Depends on type, see above. 
    '''
```
* There are a few shortcut properties that call this function
    * `pipeline_json`
    * `pipeline_rich` 
    * `pipeline_dict`
    * `pipeline_node`
    
    
## Developer interface 
### Public Attributes
* `native_coordinates`: Underlying data source coordiantes, when available
* `units`: Units associated with the output of this node
* `interpolation`: Interpolation method used with this node

### Private Attributes
* `_output`: last output from this node - property, looks in cache
* `_requested_coordinates`: last coordinates used for evaluating the node
* `_output_coordinates`: output coordinates from the last evaluation

### Additional public methods
( Some of these are already documented in the code )
* cache_path: file where object is cached
* cache_dir: directory where objects are cached
* init: Used for initialization of derived classes
* _first_init: Used to do any initialization before any of the core initialization is done
* create_output_array
* get_cached_object: similar go 'get', but specifies a particular filename? Maybe superceded by caching refactor?
* save_to_disk? 

### Testing
* I think testing should live in the common_test_utils. I.e. separate from this node. 
