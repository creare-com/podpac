# Nodes

This document describes the detailed interfaces for core node types so that a user may know what to expect. It also documents some of the available nodes implemented as part of the core library. 

In PODPAC, Nodes represent the basic unit of computation. They take inputs, produce outputs, and can represent source data, intermediate results, or final output. The base `Node` class defined a common interface for all PODPAC `Nodes`.

In particular, the base `Node` class implements:

- Caching behaviour of `Node` outputs, and interfaces with the cache system
- Serialization and deserialization of `Nodes` using our JSON format
- Saving and loading `Node` outputs
- Creating `Node` output data structures using the `create_output_array` method.
- Common interfaces required and used by all subsequent nodes:
    * `Node.eval(coordinates, output)`
    * `Node.find_coordinates()`

## DataSource

DataSource nodes interface with remote geospatial data sources (i.e. raster images, DAP servers, numpy arrays) and define how to retrieve data from these remote sources using PODPAC coordinates. PODPAC defines common generic DataSource nodes (i.e. Array, PyDAP), but advanced users can define their own DataSource nodes by defining the methods to retrieve data (`get_data(coordinates, index)`) and the method to define the `coordinates` property (`get_native_coordinates()`).

Key properties of DataSource nodes include:

- `source`: The location of the source data. Depending on the child node this can be a filepath, numpy array, or server URL).
- `coordinates`: The PODPAC coordinates of the data in `source`
- `interpolation`: Definition of the interpolation method to use with the data source.
- `nan_vals`: List of values from source data that should be interpreted as 'no data' or 'nans'.
- `boundary`: A structure defining the boundary of each data point in the data source (for example to define a point, area, or arbitrary polygon)

To evaluate data at arbitrary PODPAC coordinates, users can input `coordinates` to the eval method of the DataSource node. The DataSource `eval` process consists of the following steps:

1. Verify native coordinates support the dimensions of the requested coordinates
2. Remove extra dimensions from the requested coordinates
3. Intersect the requested coordinates with the native coordinates
4. Further down-select the necessary native coordinates if the requested coordinates are sparser than the native coordinates.
5. Retrieve data from the data source (`get_data`) at the subset of native coordinates
6. Interpolate the retrieved data to the locations of the requested coordinates

### Interpolation

The DataSource `interpolation` property defines how to handle interpolation of coordinates and data within the DataSource node. Based on a string or dictionary definition, the DataSource instantiates an Interpolation class that orchestrates the selection and use of different interpolators depending on the native and input coordinates. PODPAC natively supports an array of interpolator methods covering a wide range of use cases. Users can also write their own Interpolator class to use for specific nodes. Under the hood, PODPAC leverages interpolation methods from xarray, scipy, and rasterio to do some of the heavy lifting.

Definition of the interpolation method on a DataSource node may either be a string:

```python
interpolation = 'nearest'  # nearest neighbor interpolation
```

or a dictionary that supplies extra parameters:

```python
interpolation = {
    'method': 'nearest',
    'params': {
        'spatial_tolerance': 1.1
    }
}
```

For the most advanced users, the interpolation definition supports defining different interpolation methods for different dimensions (as of 2.0.0 this functionality is not fully implemented):

```python
interpolation = [
    {
        'method': 'bilinear',
        'dims': ['lat', 'lon']
    },
    {
        'method': 'nearest',
        'dims': ['time']
    }
]
```

When a DataSource node is created, the interpolation manager selects a list of applicable `Interpolator` classes to apply to each set of defined dimensions. When a DataSource node is being evaluated, the interpolation manager chooses the first interpolator that is capable of handling the dimensions defined in the requested coordinates and the native coordinates using the `can_interpolate` method. After selecting an interpolator for all sets of dimensions, the manager sequentially interpolates data for each set of dimensions using the `interpolate` method.

## Compositor

`Compositor` `Nodes` are used to combine multiple data files or dataset into a single interface. 

The `BaseCompositor` implements:

- The `find_coordinates` method
- The `eval` method
- The `iteroutputs` method used to iterate over all possible input data sources
- The `select_sources(coordinates`) method to sub-select input data sources BEFORE evaluating them, as an optimization
- The interface for the `composite(coordinates, data_arrays, result)` method. Child classes implement this method which determines the logic for combining data sources.

Beyond that there is the:

- `OrderedCompositor`
    - This is meant to composite disparate data sources together that might have different resolutions and coverage
    - For example, prefer a high resolution elevation model which has missing data, but fill missing values with a coarser elevation datasource
    - In practice, we use this `Compositor` to provide a single interface for a dataset that is divided into multiple files
    - Data sources are composited AFTER harmonization.
- `TileCompositor`
    - This is meant to composite a data source stored in multiple files into a single interface
    - For example, consider an elevation data source that covers the globe and is stored in 10K different files that only cover land areas
    - Data source are composited BEFORE harmonization

## Algorithm

`Algorithm` `Nodes` are the backbone of the pipeline architecture and are used to perform computations on one or many data sources or the user-requested coordinates.

The `BaseAlgorithm`, `Algorithm` (for multiple input nodes) and `UnaryAlgorithm` (for single input nodes) `Nodes` implement the basic functionality:

- The `find_coordinates` method
- The `Algorithm.eval` method for multiple input `Nodes`
- The `inputs` property that finds any PODPAC `Node` as part of the class definition
- The interfaces for the `algorith(inputs)` method which is used to implement the actual algorithm

Based on this basic interface, PODPAC implements algorithms that manipulate coordinates, does signal processing (e.g. convolutions), statistics (e.g. Mean), and completely generic, user-defined algorithms. 

In particular, the `Arithmetic` allows users to specify and `eqn` which allows nearly arbitrary point-wise computations. Also the `Generic` algorithm allows users to specify arbitrary Python code, as long as the `output` variable is set. 

## Extending Podpac with Custom Nodes

In addition to the core data sources and algorithms, you may need to write your own node to handle unique data sources or additional data processing. You can do this by subclassing a core podpac node and extending it for your needs. The DataSource node in particular is designed to be extended for new sources of data.

### Example

An example of creating a simple array-based datasource can be found in the [array-data-source](https://github.com/creare-com/podpac-examples/blob/master/notebooks/4-advanced/create-data-source.ipynb) notebook. 

### Tagging attributes

Node attributes are defined when instantiating the node. For each attribute, the class definition should include a traitlets attribute that is tagged as an `attr`, and you can optionally include a default value.

```
class MyDataSource(DataSource):
    my_attr1 = tl.Float(allow_none=False).tag(attr=True)
    my_attr2 = tl.Float(default_value=0.1).tag(attr=True)

    ...
```

You will be able to set attrs when instantiating the node:

```
node = MyDataSource(my_attr1=0.3, my_attr2=0.5)
output = node.execute(coords)
```

You will also be able to set these tagged attrs in node definitions.

## Serialization

Any podpac Node can be saved, shared, and loaded using a JSON definition. This definition describes all of the nodes required to create and evaluate the final Node.

**Serialization Properties and Methods:**

 * `save(path)`: writes the node to file.
 * `json`: provides the node definition as a JSON string.
 * `definition`: provides the node definition as an ordered dictionary.

**Deserialization Methods:**

 * `load(path)`: load a Node from file.
 * `from_json`: create a Node from a JSON definition
 * `from_definition`: create a Node from a dictionary definition
 * `from_url`: create a Node from a WMS/WCS request.

## Node Definition

The full JSON definition for a Node contains an entry for *all* of the nodes required. Input nodes must be defined before they can be referenced by later Nodes, so the final output Node must be at the end.

Individual node definition specify the node class along with its inputs and attributes. It also names the node so that it can be used as an input to other nodes later in the pipeline. The following properties are common to all podpac node definitions:

 * `node`: a path to the node class. The path is relative to the podpac module, unless `plugin` is defined. See Notes. *(string, required)*
 * `plugin`: a path to a plugin module to use (prepended node path). See Notes. *(string, optional)*
 * `attrs`: set attributes in the node for custom behavior. Each value can be a number, string, boolean, dictionary, or list. *(object, optional)*

Additional properties and examples for each of the core node types are provided below.

### DataSource

#### Sample

```
{
    "mynode": {
        "node": "algorithm.CoordData",
        "attrs": {
            "coord_name": "time"
        }
    }
}
```

### Compositor

#### Additional Properties

 * `sources`: nodes to composite *(list, required)*

#### Sample

```
{
    "SourceA": { ... },
    "SourceB": { ... },
    "SourceC": { ... },

    MyCompositor": {
        "node": "OrderedCompositor",
        "sources": ["SourceA", "SourceB", "SourceC"]
    }
}
```

### Algorithm

#### Additional Properties
 * `inputs`: node inputs to the algorithm. *(object, required)*

#### Sample

```
{
    "MyNode": { ... },
    "MyOtherNode": { ... },
    "MyThirdNode": { ... },

    "MyResult": {
        "node": "Arithmetic",
        "inputs": {
            "A": "MyNode",
            "B": "MyOtherNode",
            "C": "MyThirdNode"
        },
        "attrs": {
            "eqn": "A + {tsmtr} / {kappa} * (B - C)",
            "params": {
                "kappa": "13",
                "tsmtr": "0.3"
            }
        }
    }
}
```

### Notes

 * The `node` path should include the submodule path and the node class. The submodule path is omitted for top-level classes. For example:
   - `"node": "datalib.smap.SMAP"` is equivalent to `from podpac.datalib.smap import SMAP`.
   - `"node": "compositor.OrderedCompositor"` is equivalent to `from podpac.compositor import OrderedCompositor`.
 * The `plugin` path replaces 'podpac' in the full node path. For example
   - `"plugin": "path.to.myplugin", "node": "mymodule.MyCustomNode"` is equivalent to `from path.to.myplugin.mymodule import MyCustomNode`.
   - `"plugin": "myplugin", "node": "MyCustomNode"` is equivalent to `from myplugin import MyCustomNode`