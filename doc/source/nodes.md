# Nodes

This document describes the detailed interfaces for core node types so that a user may know what to expect. It also documents some of the available nodes implemented as part of the core library. 

... tbd ... (for now see the [DeveloperSpec](https://github.com/creare-com/podpac/blob/develop/doc/source/developer/specs/nodes.md))

## DataSource

DataSource nodes interface with remote geospatial data sources (i.e. raster images, DAP servers, numpy arrays) and define how to retrieve data from these remote sources using PODPAC coordinates. PODPAC defines common generic DataSource nodes (i.e. Array, PyDAP), but advanced users can define their own DataSource nodes by defining the methods to retrieve data (`get_data(coordinates, index)`) and the method to define the `native_coordinates` property (`get_native_coordinates()`).

Key properties of DataSource nodes include:

- `source`: The location of the source data. Depending on the child node this can be a filepath, numpy array, or server URL).
- `native_coordinates`: The PODPAC coordinates of the data in `source`
- `interpolation`: Definition of the interpolation method to use with the data source.
- `nan_vals`: List of values from source data that should be interpreted as 'no data' or 'nans'.

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
node.interpolation = 'nearest'  # nearest neighbor interpolation
```

or a dictionary that supplies extra parameters:

```python
node.interpolation = {
    'method': 'nearest',
    'params': {
        'spatial_tolerance': 1.1
    }
}
```

For the most advanced users, the interpolation definition supports defining different interpolation methods for different dimensions:

```python
node.interpolation = {
    ('lat', 'lon'): 'bilinear', 
    'time': 'nearest'
}
```

When a DataSource node is created, the interpolation manager selects a list of applicable `Interpolator` classes to apply to each set of defined dimensions. When a DataSource node is being evaluated, the interpolation manager chooses the first interpolator that is capable of handling the dimensions defined in the requested coordinates and the native coordinates using the `can_interpolate` method. After selecting an interpolator for all sets of dimensions, the manager sequentially interpolates data for each set of dimensions using the `interpolate` method.

## Compositor

... tbd ...

## Algorithm

... tbd ...

## Extending Podpac with Custom Nodes

In addition to the core data sources and algorithms, you may need to write your own node to handle unique data sources or additional data processing. You can do this by subclassing a core podpac node and extending it for your needs. The DataSource node in particular is designed to be extended for new sources of data.

### Example

An example of creating a simple array-based datasource can be found in the [array-data-source](https://github.com/creare-com/podpac/blob/master/doc/notebooks/array-data-source.ipynb) notebook. 

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

## DataSource

### Sample

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

## Compositor

### Additional Properties

 * `sources`: nodes to composite *(list, required)*

### Sample

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

## Algorithm

### Additional Properties
 * `inputs`: node inputs to the algorithm. *(object, required)*

### Sample

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

## Notes

 * The `node` path should include the submodule path and the node class. The submodule path is omitted for top-level classes. For example:
   - `"node": "datalib.smap.SMAP"` is equivalent to `from podpac.datalib.smap import SMAP`.
   - `"node": "compositor.OrderedCompositor"` is equivalent to `from podpac.compositor import OrderedCompositor`.
 * The `plugin` path replaces 'podpac' in the full node path. For example
   - `"plugin": "path.to.myplugin", "node": "mymodule.MyCustomNode"` is equivalent to `from path.to.myplugin.mymodule import MyCustomNode`.
   - `"plugin": "myplugin", "node": "MyCustomNode"` is equivalent to `from myplugin import MyCustomNode`