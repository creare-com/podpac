# Pipelines

A podpac pipeline can be defined using JSON. The pipeline definition describes the *nodes* used in the pipeline and the *output* for the pipeline.

### Attributes

 * `nodes`: node definitions  *(object, required)*
 * `output`: output definition *(object, optional)*

### Sample

```
{
    "nodes": {
        "myNode": { ... },
        "myOtherNode": { ... }
        ...
        "myResult": { ... }
    },
    "output": {
        "node": "myResult",
        "mode": "file",
        ...
    }
}
```

## Node definitions

A node definition defines the node and its inputs, attributes, and default execution parameters. It also names the node so that it can be used as an input to other nodes in the pipeline. Nodes must be defined before they are referenced in a later node.

All nodes must be one of these three basic types: *DataSource*, *Compositor*, and *Algorithm*.

### Common Attributes

 * `node`: a path to the node class. The path is relative to the podpac module, unless `plugin` is defined. See Notes. *(string, required)*
 * `plugin`: a path to a plugin module to use (prepended node path). See Notes. *(string, optional)*
 * `attrs`: set attributes in the node for custom behavior. Each value can be a number, string, boolean, dictionary, or list. *(object, optional)*
 * `params`: set default execution parameters. Each value can be a number, string, boolean, dictionary, or list. *(object, optional)*
 * `evaluate`: execute this node automatically. Setting this to `false` is useful for nodes that will be executed implicitly by a later node. *(bool, optional, default `true`)*

## DataSource

### Sample

```
{
    "nodes": {
        "sm": {
            "node": "datalib.smap.SMAP",
            "attrs": {
                "product": "SPL4SMAU.003",
                "interpolation": "bilinear"
            }
        }
    }
}
```

## Compositor

### Additional Attributes

 * `sources`: nodes to composite *(list, required)*

### Sample

```
{
    "nodes": {
        "SourceA": { ... },
        "SourceB": { ... },
        "SourceC": { ... },

        MyCompositor": {
            "node": "OrderedCompositor",
            "sources": ["SourceA", "SourceB", "SourceC"]
        }
    }
}
```

## Algorithm

### Attributes
 * `inputs`: node inputs to the algorithm. *(object, required)*

```
{
    "nodes": {
        "MyNode": { ... },
        "MyOtherNode": { ... },
        "MyThirdNode": { ... },

        "downscaled_sm": {
            "node": "Arithmetic",
            "inputs": {
                "A": "MyNode",
                "B": "MyOtherNode",
                "C": "MyThirdNode"
            },
            "params": {
                "kappa": "13",
                "tsmtr": "0.3", 
                "eqn": "A + {tsmtr} / {kappa} * (B - C)"
            }
        }
    }
}
```

### Notes

 * The `node` path should include the submodule path and the node class. The submodule path is omitted for top-level classes. For example:
   - `"node": "datalib.smap.SMAP"` is equivalent to `from podpac.datalib.smap import SMAP`.
   - `"node": "OrderedCompositor"` is equivalent to `from podpac import OrderedCompositor`.
 * The `plugin` path replaces 'podpac' in the full node path. For example
   - `"plugin": "path.to.myplugin", "node": "mymodule.MyCustomNode"` is equivalent to `from path.to.myplugin.mymodule import MyCustomNode`.
   - `"plugin": "myplugin", "node": "MyCustomNode"` is equivalent to `from myplugin import MyCustomNode`

## Output Definition

The output definition defines the node to output and, optionally, an additional output mode along with associated parameters. If an output definition is not supplied, the last defined node is used.

Podpac provides several output types: *file* and *image*. Currently custom output types are not supported.

### Common Attributes

 * `node`: The nodes to output. *(list, required)*
 * `mode`: The output mode, options are 'none' (default), 'file', 'image'. *(string, optional)*

## None (default)

No additional output. The output will be returned from the `Pipeline.execute` method.

## Files

Nodes can be output to file in a variety of formats.

### Attributes

 * `format`: file format, options are 'pickle' (default), 'geotif', 'png'. *(string, optional)*
 * `outdir`: destination path for the output file *(string, required)*

### Sample

```
{
    "nodes": {
        "MyNode1": { ... },
        "MyNode2": { ... }
    },

    "output": {
        "nodes": "MyNode2",
        "mode": "file",
        "format": "png",
        "outdir": "C:\Path\To\OutputData"
    }
}
```

## Images

Nodes can be output to a png image (in memory).

### Attributes

 * `format`: image format, options are 'png' (default). *(string, optional)*
 * `vmin`: min value for the colormap *(number, optional)*
 * `vmax`: max value for the colormap *(number, optional)*

### Sample

```
{
    "nodes": {
        "MyNode1": { ... },
        "MyNode2": { ... }
    },

    "output": {
        "nodes": "MyNode2",
        "mode": "image",
        "format": "png",
        "vmin": 0.1,
        "vmax": 0.35
    }
}
```
