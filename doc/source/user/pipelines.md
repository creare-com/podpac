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

The podpac core library includes three basic types of nodes: *DataSource*, *Compositor*, and *Algorithm*. A *Pipeline* node can also be used an an input to a pipeline. These nodes and their additional attributes are described below.

### Common Attributes

 * `node`: a path to the node class. The path is relative to the podpac module, unless `plugin` is defined. See Notes. *(string, required)*
 * `plugin`: a path to a plugin module to use (prepended node path). See Notes. *(string, optional)*
 * `attrs`: set attributes in the node for custom behavior. Each value can be a number, string, boolean, dictionary, or list. *(object, optional)*

## DataSource

### Sample

```
{
    "nodes": {
        "sm": {
            "node": "algorithm.CoordData",
            "attrs": {
                "coord_name": "time"
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

### Additional Attributes
 * `inputs`: node inputs to the algorithm. *(object, required)*

### Sample

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
            "attrs": {
                "eqn": "A + {tsmtr} / {kappa} * (B - C)",
                "params": {
                    "kappa": "13",
                    "tsmtr": "0.3"
                }
            }
        }
    }
}
```

## Pipeline

### Additional Attributes
 * `path`: path to another pipeline JSON definition. *(string, required)*

### Sample

```
{
    "nodes": {
        "MyDataSource": {
            ...
        },
        
        "MyOtherPipeline": {
            "path": "path to pipeline"
        },
        
        "result": {
            "node": "Arithmetic",
            "inputs": {
                "A": "MyDataSource",
                "B": "MyOtherPipeline",
            },
            "attrs": {
                "eqn": "A + B"
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

## Output Definition

The output definition defines the node to output and, optionally, an additional output mode along with associated parameters. If an output definition is not supplied, the last defined node is used.

Podpac provides several builtin output types, *file* and *image*. You can also define custom outputs in a plugins.

### Common Attributes

 * `node`: The nodes to output. *(list, required)*
 * `mode`: For builtin outputs, options are 'none' (default), 'file', 'image'. *(string, optional)*

## None (default)

No additional output. The output will be returned from the `Pipeline.execute` method.

## Files

Nodes can be output to file in a variety of formats.

### Additional Attributes

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

### Additional Attributes

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

## Custom Outputs

Custom outputs can be defined in a plugin by subclassing the `Output` base class found in `core.pipeline.output`. Custom 
outputs must define the `write` method with no arguments, and may define additional parameters.

### Attributes

Replace the 'mode' parameter with a plugin path and output class name:

 * `plugin`: path to a plugin module to use *(string, required)*
 * `output`: output class name *(string, required)*

### Sample Custom Output Class

File: **my_plugin/outputs.py**

```
import numpy as np
import traitlets as tl
import podpac

class NpyOutput(podpac.core.pipeline.output.Output):
    path = tl.String()
    allow_pickle = tl.Bool(True)
    fix_imports = tl.Bool(True)

    def write(self):
        numpy.save(self.path, self.node.output.data, allow_pickle=self.allow_pickle, fix_imports=self.fix_imports)
```

### Sample Pipeline

```
{
    "nodes": {
        "MyNode1": { ... },
        "MyNode2": { ... }
    },

    "output": {
        "nodes": "MyNode2",
        "plugin": "my_plugin",
        "output": "NpyOutput",
        "path": "my_pipeline_output.npy",
        "allow_pickle": false
    }
}
```
