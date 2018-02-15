
# Pipeline

A podpac pipeline can be defined using JSON. The pipeline definition describes the *nodes* used in the pipeline and, optionally, *outputs* to produce when executing the pipeline.

## Attributes

 * `nodes`: (object) node definitions
 * `outputs`: (list) output definitions, optional

## Sample

```
{
    "nodes": {
        "myNode": { ... },
        "myOtherNode": { ... }
    },
    "outputs": [
        { ... },
    ]
}
```

# Node definitions

A node definition defines the node and its inputs and parameters. It also names the node so that it can be used as an input to other nodes in the pipeline. Nodes must be defined before they are referenced in a later node.

There are three basic *node types*: DataSource, Compositor, and Algorithm.

## Common Attributes

 * `node`: (string) a path to the node class. The path is relative to the podpac module, unless `plugin` is defined. See Notes.
 * `plugin`: (string) a path to a plugin module to use (prepended node path). See Notes.
 * `attrs`: (object) explicitly set attributes in the node for custom behavior.
 * `evaluate`: (bool) if `false`, the pipeline will not execute this node automatically. This is useful for nodes that will be executed implicitly by a later node. Default: `true`.

## DataSource

###  Attributes
 * `source`: (string) the dataset source

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

### Attributes

 * `sources`: (list) nodes to composite.

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
 * `inputs`: (object) node inputs to the algorithm.
 * `params`: (object) non-node inputs to the algorithm.

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

## Notes

 * The `node` path should include the submodule path and the node class. The submodule path is omitted for top-level classes. For example:
   - `"node": "datalib.smap.SMAP"` is equivalent to `from podpac.datalib.smap import SMAP`.
   - `"node": "OrderedCompositor"` is equivalent to `from podpac import OrderedCompositor`.
 * The `plugin` path replaces 'podpac' in the full node path. For example
   - `"plugin": "path.to.myplugin", "node": "mymodule.MyCustomNode"` is equivalent to `from path.to.myplugin.mymodule import MyCustomNode`.
   - `"plugin": "myplugin", "node": "MyCustomNode"` is equivalent to `from myplugin import MyCustomNode`

# Output Definitions