# Introduction 

This document describes the detailed interfaces for Pipeline nodes so that a user may know what to expect. It also documents some of the available nodes implemented as part of the core library. 

# Nodes

... tbd ... (for now see the [DeveloperSpec](https://github.com/creare-com/podpac/blob/develop/doc/source/developer/specs/nodes.md))

## DataSource

... tbd ...

## Compositor

... tbd ...

## Algorithm

... tbd ...

## Extending Podpac with Custom Nodes

In addition to the core data sources and algorithms, you may need to write your own node to handle unique data sources or additional data processing. You can do this by subclassing a core podpac node and extending it for your needs. The DataSource node in particular is designed to be extended for new sources of data.

### Example

An example of creating a simple array-based datasource can be found in the [array-data-source](https://github.com/creare-com/podpac/blob/master/doc/notebooks/array-data-source.ipynb) notebook. 

### Tagging params

Execution parameters are passed in when the node is executed. For each parameter, the class definition should include a traitlets attribute that is tagged as a `param`. You can optionally include a default value for the param.

```
class MyDataSource(DataSource):
    my_param1 = tl.Integer(allow_none=False).tag(param=True)
    my_param2 = tl.Integer(default_value=100.0).tag(param=True)

    ...
```

You will be able to set params when instantiating the node and when executing the node:

```
node = MyDataSource(my_param1=0.5)
output = node.execute(coords, {'my_param2': 75.0})
```

You will also be able to set these tagged params in [pipelines](pipelines).

### Tagging attributes

Unlike params, node attributes are defined when instantiating the node, but cannot be set when later executing the node. Again, for each attribute, the class definition should include a traitlets attribute that is tagged as an `attr`, and you can optionally include a default value.

```
class MyDataSource(DataSource):
    my_param1 = tl.Integer(allow_none=False).tag(param=True)
    my_param2 = tl.Integer(default_value=100.0).tag(param=True)

    my_attr = tl.Integer(default_value=0.1).tag(attr=True)

    ...
```

You will be able to set attrs when instantiating the node:

```
node = MyDataSource(my_param1=0.5, my_attr=0.5)
output = node.execute(coords, {'my_param2': 75.0})
```

You will also be able to set these tagged attrs in [pipelines](pipelines).