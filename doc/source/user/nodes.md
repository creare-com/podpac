# Nodes
## Introduction 
This document describes the detailed interfaces for Pipeline nodes so that a user may know what to expect. It also documents some of the available nodes implemented as part of the core library. 

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

You will also be able to set these tagged attrs in [pipelines](pipelines).