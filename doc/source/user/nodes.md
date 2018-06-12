
# Nodes

TODO

## DataSource

TODO

## Compositor

TODO

## Algorithm

TODO

## Extending Podpac with Custom Nodes

In addition to the core data sources and algorithms, you may need to write your own node to handle unique data sources or additional data processing. You can do this by subclassing a core podpac node and extending it for your needs. The DataSource node in particular is designed to be extended for new sources of data.

### Example

TODO (DataSource example)

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

You will also be able to set these tagged params in [pipelines](pipelines.md).

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

You will also be able to set these tagged attrs in [pipelines](pipelines.md).