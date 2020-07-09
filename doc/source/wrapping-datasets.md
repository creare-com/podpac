# Wrapping Datasets

Wrapping a new dataset is challenging because you have to understand all of the quirks of the new dataset and deal with the quirks of PODPAC as well. This reference is meant to record a few rules of thumb when wrapping new datasets to help you deal with the latter. 

## Rules
1. When evaluating a node with a set of coordinates:
   a. The evaluation coordinates must include ALL of the dimensions present in the source dataset
   b. The evaluation coordinates MAY contain additional dimensions NOT present in the source dataset, and the source may ignore these
2. When returning data from a data source node:
   a. The ORDER of the evaluation coordinates MUST be preserved (see `UnitsDataArray.part_transpose`)
   b. Any multi-channel data must be returned using the `output` dimension which is ALWAYS the LAST dimension
   
## Guide
In theory, to wrap a new `DataSource`:
1. Create a new class that inherits from `podpac.core.data.DataSource` or a derived class (see the `podpac.core.data` module for generic data readers).
2. Implement a method for opening/accessing the data, or use an existing generic data node and hard-code certain attributes
3. Implement the `get_coordinates(self)` method
4. Implement the `get_data(self, coordinates, coordinates_index)` method
    a. `coordinates` is a `podpac.Coordinates` object and it's in the same coordinate system as the data source (i.e. a subset of what comes out of `get_coordinates`)
    b. `coordinates_index` is a list (or tuple?) of slices or boolean arrays or index arrays to indexes into the output of `get_coordinates()` to produce `coordinates` that come into this function. 
    
In practice, the real trick is implementing a compositor to put multiple tiles together to look like a single `DataSource`. We tend to use the `podpac.compositor.OrderedCompositor` node for this task, but it does not handle interpolation between tiles. Instead, see the `podpac.core.compositor.tile_compositor` module. 

When using compositors, it is prefered the that `sources` attribute is populated at instantiation, but on-the-fly (i.e. at eval) population of sources is also acceptible and sometimes necessary for certain datasources. 

For examples, check the `podpac.datalib` module. 

Happy wrapping!