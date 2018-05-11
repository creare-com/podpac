# Examples

See [the example notebooks](https://github.com/creare-com/podpac/doc/notebooks) for Jupyter Notebook examples. 

*Note*: Not all the examples will work because we use internal WCS sources for development. 

The general workflow for retrieving data using PODPAC is as follows: 

```python
import podpac  # import the library
c = podpac.Coordinate(time='2018-01-01 12:00:00', lat=0, lon=0)  # Create a coordinate
n = podpac.datalib.smap.SMAP()  # Create node
o = n.execute(c)  # Execute node to retrieve data at the coordinates
```

Examples of creating algorithms and pipelines can be found in the Notebooks. 
