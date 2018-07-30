# Examples

See [the example notebooks](https://github.com/creare-com/podpac/tree/master/doc/notebooks) for Jupyter Notebook examples. 

*Note*: Not all the examples will work because we use internal WCS sources for development. 

The general workflow for retrieving data using PODPAC is as follows: 

*Note*: Accessing SMAP data requires a [NASA Earth Data Account](user/earthdata)

```python
import podpac  # import the library
import podpac.datalib.smap
c = podpac.Coordinate(time='2018-01-01 12:00:00', lat=0, lon=0)  # Create a coordinate
# Possible SMAP products
smap_product_options = \
	podpac.datalib.smap.SMAP_PRODUCT_MAP.coords['product'].data.tolist()
product = smap_product_options[0] # SMAP product
n = podpac.datalib.smap.SMAP(product=product)  # Create node
o = n.execute(c)  # Execute node to retrieve data at the coordinates
```

Examples of creating algorithms and pipelines can be found in the [Notebooks](https://github.com/creare-com/podpac/tree/master/doc/notebooks). 
