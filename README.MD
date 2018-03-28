# PODPAC 

> Pipeline for Observation Data Processing Analysis and Collaboration

The basic premise is that data wranging and processing of geospatial data should be seamless
so that earth scientists can focus on science. 

The purpose of PODPAC is to facilitate the
 * Access of data products
 * Subsetting of data products
 * Projecting and interpolating data products
 * Combining/compositing data products
 * Analysis of data products
 * Sharing of algorithms and data products
 * Use of cloud computing architectures (AWS) for processing

## Installation

See [doc/install](doc/install.md)

## Usage Examples
See [doc/notebooks](doc/notebooks) for Jupyter Notebook examples. 

*Note*: Not all the examples will work because we use internal WCS sources for development. 

The general workflow for retrieving data using PODPAC is as follows: 
```python
import podpac  # import the library
c = podpac.Coordinate(time='2018-01-01 12:00:00', lat=0, lon=0)  # Create a coordinate
n = podpac.datalib.smap.SMAP()  # Create node
o = n.execute(c)  # Execute node to retrieve data at the coordinates
```

Examples of creating algorithms and pipelines can be found in the Notebooks. 

# Documentation

- To set up an Earth Data Account, used to access SMAP and other NASA data, see  [here](doc/UsingEarthDataAccountWithPODPAC.md)

- To run PODPAC in ESRI ArcGIS Python, see [here](doc/packages_EsriPlus_Python.md)

## Development / Contributor notes
See [doc/developers.md](doc/developers.md)

# Stability / Maturity

This is in the early development phase. As such:

* The code may be refactored drastically
* Interfaces may change in a backwards-incompatible ways
* Modules, Functions and classes may change names or disappear altogether
* The above may happen without notice
* Documentation and testing may be lacking

We are working towards stability so that the above would no longer be true, and we have already made significant strides towards a stable, maintainable library. 

As such use PODPAC at your own risk. 


# References
[AMS2018 Presentation using PODPAC](https://ams.confex.com/ams/98Annual/webprogram/Paper332957.html)
