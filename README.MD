# PODPAC 

[![icon](doc/source/_static/img/icon.png)](https://creare-com.github.io/podpac-docs)

> Pipeline for Observation Data Processing Analysis and Collaboration

[![Build Status](https://travis-ci.org/creare-com/podpac.svg?branch=develop)](https://travis-ci.org/creare-com/podpac)
[![Coverage Status](https://coveralls.io/repos/github/creare-com/podpac/badge.svg)](https://coveralls.io/github/creare-com/podpac)

The basic premise is that data wrangling and processing of geospatial data should be seamless 
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
For installation instructions, see [install.md](doc/source/install.md). 

A full Windows 10 Installation of PODPAC can be downloaded from [here](https://s3.amazonaws.com/podpac-s3/releases/PODPAC_latest_install_windows10.zip). To use it, extract the zip file in a folder on your machine.

For older versions, substitute `latest` in the url with the version number. For example, version `0.2.1` can be downloaded using this url [https://s3.amazonaws.com/podpac-s3/releases/PODPAC_0.2.1_install_windows10.zip](https://s3.amazonaws.com/podpac-s3/releases/PODPAC_0.2.0_install_windows10.zip)

## Documentation

The official PODPAC documentation is available here: https://creare-com.github.io/podpac-docs

For usage examples, see the [podpac_examples](https://github.com/creare-com/podpac_examples) repository. 

- To run PODPAC in ESRI ArcGIS Python, see [here](notes/packages_EsriPlus_Python.md)

## Contributing

You can find more information on contributing to PODPAC on the [Contributing page](https://creare-com.github.io/podpac-docs/contributing.html)

## Stability / Maturity

This is in the early development phase. As such:

* The code may be refactored drastically
* Interfaces may change in a backwards-incompatible ways
* Modules, Functions and classes may change names or disappear altogether
* The above may happen without notice
* Documentation and testing may be lacking

We are working towards stability so that the above would no longer be true, and we have already made significant strides towards a stable, maintainable library. 

As such use PODPAC at your own risk. 

## Acknowledgments

This material is based upon work supported by NASA under Contract No 80NSSC18C0061.

## References

For PODPAC references, see the [References page](https://creare-com.github.io/podpac-docs/references.html)
