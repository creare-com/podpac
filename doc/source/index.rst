

PODPAC
======

*Pipeline for Observational Data Processing Analysis and Collaboration*

.. sphinx_rtd_theme includes `font-awesome` already, so we don't ned to add it manually
.. raw:: html

    <a href="https://github.com/creare-com/podpac" class="fa fa-github"> View the Source</a> &mdash;
    <a href="https://github.com/creare-com/podpac-examples/tree/develop/notebooks" class="fa fa-github"> Explore Jupyter Notebooks</a>
    <br><br>


.. code-block:: python

    import podpac

    # elevation data
    terrain = podpac.data.Rasterio(source="terrain.tif")

    # soil moisture data
    soil_moisture = podpac.data.H5PY(source="smap.h5")

    # retrieve soil moisture data at elevation coordinates
    soil_moisture.eval(terrain.native_coordinates)


.. figure:: /_static/img/demo-figure.png
    :width: 100%

    Elevation (left), Soil Moisture (center), Soil Moisture at Elevation coordinates (right).

Purpose
-------

Data wrangling and processing of geospatial data should be seamless
so that earth scientists can focus on science.

The purpose of PODPAC is to facilitate

* Access of data products
* Subsetting of data products
* Projecting and interpolating data products
* Combining/compositing data products
* Analysis of data products
* Sharing of algorithms and data products
* Use of cloud computing architectures (AWS) for processing

----------

.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    why-podpac
    install
    examples
    datasets
    roadmap

.. toctree::
    :maxdepth: 1
    :caption: User Guide

    user/settings
    user/coordinates
    user/nodes
    user/pipelines
    user/cache
    user/earthdata
    user/references
    user/api

.. toctree::
    :maxdepth: 1
    :caption: Developer Guide

    developer/design
    developer/contributing
    developer/aws
    developer/docs

Acknowledgments
-----------------
This material is based upon work supported by NASA under Contract No 80NSSC18C0061.
