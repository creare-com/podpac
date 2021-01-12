

PODPAC
======

*Pipeline for Observational Data Processing Analysis and Collaboration*

.. sphinx_rtd_theme includes `font-awesome` already, so we don't ned to add it manually
.. raw:: html

    <a href="https://github.com/creare-com/podpac" class="fa fa-github"> View the Source</a> &mdash;
    <a href="https://github.com/creare-com/podpac-examples/tree/develop/notebooks" class="fa fa-github"> Explore Jupyter Notebooks</a>
    <br><br>

PODPAC is a python library that builds 
on the `scientific python ecosystem <https://www.scipy.org/about.html#the-scipy-ecosystem>`_
to enable simple, reproducible geospatial analyses that run locally or in the cloud.

.. code-block:: python

    import podpac

    # elevation
    elevation = podpac.data.Rasterio(source="elevation.tif")

    # soil moisture
    soil_moisture = podpac.data.H5PY(source="smap.h5", interpolation="bilinear")

    # evaluate soil moisture at the coordinates of the elevation data
    output = soil_moisture.eval(elevation.coordinates)

    # run evaluation in the cloud
    aws_node = podpac.managers.aws.Lambda(source=soil_moisture)
    output = aws_node.eval(elevation.coordinates)



.. figure:: /_static/img/demo-figure.png
    :width: 100%
    :align: center

    Elevation (left), Soil Moisture (center), Soil Moisture at Elevation coordinates (right).

Purpose
-------

Data wrangling and processing of geospatial data should be seamless
so that earth scientists can focus on science. 
The purpose of PODPAC is to facilitate:

* Access of data products
* Subsetting of data products
* Projecting and interpolating data products
* Combining/compositing data products
* Analysis of data products
* Sharing of algorithms and data products
* Use of cloud computing architectures (AWS) for processing

----------

.. Tutorials for installing and using podpac
.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    overview
    install
    settings
    examples
    aws

.. earthdata-tutorial

.. Deeper explorations of topics that need to get defined
.. toctree::
    :maxdepth: 1
    :caption: Topics

    why-podpac
    design
    dependencies
    nodes
    coordinates  
    cache
    datasets
    interpolation
    earthdata
    aws-development

.. Technical references that define the API and contain a deep information
.. toctree::
    :maxdepth: 1
    :caption: References

    api
    wrapping-datasets

.. Anything else clerical
.. toctree::
    :maxdepth: 1
    :caption: Support

    references
    contributing
    docs
    roadmap
    changelog


Acknowledgments
-----------------
This material is based upon work supported by NASA under Contract No 80NSSC18C0061.
