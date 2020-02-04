

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

    # elevation
    elevation = podpac.data.Rasterio(source="elevation.tif")

    # soil moisture
    soil_moisture = podpac.data.H5PY(source="smap.h5", interpolation="bilinear")

    # evaluate soil moisture at the coordinates of the elevation data
    output = soil_moisture.eval(elevation.native_coordinates)


.. figure:: /_static/img/demo-figure.png
    :width: 100%
    :align: center

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

    .. Tutorials for installing and using podpac

    install
    settings
    coordinates
    nodes
    pipelines
    cache
    earthdata
    aws

.. toctree::
    :maxdepth: 1
    :caption: Topics

    .. Deeper explorations of topics that need to get defined

    why-podpac
    examples
    datasets

.. toctree::
    :maxdepth: 1
    :caption: References

    .. Technical references that define the API and contain a deep information

    .. api

.. toctree::
    :maxdepth: 1
    :caption: Support

    .. Anything else clerical

    references
    design
    contributing
    docs
    roadmap
    changelog


Acknowledgments
-----------------
This material is based upon work supported by NASA under Contract No 80NSSC18C0061.
