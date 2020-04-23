
Design
======

> Text and figures highlighted in orange are still in development

Concepts
--------

The primary concepts in PODPAC are :ref:`design_node`, :ref:`design_coordinates`, :ref:`design_unitsdataarray` and :ref:`design_pipelines`.
The following sections describe the class hierarchy associated with each concept.

.. _design_node:

Node
""""
**Nodes** describe the components of your analysis.
These include data sources, combined data sources (**Compositors**), algorithms, and the assembly of data sources.
Nodes are assembled into :ref:`design_pipelines`, which can be output to a text file or pushed to the cloud
with minimal configuration.

.. image:: /_static/img/node.png
    :width: 100%

.. _design_coordinates:

Coordinates
"""""""""""

**Coordinates** describe the data structure and dimensions of data within a **Node**.
PODPAC is limited to 4 coordinate dimensions: ``lat``, ``lon``, ``time``, and ``alt``.

.. image:: /_static/img/coordinates.png
    :width: 70%

.. _design_unitsdataarray:

UnitsDataArray
""""""""""""""

**UnitsDataArray** is the format of all **Node** outputs.
This is a light wrapper around the `xarray.DataArray <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html>`_
class to include attributes for units, coordinate reference systems, and other geospatial specific properties.

.. image:: /_static/img/units-data-array.png
    :width: 25%

.. _design_pipelines:

Pipelines
"""""""""

**Pipelines** are assemblies of **Nodes** that can be evaluated at any PODPAC **Coordinates**.
Pipelines can be very simple, like a single data source evaluated at arbitrary coordinates:

.. image:: /_static/img/simple-pipeline.png
    :width: 50%

Pipelines can also be complex, like two data sources being combined into an algorithm:

.. image:: /_static/img/complex-pipeline.png
    :width: 85%

Pipelines are note explicitly implemented, but this functionality is available through `Nodes`.To see the representation of
a pipeline use `Node.definition`. To create a pipeline from a definition use `Node.from_definition(definition)`. 

Repository Organization
-----------------------

The directory structure is as follows:

- ``dist``: Contains installation instructions and environments for various deployments, including cloud deployment on AWS
- ``doc``: Sphinx based documentation
- ``podpac``: The PODPAC Python library
    - ``core``: The core PODPAC functionality -- contains general implementation so of classes
    - ``datalib``: Library of Nodes used to access specific data sources -- this is where the SMAP node is implemented (for example)
    - ``alglib``: Library of specific algorithms that may be limited to particular scientific domains
