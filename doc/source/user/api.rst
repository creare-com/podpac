.. _api:

API Reference
=============

.. note this must be manually updated to refer to new/changed module names

.. currentmodule:: podpac


Top Level Imports
-----------------

.. autosummary::
    :toctree: api/

    podpac.Node
    podpac.Coordinates


Nodes
-----

.. autosummary::
    :toctree: api/

    podpac.Node
    podpac.NodeException


Coordinates
-----------

.. autosummary::
    :toctree: api/

    podpac.coordinates.Coordinates
    podpac.coordinates.crange
    podpac.coordinates.clinspace
    podpac.coordinates.Coordinates1d
    podpac.coordinates.GroupCoordinates
    podpac.coordinates.merge_dims
    podpac.coordinates.concat
    podpac.coordinates.union

Data Sources
------------

.. rubric:: Data Types

.. autosummary::
    :toctree: api/

    podpac.data.Array
    podpac.data.PyDAP
    podpac.data.Rasterio
    podpac.data.WCS
    podpac.data.ReprojectedSource
    podpac.data.S3
    podpac.data.H5PY


.. rubric:: Utilities

.. autosummary::
    :toctree: api/

    podpac.data.DataSource
    podpac.data.Interpolation
    podpac.data.InterpolationException
    podpac.data.INTERPOLATION_SHORTCUTS
    podpac.data.INTERPOLATION_DEFAULT


Interoplators
-------------

.. autosummary::
    :toctree: api/

    podpac.interpolators.Interpolator
    podpac.interpolators.NearestNeighbor
    podpac.interpolators.NearestPreview
    podpac.interpolators.Rasterio
    podpac.interpolators.ScipyGrid
    podpac.interpolators.ScipyPoint


Pipelines
---------

.. autosummary::
    :toctree: api/

    podpac.pipeline.Pipeline
    podpac.pipeline.PipelineError
    podpac.pipeline.parse_pipeline_definition
    podpac.pipeline.Output
    podpac.pipeline.NoOutput
    podpac.pipeline.FileOutput
    podpac.pipeline.FTPOutput
    podpac.pipeline.S3Output
    podpac.pipeline.ImageOutput

Algorithm Nodes
---------------

.. autosummary::
    :toctree: api/

    podpac.algorithm.Algorithm
    podpac.algorithm.Arithmetic
    podpac.algorithm.SinCoords
    podpac.algorithm.Arange
    podpac.algorithm.CoordData

.. rubric:: stats

.. autosummary::
    :toctree: api/

    podpac.algorithm.Min
    podpac.algorithm.Max
    podpac.algorithm.Sum
    podpac.algorithm.Count
    podpac.algorithm.Mean
    podpac.algorithm.Median
    podpac.algorithm.Variance
    podpac.algorithm.StandardDeviation
    podpac.algorithm.Skew
    podpac.algorithm.Kurtosis
    podpac.algorithm.DayOfYear
    podpac.algorithm.GroupReduce
    podpac.algorithm.Reduce
    podpac.algorithm.Reduce2

.. rubric:: coordinates

.. autosummary::
    :toctree: api/

    podpac.algorithm.ExpandCoordinates
    podpac.algorithm.SelectCoordinates

.. rubric:: signal

.. autosummary::
    :toctree: api/

    podpac.algorithm.Convolution
    podpac.algorithm.SpatialConvolution
    podpac.algorithm.TimeConvolution

Compositor Nodes
----------------

.. autosummary::
    :toctree: api/

    podpac.compositor.Compositor
    podpac.compositor.OrderedCompositor


Datalib
-------

.. autosummary::
    :toctree: api/

    podpac.datalib.smap
    podpac.datalib.SMAP
    podpac.datalib.SMAPBestAvailable
    podpac.datalib.SMAPSource
    podpac.datalib.SMAPPorosity
    podpac.datalib.SMAPProperties
    podpac.datalib.SMAPWilt
    podpac.datalib.SMAP_PRODUCT_MAP


Utilities
---------

.. rubric:: Authentication

.. autosummary::
    :toctree: api/

    podpac.authentication.SessionWithHeaderRedirection
    podpac.authentication.EarthDataSession


.. rubric:: Settings

.. autosummary::
    :toctree: api/

    podpac.settings


.. rubric:: Utils

.. autosummary::
    :toctree: api/

    podpac.utils


.. rubric:: Version

.. autosummary::
    :toctree: api/

    podpac.version.VERSION
    podpac.version.VERSION_INFO
    podpac.version.semver
    podpac.version.version
