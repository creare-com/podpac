.. _api:

API Reference
=============

.. note this must be manually updated to refer to new/changed module names

.. currentmodule:: podpac


Top Level Imports
-----------------

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.Node
    podpac.Coordinates


Nodes
-----

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.Node
    podpac.NodeException


Coordinates
-----------

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.coordinates.Coordinates
    podpac.coordinates.Coordinates1d
    podpac.coordinates.ArrayCoordinates1d
    podpac.coordinates.UniformCoordinates1d
    podpac.coordinates.StackedCoordinates
    podpac.coordinates.GroupCoordinates


.. rubric:: Utilities

.. autosummary::
    :toctree: api/
    :template: function.rst
    
    podpac.coordinates.crange
    podpac.coordinates.clinspace
    podpac.coordinates.merge_dims
    podpac.coordinates.concat
    podpac.coordinates.union

Data Sources
------------

Generic data source wrappers

.. rubric:: Data Types

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.data.Array
    podpac.data.CSV
    podpac.data.Dataset
    podpac.data.H5PY
    podpac.data.OGR
    podpac.data.PyDAP
    podpac.data.Rasterio
    podpac.data.WCS
    podpac.data.ReprojectedSource
    podpac.data.Zarr


.. rubric:: Utilities

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.data.DataSource
    podpac.data.Interpolation
    podpac.data.INTERPOLATION_DEFAULT
    podpac.data.INTERPOLATION_METHODS


Interpolators
-------------

Classes to manage interpolation

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.interpolation.Interpolation
    podpac.interpolators.Interpolator
    podpac.interpolators.NearestNeighbor
    podpac.interpolators.NearestPreview
    podpac.interpolators.RasterioInterpolator
    podpac.interpolators.ScipyGrid
    podpac.interpolators.ScipyPoint


Algorithm Nodes
---------------

Split/Apply/Combine nodes with algorithms

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.algorithms.Algorithm

.. rubric:: General Purpose

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.algorithms.Arithmetic
    podpac.algorithms.SinCoords
    podpac.algorithms.Arange
    podpac.algorithms.CoordData

.. rubric:: Statistical Methods

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.algorithms.Min
    podpac.algorithms.Max
    podpac.algorithms.Sum
    podpac.algorithms.Count
    podpac.algorithms.Mean
    podpac.algorithms.Median
    podpac.algorithms.Variance
    podpac.algorithms.StandardDeviation
    podpac.algorithms.Skew
    podpac.algorithms.Kurtosis
    podpac.algorithms.DayOfYear
    podpac.algorithms.GroupReduce

.. rubric:: Coordinates Modification

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.algorithms.ExpandCoordinates
    podpac.algorithms.SelectCoordinates

.. rubric:: Signal Processing

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.algorithms.Convolution
    podpac.algorithms.SpatialConvolution
    podpac.algorithms.TimeConvolution

Compositor Nodes
----------------

Stitch multiple data sources together

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.compositor.OrderedCompositor
    podpac.compositor.UniformTileCompositor
    podpac.compositor.UniformTileMixin


Datalib
-------

Interfaces to external data sources

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.datalib.SMAP
    podpac.datalib.SMAPBestAvailable
    podpac.datalib.SMAPSource
    podpac.datalib.SMAPPorosity
    podpac.datalib.SMAPProperties
    podpac.datalib.SMAPWilt
    podpac.datalib.SMAP_PRODUCT_MAP

Managers
--------

Cloud computing managers

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.managers.aws
    podpac.managers.Lambda

Utilities
---------

.. rubric:: Authentication

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.authentication.RequestsSessionMixin
    podpac.authentication.S3Mixin


.. rubric:: Settings

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.settings


.. rubric:: Utils

.. autosummary::
    :toctree: api/
    :template: module.rst

    podpac.utils.create_logfile
    podpac.utils.clear_cache
    podpac.utils.cached_property
    podpac.utils.NoCacheMixin
    podpac.utils.DiskCacheMixin
    podpac.utils.NodeTrait


.. rubric:: Style

.. autosummary::
    :toctree: api/
    :template: module.rst

    podpac.style.Style


.. rubric:: Version

.. autosummary::
    :toctree: api/
    :template: function.rst

    podpac.version.semver
    podpac.version.version


.. autosummary::
    :toctree: api/
    :template: attribute.rst

    podpac.version.VERSION
    podpac.version.VERSION_INFO
