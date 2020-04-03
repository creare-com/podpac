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
    podpac.data.PyDAP
    podpac.data.Rasterio
    podpac.data.WCS
    podpac.data.ReprojectedSource
    podpac.data.S3
    podpac.data.H5PY


.. rubric:: Utilities

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.data.DataSource
    podpac.data.Interpolation
    podpac.data.InterpolationException
    podpac.data.INTERPOLATION_SHORTCUTS
    podpac.data.INTERPOLATION_DEFAULT


Interpolators
-------------

Classes to manage interpolation

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.interpolators.Interpolator
    podpac.interpolators.NearestNeighbor
    podpac.interpolators.NearestPreview
    podpac.interpolators.Rasterio
    podpac.interpolators.ScipyGrid
    podpac.interpolators.ScipyPoint


Algorithm Nodes
---------------

Split/Apply/Combine nodes with algorithms

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.algorithm.Algorithm

.. rubric:: General Purpose

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.algorithm.Arithmetic
    podpac.algorithm.SinCoords
    podpac.algorithm.Arange
    podpac.algorithm.CoordData

.. rubric:: Statistical Methods

.. autosummary::
    :toctree: api/
    :template: class.rst

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

.. rubric:: Coordinates Modification

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.algorithm.ExpandCoordinates
    podpac.algorithm.SelectCoordinates

.. rubric:: Signal Processing

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.algorithm.Convolution
    podpac.algorithm.SpatialConvolution
    podpac.algorithm.TimeConvolution

Compositor Nodes
----------------

Stitch multiple data sources together

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.compositor.OrderedCompositor


Datalib
-------

Interfaces to external data sources

.. autosummary::
    :toctree: api/
    :template: class.rst

    podpac.datalib.smap
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
