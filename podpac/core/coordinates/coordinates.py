"""
Multidimensional Coordinates
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import warnings
from copy import deepcopy
import sys
import itertools
import json
from collections import OrderedDict
from hashlib import md5 as hash_alg
import re

import numpy as np
import traitlets as tl
import pandas as pd
import xarray as xr
import xarray.core.coordinates
from six import string_types
import pyproj
import logging

import podpac
from podpac.core.settings import settings
from podpac.core.utils import OrderedDictTrait, _get_query_params_from_url, _get_param
from podpac.core.coordinates.base_coordinates import BaseCoordinates
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.dependent_coordinates import ArrayCoordinatesNd
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates
from podpac.core.coordinates.dependent_coordinates import DependentCoordinates
from podpac.core.coordinates.rotated_coordinates import RotatedCoordinates
from podpac.core.coordinates.cfunctions import clinspace

# Optional dependencies
from lazy_import import lazy_module, lazy_class

rasterio = lazy_module("rasterio")
affine_module = lazy_module("affine")


# Set up logging
_logger = logging.getLogger(__name__)


class Coordinates(tl.HasTraits):
    """
    Multidimensional Coordinates.

    Coordinates are used to evaluate Nodes and to define the coordinates of a DataSource nodes. The API is modeled after
    coords in `xarray <http://xarray.pydata.org/en/stable/data-structures.html>`_:

     * Coordinates are created from a list of coordinate values and dimension names.
     * Coordinate values are always either ``float`` or ``np.datetime64``. For convenience, podpac
       automatically converts datetime strings such as ``'2018-01-01'`` to ``np.datetime64``.
     * The allowed dimensions are ``'lat'``, ``'lon'``, ``'time'``, and ``'alt'``.
     * Coordinates from multiple dimensions can be stacked together to represent a *list* of coordinates instead of a
       *grid* of coordinates. The name of the stacked coordinates uses an underscore to combine the underlying
       dimensions, e.g. ``'lat_lon'``.

    Coordinates are dict-like, for example:

     * get coordinates by dimension name: ``coords['lat']``
     * get iterable dimension keys and coordinates values: ``coords.keys()``, ``coords.values()``
     * loop through dimensions: ``for dim in coords: ...``

    Parameters
    ----------
    dims
        Tuple of dimension names, potentially stacked.
    udims
        Tuple of individual dimension names, always unstacked.
    """

    crs = tl.Unicode(read_only=True, allow_none=True)
    crs_desc = tl.Unicode(allow_none=True)

    _coords = OrderedDictTrait(trait=tl.Instance(BaseCoordinates), default_value=OrderedDict())

    def __init__(self, coords, dims=None, crs=None, crs_desc=None, ctype=None, validate_crs=True):
        """
        Create multidimensional coordinates.

        Arguments
        ---------
        coords : list
            List of coordinate values for each dimension. Valid coordinate values:

             * single coordinate value (number, datetime64, or str)
             * array of coordinate values
             * list of stacked coordinate values
             * :class:`Coordinates1d` or :class:`StackedCoordinates` object
        dims : list of str, optional
            List of dimension names. Optional if all items in ``coords`` are named. Valid names are

             * 'lat', 'lon', 'alt', or 'time' for unstacked coordinates
             * dimension names joined by an underscore for stacked coordinates
        crs : str, optional
            Coordinate reference system. Supports PROJ4 and WKT.
        crs_desc : str, optional
            Abbreviated display description for the crs.
        ctype : str, optional
            Default coordinates type. One of 'point', 'midpoint', 'left', 'right'.
        validate_crs : bool, optional
            Use False to skip crs validation. Default True.
        """

        if not isinstance(coords, (list, tuple, np.ndarray, xr.DataArray)):
            raise TypeError("Invalid coords, expected list or array, not '%s'" % type(coords))

        if dims is not None and not isinstance(dims, (tuple, list)):
            raise TypeError("Invalid dims type '%s'" % type(dims))

        if dims is None:
            for i, c in enumerate(coords):
                if not isinstance(c, (BaseCoordinates, xr.DataArray)):
                    raise TypeError(
                        "Cannot get dim for coordinates at position %d with type '%s'"
                        "(expected 'Coordinates1d' or 'DataArray')" % (i, type(c))
                    )

            dims = [c.name for c in coords]

        if len(dims) != len(coords):
            raise ValueError("coords and dims size mismatch, %d != %d" % (len(dims), len(coords)))

        # get/create coordinates
        dcoords = OrderedDict()
        for i, dim in enumerate(dims):
            if isinstance(dim, (tuple, list)):
                dim = "_".join(dim)

            if dim in dcoords:
                raise ValueError("Duplicate dimension '%s' at position %d" % (dim, i))

            # coerce
            if isinstance(coords[i], BaseCoordinates):
                c = coords[i]
            elif "_" in dim:
                c = StackedCoordinates(coords[i])
            elif "," in dim:
                c = DependentCoordinates(coords[i])
            else:
                c = ArrayCoordinates1d(coords[i])

            # propagate properties and name
            c._set_name(dim)
            if ctype is not None:
                c._set_ctype(ctype)

            # set coords
            dcoords[dim] = c

        self.set_trait("_coords", dcoords)

        if crs is not None:
            # validate
            if validate_crs:
                # raises pyproj.CRSError if invalid
                CRS = pyproj.CRS(crs)

                # make sure CRS defines vertical units
                if "alt" in self.udims and not CRS.is_vertical:
                    raise ValueError("Altitude dimension is defined, but CRS does not contain vertical unit")

            crs = self.set_trait("crs", crs)

        super(Coordinates, self).__init__(crs_desc=crs_desc)

    @tl.validate("_coords")
    def _validate_coords(self, d):
        val = d["value"]

        if len(val) == 0:
            return val

        for dim, c in val.items():
            if dim != c.name:
                raise ValueError("Dimension mismatch, '%s' != '%s'" % (dim, c.name))

        dims = [dim for c in val.values() for dim in c.dims]
        for dim in dims:
            if dims.count(dim) != 1:
                raise ValueError("Duplicate dimension '%s' in dims %s" % (dim, tuple(val.keys())))

        return val

    @tl.default("crs")
    def _default_crs(self):
        return settings["DEFAULT_CRS"]

    # ------------------------------------------------------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _coords_from_dict(d, order=None):
        if sys.version < "3.6":
            if order is None and len(d) > 1:
                raise TypeError("order required")

        if order is not None:
            if set(order) != set(d):
                raise ValueError("order %s does not match dims %s" % (order, d))
        else:
            order = d.keys()

        coords = []
        for dim in order:
            if isinstance(d[dim], Coordinates1d):
                c = d[dim].copy(name=dim)
            elif isinstance(d[dim], tuple):
                c = UniformCoordinates1d.from_tuple(d[dim], name=dim)
            else:
                c = ArrayCoordinates1d(d[dim], name=dim)
            coords.append(c)

        return coords

    @classmethod
    def grid(cls, dims=None, crs=None, ctype=None, **kwargs):
        """
        Create a grid of coordinates.

        Valid coordinate values:

         * single coordinate value (number, datetime64, or str)
         * array of coordinate values
         * ``(start, stop, step)`` tuple for uniformly-spaced coordinates
         * Coordinates1d object

        This is equivalent to creating unstacked coordinates with a list of coordinate values::

            podpac.Coordinates.grid(lat=[0, 1, 2], lon=[10, 20], dims=['lat', 'lon'])
            podpac.Coordinates([[0, 1, 2], [10, 20]], dims=['lan', 'lon'])

        Arguments
        ---------
        lat : optional
            coordinates for the latitude dimension
        lon : optional
            coordinates for the longitude dimension
        alt : optional
            coordinates for the altitude dimension
        time : optional
            coordinates for the time dimension
        dims : list of str, optional in Python>=3.6
            List of dimension names, must match the provided keyword arguments. In Python 3.6 and above, the ``dims``
            argument is optional, and the dims will match the order of the provided keyword arguments.
        crs : str, optional
            Coordinate reference system. Supports any PROJ4 or PROJ6 compliant string (https://proj.org).
        ctype : str, optional
            Default coordinates type. One of 'point', 'midpoint', 'left', 'right'.

        Returns
        -------
        :class:`Coordinates`
            podpac Coordinates

        See Also
        --------
        points
        """

        coords = cls._coords_from_dict(kwargs, order=dims)
        return cls(coords, crs=crs, ctype=ctype)

    @classmethod
    def points(cls, crs=None, ctype=None, dims=None, **kwargs):
        """
        Create a list of multidimensional coordinates.

        Valid coordinate values:

         * single coordinate value (number, datetime64, or str)
         * array of coordinate values
         * ``(start, stop, step)`` tuple for uniformly-spaced coordinates
         * Coordinates1d object

        Note that the coordinates for each dimension must be the same size.

        This is equivalent to creating stacked coordinates with a list of coordinate values and a stacked dimension
        name::

            podpac.Coordinates.points(lat=[0, 1, 2], lon=[10, 20, 30], dims=['lat', 'lon'])
            podpac.Coordinates([[[0, 1, 2], [10, 20, 30]]], dims=['lan_lon'])

        Arguments
        ---------
        lat : optional
            coordinates for the latitude dimension
        lon : optional
            coordinates for the longitude dimension
        alt : optional
            coordinates for the altitude dimension
        time : optional
            coordinates for the time dimension
        dims : list of str, optional in Python>=3.6
            List of dimension names, must match the provided keyword arguments. In Python 3.6 and above, the ``dims``
            argument is optional, and the dims will match the order of the provided keyword arguments.
        crs : str, optional
            Coordinate reference system. Supports any PROJ4 or PROJ6 compliant string (https://proj.org/).
        ctype : str, optional
            Default coordinates type. One of 'point', 'midpoint', 'left', 'right'.

        Returns
        -------
        :class:`Coordinates`
            podpac Coordinates

        See Also
        --------
        grid
        """

        coords = cls._coords_from_dict(kwargs, order=dims)
        stacked = StackedCoordinates(coords)
        return cls([stacked], crs=crs, ctype=ctype)

    @classmethod
    def from_xarray(cls, xcoord, crs=None, ctype=None):
        """
        Create podpac Coordinates from xarray coords.

        Arguments
        ---------
        xcoord : xarray.core.coordinates.DataArrayCoordinates
            xarray coords
        crs : str, optional
            Coordinate reference system. Supports any PROJ4 or PROJ6 compliant string (https://proj.org/).
        ctype : str, optional
            Default coordinates type. One of 'point', 'midpoint', 'left', 'right'.

        Returns
        -------
        :class:`Coordinates`
            podpac Coordinates
        """

        if not isinstance(xcoord, xarray.core.coordinates.DataArrayCoordinates):
            raise TypeError("Coordinates.from_xarray expects xarray DataArrayCoordinates, not '%s'" % type(xcoord))

        coords = []
        for dim in xcoord.dims:
            if dim == "output":
                continue
            if isinstance(xcoord.indexes[dim], (pd.DatetimeIndex, pd.Float64Index, pd.Int64Index)):
                c = ArrayCoordinates1d.from_xarray(xcoord[dim])
            elif isinstance(xcoord.indexes[dim], pd.MultiIndex):
                c = StackedCoordinates.from_xarray(xcoord[dim])
            else:
                raise NotImplementedError
            coords.append(c)

        return cls(coords, crs=crs, ctype=ctype)

    @classmethod
    def from_json(cls, s):
        """
        Create podpac Coordinates from a coordinates JSON definition.

        Example JSON definition::

            [
                {
                    "name": "lat",
                    "start": 1,
                    "stop": 10,
                    "step": 0.5,
                },
                {
                    "name": "lon",
                    "start": 1,
                    "stop": 2,
                    "size": 100
                },
                {
                    "name": "time",
                    "ctype": "left"
                    "values": [
                        "2018-01-01",
                        "2018-01-03",
                        "2018-01-10"
                    ]
                }
            ]

        Arguments
        ---------
        s : str
            coordinates JSON definition

        Returns
        -------
        :class:`Coordinates`
            podpac Coordinates

        See Also
        --------
        json
        """

        d = json.loads(s)
        return cls.from_definition(d)

    @classmethod
    def from_url(cls, url):
        """
        Create podpac Coordinates from a WMS/WCS request.

        Arguments
        ---------
        url : str, dict
            The raw WMS/WCS request url, or a dictionary of query parameters

        Returns
        -------
        :class:`Coordinates`
            podpac Coordinates
        """
        params = _get_query_params_from_url(url)

        # The ordering here is lat/lon or y/x for WMS 1.3.0
        # The ordering here is lon/lat or x/y for WMS 1.1
        # See https://docs.geoserver.org/stable/en/user/services/wms/reference.html
        # and https://docs.geoserver.org/stable/en/user/services/wms/basics.html

        bbox = np.array(_get_param(params, "BBOX").split(","), float)

        # I don't seem to need to reverse one of these... perhaps one of my test servers did not implement the spec?
        if _get_param(params, "VERSION").startswith("1.1"):
            r = -1
        elif _get_param(params, "VERSION").startswith("1.3"):
            r = 1

        # Extract bounding box information and translate to PODPAC coordinates
        start = bbox[:2][::r]
        stop = bbox[2::][::r]
        size = np.array([_get_param(params, "WIDTH"), _get_param(params, "HEIGHT")], int)[::r]

        coords = OrderedDict()

        # Note, version 1.1 used "SRS" and 1.3 uses 'CRS'
        coords["crs"] = _get_param(params, "SRS")
        if coords["crs"] is None:
            coords["crs"] = _get_param(params, "CRS")

        coords["coords"] = [
            {"name": "lat", "start": start[0], "stop": stop[0], "size": size[0]},
            {"name": "lon", "start": start[1], "stop": stop[1], "size": size[1]},
        ]

        if "TIME" in params:
            coords["coords"].append({"name": "time", "values": [_get_param(params, "TIME")]})

        return cls.from_definition(coords)

    @classmethod
    def from_geotransform(cls, geotransform, shape, crs=None):
        """ Creates Coordinates from GDAL Geotransform. 
        
        """
        tol = 1e-15  # tolerance for deciding when a number is zero
        # Handle the case of rotated coordinates
        try:
            rcoords = RotatedCoordinates.from_geotransform(geotransform, shape, dims=["lat", "lon"])
        except affine_module.UndefinedRotationError:
            rcoords = None
            _logger.debug("Rasterio source dataset does not have Rotated Coordinates")

        if rcoords is not None and np.abs(rcoords.theta % (np.pi / 2)) > tol:
            # These are Rotated coordinates and we can return
            coords = Coordinates([rcoords], dims=["lat,lon"], crs=crs)
            return coords

        # Handle the case of uniform coordinates (not rotated, but N-S E-W aligned)
        affine = rasterio.Affine.from_gdal(*geotransform)
        if affine.e <= tol and affine.a <= tol:
            order = -1
            step = np.array([affine.d, affine.b])
        else:
            order = 1
            step = np.array([affine.e, affine.a])

        origin = affine.f + step[0] / 2, affine.c + step[1] / 2
        end = origin[0] + step[0] * (shape[::order][0] - 1), origin[1] + step[1] * (shape[::order][1] - 1)
        coords = Coordinates(
            [
                podpac.clinspace(origin[0], end[0], shape[::order][0], "lat"),
                podpac.clinspace(origin[1], end[1], shape[::order][1], "lon"),
            ][::order],
            crs=crs,
        )
        return coords

    @classmethod
    def from_definition(cls, d):
        """
        Create podpac Coordinates from a coordinates definition.

        Arguments
        ---------
        d : list
            coordinates definition

        Returns
        -------
        :class:`Coordinates`
            podpac Coordinates

        See Also
        --------
        from_json, definition
        """

        if not isinstance(d, dict):
            raise TypeError("Could not parse coordinates definition, expected type 'dict', got '%s'" % type(d))

        if "coords" not in d:
            raise ValueError("Could not parse coordinates definition, 'coords' required")

        if not isinstance(d["coords"], list):
            raise TypeError(
                "Could not parse coordinates definition, expected 'coords' of type 'list', got '%s'"
                % (type(d["coords"]))
            )

        coords = []
        for e in d["coords"]:
            if isinstance(e, list):
                c = StackedCoordinates.from_definition(e)
            elif "start" in e and "stop" in e and ("step" in e or "size" in e):
                c = UniformCoordinates1d.from_definition(e)
            elif "name" in e and "values" in e:
                c = ArrayCoordinates1d.from_definition(e)
            elif "dims" in e and "values" in e:
                c = DependentCoordinates.from_definition(e)
            elif "dims" in e and "shape" in e and "theta" in e and "origin" in e and ("step" in e or "corner" in e):
                c = RotatedCoordinates.from_definition(e)
            else:
                raise ValueError("Could not parse coordinates definition item with keys %s" % e.keys())

            coords.append(c)

        kwargs = {k: v for k, v in d.items() if k != "coords"}
        return cls(coords, **kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    # standard dict-like methods
    # ------------------------------------------------------------------------------------------------------------------

    def keys(self):
        """ dict-like keys: dims """
        return self._coords.keys()

    def values(self):
        """ dict-like values: coordinates for each key/dimension """
        return self._coords.values()

    def items(self):
        """ dict-like items: (dim, coordinates) pairs """
        return self._coords.items()

    def __iter__(self):
        return iter(self._coords)

    def get(self, dim, default=None):
        """ dict-like get: get coordinates by dimension name with an optional """
        try:
            return self[dim]
        except KeyError:
            return default

    def __getitem__(self, index):
        if isinstance(index, string_types):
            dim = index
            if dim in self._coords:
                return self._coords[dim]

            # extracts individual coords from stacked and dependent coordinates
            for c in self._coords.values():
                if isinstance(c, (StackedCoordinates, DependentCoordinates)) and dim in c.dims:
                    return c[dim]

            raise KeyError("Dimension '%s' not found in Coordinates %s" % (dim, self.dims))

        else:
            # extend index to a tuple of the correct length
            if not isinstance(index, tuple):
                index = (index,)
            index = index + tuple(slice(None) for i in range(self.ndim - len(index)))

            # bundle dependent coordinates indices
            indices = []
            i = 0
            for c in self._coords.values():
                if isinstance(c, DependentCoordinates):
                    indices.append(tuple(index[i : i + len(c.dims)]))
                    i += len(c.dims)
                else:
                    indices.append(index[i])
                    i += 1

            return Coordinates(
                [c[I] for c, I in zip(self._coords.values(), indices)], validate_crs=False, **self.properties
            )

    def __setitem__(self, dim, c):

        # coerce
        if isinstance(c, BaseCoordinates):
            pass
        elif isinstance(c, Coordinates):
            c = c[dim]
        elif "_" in dim:
            c = StackedCoordinates(c)
        elif "," in dim:
            c = DependentCoordinates(c)
        else:
            c = ArrayCoordinates1d(c)

        c._set_name(dim)

        if dim in self.dims:
            d = self._coords.copy()
            d[dim] = c
            self._coords = d

        elif dim in self.udims:
            stacked_dim = [sd for sd in self.dims if dim in sd][0]
            self._coords[stacked_dim][dim] = c
        else:
            raise KeyError("Cannot set dimension '%s' in Coordinates %s" % (dim, self.dims))

    def __delitem__(self, dim):
        if not dim in self.dims:
            raise KeyError("Cannot delete dimension '%s' in Coordinates %s" % (dim, self.dims))

        del self._coords[dim]

    def __len__(self):
        return len(self._coords)

    def update(self, other):
        """ dict-like update: add/replace coordinates using another Coordinates object """
        if not isinstance(other, Coordinates):
            raise TypeError("Cannot update Coordinates with object of type '%s'" % type(other))

        d = self._coords.copy()
        d.update(other._coords)
        self._coords = d

    def __eq__(self, other):
        if not isinstance(other, Coordinates):
            return False

        # shortcuts
        if self.dims != other.dims:
            return False

        if self.shape != other.shape:
            return False

        # properties
        if self.CRS != other.CRS:
            return False

        # full check of underlying coordinates
        if self._coords != other._coords:
            return False

        return True

    if sys.version < "3":

        def __ne__(self, other):
            return not self.__eq__(other)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def dims(self):
        """:tuple: Tuple of dimension names, potentially stacked.

        See Also
        --------
        udims
        """

        return tuple(c.name for c in self._coords.values())

    @property
    def idims(self):
        """:tuple: Tuple of indexing dimension names.

        Unless there are dependent coordinates, this will match the ``dims``. For dependent coordinates, indexing
        dimensions `'i'`, `'j'`, etc are used by default.
        """

        return tuple(dim for c in self._coords.values() for dim in c.idims)

    @property
    def udims(self):
        """:tuple: Tuple of unstacked dimension names.

        If there are no stacked dimensions, then ``dims`` and ``udims`` will be the same::

            In [1]: lat = [0, 1]

            In [2]: lon = [10, 20]

            In [3]: time = '2018-01-01'

            In [4]: c = podpac.Coordinates([lat, lon, time], dims=['lat', 'lon', 'time'])

            In [5]: c.dims
            Out[5]: ('lat', 'lon', 'time')

            In [6]: c.udims
            Out[6]: ('lat', 'lon', 'time')


        If there are stacked dimensions, then ``udims`` contains the individual dimension names::

            In [7]: c = podpac.Coordinates([[lat, lon], time], dims=['lat_lon', 'time'])

            In [8]: c.dims
            Out[8]: ('lat_lon', 'time')

            In [9]: c.udims
            Out[9]: ('lat', 'lon', 'time')

        See Also
        --------
        dims
        """

        return tuple(dim for c in self._coords.values() for dim in c.udims)

    @property
    def shape(self):
        """:tuple: Tuple of the number of coordinates in each dimension."""

        return tuple(size for c in self._coords.values() for size in c.shape)

    @property
    def ushape(self):
        return tuple(self[dim].size for dim in self.udims)

    @property
    def ndim(self):
        """:int: Number of dimensions. """

        return len(self.shape)

    @property
    def size(self):
        """:int: Total number of coordinates."""

        if len(self.shape) == 0:
            return 0
        return np.prod(self.shape)

    @property
    def bounds(self):
        """:dict: Dictionary of (low, high) coordinates bounds in each unstacked dimension"""
        return {dim: self[dim].bounds for dim in self.udims}

    @property
    def area_bounds(self):
        """:dict: Dictionary of (low, high) coordinates area_bounds in each unstacked dimension"""
        return {dim: self[dim].area_bounds for dim in self.udims}

    @property
    def coords(self):
        """
        :xarray.core.coordinates.DataArrayCoordinates: xarray coords, a dictionary-like container of coordinate arrays.
        """

        coords = OrderedDict()
        for c in self._coords.values():
            coords.update(c.coords)
        return coords

    @property
    def CRS(self):
        return pyproj.CRS(self.crs)

    # TODO: add a convience property for displaying altitude units for the CRS
    # @property
    # def alt_units(self):
    #     CRS = self.CRS

    #     if CRS.is_vertical:
    #         alt_units = <unsure how to get this from CRS>
    #     else:
    #         raise ValueError("CRS does not contain vertical component")

    #     return alt_units

    @property
    def properties(self):
        """:dict: Dictionary of the coordinate properties. """

        d = OrderedDict()
        d["crs"] = self.crs
        return d

    @property
    def definition(self):
        """:list: Serializable coordinates definition."""

        d = OrderedDict()
        d["coords"] = [c.definition for c in self._coords.values()]
        d.update(self.properties)
        return d

    @property
    def full_definition(self):
        """:list: Serializable coordinates definition, containing all properties. For internal use."""

        d = OrderedDict()
        d["coords"] = [c.full_definition for c in self._coords.values()]
        d[
            "crs"
        ] = (
            self.CRS.to_wkt()
        )  # use "wkt" which is suggested as best format: https://proj.org/faq.html#what-is-the-best-format-for-describing-coordinate-reference-systems
        return d

    @property
    def json(self):
        """:str: JSON-serialized coordinates definition.

        The ``json`` can be used to create new Coordinates::

            c = podapc.Coordinates(...)
            c2 = podpac.Coordinates.from_json(c.definition)

        The serialized definition is used to define coordinates in node definitions and to transport coordinates, e.g.
        over HTTP and in AWS lambda functions. It also provides a consistent hashable value.

        See Also
        --------
        from_json
        """

        return json.dumps(self.definition, separators=(",", ":"), cls=podpac.core.utils.JSONEncoder)

    @property
    def hash(self):
        """:str: Coordinates hash value."""
        # We can't use self.json for the hash because the CRS is not standardized.
        # As such, we json.dumps the full definition.
        json_d = json.dumps(self.full_definition, separators=(",", ":"), cls=podpac.core.utils.JSONEncoder)
        return hash_alg(json_d.encode("utf-8")).hexdigest()

    @property
    def geotransform(self):
        """ :tuple: GDAL geotransform. """
        # Make sure we only have 1 time and alt dimension
        if "time" in self.udims and self["time"].size > 1:
            raise TypeError(
                'Only 2-D coordinates have a GDAL transform. This array has a "time" dimension of {} > 1'.format(
                    self["time"].size
                )
            )
        if "alt" in self.udims and self["alt"].size > 1:
            raise TypeError(
                'Only 2-D coordinates have a GDAL transform. This array has a "alt" dimension of {} > 1'.format(
                    self["alt"].size
                )
            )

        # Do the uniform coordinates case
        if (
            "lat" in self.dims
            and isinstance(self._coords["lat"], UniformCoordinates1d)
            and "lon" in self.dims
            and isinstance(self._coords["lon"], UniformCoordinates1d)
        ):
            if self.dims.index("lon") < self.dims.index("lat"):
                first, second = "lat", "lon"
            else:
                first, second = "lon", "lat"  # This case will have the exact correct geotransform
            transform = rasterio.transform.Affine.translation(
                self[first].start - self[first].step / 2, self[second].start - self[second].step / 2
            ) * rasterio.transform.Affine.scale(self[first].step, self[second].step)
            transform = transform.to_gdal()
        # Do the rotated coordinates cases
        elif "lat,lon" in self.dims and isinstance(self._coords["lat,lon"], RotatedCoordinates):
            transform = self._coords["lat,lon"].geotransform
        elif "lon,lat" in self.dims and isinstance(self._coords["lon,lat"], RotatedCoordinates):
            transform = self._coords["lon,lat"].geotransform
        else:
            raise TypeError(
                "Only 2-D coordinates that are uniform or rotated have a GDAL transform. These coordinates "
                "{} do not.".format(self)
            )
        if self.udims.index("lon") < self.udims.index("lat"):
            # transform = (transform[3], transform[5], transform[4], transform[0], transform[2], transform[1])
            transform = transform[3:] + transform[:3]

        return transform

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def drop(self, dims, ignore_missing=False):
        """
        Remove the given dimensions from the Coordinates `dims`.

        Parameters
        ----------
        dims : str, list
            Dimension(s) to drop.
        ignore_missing : bool, optional
            If True, do not raise an exception if a given dimension is not in ``dims``. Default ``False``.

        Returns
        -------
        :class:`Coordinates`
            Coordinates object with the given dimensions removed

        Raises
        ------
        KeyError
            If a given dimension is missing in the Coordinates (and ignore_missing is ``False``).

        See Also
        --------
        udrop
        """

        if not isinstance(dims, (tuple, list)):
            dims = (dims,)

        for dim in dims:
            if not isinstance(dim, string_types):
                raise TypeError("Invalid drop dimension type '%s'" % type(dim))
            if dim not in self.dims and not ignore_missing:
                raise KeyError("Dimension '%s' not found in Coordinates with dims %s" % (dim, self.dims))

        return Coordinates(
            [c for c in self._coords.values() if c.name not in dims], validate_crs=False, **self.properties
        )

    # do we ever need this?
    def udrop(self, dims, ignore_missing=False):
        """
        Remove the given individual dimensions from the Coordinates `udims`.

        Unlike `drop`, ``udrop`` will remove parts of stacked coordinates::

            In [1]: c = podpac.Coordinates([[[0, 1], [10, 20]], '2018-01-01'], dims=['lat_lon', 'time'])

            In [2]: c
            Out[2]:
            Coordinates
                lat_lon[lat]: ArrayCoordinates1d(lat): Bounds[0.0, 1.0], N[2], ctype['midpoint']
                lat_lon[lon]: ArrayCoordinates1d(lon): Bounds[10.0, 20.0], N[2], ctype['midpoint']
                time: ArrayCoordinates1d(time): Bounds[2018-01-01, 2018-01-01], N[1], ctype['midpoint']

            In [3]: c.udrop('lat')
            Out[3]:
            Coordinates
                lon: ArrayCoordinates1d(lon): Bounds[10.0, 20.0], N[2], ctype['midpoint']
                time: ArrayCoordinates1d(time): Bounds[2018-01-01, 2018-01-01], N[1], ctype['midpoint']

        Parameters
        ----------
        dims : str, list
            Individual dimension(s) to drop.
        ignore_missing : bool, optional
            If True, do not raise an exception if a given dimension is not in ``udims``. Default ``False``.

        Returns
        -------
        :class:`Coordinates`
            Coordinates object with the given dimensions removed.

        Raises
        ------
        KeyError
            If a given dimension is missing in the Coordinates (and ignore_missing is ``False``).

        See Also
        --------
        drop
        """

        if not isinstance(dims, (tuple, list)):
            dims = (dims,)

        for dim in dims:
            if not isinstance(dim, string_types):
                raise TypeError("Invalid drop dimension type '%s'" % type(dim))
            if dim not in self.udims and not ignore_missing:
                raise KeyError("Dimension '%s' not found in Coordinates with udims %s" % (dim, self.udims))

        cs = []
        for c in self._coords.values():
            if isinstance(c, Coordinates1d):
                if c.name not in dims:
                    cs.append(c)
            elif isinstance(c, StackedCoordinates):
                stacked = [s for s in c if s.name not in dims]
                if len(stacked) > 1:
                    cs.append(StackedCoordinates(stacked))
                elif len(stacked) == 1:
                    cs.append(stacked[0])

        return Coordinates(cs, validate_crs=False, **self.properties)

    def intersect(self, other, dims=None, outer=False, return_indices=False):
        """
        Get the coordinate values that are within the bounds of a given coordinates object.

        The intersection is calculated in each dimension separately.

        The default intersection selects coordinates that are within the other coordinates bounds::

            In [1]: coords = Coordinates([[0, 1, 2, 3]], dims=['lat'])

            In [2]: other = Coordinates([[1.5, 2.5]], dims=['lat'])

            In [3]: coords.intersect(other).coords
            Out[3]:
            Coordinates:
              * lat      (lat) float64 2.0

        The *outer* intersection selects the minimal set of coordinates that contain the other coordinates::

            In [4]: coords.intersect(other, outer=True).coords
            Out[4]:
            Coordinates:
              * lat      (lat) float64 1.0 2.0 3.0

        The *outer* intersection also selects a boundary coordinate if the other coordinates are outside this
        coordinates bounds but *inside* its area bounds::

            In [5]: other_near = Coordinates([[3.25]], dims=['lat'])

            In [6]: other_far = Coordinates([[10.0]], dims=['lat'])

            In [7]: coords.intersect(other_near, outer=True).coords
            Coordinates:
              * lat      (lat) float64 3.0

            In [8]: coords.intersect(other_far, outer=True).coords
            Coordinates:
              * lat      (lat) float64

        Parameters
        ----------
        other : :class:`Coordinates1d`, :class:`StackedCoordinates`, :class:`Coordinates`
            Coordinates to intersect with.
        dims : list, optional
            Restrict intersection to the given dimensions. Default is all available dimensions.
        outer : bool, optional
            If True, do an *outer* intersection. Default False.
        return_indices : bool, optional
            If True, return slice or indices for the selection in addition to coordinates. Default False.

        Returns
        -------
        intersection : :class:`Coordinates`
            Coordinates object consisting of the intersection in each dimension.
        idx : list
            List of indices for each dimension that produces the intersection, only if ``return_indices`` is True.
        """

        if not isinstance(other, Coordinates):
            raise TypeError("Coordinates cannot be intersected with type '%s'" % type(other))

        if other.crs.lower() != self.crs.lower():
            other = other.transform(self.crs)

        bounds = other.bounds
        if dims is not None:
            bounds = {dim: bounds[dim] for dim in dims}  # if dim in bounds}

        return self.select(bounds, outer=outer, return_indices=return_indices)

    def select(self, bounds, return_indices=False, outer=False):
        """
        Get the coordinate values that are within the given bounds for each dimension.

        The default selection returns coordinates that are within the bounds::

            In [1]: c = Coordinates([[0, 1, 2, 3], [10, 20, 30, 40]], dims=['lat', 'lon'])

            In [2]: c.select({'lat': [1.5, 3.5]})
            Out[2]:
            Coordinates
                    lat: ArrayCoordinates1d(lat): Bounds[2.0, 3.0], N[2], ctype['midpoint']
                    lon: ArrayCoordinates1d(lon): Bounds[10.0, 40.0], N[4], ctype['midpoint']

            In [3]: c.select({'lat': [1.5, 3.5], 'lon': [25, 45]})
            Out[3]:
            Coordinates
                    lat: ArrayCoordinates1d(lat): Bounds[2.0, 3.0], N[2], ctype['midpoint']
                    lon: ArrayCoordinates1d(lon): Bounds[30.0, 40.0], N[2], ctype['midpoint']

        The *outer* selection returns the minimal set of coordinates that contain the bounds::

            In [4]: c.select({'lat':[1.5, 3.5]}, outer=True)
            Out[4]:
            Coordinates
                    lat: ArrayCoordinates1d(lat): Bounds[1.0, 3.0], N[3], ctype['midpoint']
                    lon: ArrayCoordinates1d(lon): Bounds[10.0, 40.0], N[4], ctype['midpoint']

        Parameters
        ----------
        bounds : dict
            Selection bounds for the desired coordinates.
        outer : bool, optional
            If True, do *outer* selections. Default False.
        return_indices : bool, optional
            If True, return slice or indices for the selections in addition to coordinates. Default False.

        Returns
        -------
        selection : :class:`Coordinates`
            Coordinates object with coordinates within the given bounds.
        I : list of indices (slices/lists)
            index or slice for the selected coordinates in each dimension (only if return_indices=True)
        """

        selections = [c.select(bounds, outer=outer, return_indices=return_indices) for c in self._coords.values()]
        return self._make_selected_coordinates(selections, return_indices)

    def _make_selected_coordinates(self, selections, return_indices):
        if return_indices:
            coords = Coordinates([c for c, I in selections], validate_crs=False, **self.properties)
            # unbundle DepedentCoordinates indices
            I = [I if isinstance(c, DependentCoordinates) else [I] for c, I in selections]
            I = [e for l in I for e in l]
            return coords, tuple(I)
        else:
            return Coordinates(selections, validate_crs=False, **self.properties)

    def unique(self, return_indices=False):
        """
        Remove duplicate coordinate values from each dimension.

        Arguments
        ---------
        return_indices : bool, optional
            If True, return indices for the unique coordinates in addition to the coordinates. Default False.
        Returns
        -------
        coords : Coordinates
            New Coordinates object with unique, sorted coordinate values in each dimension.
        I : list of indices
            index for the unique coordinates in each dimension (only if return_indices=True)
        """

        I = tuple(np.unique(c.coordinates, return_index=True)[1] for c in self.values())

        if return_indices:
            return self[I], I
        else:
            return self[I]

    def unstack(self):
        """
        Unstack the coordinates of all of the dimensions.

        Returns
        -------
        unstacked : :class:`Coordinates`
            A new Coordinates object with unstacked coordinates.

        See Also
        --------
        xr.DataArray.unstack
        """

        return Coordinates([self[dim] for dim in self.udims], validate_crs=False, **self.properties)

    def iterchunks(self, shape, return_slices=False):
        """
        Get a generator that yields Coordinates no larger than the given shape until the entire Coordinates is covered.

        Parameters
        ----------
        shape : tuple
            The maximum shape of the chunk, with sizes corresponding to the `dims`.
        return_slice : boolean, optional
            Return slice in addition to Coordinates chunk.

        Yields
        ------
        coords : :class:`Coordinates`
            A Coordinates object with one chunk of the coordinates.
        slices : list
            slices for this Coordinates chunk, only if ``return_slices`` is True
        """

        l = [[slice(i, i + n) for i in range(0, m, n)] for m, n in zip(self.shape, shape)]
        for slices in itertools.product(*l):
            coords = Coordinates(
                [self._coords[dim][slc] for dim, slc in zip(self.dims, slices)], validate_crs=False, **self.properties
            )
            if return_slices:
                yield coords, slices
            else:
                yield coords

    def transpose(self, *dims, **kwargs):
        """
        Transpose (re-order) the dimensions of the Coordinates.

        Parameters
        ----------
        dim_1, dim_2, ... : str, optional
            Reorder dims to this order. By default, reverse the dims.
        in_place : boolean, optional
            If True, transpose the dimensions in-place.
            Otherwise (default), return a new, transposed Coordinates object.

        Returns
        -------
        transposed : :class:`Coordinates`
            The transposed Coordinates object.

        See Also
        --------
        xarray.DataArray.transpose : return a transposed DataArray

        """

        in_place = kwargs.get("in_place", False)

        if len(dims) == 0:
            dims = list(self._coords.keys())[::-1]

        if len(dims) != len(self.dims):
            raise ValueError("Invalid transpose dimensions, input %s does not match dims %s" % (dims, self.dims))

        coords = []
        for dim in dims:
            if dim in self._coords:
                coords.append(self._coords[dim])
            elif "," in dim and dim.split(",")[0] in self.udims:
                target_dims = dim.split(",")
                source_dim = [_dim for _dim in self.dims if target_dims[0] in _dim][0]
                coords.append(self._coords[source_dim].transpose(*target_dims, in_place=in_place))
            elif "_" in dim and dim.split("_")[0] in self.udims:
                target_dims = dim.split("_")
                source_dim = [_dim for _dim in self.dims if target_dims[0] in _dim][0]
                coords.append(self._coords[source_dim].transpose(*target_dims, in_place=in_place))
            else:
                raise ValueError("Invalid transpose dimensions, input %s does match any dims in %s" % (dim, self.dims))

        if in_place:
            self._coords = OrderedDict(zip(dims, coords))
            return self
        else:
            return Coordinates(coords, validate_crs=False, **self.properties)

    def transform(self, crs):
        """
        Transform coordinate dimensions (`lat`, `lon`, `alt`) into a different coordinate reference system.
        Uses PROJ syntax for coordinate reference systems and units.

        See `PROJ Documentation <https://proj.org/usage/projections.html#cartographic-projection>`_ for
        more information about creating PROJ4 strings. See `PROJ4 Distance Units
        <https://proj4.org/operations/conversions/unitconvert.html#distance-units>`_ for unit string
        references.

        Examples
        --------
        Transform gridded coordinates::

            c = Coordinates([np.linspace(-10, 10, 21), np.linspace(-30, -10, 21)], dims=['lat', 'lon'])
            c.crs

            >> 'EPSG:4326'

            c.transform('EPSG:2193')

            >> Coordinates
                lat: ArrayCoordinates1d(lat): Bounds[-9881992.849134896, 29995929.885877542], N[21], ctype['point']
                lon: ArrayCoordinates1d(lon): Bounds[1928928.7360588573, 4187156.434405213], N[21], ctype['midpoint']

        Transform stacked coordinates::

            c = Coordinates([(np.linspace(-10, 10, 21), np.linspace(-30, -10, 21))], dims=['lat_lon'])
            c.transform('EPSG:2193')

            >> Coordinates
                lat_lon[lat]: ArrayCoordinates1d(lat): Bounds[-9881992.849134896, 29995929.885877542], N[21], ctype['point']
                lat_lon[lon]: ArrayCoordinates1d(lon): Bounds[1928928.7360588573, 4187156.434405213], N[21], ctype['midpoint']

        Transform coordinates using a PROJ4 string::

            c = Coordinates([np.linspace(-10, 10, 21), np.linspace(-30, -10, 21)], dims=['lat', 'lon'])
            c.transform('+proj=merc +lat_ts=56.5 +ellps=GRS80')

            >> Coordinates
                lat: ArrayCoordinates1d(lat): Bounds[-1847545.541169525, -615848.513723175], N[21], ctype['midpoint']
                lon: ArrayCoordinates1d(lon): Bounds[-614897.0725896168, 614897.0725896184], N[21], ctype['midpoint']

        Parameters
        ----------
        crs : str
            PROJ4 compatible coordinate reference system string.

        Returns
        -------
        :class:`podpac.Coordinates`
            Transformed Coordinates

        Raises
        ------
        ValueError
            Coordinates must have both lat and lon dimensions if either is defined
        """
        from_crs = self.CRS
        to_crs = pyproj.CRS(crs)

        # no transform needed
        if from_crs == to_crs:
            return deepcopy(self)

        # make sure the CRS defines vertical units
        if "alt" in self.udims and not to_crs.is_vertical:
            raise ValueError("Altitude dimension is defined, but CRS to transform does not contain vertical unit")

        if "lat" in self.dims and "lon" not in self.dims:
            raise ValueError("Cannot transform lat coordinates without lon coordinates")

        if "lon" in self.dims and "lat" not in self.dims:
            raise ValueError("Cannot transform lon coordinates without lon coordinates")

        if "lat" in self.dims and "lon" in self.dims and abs(self.dims.index("lat") - self.dims.index("lon")) != 1:
            raise ValueError("Cannot transform coordinates with nonadjacent lat and lon, transpose first")

        transformer = pyproj.Transformer.from_proj(from_crs, to_crs, always_xy=True)

        # Collect the individual coordinates
        cs = [c for c in self.values()]

        if "lat" in self.dims and "lon" in self.dims:
            # try to do a simplified transform (resulting in unstacked lat-lon coordinates)
            tc = self._simplified_transform(crs, transformer)
            if tc:
                cs[self.dims.index("lat")] = tc[0]
                cs[self.dims.index("lon")] = tc[1]

            # otherwise convert lat-lon to dependent coordinates
            else:
                ilat = self.dims.index("lat")
                ilon = self.dims.index("lon")
                if ilat == ilon - 1:
                    c1, c2 = self["lat"], self["lon"]
                elif ilon == ilat - 1:
                    c1, c2 = self["lon"], self["lat"]

                c = DependentCoordinates(
                    np.meshgrid(c1.coordinates, c2.coordinates, indexing="ij"),
                    dims=[c1.name, c2.name],
                    ctypes=[c1.ctype, c2.ctype],
                    segment_lengths=[c1.segment_lengths, c2.segment_lengths],
                )

                # replace 'lat' and 'lon' entries with single 'lat,lon' entry
                i = min(ilat, ilon)
                cs.pop(i)
                cs.pop(i)
                cs.insert(i, c)

        # transform
        # note that unstacked lat and lon coordinates are ignored in the _transform method
        # so the simplified transform above will be preserved
        ts = []
        for c in cs:
            tc = c._transform(transformer)
            if isinstance(tc, list):
                ts.extend(tc)
            else:
                ts.append(tc)

        return Coordinates(ts, crs=crs, validate_crs=False)

    def _simplified_transform(self, crs, transformer):
        """ Transform coordinates to simple Coordinates1d (instead of DependentCoordinates) if possible """

        # TODO: ADD ALTITUDE
        def make_uniform(coords, size, name):
            tc = None
            dcoords = np.diff(coords)
            if np.allclose(dcoords[1:], dcoords[0], atol=1e-7):
                # We've actually already done the needed transformation, so we can just go ahead and create the
                # coordinates
                tc = clinspace(coords[0], coords[-1], size, name)
            return tc

        lat = clinspace(self["lat"].coordinates[0], self["lat"].coordinates[-1], 5)
        lon = clinspace(self["lon"].coordinates[0], self["lon"].coordinates[-1], 5)
        c = DependentCoordinates(
            np.meshgrid(lat.coordinates, lon.coordinates, indexing="ij"),
            dims=["lat", "lon"],
            ctypes=[self["lat"].ctype, self["lon"].ctype],
            segment_lengths=[self["lat"].segment_lengths, self["lon"].segment_lengths],
        )
        t = c._transform(transformer)
        if not isinstance(t, list):
            # Then we are not independent -- this was checked in the Dependent Coordinates
            return

        # Great, we CAN simplify the transformed coordinates. Now let's test if these are uniform or not
        # Note, the dependent coordinates may already give UniformCoordinates, but these are not the correct size

        # Lat uniform test and transform
        if isinstance(t[0], UniformCoordinates1d):
            t_lat = make_uniform(t[0].coordinates, self["lat"].size, "lat")
        else:
            # Need to compute the non-uniform transformed coordinates
            temp_coords = Coordinates(
                [[self["lat"].coordinates, self["lat"].coordinates * 0 + self["lon"].coordinates.mean()]],
                ["lat_lon"],
                crs=self.crs,
            )
            t_lat = temp_coords.transform(crs)["lat"]

            # There is a small possibility here that these are again uniform (i.e. non-uniform array input gives
            # uniform array output). So let's check that...
            t_lat_2 = make_uniform(t_lat.coordinates, self["lat"].size, "lat")
            if t_lat_2 is not None:
                t_lat = t_lat_2

        # Lon uniform test and transform
        if isinstance(t[1], UniformCoordinates1d):
            t_lon = make_uniform(t[1].coordinates, self["lon"].size, "lon")
        else:
            # Need to compute the non-uniform coordinates
            temp_coords = Coordinates(
                [[self["lon"].coordinates, self["lon"].coordinates * 0 + self["lat"].coordinates.mean()]],
                ["lon_lat"],
                crs=self.crs,
            )
            t_lon = temp_coords.transform(crs)["lon"]

            # There is a small possibility here that these are again uniform (i.e. non-inform array input gives
            # uniform array output). So let's check that...
            t_lon_2 = make_uniform(t_lon.coordinates, self["lon"].size, "lon")
            if t_lon_2 is not None:
                t_lon = t_lon_2

        return t_lat, t_lon

    # ------------------------------------------------------------------------------------------------------------------
    # Operators/Magic Methods
    # ------------------------------------------------------------------------------------------------------------------

    def __repr__(self):
        rep = str(self.__class__.__name__)
        if self.crs_desc:
            rep += " ({})".format(self.crs_desc)
        elif self.crs:
            rep += " ({})".format(self.crs)
        for c in self._coords.values():
            if isinstance(c, Coordinates1d):
                rep += "\n\t%s: %s" % (c.name, c)
            elif isinstance(c, StackedCoordinates):
                for dim in c.dims:
                    rep += "\n\t%s[%s]: %s" % (c.name, dim, c[dim])
            elif isinstance(c, DependentCoordinates):
                for dim in c.dims:
                    rep += "\n\t%s[%s]: %s" % (c.name, dim, c._rep(dim))
        return rep


def merge_dims(coords_list):
    """
    Merge the coordinates.

    Arguments
    ---------
    coords_list : list
        List of :class:`Coordinates` with unique dimensions.

    Returns
    -------
    coords : :class:`Coordinates`
        Coordinates with merged dimensions.

    Raises
    ------
    ValueError
        If dimensions are duplicated.
    """

    coords_list = list(coords_list)
    for coords in coords_list:
        if not isinstance(coords, Coordinates):
            raise TypeError("Cannot merge '%s' with Coordinates" % type(coords))

    if len(coords_list) == 0:
        return Coordinates([], crs=None)

    # check crs
    crs = coords_list[0].crs
    if not all(coords.crs == crs for coords in coords_list):
        raise ValueError("Cannot merge Coordinates, crs mismatch")

    # merge
    coords = sum([list(coords.values()) for coords in coords_list], [])
    return Coordinates(coords, crs=crs, validate_crs=False)


def concat(coords_list):
    """
    Combine the given coordinates by concatenating coordinate values in each dimension.

    Arguments
    ---------
    coords_list : list
        List of :class:`Coordinates`.

    Returns
    -------
    coords : :class:`Coordinates`
        Coordinates with concatenated coordinate values in each dimension.

    See Also
    --------
    :class:`union`
    """

    coords_list = list(coords_list)
    for coords in coords_list:
        if not isinstance(coords, Coordinates):
            raise TypeError("Cannot concat '%s' with Coordinates" % type(coords))

    if not coords_list:
        return Coordinates([], crs=None)

    # check crs
    crs = coords_list[0].crs
    if not all(coords.crs == crs for coords in coords_list):
        raise ValueError("Cannot concat Coordinates, crs mismatch")

    # concatenate
    d = OrderedDict()
    for coords in coords_list:
        for dim, c in coords.items():
            if isinstance(c, Coordinates1d):
                if dim not in d:
                    d[dim] = c.coordinates
                else:
                    d[dim] = np.concatenate([d[dim], c.coordinates])
            elif isinstance(c, StackedCoordinates):
                if dim not in d:
                    d[dim] = [s.coordinates for s in c]
                else:
                    d[dim] = [np.concatenate([d[dim][i], s.coordinates]) for i, s in enumerate(c)]

    return Coordinates(list(d.values()), dims=list(d.keys()), crs=crs, validate_crs=False)


def union(coords_list):
    """
    Combine the given coordinates by collecting the unique, sorted coordinate values in each dimension.

    Arguments
    ---------
    coords_list : list
        List of :class:`Coordinates`.

    Returns
    -------
    coords : :class:`Coordinates`
        Coordinates with unique, sorted coordinate values in each dimension.

    See Also
    --------
    :class:`concat`
    """

    return concat(coords_list).unique()
