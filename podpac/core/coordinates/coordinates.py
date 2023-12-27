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

import numpy as np
import traitlets as tl
import pandas as pd
import xarray as xr
import xarray.core.coordinates
from six import string_types
import pyproj
import logging
from scipy import spatial

import podpac
from podpac.core.settings import settings
from podpac.core.utils import OrderedDictTrait, _get_query_params_from_url, _get_param, cached_property
from podpac.core.coordinates.utils import has_alt_units
from podpac.core.coordinates.utils import VALID_DIMENSION_NAMES
from podpac.core.coordinates.base_coordinates import BaseCoordinates
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates
from podpac.core.coordinates.affine_coordinates import AffineCoordinates
from podpac.core.coordinates.cfunctions import clinspace
from podpac.core.utils import hash_alg

# Optional dependencies
from lazy_import import lazy_module, lazy_class

rasterio = lazy_module("rasterio")

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

    _coords = OrderedDictTrait(value_trait=tl.Instance(BaseCoordinates), default_value=OrderedDict())

    def __init__(self, coords, dims=None, crs=None, validate_crs=True):
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
            else:
                c = ArrayCoordinates1d(coords[i])

            # propagate name
            c._set_name(dim)

            # set coords
            dcoords[dim] = c

        self.set_trait("_coords", dcoords)

        if crs is not None:
            # validate
            if validate_crs:
                # raises pyproj.CRSError if invalid
                CRS = pyproj.CRS(crs)

                # make sure CRS defines vertical units
                if "alt" in self.udims and not has_alt_units(CRS):
                    raise ValueError("Altitude dimension is defined, but CRS does not contain vertical unit")

            crs = self.set_trait("crs", crs)

        super(Coordinates, self).__init__()

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
    def grid(cls, dims=None, crs=None, **kwargs):
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

        Returns
        -------
        :class:`Coordinates`
            podpac Coordinates

        See Also
        --------
        points
        """

        coords = cls._coords_from_dict(kwargs, order=dims)
        return cls(coords, crs=crs)

    @classmethod
    def points(cls, crs=None, dims=None, **kwargs):
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
        return cls([stacked], crs=crs)

    @classmethod
    def from_xarray(cls, x, crs=None, validate_crs=False):
        """
        Create podpac Coordinates from xarray coords.

        Arguments
        ---------
        x : DataArray, Dataset, DataArrayCoordinates, DatasetCoordinates
            DataArray, Dataset, or xarray coordinates
        crs : str, optional
            Coordinate reference system. Supports any PROJ4 or PROJ6 compliant string (https://proj.org/).
            If not provided, the crs will be loaded from ``x.attrs`` if possible.
        validate_crs: bool, optional
            Default is False. If True, the crs will be validated.

        Returns
        -------
        coords : :class:`Coordinates`
            podpac Coordinates
        """
        d = OrderedDict()
        if isinstance(x, (xr.DataArray, xr.Dataset)):
            # only pull crs from the DataArray attrs if the crs is not specified
            if crs is None:
                crs = x.attrs.get("crs")

            xcoords = x.coords
            if "geotransform" in x.attrs:
                other = cls.from_xarray(xcoords, crs=crs, validate_crs=validate_crs).udrop(["lat", "lon"])
                latshape = xcoords["lat"].shape
                lonshape = xcoords["lon"].shape
                if latshape == lonshape and len(latshape) == 2:
                    shape = latshape
                else:
                    shape = [latshape[0], lonshape[0]]
                    xdims = list(xcoords.keys())
                    if xdims.index("lat") > xdims.index("lon"):
                        shape = shape[::-1]
                lat_lon = cls.from_geotransform(x.geotransform, shape=shape, crs=crs, validate_crs=validate_crs)
                coords = merge_dims([other, lat_lon])

                # These dims might have something like lat_lon-1, lat_lon-2, so eliminate the '-' ...
                dims = [d.split("-")[0] for d in xcoords.dims if d != "output"]
                # ... and make sure it's all unique without changing order (np.unique would change order...)
                dims = [d for i, d in enumerate(dims) if d not in dims[:i]]
                coords = coords.transpose(*dims)
                return coords

        elif isinstance(x, (xarray.core.coordinates.DataArrayCoordinates, xarray.core.coordinates.DatasetCoordinates)):
            xcoords = x
        else:
            raise TypeError(
                "Coordinates.from_xarray expects an xarray DataArray or DataArrayCoordinates, not '%s'" % type(x)
            )

        # warn if crs is not provided as an argument OR in the data array
        if crs is None:
            warnings.warn("using default crs for podpac coordinates loaded from xarray because no crs was provided")

        for dim in xcoords.dims:
            if dim in d:
                continue
            if dim == "output":
                continue

            if "-" in dim:
                dim, _ = dim.split("-")

            if dim in xcoords.indexes and isinstance(xcoords.indexes[dim], pd.MultiIndex):
                # 1d stacked
                d[dim] = StackedCoordinates.from_xarray(xcoords[dim])
            elif "_" in dim:
                # nd stacked
                d[dim] = StackedCoordinates([xcoords[k] for k in dim.split("_")], name=dim)
            else:
                # unstacked
                d[dim] = ArrayCoordinates1d.from_xarray(xcoords[dim])

        coords = cls(list(d.values()), crs=crs, validate_crs=validate_crs)
        return coords

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
        coords = OrderedDict()

        # The ordering here is lat/lon or y/x for WMS 1.3.0
        # The ordering here is lon/lat or x/y for WMS 1.1
        # See https://docs.geoserver.org/stable/en/user/services/wms/reference.html
        # and https://docs.geoserver.org/stable/en/user/services/wms/basics.html

        bbox = np.array(_get_param(params, "BBOX").split(","), float)

        # Note, version 1.1 used "SRS" and 1.3 uses 'CRS'
        coords["crs"] = _get_param(params, "SRS")
        if coords["crs"] is None:
            coords["crs"] = _get_param(params, "CRS")

        if _get_param(params, "SERVICE") == "WCS":
            r = -1
        elif _get_param(params, "VERSION").startswith("1.1"):
            r = -1
        elif _get_param(params, "VERSION").startswith("1.3"):
            r = 1
            try:
                crs = pyproj.CRS(coords["crs"])
                if crs.axis_info[0].direction != "north":
                    r = -1
            except:
                pass
        else:
            r = 1

        # Extract bounding box information and translate to PODPAC coordinates
        # Not, size does not get re-ordered, Height == Lat and width = lon -- ALWAYS
        size = np.array([_get_param(params, "HEIGHT"), _get_param(params, "WIDTH")], int)
        start = bbox[:2][::r]
        stop = bbox[2:][::r]

        # The bbox gives the edges of the pixels, but our coordinates use the
        # box centers -- so we have to shrink the start/stop portions
        dx = (stop - start) / (size)  # This should take care of signs
        start = start + dx / 2
        stop = stop - dx / 2

        coords["coords"] = [
            {"name": "lat", "start": stop[0], "stop": start[0], "size": size[0]},
            {"name": "lon", "start": start[1], "stop": stop[1], "size": size[1]},
        ]

        if "TIME" in params:
            coords["coords"].append({"name": "time", "values": [_get_param(params, "TIME")]})

        other_params = _get_param(params, "PARAMS")
        if other_params:
            other_params = json.loads(other_params)
            # check the param keys for any dimension in VALID_DIMENSIONS. If yes, add it to the coords
            for key in other_params:
                if key in VALID_DIMENSION_NAMES:
                    coords["coords"].append({"name": key, "values": [other_params[key]]})

        return cls.from_definition(coords)

    @classmethod
    def from_geotransform(cls, geotransform, shape, crs=None, validate_crs=True):
        """Creates Coordinates from GDAL Geotransform."""

        cs = AffineCoordinates(geotransform, shape).simplify()
        if isinstance(cs, AffineCoordinates):
            cs = [cs]
        return Coordinates(cs, crs=crs, validate_crs=validate_crs)

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
            elif "geotransform" in e and "shape" in e:
                c = AffineCoordinates.from_definition(e)
            else:
                raise ValueError("Could not parse coordinates definition item with keys %s" % e.keys())

            coords.append(c)

        kwargs = {k: v for k, v in d.items() if k != "coords"}
        return cls(coords, **kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    # standard dict-like methods
    # ------------------------------------------------------------------------------------------------------------------

    def keys(self):
        """dict-like keys: dims"""
        return self._coords.keys()

    def values(self):
        """dict-like values: coordinates for each key/dimension"""
        return self._coords.values()

    def items(self):
        """dict-like items: (dim, coordinates) pairs"""
        return self._coords.items()

    def __iter__(self):
        return iter(self._coords)

    def get(self, dim, default=None):
        """dict-like get: get coordinates by dimension name with an optional"""
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
                if isinstance(c, StackedCoordinates) and dim in c.dims:
                    return c[dim]

            raise KeyError("Dimension '%s' not found in Coordinates %s" % (dim, self.dims))

        else:
            # extend index to a tuple of the correct length
            if not isinstance(index, tuple):
                index = (index,)
            index = index + tuple(slice(None) for i in range(self.ndim - len(index)))

            # bundle shaped coordinates indices
            indices = []
            i = 0
            for c in self._coords.values():
                if c.ndim == 1:
                    indices.append(index[i])
                else:
                    indices.append(tuple(index[i : i + c.ndim]))
                i += c.ndim

            cs = [c[I] for c, I in zip(self._coords.values(), indices)]
            return Coordinates(cs, validate_crs=False, **self.properties)

    def __setitem__(self, dim, c):

        # coerce
        if isinstance(c, BaseCoordinates):
            pass
        elif isinstance(c, Coordinates):
            c = c[dim]
        elif "_" in dim:
            c = StackedCoordinates(c)
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
        """dict-like update: add/replace coordinates using another Coordinates object"""
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
        # TODO check transform instead
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
    def xdims(self):
        """:tuple: Tuple of indexing dimension names used to make xarray DataArray.

        Unless there are shaped (ndim>1) coordinates, this will match the ``dims``.
        """

        return tuple(dim for c in self._coords.values() for dim in c.xdims)

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
        """:int: Number of dimensions."""

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
    def xcoords(self):
        """
        :dict: xarray coords
        """

        xcoords = OrderedDict()
        for c in self._coords.values():
            xcoords.update(c.xcoords)
        return xcoords

    @property
    def CRS(self):
        return pyproj.CRS(self.crs)

    @property
    def alt_units(self):
        CRS = self.CRS

        if not has_alt_units(CRS):
            return None

        # try to get vunits
        d = CRS.to_dict()
        if "vunits" in d:
            return d["vunits"]

        # get from axis info (is this is ever useful)
        for axis in self.CRS.axis_info:
            if axis.direction == "up":
                return axis.unit_name  # may need to be converted, e.g. "centimetre" > "cm"

        raise RuntimeError("Could not get alt_units from crs '%s'" % self.crs)

    @property
    def properties(self):
        """:dict: Dictionary of the coordinate properties."""

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
        # "wkt" is suggested as best format: https://proj.org/faq.html#what-is-the-best-format-for-describing-coordinate-reference-systems
        d["crs"] = self.CRS.to_wkt()
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

    @cached_property
    def hash(self):
        """:str: Coordinates hash value."""
        # We can't use self.json for the hash because the CRS is not standardized.
        # As such, we json.dumps the full definition.
        json_d = json.dumps(self.full_definition, separators=(",", ":"), cls=podpac.core.utils.JSONEncoder)
        return hash_alg(json_d.encode("utf-8")).hexdigest()

    @property
    def geotransform(self):
        """:tuple: GDAL geotransform."""
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
            and "lon" in self.dims
            and self._coords["lat"].is_uniform
            and self._coords["lon"].is_uniform
        ):
            if self.dims.index("lon") < self.dims.index("lat"):
                first, second = "lat", "lon"
            else:
                first, second = "lon", "lat"  # This case will have the exact correct geotransform
            transform = rasterio.transform.Affine.translation(
                self[first].start - self[first].step / 2, self[second].start - self[second].step / 2
            ) * rasterio.transform.Affine.scale(self[first].step, self[second].step)
            transform = transform.to_gdal()
        elif "lat_lon" in self.dims and isinstance(self._coords["lat_lon"], AffineCoordinates):
            transform = self._coords["lat_lon"].geotransform
        elif "lon_lat" in self.dims and isinstance(self._coords["lon_lat"], AffineCoordinates):
            transform = self._coords["lon_lat"].geotransform
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

    def get_area_bounds(self, boundary):
        """Get coordinate area bounds, including segment information, for each unstacked dimension.

        Arguments
        ---------
        boundary : dict
            dictionary of boundary offsets for each unstacked dimension. Non-segment dimensions can be omitted.

        Returns
        -------
        area_bounds : dict
            Dictionary of (low, high) coordinates area_bounds in each unstacked dimension
        """

        area_bounds = {}
        for dim, c in self._coords.items():
            if isinstance(c, StackedCoordinates):
                area_bounds.update(c.get_area_bounds(boundary))
            else:
                area_bounds[dim] = c.get_area_bounds(boundary.get(dim))
        return area_bounds

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

    def udrop(self, dims, ignore_missing=False):
        """
        Remove the given individual dimensions from the Coordinates `udims`.

        Unlike `drop`, ``udrop`` will remove parts of stacked coordinates::

            In [1]: c = podpac.Coordinates([[[0, 1], [10, 20]], '2018-01-01'], dims=['lat_lon', 'time'])

            In [2]: c
            Out[2]:
            Coordinates
                lat_lon[lat]: ArrayCoordinates1d(lat): Bounds[0.0, 1.0], N[2]
                lat_lon[lon]: ArrayCoordinates1d(lon): Bounds[10.0, 20.0], N[2]
                time: ArrayCoordinates1d(time): Bounds[2018-01-01, 2018-01-01], N[1]

            In [3]: c.udrop('lat')
            Out[3]:
            Coordinates
                lon: ArrayCoordinates1d(lon): Bounds[10.0, 20.0], N[2]
                time: ArrayCoordinates1d(time): Bounds[2018-01-01, 2018-01-01], N[1]

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
                if len(stacked) == len(c):
                    # preserves parameterized stacked coordinates such as AffineCoordinates
                    cs.append(c)
                elif len(stacked) > 1:
                    cs.append(StackedCoordinates(stacked))
                elif len(stacked) == 1:
                    cs.append(stacked[0])

        return Coordinates(cs, validate_crs=False, **self.properties)

    def intersect(self, other, dims=None, outer=False, return_index=False):
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
        return_index : bool, optional
            If True, return index for the selection in addition to coordinates. Default False.

        Returns
        -------
        intersection : :class:`Coordinates`
            Coordinates object consisting of the intersection in each dimension.
        selection_index : list
            List of indices for each dimension that produces the intersection, only if ``return_index`` is True.
        """

        if not isinstance(other, Coordinates):
            raise TypeError("Coordinates cannot be intersected with type '%s'" % type(other))

        if other.crs.lower() != self.crs.lower():
            other = other.transform(self.crs)

        bounds = other.bounds
        if dims is not None:
            bounds = {dim: bounds[dim] for dim in dims}  # if dim in bounds}

        return self.select(bounds, outer=outer, return_index=return_index)

    def select(self, bounds, return_index=False, outer=False):
        """
        Get the coordinate values that are within the given bounds for each dimension.

        The default selection returns coordinates that are within the bounds::

            In [1]: c = Coordinates([[0, 1, 2, 3], [10, 20, 30, 40]], dims=['lat', 'lon'])

            In [2]: c.select({'lat': [1.5, 3.5]})
            Out[2]:
            Coordinates
                    lat: ArrayCoordinates1d(lat): Bounds[2.0, 3.0], N[2]
                    lon: ArrayCoordinates1d(lon): Bounds[10.0, 40.0], N[4]

            In [3]: c.select({'lat': [1.5, 3.5], 'lon': [25, 45]})
            Out[3]:
            Coordinates
                    lat: ArrayCoordinates1d(lat): Bounds[2.0, 3.0], N[2]
                    lon: ArrayCoordinates1d(lon): Bounds[30.0, 40.0], N[2]

        The *outer* selection returns the minimal set of coordinates that contain the bounds::

            In [4]: c.select({'lat':[1.5, 3.5]}, outer=True)
            Out[4]:
            Coordinates
                    lat: ArrayCoordinates1d(lat): Bounds[1.0, 3.0], N[3]
                    lon: ArrayCoordinates1d(lon): Bounds[10.0, 40.0], N[4]

        Parameters
        ----------
        bounds : dict
            Selection bounds for the desired coordinates.
        outer : bool, optional
            If True, do *outer* selections. Default False.
        return_index : bool, optional
            If True, return index for the selections in addition to coordinates. Default False.

        Returns
        -------
        selection : :class:`Coordinates`
            Coordinates object with coordinates within the given bounds.
        selection_index : list
            index for the selected coordinates in each dimension (only if return_index=True)
        """

        selections = [c.select(bounds, outer=outer, return_index=return_index) for c in self._coords.values()]
        return self._make_selected_coordinates(selections, return_index)

    def _make_selected_coordinates(self, selections, return_index):
        if return_index:
            coords = Coordinates([c for c, I in selections], validate_crs=False, **self.properties)
            # unbundle shaped indices
            I = [I if c.ndim > 1 else [I] for c, I in selections]
            I = [e for l in I for e in l]
            return coords, tuple(I)
        else:
            return Coordinates(selections, validate_crs=False, **self.properties)

    def unique(self, return_index=False):
        """
        Remove duplicate coordinate values from each dimension.

        Arguments
        ---------
        return_index : bool, optional
            If True, return index for the unique coordinates in addition to the coordinates. Default False.
        Returns
        -------
        unique : :class:`podpac.Coordinates`
            New Coordinates object with unique, sorted coordinate values in each dimension.
        unique_index : list of indices
            index for the unique coordinates in each dimension (only if return_index=True)
        """

        if self.ndim == 0:
            if return_index:
                return self[:], tuple()
            else:
                return self[:]

        cs, I = zip(*[c.unique(return_index=True) for c in self.values()])
        unique = Coordinates(cs, validate_crs=False, **self.properties)

        if return_index:
            return unique, I
        else:
            return unique

    def unstack(self):
        """
        Unstack the coordinates of all of the dimensions.

        Returns
        -------
        unstacked : :class:`podpac.Coordinates`
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

    def transform_time(self, units):
        if "time" not in self.dims:
            raise ValueError("Time dimension is required to do a time transformation.")

        time_coords = self["time"].coordinates
        xr_time = xr.Dataset({"time": time_coords})
        new_time = getattr(xr_time.time.dt, units).data

        new_time_coord = ArrayCoordinates1d(coordinates=new_time, name="time").simplify()
        coords = (self).drop("time")
        # transpose will make a copy
        coords = merge_dims([coords, Coordinates([new_time_coord], crs=self.crs)]).transpose(*self.dims, in_place=False)
        return coords

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
                lat: ArrayCoordinates1d(lat): Bounds[-9881992.849134896, 29995929.885877542], N[21]
                lon: ArrayCoordinates1d(lon): Bounds[1928928.7360588573, 4187156.434405213], N[21]

        Transform stacked coordinates::

            c = Coordinates([(np.linspace(-10, 10, 21), np.linspace(-30, -10, 21))], dims=['lat_lon'])
            c.transform('EPSG:2193')

            >> Coordinates
                lat_lon[lat]: ArrayCoordinates1d(lat): Bounds[-9881992.849134896, 29995929.885877542], N[21]
                lat_lon[lon]: ArrayCoordinates1d(lon): Bounds[1928928.7360588573, 4187156.434405213], N[21]

        Transform coordinates using a PROJ4 string::

            c = Coordinates([np.linspace(-10, 10, 21), np.linspace(-30, -10, 21)], dims=['lat', 'lon'])
            c.transform('+proj=merc +lat_ts=56.5 +ellps=GRS80')

            >> Coordinates
                lat: ArrayCoordinates1d(lat): Bounds[-1847545.541169525, -615848.513723175], N[21]
                lon: ArrayCoordinates1d(lon): Bounds[-614897.0725896168, 614897.0725896184], N[21]

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
        if "alt" in self.udims and not has_alt_units(to_crs):
            raise ValueError("Altitude dimension is defined, but CRS to transform does not contain vertical unit")

        if "lat" in self.udims and "lon" not in self.udims:
            raise ValueError("Cannot transform lat coordinates without lon coordinates")

        if "lon" in self.udims and "lat" not in self.udims:
            raise ValueError("Cannot transform lon coordinates without lat coordinates")

        if "lat" in self.dims and "lon" in self.dims and abs(self.dims.index("lat") - self.dims.index("lon")) != 1:
            raise ValueError("Cannot transform coordinates with nonadjacent lat and lon, transpose first")

        transformer = pyproj.Transformer.from_proj(from_crs, to_crs, always_xy=True)

        # Collect the individual coordinates
        cs = [c for c in self.values()]

        if "lat" in self.dims and "lon" in self.dims:
            st = self._simplified_transform(transformer, cs)

            if st:  # We could do the shortcut, and we have the result already
                cs = st
            # otherwise, replace lat and lon coordinates with a single stacked lat_lon:
            else:  # Have to convert every coordinate
                ilat = self.dims.index("lat")
                ilon = self.dims.index("lon")
                if ilat == ilon - 1:
                    c1, c2 = self["lat"], self["lon"]
                elif ilon == ilat - 1:
                    c1, c2 = self["lon"], self["lat"]
                else:
                    raise RuntimeError("lat and lon dimensions should be adjacent")

                c = StackedCoordinates(
                    np.meshgrid(c1.coordinates, c2.coordinates, indexing="ij"), dims=[c1.name, c2.name]
                )

                # replace 'lat' and 'lon' entries with single 'lat_lon' entry
                i = min(ilat, ilon)
                cs.pop(i)
                cs.pop(i)
                cs.insert(i, c)

        # transform remaining altitude or stacked spatial dimensions if needed
        ts = []
        for c in cs:
            tc = c._transform(transformer)
            if isinstance(tc, list):
                ts.extend(tc)
            else:
                ts.append(tc)

        return Coordinates(ts, crs=crs, validate_crs=False)

    def _simplified_transform(self, transformer, cs):
        lat_sample = np.linspace(self["lat"].bounds[0], self["lat"].bounds[1], 5)
        lon_sample = np.linspace(self["lon"].bounds[0], self["lon"].bounds[1], 5)
        sample = StackedCoordinates(np.meshgrid(lat_sample, lon_sample, indexing="ij"), dims=["lat", "lon"])
        # The sample tests if the crs transform is linear, or non-linear. The results are as follows:
        #
        # Start from "uniform stacked"
        # 1. Returns "uniform unstacked"  <-- simple scaling between crs's
        # 2. Returns "array unstacked" <-- Orthogonal coordinates still, but non-linear in this dim
        # 3. Returns "Stacked" <-- not orthogonal from one crs to the other
        #
        t = sample._transform(transformer)

        if isinstance(t, StackedCoordinates):  # Need to transform ALL the coordinates
            return
        # Then we can do a faster transform, either already done or just the diagonal
        for i, j in zip([0, 1], [1, 0]):
            if isinstance(t[i], UniformCoordinates1d) and isinstance(cs[i], UniformCoordinates1d):  # already done
                start = t[i].start
                stop = t[i].stop
                if self[t[i].name].is_descending:
                    start, stop = stop, start
                cs[self.dims.index(t[i].name)] = clinspace(start, stop, self[t[i].name].size, name=t[i].name)
                continue
            # Transform all of the points for this dimension (either lat or lon) and record result
            this = self[t[i].name]
            that = self[t[j].name]
            if this.size > 1 and that.size > 1:
                other = clinspace(that.bounds[0], that.bounds[1], this.size)
            elif this.size == that.size:
                other = that.coordinates
            else:
                other = np.zeros(this.size) + that.coordinates.mean()
            diagonal = StackedCoordinates([this.coordinates, other], dims=[this.name, that.name])
            t_diagonal = diagonal._transform(transformer)
            cs[self.dims.index(this.name)] = t_diagonal[this.name]
        return cs

    def simplify(self):
        """Simplify coordinates in each dimension.

        Returns
        -------
        simplified : Coordinates
            Simplified coordinates.
        """

        cs = []
        for c in self._coords.values():
            c2 = c.simplify()
            if isinstance(c2, list):
                cs += c2
            else:
                cs.append(c2)
        return Coordinates(cs, **self.properties)

    def issubset(self, other):
        """Report whether other Coordinates contains these coordinates.

        Note that the dimension order and stacking is ignored.

        Arguments
        ---------
        other : Coordinates
            Other coordinates to check

        Returns
        -------
        issubset : bool
            True if these coordinates are a subset of the other coordinates in every dimension.
        """

        if set(self.udims) != set(other.udims):
            return False

        return all(c.issubset(other) for c in self.values())

    def is_stacked(self, dim):  # re-wrote to be able to iterate through c.dims
        value = (dim in self.dims) + (dim in self.udims)
        if value == 0:
            raise ValueError("Dimension {} is not in self.dims={}".format(dim, self.dims))
        elif value == 1:  # one true, one false
            return True
        elif value == 2:  # both true
            return False

    def horizontal_resolution(self, units="meter", restype="nominal"):
        """
        Returns horizontal resolution of coordinate system.

        Parameters
        ----------
        units : str
            The desired unit the returned resolution should be in. Supports any unit supported by podpac.units (i.e. pint). Default is 'meter'.
        restype : str
            The kind of horizontal resolution that should be returned. Supported values are:
            - "nominal" <-- Returns a number. Gives a 'nominal' resolution over the entire domain. This is wrong but fast.
            - "summary" <-- Returns a tuple (mean, standard deviation). Gives the exact mean and standard deviation for unstacked coordinates, some error for stacked coordinates
            - "full" <-- Returns a 1 or 2-D array. Gives exact grid differences if unstacked coordinates or distance matrix if stacked coordinates

        Returns
        -------
        OrderedDict
            A dictionary with:
            keys : str
                dimension names
            values
                resolution (format determined by 'type' parameter)

        Raises
        ------
        ValueError
            If the 'restype' is not one of the supported resolution types


        """
        # This function handles mainly edge case sanitation.
        # It calls StackedCoordinates and Coordinates1d 'horizontal_resolution' methods to get the actual values.

        if "lat" not in self.udims:  # require latitude
            raise ValueError("Latitude required for horizontal resolution.")

        # ellipsoid tuple to pass to geodesic
        ellipsoid_tuple = (
            self.CRS.ellipsoid.semi_major_metre / 1000,
            self.CRS.ellipsoid.semi_minor_metre / 1000,
            1 / self.CRS.ellipsoid.inverse_flattening,
        )

        # main execution loop
        resolutions = OrderedDict()  # To return
        for name, dim in self.items():
            if dim.is_stacked:
                if "lat" in dim.dims and "lon" in dim.dims:
                    resolutions[name] = dim.horizontal_resolution(
                        None, ellipsoid_tuple, self.CRS.coordinate_system.name, restype, units
                    )
                elif "lat" in dim.dims:
                    # Calling self['lat'] forces UniformCoordinates1d, even if stacked
                    resolutions["lat"] = self["lat"].horizontal_resolution(
                        self["lat"], ellipsoid_tuple, self.CRS.coordinate_system.name, restype, units
                    )
                elif "lon" in dim.dims:
                    # Calling self['lon'] forces UniformCoordinates1d, even if stacked
                    resolutions["lon"] = self["lon"].dim.horizontal_resolution(
                        self["lat"], ellipsoid_tuple, self.CRS.coordinate_system.name, restype, units
                    )
            elif (
                name == "lat" or name == "lon"
            ):  # need to do this inside of loop in case of stacked [[alt,time]] but unstacked [lat, lon]
                resolutions[name] = dim.horizontal_resolution(
                    self["lat"], ellipsoid_tuple, self.CRS.coordinate_system.name, restype, units
                )

        return resolutions

    # ------------------------------------------------------------------------------------------------------------------
    # Operators/Magic Methods
    # ------------------------------------------------------------------------------------------------------------------

    def __repr__(self):
        rep = str(self.__class__.__name__)
        if self.crs:
            rep += " ({})".format(self.crs)
        for c in self._coords.values():
            if isinstance(c, Coordinates1d):
                rep += "\n\t%s: %s" % (c.name, c)
            elif isinstance(c, AffineCoordinates):
                rep += "\n\t%s: %s" % (c.name, c)
            elif isinstance(c, StackedCoordinates):
                for dim in c.dims:
                    rep += "\n\t%s[%s]: %s" % (c.name, dim, c[dim])
        return rep


def merge_dims(coords_list, validate_crs=True):
    """
    Merge the coordinates.

    Arguments
    ---------
    coords_list : list
        List of :class:`Coordinates` with unique dimensions.

    validate_crs : bool, optional
        Default is True. If False, the coordinates will not be checked for a common crs,
        and the crs of the first item in the list will be used.

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
    if validate_crs and not all(coords.crs == crs for coords in coords_list):
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
