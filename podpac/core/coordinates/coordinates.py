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

import numpy as np
import traitlets as tl
import pandas as pd
import xarray as xr
import xarray.core.coordinates
from six import string_types
import pyproj

import podpac
from podpac.core.utils import OrderedDictTrait
from podpac.core.coordinates.base_coordinates import BaseCoordinates
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates

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

    _coords = OrderedDictTrait(trait=tl.Instance(BaseCoordinates), default_value=OrderedDict())

    def __init__(self, coords, dims=None, coord_ref_sys=None, ctype=None, distance_units=None):
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
        coord_ref_sys : str, optional
            Default coordinates reference system. Supports any PROJ4 compliant string (https://proj4.org/index.html).
        ctype : str, optional
            Default coordinates type. One of 'point', 'midpoint', 'left', 'right'.
        distance_units : Units
            Default distance units.
        """

        if not isinstance(coords, (list, tuple, np.ndarray, xr.DataArray)):
            raise TypeError("Invalid coords, expected list or array, not '%s'" % type(coords))

        if dims is not None and not isinstance(dims, (tuple, list)):
            raise TypeError("Invalid dims type '%s'" % type(dims))

        if dims is None:
            for i, c in enumerate(coords):
                if not isinstance(c, (BaseCoordinates, xr.DataArray)):
                    raise TypeError("Cannot get dim for coordinates at position %d with type '%s'"
                                    "(expected 'Coordinates1d' or 'DataArray')" % (i, type(c)))

            dims = [c.name for c in coords]

        if len(dims) != len(coords):
            raise ValueError("coords and dims size mismatch, %d != %d" % (len(dims), len(coords)))

        # get/create coordinates
        dcoords = OrderedDict()
        for i, dim in enumerate(dims):
            if dim in dcoords:
                raise ValueError("Duplicate dimension name '%s' at position %d" % (dim, i))

            if isinstance(coords[i], BaseCoordinates):
                c = coords[i]
            elif '_' in dim:
                cs = [val if isinstance(val, Coordinates1d) else ArrayCoordinates1d(val) for val in coords[i]]
                c = StackedCoordinates(cs)
            else:
                c = ArrayCoordinates1d(coords[i])

            dcoords[dim] = c
            self._set_properties(c, dim, ctype, distance_units, coord_ref_sys, i)
        
        self.set_trait('_coords', dcoords)
        super(Coordinates, self).__init__()

    @tl.validate('_coords')
    def _validate_coords(self, d):
        val = d['value']

        if len(val) == 0:
            return val

        for dim, c in val.items():
            if dim != c.name:
                raise ValueError("Dimension name mismatch, '%s' != '%s'" % (dim, c.name))

        dims = [dim for c in val.values() for dim in c.dims]
        for dim in dims:
            if dims.count(dim) != 1:
                raise ValueError("Duplicate dimension name '%s' in dims %s" % (dim, tuple(val.keys())))

        crs = list(val.values())[0].coord_ref_sys
        for i, c in enumerate(val.values()):
            if c.coord_ref_sys != crs:
                raise ValueError("coord_ref_sys mismatch '%s' != '%s' at pos %d" % (c.coord_ref_sys, crs, i))

        return val

    def _set_properties(self, c, name, ctype, distance_units, coord_ref_sys, pos):
        if isinstance(c, Coordinates1d):
            cs = [c]
            names = [name]
        else:
            cs = list(c)
            names = name.split('_')

        for c, name in zip(cs, names):
            # set or check the coord_ref_sys
            if coord_ref_sys is not None:
                if 'coord_ref_sys' not in c.properties:
                    c.set_trait('coord_ref_sys', coord_ref_sys)
                elif coord_ref_sys != c.coord_ref_sys:
                    raise ValueError("coord_ref_sys mismatch %s != %s at pos %d" % (coord_ref_sys, c.coord_ref_sys, pos))

            # only set name, ctype, and units if they aren't already set
            if name is not None and 'name' not in c.properties:
                c.name = name
            if ctype is not None and 'ctype' not in c.properties:
                c.set_trait('ctype', ctype)
            if distance_units is not None and c.name in ['lat', 'lon', 'alt'] and 'units' not in c.properties:
                c.set_trait('units', distance_units)

    # ------------------------------------------------------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _coords_from_dict(d, order=None):
        if sys.version < '3.6':
            if order is None and len(d) > 1:
                raise TypeError('order required')

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
    def grid(cls, dims=None, coord_ref_sys=None, ctype=None, distance_units=None, **kwargs):
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
        coord_ref_sys : str, optional
            Default coordinates reference system
        ctype : str, optional
            Default coordinates type. One of 'point', 'midpoint', 'left', 'right'.
        distance_units : Units
            Default distance units.

        Returns
        -------
        :class:`Coordinates`
            podpac Coordinates

        See Also
        --------
        points
        """

        coords = cls._coords_from_dict(kwargs, order=dims)
        return cls(coords, coord_ref_sys=coord_ref_sys, ctype=ctype, distance_units=distance_units)

    @classmethod
    def points(cls, coord_ref_sys=None, ctype=None, distance_units=None, dims=None, **kwargs):
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
        coord_ref_sys : str, optional
            Default coordinates reference system
        ctype : str, optional
            Default coordinates type. One of 'point', 'midpoint', 'left', 'right'.
        distance_units : Units
            Default distance units.

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
        return cls([stacked], coord_ref_sys=coord_ref_sys, ctype=ctype, distance_units=distance_units)

    @classmethod
    def from_xarray(cls, xcoord, coord_ref_sys=None, ctype=None, distance_units=None):
        """
        Create podpac Coordinates from xarray coords.

        Arguments
        ---------
        xcoord : xarray.core.coordinates.DataArrayCoordinates
            xarray coords
        coord_ref_sys : str, optional
            Default coordinates reference system
        ctype : str, optional
            Default coordinates type. One of 'point', 'midpoint', 'left', 'right'.
        distance_units : Units
            Default distance units.

        Returns
        -------
        :class:`Coordinates`
            podpac Coordinates
        """

        if not isinstance(xcoord, xarray.core.coordinates.DataArrayCoordinates):
            raise TypeError("Coordinates.from_xarray expects xarray DataArrayCoordinates, not '%s'" % type(xcoord))

        coords = []
        for dim in xcoord.dims:
            if isinstance(xcoord.indexes[dim], (pd.DatetimeIndex, pd.Float64Index, pd.Int64Index)):
                c = ArrayCoordinates1d.from_xarray(xcoord[dim])
            elif isinstance(xcoord.indexes[dim], pd.MultiIndex):
                c = StackedCoordinates.from_xarray(xcoord[dim])
            coords.append(c)

        return cls(coords, coord_ref_sys=coord_ref_sys, ctype=ctype, distance_units=distance_units)

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

        if not isinstance(d, list):
            raise TypeError("Could not parse coordinates definition of type '%s'" % type(d))

        coords = []
        for elem in d:
            if isinstance(elem, list):
                c = StackedCoordinates.from_definition(elem)
            elif 'start' in elem and 'stop' in elem and ('step' in elem or 'size' in elem):
                c = UniformCoordinates1d.from_definition(elem)
            elif 'values' in elem:
                c = ArrayCoordinates1d.from_definition(elem)
            else:
                raise ValueError("Could not parse coordinates definition item with keys %s" % elem.keys())

            coords.append(c)

        return cls(coords)

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

    def get(self, dim, default=None):
        """ dict-like get: get coordinates by dimension name with an optional """
        try:
            return self[dim]
        except KeyError:
            return default

    def __iter__(self):
        return iter(self._coords)

    def __getitem__(self, dim):
        if dim in self._coords:
            return self._coords[dim]

        # extracts individual coords from stacked coords
        for c in self._coords.values():
            if isinstance(c, StackedCoordinates) and dim in c.dims:
                return c[dim]

        raise KeyError("Dimension '%s' not found in Coordinates %s" % (dim, self.dims))

    def __setitem__(self, dim, c):

        # cast input coordinates
        #             if isinstance(coords[i], BaseCoordinates):
        if isinstance(c, BaseCoordinates):
            pass
        elif isinstance(c, Coordinates):
            c = c[dim]
        elif isinstance(c, (list, tuple, np.ndarray)):
            if '_' in dim:
                cs = [val if isinstance(val, Coordinates1d) else ArrayCoordinates1d(val) for val in c]
                c = StackedCoordinates(cs)
            else:
                c = ArrayCoordinates1d(c)
        else:
            raise TypeError("Invalid coords, expected list, array, " +
                            "or class implementing BaseCoordinates, not '%s'" % type(c))

        if c.name is None:
            c.name = dim

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

        # full check of underlying coordinates
        if self._coords != other._coords:
            return False

        return True

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
    def shape(self):
        """:tuple: Tuple of the number of coordinates in each dimension."""
        
        return tuple(c.size for c in self._coords.values())

    @property
    def ndim(self):
        """:int: Number of dimensions. """
        
        return len(self.dims)

    @property
    def size(self):
        """:int: Total number of coordinates."""

        if len(self.shape) == 0:
            return 0
        return np.prod(self.shape)

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

        return tuple(dim for c in self._coords.values() for dim in c.dims)

    @property
    def coords(self):
        """
        :xarray.core.coordinates.DataArrayCoordinates: xarray coords, a dictionary-like container of coordinate arrays.
        """

        x = xr.DataArray(np.empty(self.shape), coords=[c.coordinates for c in self._coords.values()], dims=self.dims)
        return x.coords

    @property
    def definition(self):
        """
        Serializable coordinates definition.

        The ``definition`` can be used to create new Coordinates::

            c = podpac.Coordinates(...)
            c2 = podpac.Coordinates.from_definition(c.definition)

        See Also
        --------
        from_definition, json
        """

        return [c.definition for c in self._coords.values()]

    @property
    def json(self):
        """:str: JSON-serialized coordinates definition.

        The ``json`` can be used to create new Coordinates::

            c = podapc.Coordinates(...)
            c2 = podpac.Coordinates.from_json(c.definition)

        The serialized definition is used to define coordinates in pipelines and to transport coordinates, e.g.
        over HTTP and in AWS lambda functions. It also provides a consistent hashable value.

        See Also
        --------
        from_json
        """

        return json.dumps(self.definition, cls=podpac.core.utils.JSONEncoder)

    @property
    def hash(self):
        """
        Coordinates hash.

        *Note: To be replaced with the __hash__ method.*
        """

        return hash_alg(self.json.encode('utf-8')).hexdigest()

    @property
    def coord_ref_sys(self):
        """
        :str: coordinate reference system.

        .. deprecated:: 1.0.0
              `coord_ref_sys` will be removed in podpac 1.0.0, it is replaced by
              `crs` for consistency
        """

        warnings.warn('`coord_ref_sys` has been replaced with `crs` and will be deprecated in future releases', DeprecationWarning)
        return self.crs
    
    @property
    def crs(self):
        """:str: coordinate reference system."""
        if not self._coords:
            return None
        
        # the coord_ref_sys is the same for all coords
        return list(self._coords.values())[0].coord_ref_sys

        # key = 'lat' if 'lat' in self.udims else 'lon'
        # if key == 'lon' and 'lon' not in self.udims:
        #     return None

        # return GDAL_CRS.get(self[key].coord_ref_sys, self[key].coord_ref_sys)

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

        return Coordinates([c for c in self._coords.values() if c.name not in dims])

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

        return Coordinates(cs)

    def intersect(self, other, outer=False, return_indices=False):
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
        if other.crs != self.crs:
            other = other.transform(self.crs)

        intersections = [c.intersect(other, outer=outer, return_indices=return_indices) for c in self.values()]
        if return_indices:
            coords = Coordinates([c for c, I in intersections])
            idx = [I for c, I in intersections]
            return coords, tuple(idx)
        else:
            return Coordinates(intersections)

    def unique(self):
        """
        Remove duplicate coordinate values from each dimension.

        Returns
        -------
        coords : Coordinates
            New Coordinates object with unique, sorted coordinate values in each dimension.
        """

        return Coordinates([c[np.unique(c.coordinates, return_index=True)[1]] for c in self.values()])

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

        return Coordinates([self[dim] for dim in self.udims])

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

        l = [[slice(i, i+n) for i in range(0, m, n)] for m, n in zip(self.shape, shape)]
        for slices in itertools.product(*l):
            coords = Coordinates([self._coords[dim][slc] for dim, slc in zip(self.dims, slices)])
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

        in_place = kwargs.get('in_place', False)

        if len(dims) == 0:
            dims = list(self._coords.keys())[::-1]

        if len(dims) != self.ndim:
            raise ValueError("Invalid transpose dimensions, input %s does not match dims %s" % (dims, self.dims))

        if in_place:
            self._coords = OrderedDict([(dim, self._coords[dim]) for dim in dims])
            return self

        else:
            return Coordinates([self._coords[dim] for dim in dims])

    def transform(self, crs=None, alt_units=None):
        """
        Transform coordinate dimensions (`lat`, `lon`, `alt`) into a different coordinate reference system.
        Uses PROJ4 syntax for coordinate reference systems and units.
        
        See `PROJ4 Documentation <https://proj4.org/usage/projections.html#cartographic-projection>`_ for
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
        
        Transform coordinates with altitude::
        
            # include alt units in proj4 string
            c = Coordinates([[0, 1, 2], [0, 1, 2], [1, 2, 3]], dims=['lat', 'lon', 'alt'])
            c.transform('+init=epsg:2193 +vunits=ft')
        
            >> Coordinates
                lat: ArrayCoordinates1d(lat): Bounds[594971.8894642257, 819117.0608407748], N[3], ctype['midpoint']
                lon: ArrayCoordinates1d(lon): Bounds[29772096.71234478, 29995929.885877542], N[3], ctype['midpoint']
                alt: ArrayCoordinates1d(alt): Bounds[3.280839895013123, 9.842519685039369], N[3], ctype['midpoint']


            # specify alt units seperately
            c.transform('EPSG:2193', alt_units='ft')
        
            >> Coordinates
                lat: ArrayCoordinates1d(lat): Bounds[594971.8894642257, 819117.0608407748], N[3], ctype['midpoint']
                lon: ArrayCoordinates1d(lon): Bounds[29772096.71234478, 29995929.885877542], N[3], ctype['midpoint']
                alt: ArrayCoordinates1d(alt): Bounds[3.280839895013123, 9.842519685039369], N[3], ctype['midpoint']

            # using alt_units will save the property `crs` as a proj4 string:
            ct = c.transform('EPSG:2193', alt_units='ft')
            ct.crs

            >> '+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +units=m +no_defs +type=crs +vunits=ft'
        
        Parameters
        ----------
        crs : str, optional
            PROJ4 compatible coordinate reference system string.
            Defaults to the current `crs`
        alt_units : str, optional
            Override the alt units defined in `crs` string.
            This is implemented to provide a shorthand for transforming alt units
            without specifying the whole proj4 string.
        
        Returns
        -------
        :class:`podpac.Coordinates`
            Transformed Coordinates
        
        Raises
        ------
        ValueError
            Coordinates must have both lat and lon dimensions if either is defined
        """

        t_coords = deepcopy(self)

        if crs is None:
            crs = self.crs

        from_crs = pyproj.CRS(self.crs)
        to_crs = pyproj.CRS(crs)

        # convert alt units into proj4 syntax
        if alt_units is not None:
            to_crs = pyproj.CRS('{} +vunits={}'.format(to_crs.to_proj4(), alt_units))

        # create proj4 transformer
        transformer = pyproj.Transformer.from_crs(from_crs, to_crs)

        # update crs on the individual coords - this must be done before assigning new values
        # note using `srs` here so it captures the user input (i.e. EPSG:4193)
        # if alt_units included, this will be a whole proj4 string
        for dim in t_coords.udims:
            t_coords[dim].coord_ref_sys = to_crs.srs

        # if lat or lon is present, coordinates MUST have both, even if stacked:
        if ('lat' in self.udims or 'lon' in self.udims):
            
            if (set(['lat', 'lon']) - set(self.udims)):
                raise ValueError('Coordinates must have both lat and lon dimensions to transform coordinate reference systems')

            (lat, lon) = transformer.transform(self.coords['lat'].values, self.coords['lon'].values)
            t_coords['lat'] = ArrayCoordinates1d(lat, coord_ref_sys=t_coords.crs)
            t_coords['lon'] = ArrayCoordinates1d(lon, coord_ref_sys=t_coords.crs)

        # by keeping these seperate, we can handle altitude dimensions that are a different length from lat/lon
        if 'alt' in self.udims:
  
            dummy = np.zeros(len(self.coords['alt'].values))  # must be same length as alt
            (lat, lon, alt) = transformer.transform(dummy, dummy, self.coords['alt'].values)
            t_coords['alt'] = ArrayCoordinates1d(alt, coord_ref_sys=t_coords.crs)

        return t_coords


    # ------------------------------------------------------------------------------------------------------------------
    # Operators/Magic Methods
    # ------------------------------------------------------------------------------------------------------------------

    def __repr__(self):
        # TODO JXM
        rep = str(self.__class__.__name__)
        for c in self._coords.values():
            if isinstance(c, Coordinates1d):
                rep += '\n\t%s: %s' % (c.name, c)
            elif isinstance(c, StackedCoordinates):
                for _c in c:
                    rep += '\n\t%s[%s]: %s' % (c.name, _c.name, _c)
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

    for coords in coords_list:
        if not isinstance(coords, Coordinates):
            raise TypeError("Cannot merge '%s' with Coordinates" % type(coords))

    coords = sum([list(coords.values()) for coords in coords_list], [])
    return Coordinates(coords)

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

    return Coordinates(list(d.values()), list(d.keys()))

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
