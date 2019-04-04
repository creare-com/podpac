"""
Stacked Coordinates

Multiple coordinates aligned for a single data point
"""
from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict

import numpy as np
import xarray as xr
import pandas as pd
import traitlets as tl
from six import string_types

from podpac.core.coordinates.base_coordinates import BaseCoordinates
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.utils import OrderedDictTrait

class StackedCoordinates(BaseCoordinates):
    """
    Stacked coordinates.

    StackedCoordinates contain coordinates from two or more different dimensions that are stacked together to form a
    list of points (rather than a grid). The underlying coordinates values are :class:`Coordinates1d` objects of equal
    size. The name for the stacked coordinates combines the underlying dimensions with underscores, e.g. ``'lat_lon'``
    or ``'lat_lon_time'``.

    When creating :class:`Coordinates`, podpac automatically detects StackedCoordinates. The following Coordinates
    contain 3 stacked lat-lon coordinates and 2 time coordinates in a 3 x 2 grid::

        >>> lat = [0, 1, 2]
        >>> lon = [10, 20, 30]
        >>> time = ['2018-01-01', '2018-01-02']
        >>> podpac.Coordinates([[lat, lon], time], dims=['lat_lon', 'time'])
        Coordinates
            lat_lon[lat]: ArrayCoordinates1d(lat): Bounds[0.0, 2.0], N[3], ctype['midpoint']
            lat_lon[lon]: ArrayCoordinates1d(lon): Bounds[10.0, 30.0], N[3], ctype['midpoint']
            time: ArrayCoordinates1d(time): Bounds[2018-01-01, 2018-01-02], N[2], ctype['midpoint']

    For convenience, you can also create uniformly-spaced stacked coordinates using :class:`clinspace`::

        >>> lat_lon = podpac.clinspace((0, 10), (2, 30), 3)
        >>> time = ['2018-01-01', '2018-01-02']
        >>> podpac.Coordinates([lat_lon, time], dims=['lat_lon', 'time'])
        Coordinates
            lat_lon[lat]: ArrayCoordinates1d(lat): Bounds[0.0, 2.0], N[3], ctype['midpoint']
            lat_lon[lon]: ArrayCoordinates1d(lon): Bounds[10.0, 30.0], N[3], ctype['midpoint']
            time: ArrayCoordinates1d(time): Bounds[2018-01-01, 2018-01-02], N[2], ctype['midpoint']

    Parameters
    ----------
    dims : tuple
        Tuple of dimension names.
    name : str
        Stacked dimension name.
    coords : dict-like
        xarray coordinates (container of coordinate arrays)
    coordinates : pandas.MultiIndex
        MultiIndex of stacked coordinates values.
        
    """

    _coords = OrderedDictTrait(trait=tl.Instance(Coordinates1d), default_value=OrderedDict())

    def __init__(self, coords, coord_ref_sys=None, ctype=None, distance_units=None):
        """
        Initialize a multidimensional coords object.

        Parameters
        ----------
        coords : list, :class:`StackedCoordinates`
            Coordinate values in a list, or a StackedCoordinates object to copy.
        coord_ref_sys : str, optional
            Default coordinates reference system.
        ctype : str, optional
            Default coordinates type.
        distance_units : Units, optional
            Default distance units.

        See Also
        --------
        clinspace, crange
        """

        if isinstance(coords, StackedCoordinates):
            coords = [c.copy() for c in coords]
        elif not isinstance(coords, (list, tuple)):
            raise TypeError("Unrecognized coords type '%s'" % type(coords))

        if len(coords) < 2:
            raise ValueError('Stacked coords must have at least 2 coords, got %d' % len(coords))

        # input coords must be Coordinates1d type
        for i, c in enumerate(coords):
            if not isinstance(c, Coordinates1d):
                raise TypeError("Invalid coordinates of type '%s' in stacked coords at position %d" % (type(c), i))

        # stacked coordinates must have the same sizes, names, and coordinate reference systems
        self._check_sizes([c.size for c in coords])
        self._check_names([c.name for c in coords])
        self._check_coord_ref_sys(coords, coord_ref_sys)

        # validation is complete, set properties
        self._set_properties(coords, ctype, distance_units, coord_ref_sys)

        # create ordered dict from stacked coordinates
        dcoords = OrderedDict()
        for c in coords:
            dcoords[c.name] = c

        self.set_trait('_coords', dcoords)

        super(StackedCoordinates, self).__init__()

    def _check_names(self, names):
        for i, name in enumerate(names):
            if name is None:
                raise ValueError("Dim for stacked coordinates at position %d is not defined" % (i))

            if name in names[:i]:
                raise ValueError("Duplicate dimension name '%s' in stacked coords at position %d" % (name, i))

    def _check_sizes(self, sizes):
        for i, size in enumerate(sizes):
            if size != sizes[0]:
                raise ValueError("Size mismatch in stacked coords %d != %d at position %d" % (size, sizes[0], i))

    def _check_coord_ref_sys(self, coords, crs=None):
        # the coord_ref_sys should be the same, and should match the input coord_ref_sys if defined
        if crs is None:
            crs = coords[0].coord_ref_sys

        for i, c in enumerate(coords):
            if 'coord_ref_sys' in c.properties and c.coord_ref_sys != crs:
                raise ValueError("coord_ref_sys mismatch in stacked coords %s != %s at position %d" % (
                    c.coord_ref_sys, crs, i))

    def _set_properties(self, coords, ctype, distance_units, coord_ref_sys):
        for c in coords:
            # NOTE this is nearly identical to the Coordinates._set_properties private method
            # set or check the coord_ref_sys
            if coord_ref_sys is not None:
                if 'coord_ref_sys' not in c.properties:
                    c.set_trait('coord_ref_sys', coord_ref_sys)
                elif coord_ref_sys != c.coord_ref_sys:
                    raise ValueError("coord_ref_sys mismatch %s != %s at pos %d" % (coord_ref_sys, c.coord_ref_sys, pos))

            # only set name, ctype, and units if they aren't already set
            if ctype is not None and 'ctype' not in c.properties:
                c.set_trait('ctype', ctype)
            if distance_units is not None and c.name in ['lat', 'lon', 'alt'] and 'units' not in c.properties:
                c.set_trait('units', distance_units)

    # ------------------------------------------------------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_xarray(cls, xcoords, coord_ref_sys=None, ctype=None, distance_units=None):
        """
        Convert an xarray coord to StackedCoordinates

        Parameters
        ----------
        xcoords : DataArrayCoordinates
            xarray coords attribute to convert
        coord_ref_sys : str, optional
            Default coordinates reference system.
        ctype : str, optional
            Default coordinates type.
        distance_units : Units, optional
            Default distance units.

        Returns
        -------
        coord : :class:`StackedCoordinates`
            stacked coordinates object
        """

        dims = xcoords.indexes[xcoords.dims[0]].names
        coords = [ArrayCoordinates1d.from_xarray(xcoords[dims]) for dims in dims]
        return cls(coords, coord_ref_sys=coord_ref_sys, ctype=ctype, distance_units=distance_units)

    @classmethod
    def from_definition(cls, d):
        """
        Create StackedCoordinates from a stacked coordinates definition.

        Arguments
        ---------
        d : list
            stacked coordinates definition

        Returns
        -------
        :class:`StackedCoordinates`
            stacked coordinates object

        See Also
        --------
        definition
        """

        coords = []
        for elem in d:
            if 'start' in elem and 'stop' in elem and ('step' in elem or 'size' in elem):
                c = UniformCoordinates1d.from_definition(elem)
            elif 'values' in elem:
                c = ArrayCoordinates1d.from_definition(elem)
            else:
                raise ValueError("Could not parse coordinates definition with keys %s" % elem.keys())

            coords.append(c)

        return cls(coords)

    def copy(self):
        """
        Make a copy of the stacked coordinates.

        Returns
        -------
        :class:`StackedCoordinates`
            Copy of the stacked coordinates, with provided properties and name.
        """

        return StackedCoordinates(list(self._coords.values()))

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

    def update(self, other):
        """ dict-like update: add/replace coordinates using another StackedCoordinates object """
        if not isinstance(other, StackedCoordinates):
            raise TypeError("Cannot update StackedCoordinates with object of type '%s'" % type(other))

        d = self._coords.copy()
        d.update(other._coords)
        self._coords = d

    def __repr__(self):
        rep = str(self.__class__.__name__)
        for c in self._coords.values():
            rep += '\n\t%s[%s]: %s' % (self.name, c.name or '?', c)
        return rep

    def __iter__(self):
        return iter(self._coords)

    def __len__(self):
        return len(self._coords)

    def __getitem__(self, dim):
        if dim in self._coords:
            return self._coords[dim]

        raise KeyError("Dimension '%s' not found in StackedCoordinates %s" % (dim, self.dims))

    def __setitem__(self, dim, c):
        if not dim in self.dims:
            raise KeyError("Cannot set dimension '%s' in StackedCoordinates %s" % (dim, self.dims))

        # try to cast to ArrayCoordinates1d
        if not isinstance(c, Coordinates1d):
            c = ArrayCoordinates1d(c)

        if c.name is None:
            c.name = dim

        d = self._coords.copy()
        d[dim] = c

        # check consistency
        self._check_sizes([c.size for c in d])
        self._check_names([c.name for c in d])
        self._check_coord_ref_sys(d)

        self._coords = d


    def __eq__(self, other):
        if not isinstance(other, StackedCoordinates):
            return False

        # shortcuts
        if self.dims != other.dims:
            return False

        if self.size != other.size:
            return False

        # full check of underlying coordinates
        if self._coords != other._coords:
            return False

        return True

    def __delitem__(self, dim):
        if not dim in self.dims:
            raise KeyError("Cannot delete dimension '%s' in StackedCoordinates %s" % (dim, self.dims))

        del self._coords[dim]

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def dims(self):
        """:tuple: Tuple of dimension names."""
        return tuple(c.name for c in self._coords.values())

    @property
    def udims(self):
        return self.dims

    @property
    def name(self):
        """:str: Stacked dimension name.

        Stacked dimension names are the individual `dims` joined by an underscore.
        """
        if any(self.dims):
            return '_'.join(dim or '?' for dim in self.dims)

    @name.setter
    def name(self, value):
        names = value.split('_')
        if len(names) != len(self._coords):
            raise ValueError("Invalid name '%s' for StackedCoordinates with length %d" % (value, len(self._coords)))
        
        self._check_names(names)

        for c, name in zip(self._coords.values(), names):
            c.name = name

    @property
    def size(self):
        """ Number of stacked coordinates. """
        return list(self._coords.values())[0].size

    @property
    def coordinates(self):
        """:pandas.MultiIndex: MultiIndex of stacked coordinates values."""

        return pd.MultiIndex.from_arrays([np.array(c.coordinates) for c in self._coords.values()], names=self.dims)

    @property
    def coords(self):
        """:dict-like: xarray coordinates (container of coordinate arrays)"""

        x = xr.DataArray(np.empty(self.size), coords=[self.coordinates], dims=self.name)
        return x[self.name].coords

    @property
    def coord_ref_sys(self):
        """:str: coordinate reference system."""

        # the coord_ref_sys is the same for all coords
        return list(self._coords.values())[0].coord_ref_sys

    @property
    def definition(self):
        """:list: Serializable stacked coordinates definition.

        The ``definition`` can be used to create new StackedCoordinates::

            c = podpac.StackedCoordinates(...)
            c2 = podpac.StackedCoordinates.from_definition(c.definition)

        See Also
        --------
        from_definition
        """
        return [c.definition for c in self._coords.values()]

    # -----------------------------------------------------------------------------------------------------------------
    # Methods
    # -----------------------------------------------------------------------------------------------------------------

    def intersect(self, other, outer=False, return_indices=False):
        """
        Get the stacked coordinate values that are within the bounds of a given coordinates object in all dimensions.

        *Note: you should not generally need to call this method directly.*
        
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
        intersection : :class:`StackedCoordinates`
            StackedCoordinates object consisting of the intersection in all dimensions.
        idx : slice or list
            Slice or index for the intersected coordinates, only if ``return_indices`` is True.
        """

        # intersections in each dimension
        Is = [c.intersect(other, outer=outer, return_indices=True)[1] for c in self._coords.values()]

        # logical AND of the intersections
        I = Is[0]
        for J in Is[1:]:
            if isinstance(I, slice) and isinstance(J, slice):
                I = slice(max(I.start or 0, J.start or 0), min(I.stop or self.size, J.stop or self.size))
            else:
                if isinstance(I, slice):
                    I = np.arange(self.size)[I]
                if isinstance(J, slice):
                    J = np.arange(self.size)[I]
                I = [i for i in I if i in J]

        # for consistency
        if isinstance(I, slice) and I.start == 0 and I.stop == self.size:
            I = slice(None, None)

        if return_indices:
            return self[I], I
        else:
            return self[I]
