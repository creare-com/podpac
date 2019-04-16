
from __future__ import division, unicode_literals, print_function, absolute_import

import copy

import numpy as np
import xarray as xr
import pandas as pd
import traitlets as tl
from six import string_types

from podpac.core.coordinates.base_coordinates import BaseCoordinates
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d

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

    _coords = tl.List(trait=tl.Instance(Coordinates1d), read_only=True)

    def __init__(self, coords, name=None, dims=None, crs=None, ctype=None, distance_units=None):
        """
        Initialize a multidimensional coords object.

        Parameters
        ----------
        coords : list, :class:`StackedCoordinates`
            Coordinate values in a list, or a StackedCoordinates object to copy.
        crs : str, optional
            Default coordinates reference system.
        ctype : str, optional
            Default coordinates type.
        distance_units : Units, optional
            Default distance units.

        See Also
        --------
        clinspace, crange
        """

        if not isinstance(coords, (list, tuple)):
            raise TypeError("Unrecognized coords type '%s'" % type(coords))

        if len(coords) < 2:
            raise ValueError('Stacked coords must have at least 2 coords, got %d' % len(coords))

        # coerce
        coords = tuple(c if isinstance(c, Coordinates1d) else ArrayCoordinates1d(c) for c in coords)
        
        # set coords
        self.set_trait('_coords', coords)

        # propagate properties
        if dims is not None and name is not None:
            raise TypeError("StackedCoordinates expected 'dims' or 'name', not both")
        if dims is not None:
            self._set_dims(dims)
        if name is not None:
            self._set_name(name)
        if crs is not None:
            self._set_crs(crs)
        if ctype is not None:
            self._set_ctype(ctype)
        if distance_units is not None:
            self._set_distance_units(distance_units)

        # finalize
        super(StackedCoordinates, self).__init__()

    @tl.validate('_coords')
    def _validate_coords(self, d):
        val = d['value']

        # check sizes
        size = val[0].size
        for c in val[1:]:
            if c.size != size:
                raise ValueError("Size mismatch in stacked coords %d != %d" % (c.size, size))
        
        # check dims
        dims = [c.name for c in val]
        for i, dim in enumerate(dims):
            if dim is not None and dim in dims[:i]:
                raise ValueError("Duplicate dimension '%s' in stacked coords" % dim)

        # check crs
        for c in val:
            if c.crs != val[0].crs:
                raise ValueError("crs mismatch in stacked_coords %s != %s" % (c.crs, val[0].crs))

        return val

    def _set_name(self, value):
        dims = value.split('_')
        
        # check size
        if len(dims) != len(self._coords):
            raise ValueError("Invalid name '%s' for StackedCoordinates with length %d" % (value, len(self._coords)))
        
        self._set_dims(dims)

    def _set_dims(self, dims):
        # check size
        if len(dims) != len(self._coords):
            raise ValueError("Invalid dims '%s' for StackedCoordinates with length %d" % (dims, len(self._coords)))
        
        # set names, checking for duplicates
        for i, (c, dim) in enumerate(zip(self._coords, dims)):
            if dim is None:
                continue
            c._set_name(dim)

    def _set_crs(self, value):
        for c in self._coords:
            c._set_crs(value)

    def _set_ctype(self, value):
        for c in self._coords:
            c._set_ctype(value)

    def _set_distance_units(self, value):
        for c in self._coords:
            c._set_distance_units(value)

    # ------------------------------------------------------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_xarray(cls, xcoords, crs=None, ctype=None, distance_units=None):
        """
        Convert an xarray coord to StackedCoordinates

        Parameters
        ----------
        xcoords : DataArrayCoordinates
            xarray coords attribute to convert
        crs : str, optional
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
        return cls(coords, crs=crs, ctype=ctype, distance_units=distance_units)

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

    # ------------------------------------------------------------------------------------------------------------------
    # standard methods, list-like
    # ------------------------------------------------------------------------------------------------------------------

    def __repr__(self):
        rep = str(self.__class__.__name__)
        for c in self._coords:
            rep += '\n\t%s[%s]: %s' % (self.name, c.name or '?', c)
        return rep

    def __iter__(self):
        return iter(self._coords)

    def __len__(self):
        return len(self._coords)

    def __getitem__(self, index):
        if isinstance(index, string_types):
            if index not in self.dims:
                raise KeyError("Dimension '%s' not found in dims %s" % (index, self.dims))
            
            return self._coords[self.dims.index(index)]

        else:
            return StackedCoordinates([c[index] for c in self._coords])

    def __setitem__(self, dim, c):
        if not dim in self.dims:
            raise KeyError("Cannot set dimension '%s' in StackedCoordinates %s" % (dim, self.dims))

        # try to cast to ArrayCoordinates1d
        if not isinstance(c, Coordinates1d):
            c = ArrayCoordinates1d(c)

        if c.name is None:
            c.name = dim

        # replace the element of the coords list
        idx = self.dims.index(dim)
        coords = list(self._coords)
        coords[idx] = c

        # set (and check) new coords list
        self.set_trait('_coords', coords)

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

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def dims(self):
        """:tuple: Tuple of dimension names."""
        return tuple(c.name for c in self._coords)

    @property
    def udims(self):
        """:tuple: Tuple of unstacked dimension names, for compatibility. This is the same as the dims."""
        return self.dims

    @property
    def idims(self):
        """:tuple: Tuple of indexing dimensions.

        For stacked coordinates, this is a singleton of the stacked coordinates name ``(self.name,)``.
        """

        return (self.name,)

    @property
    def name(self):
        """:str: Stacked dimension name. Stacked dimension names are the individual `dims` joined by an underscore."""

        if any(self.dims):
            return '_'.join(dim or '?' for dim in self.dims)

    @property
    def size(self):
        """:int: Number of stacked coordinates. """
        return self._coords[0].size

    @property
    def shape(self):
        """:tuple: Shape of the stacked coordinates."""
        return (self.size,)

    @property
    def bounds(self):
        """:dict: Dictionary of (low, high) coordinates bounds in each dimension"""
        if None in self.dims:
            raise ValueError("Cannot get bounds for StackedCoordinates with un-named dimensions")
        return {dim: self[dim].bounds for dim in self.udims}

    @property
    def area_bounds(self):
        """:dict: Dictionary of (low, high) coordinates area_bounds in each dimension"""
        if None in self.dims:
            raise ValueError("Cannot get area_bounds for StackedCoordinates with un-named dimensions")
        return {dim: self[dim].area_bounds for dim in self.udims}

    @property
    def coordinates(self):
        """:pandas.MultiIndex: MultiIndex of stacked coordinates values."""

        return pd.MultiIndex.from_arrays([np.array(c.coordinates) for c in self._coords], names=self.dims)

    @property
    def values(self):
        """:pandas.MultiIndex: MultiIndex of stacked coordinates values."""

        return self.coordinates

    @property
    def coords(self):
        """:dict-like: xarray coordinates (container of coordinate arrays)"""
        if None in self.dims:
            raise ValueError("Cannot get coords for StackedCoordinates with un-named dimensions")
        return {self.name: self.coordinates}

    @property
    def crs(self):
        """:str: coordinate reference system."""

        # the crs is the same for all coords
        return self._coords[0].crs

    @property
    def definition(self):
        """:list: Serializable stacked coordinates definition. """

        return [c.definition for c in self._coords]

    @property
    def full_definition(self):
        """:list: Serializable stacked coordinates definition, containing all properties. For internal use."""
        
        return [c.full_definition for c in self._coords]

    # -----------------------------------------------------------------------------------------------------------------
    # Methods
    # -----------------------------------------------------------------------------------------------------------------
            
    def copy(self):
        """
        Make a copy of the stacked coordinates.

        Returns
        -------
        :class:`StackedCoordinates`
            Copy of the stacked coordinates.
        """

        return StackedCoordinates(self._coords)

    def intersect(self, other, outer=False, return_indices=False):
        """
        Get the stacked coordinate values that are within the bounds of a given coordinates object in all dimensions.

        *Note: you should not generally need to call this method directly.*
        
        Parameters
        ----------
        other : :class:`BaseCoordinates1d`, :class:`Coordinates`
            Coordinates to intersect with.
        outer : bool, optional
            If True, do an *outer* intersection. Default False.
        return_indices : bool, optional
            If True, return slice or indices for the selection in addition to coordinates. Default False.

        Returns
        -------
        intersection : :class:`StackedCoordinates`
            StackedCoordinates object consisting of the intersection in all dimensions.
        I : slice or list
            Slice or index for the intersected coordinates, only if ``return_indices`` is True.
        """

        # logical AND of the intersection in each dimension
        indices = [c.intersect(other, outer=outer, return_indices=True)[1] for c in self._coords]
        I = self._and_indices(indices)

        if return_indices:
            return self[I], I
        else:
            return self[I]

    def select(self, bounds, return_indices=False, outer=False):
        """
        Get the coordinate values that are within the given bounds in all dimensions.

        *Note: you should not generally need to call this method directly.*

        Parameters
        ----------
        bounds : dict
            dictionary of dim -> (low, high) selection bounds
        outer : bool, optional
            If True, do *outer* selections. Default False.
        return_indices : bool, optional
            If True, return slice or indices for the selections in addition to coordinates. Default False.

        Returns
        -------
        selection : :class:`StackedCoordinates`
            StackedCoordinates object consisting of the selection in all dimensions.
        I : slice or list
            Slice or index for the selected coordinates, only if ``return_indices`` is True.
        """

        # logical AND of the selection in each dimension
        indices = [c.select(bounds, outer=outer, return_indices=True)[1] for c in self._coords]
        I = self._and_indices(indices)

        if return_indices:
            return self[I], I
        else:
            return self[I]

    def _and_indices(self, indices):
        # logical AND of the selected indices
        I = indices[0]
        for J in indices[1:]:
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

        return I
