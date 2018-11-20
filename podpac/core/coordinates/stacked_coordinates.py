
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

    See Also
    --------
    :class:`clinspace`
    """

    # TODO dict vs tuple?
    _coords = tl.Tuple(trait=tl.Instance(Coordinates1d))

    # TODO default coord_ref_sys, ctype, distance_units, time_units

    def __init__(self, coords, coord_ref_sys=None, ctype=None, distance_units=None):
        """
        Initialize a multidimensional coords object.

        Parameters
        ----------
        coords : list, dict, or Coordinates
            Coordinates, either
             * list of named BaseCoordinates objects
             * dictionary of BaseCoordinates objects, with dimension names as the keys
             * Coordinates object to be copied
        ctype : str
            Default coordinates type (optional).
        coord_ref_sys : str
            Default coordinates reference system (optional)
        """


        if isinstance(coords, StackedCoordinates):
            coords = copy.deepcopy(coords._coords)

        elif not isinstance(coords, (list, tuple)):
            raise TypeError("Unrecognized coords type '%s'" % type(coords))

        # set 1d coordinates defaults
        # TODO JXM factor out, etc, maybe move to observe so that it gets validated first
        for c in coords:
            if 'ctype' not in c._trait_values and ctype is not None:
                c.ctype = ctype
            if 'coord_ref_sys' not in c._trait_values and coord_ref_sys is not None:
                c.coord_ref_sys = coord_ref_sys
            if 'units' not in c._trait_values and distance_units is not None and c.name in ['lat', 'lon', 'alt']:
                c.units = distance_units

        super(StackedCoordinates, self).__init__(_coords=coords)

    @tl.validate('_coords')
    def _validate_coords(self, d):
        val = d['value']
        if len(val) < 2:
            raise ValueError('stacked coords must have at least 2 coords, got %d' % len(val))

        names = []
        for i, c in enumerate(val):
            if c.size != val[0].size:
                raise ValueError("mismatch size in stacked coords %d != %d at position %d" % (c.size, val[0].size, i))

            if c.name is not None:
                if c.name in names:
                    raise ValueError("duplicate dimension name '%s' in stacked coords at position %d" % (c.name, i))
                names.append(c.name)

        return val

    # ------------------------------------------------------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_xarray(cls, xcoord, coord_ref_sys=None, ctype=None, distance_units=None, **kwargs):
        """
        Convert an xarray coord to Stacked

        Parameters
        ----------
        xcoord : DataArrayCoordinates
            xarray coord attribute to convert

        Returns
        -------
        coord : Coordinates
            podpact Coordinates object

        Raises
        ------
        TypeError
            Description
        """

        dims = xcoord.indexes[xcoord.dims[0]].names
        return cls([ArrayCoordinates1d.from_xarray(xcoord[dims]) for dims in dims], **kwargs)

    @classmethod
    def from_definition(cls, d):
        coords = []
        for elem in d:
            if 'start' in elem and 'stop' in elem and 'step' in elem:
                c = UniformCoordinates1d.from_json(elem)
            elif 'values' in elem:
                c = ArrayCoordinates1d.from_json(elem)
            else:
                raise ValueError("Could not parse coordinates definition with keys %s" % elem.keys())

            coords.append(c)

        return cls(coords)

    def copy(self, name=None, **kwargs):
        c = StackedCoordinates([c.copy() for c in self._coords], **kwargs)
        if name is not None:
            c.name = name
        return c

    # ------------------------------------------------------------------------------------------------------------------
    # standard methods, tuple-like
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
                raise KeyError("dim '%s' not found todo better message" % index)
            return self._coords[self.dims.index(index)]

        else:
            return StackedCoordinates([c[index] for c in self._coords])

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def dims(self):
        return tuple(c.name for c in self._coords)

    @property
    def name(self):
        return '_'.join(dim or '?' for dim in self.dims)

    @name.setter
    def name(self, value):
        names = value.split('_')
        if len(names) != len(self._coords):
            raise ValueError("Invalid name '%s' for StackedCoordinates with length %d" % (value, len(self._coords)))
        for c, name in zip(self._coords, names):
            c.name = name

    @property
    def size(self):
        return self._coords[0].size

    @property
    def coordinates(self):
        # TODO don't recompute this every time (but also don't compute it until requested)
        return pd.MultiIndex.from_arrays([np.array(c.coordinates) for c in self._coords], names=self.dims)

    @property
    def coords(self):
        # TODO don't recompute this every time (but also don't compute it until requested)
        x = xr.DataArray(np.empty(self.size), coords=[self.coordinates], dims=self.name)
        return x[self.name].coords

    @property
    def definition(self):
        return [c.definition for c in self._coords]

    # -----------------------------------------------------------------------------------------------------------------
    # Methods
    # -----------------------------------------------------------------------------------------------------------------

    def intersect(self, other, outer=False, return_indices=False):
        Is = [c.intersect(other, outer=outer, return_indices=True)[1] for c in self._coords]

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
