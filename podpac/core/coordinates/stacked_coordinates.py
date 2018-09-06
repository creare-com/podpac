
from __future__ import division, unicode_literals, print_function, absolute_import

import copy

import numpy as np
import pandas as pd
import traitlets as tl
from six import string_types

from podpac.core.coordinates.coordinates1d import BaseCoordinates1d, Coordinates1d

class StackedCoordinates(BaseCoordinates1d):
   
    # TODO dict vs tuple
    _coords = tl.Tuple(trait=tl.Instance(Coordinates1d))

    # TODO default coord_ref_sys, ctype, distance_units, time_units

    def __init__(self, coords, coord_ref_sys=None, ctype=None, distance_units=None):
        """
        Initialize a multidimensional coords object.

        Parameters
        ----------
        coords : list, dict, or Coordinates
            Coordinates, either
             * list of named BaseCoordinates1d objects
             * dictionary of BaseCoordinates1d objects, with dimension names as the keys
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

    @tl.validate('coords')
    def _validate_coords(self, d):
        val = d['val']
        if len(val) < 2:
            raise ValueError('stacked coords must have at least 2 coords, got %d' % len(val))

        names = []
        for i, c in enumerate(val):
            if c.name is None:
                raise ValueError("missing dimension name in coords list at position %d" % i)
            if c.name in names:
                raise ValueError("duplicate dimension name '%s' in stacked coords at position %d" % (c.name, i))
            if c.size != val[0].size:
                raise ValueError("mismatch size in stacked coords %d != %d at position %d" % (c.size, val[0].size, i))
            names.append(c.name)

        return val
    
    # ------------------------------------------------------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_xarray(cls, xcoord, coord_ref_sys=None, ctype=None, distance_units=None):
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

        dim = xcoord.dims[0]
        return cls([from_xarray_1d(xcoord[name]) for name in xcoord.indexes[dim].names], **kwargs)
    
    # ------------------------------------------------------------------------------------------------------------------
    # standard (tuple-like) methods
    # ------------------------------------------------------------------------------------------------------------------

    def __getitem__(self, index):
        return StackedCoordinates([c[index] for c in self._coords])

    def __repr__(self):
        # TODO
        raise NotImplementedError

    def __iter__(self):
        return iter(self._coords)

    def __len__(self):
        return len(self._coords)

    # TODO [] vs get/isel?
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
        return '_'.join(self.dims)

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