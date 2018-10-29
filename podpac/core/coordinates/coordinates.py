"""
Multidimensional Coordinates
"""


from __future__ import division, unicode_literals, print_function, absolute_import

import copy
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

from podpac.core.coordinates.utils import GDAL_CRS
from podpac.core.coordinates.base_coordinates import BaseCoordinates
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates

class OrderedDictTrait(tl.Dict):
    """ OrderedDict trait for Python < 3.6 (including Python 2) compatibility """
    
    default_value = OrderedDict()
    def validate(self, obj, value):
        if not isinstance(value, OrderedDict):
            raise tl.TraitError('...')
        super(OrderedDictTrait, self).validate(obj, value)
        return value

class Coordinates(tl.HasTraits):
    """
    Multidimensional Coordinates.
    
    Attributes
    ----------
    coords
    dims
    ndim
    shape
    coordinates
    """

    if sys.version < '3.6':
        _coords = OrderedDictTrait(trait=tl.Instance(BaseCoordinates))
    else:
        _coords = tl.Dict(trait=tl.Instance(BaseCoordinates))

    def __init__(self, coords=[], dims=None, coord_ref_sys=None, ctype=None, distance_units=None):
        """
        Initialize a multidimensional coords object.

        Parameters
        ----------
        coords : list, dict, or Coordinates
            List of named BaseCoordinates objects
        ctype : str
            Default coordinates type (optional).
        coord_ref_sys : str
            Default coordinates reference system (optional)
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

        dcoords = OrderedDict()
        for i, dim in enumerate(dims):
            if dim in dcoords:
                raise ValueError("duplicate dimension name '%s' at position %d" % (dim, i))

            # TODO default properties
            if isinstance(coords[i], BaseCoordinates):
                c = coords[i].copy()
            elif '_' in dim:
                c = StackedCoordinates([ArrayCoordinates1d(values) for values in coords[i]])
            else:
                c = ArrayCoordinates1d(coords[i])

            c.name = dim
            dcoords[dim] = c

        # set 1d coordinates defaults
        # TODO factor out, store as default_* traits, and pass on through StackedCoordinates as well
        # maybe move to observe so that it gets validated first
        # for c in coords.values():
        #     if 'ctype' not in c._trait_values and ctype is not None:
        #         c.ctype = ctype
        #     if 'coord_ref_sys' not in c._trait_values and coord_ref_sys is not None:
        #         c.coord_ref_sys = coord_ref_sys
        #     if 'units' not in c._trait_values and distance_units is not None and c.name in ['lat', 'lon', 'alt']:
        #         c.units = distance_units
        
        super(Coordinates, self).__init__(_coords=dcoords)

    @tl.validate('_coords')
    def _validate_coords(self, d):
        val = d['value']
        for dim, c in val.items():
            if dim != c.name:
                raise ValueError("dimension name mismatch, '%s' != '%s'" % (dim, c.name))

        dims = [dim for c in self._coords.values() for dim in c.dims]
        for dim in dims:
            if dims.count(dim) != 1:
                raise ValueError("duplicate dimension '%s' in dims %s" % (dim, val.keys()))

        return val

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
    def grid(cls, coord_ref_sys=None, ctype=None, distance_units=None, dims=None, **kwargs):
        coords = cls._coords_from_dict(kwargs, dims)
        return cls(coords, coord_ref_sys=coord_ref_sys, ctype=ctype, distance_units=distance_units)

    @classmethod
    def points(cls, coord_ref_sys=None, ctype=None, distance_units=None, dims=None, **kwargs):
        coords = cls._coords_from_dict(kwargs, dims)
        stacked = StackedCoordinates(coords)
        return cls([stacked], coord_ref_sys=coord_ref_sys, ctype=ctype, distance_units=distance_units)

    @classmethod
    def from_xarray(cls, xcoord, coord_ref_sys=None, ctype=None, distance_units=None):
        """
        Convert an xarray coord to podpac Coordinates.
        
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

        if not isinstance(xcoord, xarray.core.coordinates.DataArrayCoordinates):
            raise TypeError("input must be an xarray DataArrayCoordinate, not '%s'" % type(xcoord))

        coords = []
        for dim in xcoord.dims:
            if isinstance(xcoord.indexes[dim], (pd.DatetimeIndex, pd.Float64Index, pd.Int64Index)):
                c = ArrayCoordinates1d.from_xarray(xcoord[dim])
            elif isinstance(xcoord.indexes[dim], pd.MultiIndex):
                c = StackedCoordinates.from_xarray(xcoord[dim])
            coords.append(c)

        return cls(coords, coord_ref_sys=coord_ref_sys, ctype=ctype, distance_units=distance_units)

    @classmethod
    def from_json(cls, d):
        if isinstance(d, str):
            d = json.loads(d)

        coords = []
        for elem in d:
            if isinstance(elem, list):
                c = StackedCoordinates.from_json(elem)
            elif 'start' in elem and 'stop' in elem and 'step' in elem:
                c = UniformCoordinates1d.from_json(elem)
            elif 'values' in elem:
                c = ArrayCoordinates1d.from_json(elem)
            else:
                raise ValueError("Could not parse coordinates definition with keys %s" % elem.keys())
            
            coords.append(c)

        return cls(coords)
    
    # ------------------------------------------------------------------------------------------------------------------
    # standard dict-like methods
    # ------------------------------------------------------------------------------------------------------------------

    def values(self):
        return self._coords.values()

    def keys(self):
        return self._coords.keys()

    def items(self):
        return self._coords.items()

    def get(self, dim, default=None):
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

        raise KeyError("dimension '%s' not found in Coordinates %s" % (dim, self.dims))

    def __setitem__(self, dim, c):
        if not dim in self.dims:
            raise KeyError("cannot set dimension '%s' in Coordinates %s" % (dim, self.dims))
            
        # TODO allow setting an array (cast it to ArrayCoordinates1d)
        if not isinstance(c, BaseCoordinates):
            raise TypeError("todo")


        if c.name is None:
            c.name = dim
        elif c.name != dim:
            raise ValueError("dimension name mismatch, '%s' != '%s'" % (dim, c.name))
        
        # TODO ctype, etc defaults
        
        self._coords[dim] = c
        
        # TODO we could also support setting individal coords in stacked coords
        # if dim in self.udims:
        #     for _c in self._coords.values():
        #         if isinstance(c, StackedCoordinates) and dim in c.dims:
        #             _c[dim] = c # this will validate the size
        
        # TODO we could also support adding new coords (need to check for duplicate dimensions)
        # else:
        #     self._coords[dim] = c

    def __delitem__(self, dim):
        if not dim in self.dims:
            raise KeyError("cannot delete dimension '%s' in Coordinates %s" % (dim, self.dims))

        del self._coords[dim]

        # TODO we could also support deleting individal coords within stacked coords

    def __contains__(self, dim):
        raise NotImplementedError("not sure if this should check dims or udims")

    def __len__(self):
        return len(self._coords)

    def update(self, other):
        if not isinstance(other, Coordinates):
            raise TypeError("Cannot update '%s' with Coordinates" % type(other))

        d = c._coords
        d.update(other._coords)
        self._coords = d

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def dims(self):
        return tuple(c.name for c in self._coords.values())

    @property
    def shape(self):
        return tuple(c.size for c in self._coords.values())

    @property
    def ndim(self):
        return len(self.dims)

    @property
    def size(self):
        if len(self.shape) == 0:
            return 0
        return np.prod(self.shape)

    @property
    def udims(self):
        return tuple(dim for c in self._coords.values() for dim in c.dims)

    @property
    def coords(self):
        # TODO don't recompute this every time (but also don't compute it until requested)
        x = xr.DataArray(np.empty(self.shape), coords=[c.coordinates for c in self._coords.values()], dims=self.dims)
        return x.coords

    @property
    def latlon_bounds_str(self):
        if 'lat' in self.udims and 'lon' in self.udims:
            # Where is this really used? Shouldn't this be area_bounds?
            return '%s_%s_x_%s_%s' % (
                self['lat'].bounds[0],
                self['lon'].bounds[0],
                self['lat'].bounds[1],
                self['lon'].bounds[1])
        else:
            return 'NA'

    @property
    def json(self):
        return json.dumps([c.json for c in self._coords.values()])

    @property
    def hash(self):
        return hash(json.dumps(self.json))
    
    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def drop(self, dims, ignore_missing=False):
        """
        Remove the given dimensions from the Coordinates.
        
        Parameters
        ----------
        dims : str, list
            Description
        ignore_missing : bool

        """

        if not isinstance(dims, (tuple, list)):
            dims = (dims,)

        for dim in dims:
            if not isinstance(dim, string_types):
                raise TypeError("Invalid drop dimension type '%s'" % type(dim))
            if dim not in self.dims and not ignore_missing:
                raise KeyError("Dimension '%s' not found in Coordinates with %s" % (dim, self.dims))

        return Coordinates([c for c in self._coords.values() if c.name not in dims])

    # do we ever need this?
    def udrop(self, dims, ignore_missing=False):
        if not isinstance(dims, (tuple, list)):
            dims = (dims,)

        for dim in dims:
            if not isinstance(dim, str):
                raise TypeError("Invalid drop dimension type '%s'" % type(dim))
            if dim not in self.udims and not ignore_missing:
                raise KeyError("Dimension '%s' not found in Coordinates with %s" % (dim, self.udims))

        cs = []
        for c in self.coords.values():
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
        TODO
        """

        intersections = [c.intersect(other, outer=outer, return_indices=return_indices) for c in self.values()]
        if return_indices:
            coords = Coordinates([c for c, I in intersections])
            idx = [I for c, I in intersections]
            return coords, idx
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

    def unstack(self):
        """
        Unstack the coordinates of all of the dimensions.
        
        Returns
        -------
        unstacked : Coordinates
            A new coordinate object with unstacked dimensions.

        See Also
        --------
        xr.DataArray.unstack
        """

        return Coordinates([self[dim] for dim in self.udims], **self.properties)
    
    @property
    def properties(self):
        '''
        Dictionary specifying the coordinate properties.
        
        Returns
        -------
        TYPE
            Description
        '''

        # TODO JXM
        # return {
        #     'coord_ref_sys': self.coord_ref_sys,
        #     'ctype': self.ctype
        # }
        
        c = self[self.udims[0]]
        return {
            'coord_ref_sys': c.coord_ref_sys,
            'ctype': c.ctype
        }

    
    # #@property
    # #def gdal_transform(self):
    #     if self['lon'].regularity == 'regular' and self['lat'].regularity == 'regular':
    #         lon_bounds = self['lon'].area_bounds
    #         lat_bounds = self['lat'].area_bounds
    #         transform = [lon_bounds[0], self['lon'].delta, 0, lat_bounds[0], 0, -self['lat'].delta]
    #     else:
    #         raise NotImplementedError
    #     return transform
    
    @property
    def gdal_crs(self):
        """GDAL coordinate reference system.
        
        Returns
        -------
        TYPE
            Description
        """

        # TODO enforce all have the same coord ref sys, possibly make that read-only and always passed from here
        # return GDAL_CRS[self.coord_ref_sys]
        return GDAL_CRS[self[self.udims[0]].coord_ref_sys]

    def iterchunks(self, shape, return_slices=False):
        """
        TODO
        
        Parameters
        ----------
        shape : tuple
            TODO
        return_slice : boolean, optional
            Return slice in addition to Coordinates chunk.
        
        Yields
        ------
        coords : Coordinates
            A Coordinates object with one chunk of the coordinates.
        slices : list
            slices for this Coordinates chunk, only if return_slices is True
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
        Transpose (re-order) the Coordinates dimensions.
              
        Parameters
        ----------
        in_place : boolean, optional
            If False, return a new, transposed Coordinates object (default).
            If True, transpose the dimensions in-place.
        *dims : str, optional
            Reorder dims to this order. By default, reverse the dims.

        Returns
        -------
        transposed : Coordinates
            The transposed Coordinates object.
        
        See Also
        --------
        xarray.DataArray.transpose : return a transposed DataArray
        
        """

        if len(dims) == 0:
            dims = list(self._coords.keys())[::-1]

        if kwargs.get('in_place', False):
            self._coords = OrderedDict([(dim, self._coords[dim]) for dim in dims])
            return self

        else:
            return Coordinates([self._coords[dim] for dim in dims], **self.properties)

def merge_dims(coords_list):
    """
    Merge the dimensions of the given coordinates. Throws an error if dimensions are duplicated.

    Arguments
    ---------
    coords_list : list
        List of Coordinates objects

    Returns
    -------
    coords : Coordinates
        Coordinates object with the dimension(s) from each set of coordinates in the list.
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
        List of Coordinates objects.

    Returns
    -------
    coords : Coordinates
        Coordinates object with concatenated coordinate values in each dimension.
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
        List of Coordinates objects

    Returns
    -------
    coords : Coordinates
        Coordinates object with unique, sorted coordinate values in each dimension.
    """

    return concat(coords_list).unique()
