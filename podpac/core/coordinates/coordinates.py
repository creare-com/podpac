"""
Multidimensional Coordinates
"""


from __future__ import division, unicode_literals, print_function, absolute_import

import copy
import sys
import itertools
from collections import OrderedDict

import numpy as np
import traitlets as tl
import pandas as pd
import xarray as xr
import xarray.core.coordinates

from podpac.core.coordinates.base_coordinates1d import BaseCoordinates1d
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
    
        >>> Coordinates(lat=1)                                       # doctest: +SKIP
        >>> Coordinates(lat_lon=((1, 2)))                            # doctest: +SKIP
        >>> Coordinates(lat=(49.1, 50.2, 100))                       # doctest: +SKIP
        >>> Coordinates(lat_lon=((49.1, -120), (50.2, -122), 100))   # doctest: +SKIP
        >>> Coordinates(lat=(49.1, 50.1, 0.1))                       # doctest: +SKIP
        >>> Coordinates(lat=np.array([50, 50.1, 50.4, 50.8, 50.9]))  # doctest: +SKIP
        >>> Coordinates(lat_lon=([50, 50.1, 50.4, 50.8, 50.9], [-120, -125, -126, -127, -130])) # doctest: +SKIP

    Attributes
    ----------
    coords
    dims
    ndim
    shape
    coordinates
    """

    if sys.version < '3.6':
        _coords = OrderedDictTrait(trait=tl.Instance(BaseCoordinates1d))
    else:
        _coords = tl.Dict(trait=tl.Instance(BaseCoordinates1d))

    def __init__(self, coords=[], coord_ref_sys=None, ctype=None, distance_units=None):
        """
        Initialize a multidimensional coords object.

        Parameters
        ----------
        coords : list, dict, or Coordinates
            List of named BaseCoordinates1d objects
        ctype : str
            Default coordinates type (optional).
        coord_ref_sys : str
            Default coordinates reference system (optional)
        """

        if isinstance(coords, BaseCoordinates1d):
            coords = [coords]

        if isinstance(coords, list):
            d = OrderedDict()
            for i, c in enumerate(coords):
                if c.name is None:
                    raise ValueError("missing dimension name in coords list at position %d" % i)
                if c.name in d:
                    raise ValueError("duplicate dimension name '%s' in coords list at position %d" % (c.name, i))
                d[c.name] = c
            coords = d

        else:
            raise TypeError("Unrecognized coords type '%s'" % type(coords))

        # set 1d coordinates defaults
        # TODO factor out, store as default_* traits, and pass on through StackedCoordinates as well
        # maybe move to observe so that it gets validated first
        for c in coords.values():
            if 'ctype' not in c._trait_values and ctype is not None:
                c.ctype = ctype
            if 'coord_ref_sys' not in c._trait_values and coord_ref_sys is not None:
                c.coord_ref_sys = coord_ref_sys
            if 'units' not in c._trait_values and distance_units is not None and c.name in ['lat', 'lon', 'alt']:
                c.units = distance_units
        
        super(Coordinates, self).__init__(_coords=coords)

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
            if isinstance(d[dim], tuple):
                c = UniformCoordinates1d.from_tuple(d[dim], name=dim)
            else:
                c = ArrayCoordinates1d(d[dim], name=dim)
            coords.append(c)

        return coords

    @classmethod
    def grid(cls, coord_ref_sys=None, ctype=None, distance_units=None, order=None, **kwargs):
        coords = cls._coords_from_dict(kwargs, order)
        return cls(coords, coord_ref_sys=coord_ref_sys, ctype=ctype, distance_units=distance_units)

    @classmethod
    def point(cls, coord_ref_sys=None, ctype=None, distance_units=None, order=None, **kwargs):
        coords = cls._coords_from_dict(kwargs, order)
        stacked = StackedCoordinates(coords)
        return cls(stacked, coord_ref_sys=coord_ref_sys, ctype=ctype, distance_units=distance_units)

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
    
    # ------------------------------------------------------------------------------------------------------------------
    # standard dict-like methods
    # ------------------------------------------------------------------------------------------------------------------

    def values(self):
        return self._coords.values()

    def keys(self):
        return self._coords.keys()

    def items(self):
        return self._coords.items()

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
            
        if not isinstance(c, BaseCoordinates1d):
            raise TypeError("todo")

        if c.name is None:
            c.name = dim
        elif c.name != dim:
            raise ValueError("dimension name mismatch, '%s' != '%s'" % (dim, c.name))
        
        # TODO ctype, etc defaults
        
        self._coords[dim] = c
        
        # TODO we could also support setting individal coords in stacked coords
        # if dim in self.udims:
        #     for c in self._coords.values():
        #         if isinstance(c, StackedCoordinates) and dim in c.dims:
        #             c[dim] = c
        
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

    def get(self, dim):
        try:
            return self[dim]
        except KeyError:
            return None

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
        return np.prod(self.shape)

    @property
    def udims(self):
        return tuple(dim for c in self._coords.values() for dim in c.dims)

    @property
    def coords(self):
        # TODO don't recompute this every time (but also don't compute it until requested)
        x = xr.DataArray(np.empty(self.shape), coords=[c.coordinates for c in self._coords.values()], dims=self.dims)
        return x.coords
    
    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def drop(self, dims):
        """
        Remove the given dimensions from the Coordinates.
        
        Parameters
        ----------
        dims
            Description
        """

        if not isinstance(dims, (tuple, list)):
            dims = [dims]

        for dim in dims:
            if dim not in self.dims:
                raise ValueError("Dimension '%s' not found in Coordinates with %s" % (dim, self.dims))

        return Coordinates([c for c in self._coords.values() if c.name not in dims])

    # do we ever need this?
    def drop2(self, dims):
        if not isinstance(dims, (tuple, list)):
            dims = [dims]

        for dim in dims:
            if dim not in self.dims:
                raise ValueError("Dimension '%s' not found in Coordinates with %s" % (dim, self.dims))

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

    def intersect(self, other, pad=1, return_indices=False):
        intersections = [c.intersect(other, pad=pad, return_indices=return_indices) for c in self.values()]
        if return_indices:
            return Coordinates([c for c, I in intersections]), [I for c, I in intersections]
        else:
            return Coordinates(intersections)

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


    # def get_dims_map(self, coords=None):
    #     """
    #     TODO
        
    #     Parameters
    #     ----------
    #     coords : dict, optional
    #         TODO
        
    #     Returns
    #     -------
    #     stacked_coords : OrderedDict
    #         TODO
    #     """

    #     if coords is None:
    #         coords = self.coords

    #     stacked_coords = OrderedDict()
    #     for c in coords:
    #         if '_' in c:
    #             for cc in c.split('_'):
    #                 stacked_coords[cc] = c
    #         else:
    #             stacked_coords[c] = c
    #     return stacked_coords
    
    # def unstack_dict(self, coords=None, check_dim_repeat=False):
    #     """
    #     TODO
        
    #     Parameters
    #     ----------
    #     coords : dict, optional
    #         TODO
    #     check_dim_repeat : boolean
    #         TODO
        
    #     Returns
    #     -------
    #     new_crds : OrderedDict
    #         TODO
        
    #     Raises
    #     ------
    #     TypeError
    #         Description
    #     ValueError
    #         Description
    #     """

    #     if coords is None:
    #         coords = self._coords

    #     dims_map = self.get_dims_map(coords)
       
    #     new_crds = OrderedDict()
    #     seen_dims = []
    #     for key, val in coords.items():
    #         if key not in self.dims_map:  # stacked
    #             keys = key.split('_')
    #             for i, k in enumerate(keys):
    #                 new_crds[k] = val[i]

    #                 # parse uniform coords tuple and append size
    #                 if isinstance(val[i], tuple):
    #                     if len(val) != len(keys) + 1:
    #                         raise ValueError("missing size for stacked uniform coordinates")
    #                     if (not isinstance(val[-1], (int, np.long, np.integer)) or
    #                         isinstance(val[-1], (np.timedelta64))):
    #                         raise TypeError("invalid size for stacked uniform coordinates \
    #                                          (expected integer, not '%s')" % type(val[-1]))
    #                     new_crds[k] += (val[-1],)
                    
    #                 if check_dim_repeat and k in seen_dims:
    #                     raise ValueError(
    #                         "The dimension '%s' cannot be repeated." % dim)
    #                 seen_dims.append(k)
    #         else:
    #             new_crds[key] = val
    #             if check_dim_repeat and key in seen_dims:
    #                 raise ValueError(
    #                     "The dimension '%s' cannot be repeated." % key)
    #             seen_dims.append(key)

    #     return new_crds

    # def stack_dict(self, coords=None, dims_map=None):
    #     """
    #     TODO
        
    #     Parameters
    #     ----------
    #     coords : OrderedDict, optional
    #         TODO
    #     dims_map : OrderedDict, optional
    #         TODO
        
    #     Returns
    #     -------
    #     stacked_coords : OrderedDict
    #         TODO
    #     """

    #     if coords is None:
    #         coords = self._coords
    #     if dims_map is None:
    #         dims_map = self.dims_map

    #     stacked_coords = OrderedDict()
    #     for key, val in dims_map.items():
    #         if val in stacked_coords:
    #             temp = stacked_coords[val]
    #             if not isinstance(temp, list):
    #                 temp = [temp]
    #             temp.append(coords[key])
    #             stacked_coords[val] = temp
    #         else:
    #             stacked_coords[val] = coords[key]
    #     return stacked_coords
   
    # @property
    # def is_stacked(self):
    #     """
    #     True if any of the coordinates are stacked; False if none are stacked.
        
    #     Returns
    #     -------
    #     TYPE
    #         Description
    #     """

    #     for k, v in self.dims_map.items():
    #         if k != v:
    #             return True
    #     return False
   
    # def stack(self, stack_dims, copy=True):
    #     """
    #     Stack the coordinates in of given dimensions.
        
    #     Parameters
    #     ----------
    #     stack_dims : list
    #         dimensions to stack
    #     copy : boolean, optional
    #         If True, stack dimensions in-place.
        
    #     Returns
    #     -------
    #     coord : Coordinates
    #         If copy=False, a new coordinate object with stacked dimensions.
    #         If copy=True, this object with its dimensions stacked.
    #     """

    #     stack_dim = '_'.join(stack_dims)
    #     dims_map = {k:v for k,v in self.dims_map.items()}
    #     for k in stack_dims:
    #         dims_map[k] = stack_dim
    #     stack_dict = self.stack_dict(self._coords.copy(), dims_map=dims_map)
    #     if copy:
    #         return self.__class__(coords=stack_dict, **self.kwargs)
    #     else:
    #         # Check for correct dimensions
    #         tmp = self.dims_map
    #         self.dims_map = dims_map
    #         try:
    #             self._coords_validate({'value': self._coords})
    #         except Exception as e:
    #             self.dims_map = tmp
    #             raise(e)
            
    #         return self

    # def unstack(self, copy=True):
    #     """
    #     Unstack the coordinates of all of the dimensions.
        
    #     Parameters
    #     ----------
    #     copy : boolean, optional
    #         If True, unstack dimensions in-place.
        
    #     Returns
    #     -------
    #     coord : Coordinates
    #         If copy=False, a new coordinate object with unstacked dimensions.
    #         If copy=True, this object with its dimensions unstacked.
    #     """

    #     if copy:
    #         return self.__class__(coords=self._coords.copy())
    #     else:
    #         self.dims_map = {v:v for v in self.dims_map}
    #         return self
    
    # @property
    # def kwargs(self):
    #     '''
    #     Dictionary specifying the coordinate properties.
        
    #     Returns
    #     -------
    #     TYPE
    #         Description
    #     '''

    #     return {
    #         'coord_ref_sys': self.coord_ref_sys,
    #         'ctype': self.ctype
    #     }
    
    # def replace_coords(self, other, copy=True):
    #     '''
    #     TODO
        
    #     Parameters
    #     ----------
    #     other : Coordinates
    #         Replacement Coordinates.
    #     copy : boolean, optional
    #         If True (default), make a new Coordinates object. If False, replace
    #         coordinates in-place.
        
    #     Returns
    #     -------
    #     coord : Coordinates
    #         If copy=True, a new Coordinates with replaced coordinates.
    #         If copy=False, this object with its coordinates replaced.
    #     '''

    #     if copy:
    #         coords = self._coords.copy()
    #         dims_map = self.dims_map.copy()
    #     else:
    #         coords = self._coords
    #         dims_map = self.dims_map
            
    #     for c in coords:
    #         if c in other._coords:
    #             coords[c] = other._coords[c]
    #             old_stack = dims_map[c]
    #             dims_map[c] = other.dims_map[c]
    #             fix_stack = [o for o in old_stack.split('_') 
    #                          if o not in dims_map[c].split('_')]
    #             for f in fix_stack:
    #                 dims_map[f] = '_'.join(fix_stack)
        
    #     if copy:
    #         stack_dict = self.stack_dict(coords, dims_map=dims_map)
    #         return self.__class__(coords=stack_dict)
    #     else:
    #         return self   
 
    # def get_shape(self, other_coords=None):
    #     """
    #     Coordinates shape, corresponding to dims attribute.
        
    #     Parameters
    #     ----------
    #     other_coords : Coordinates, optional
    #         TODO
        
    #     Returns
    #     -------
    #     shape : tuple
    #         Number of coordinates in each dimension.
    #     """

    #     if other_coords is None:
    #         other_coords = self
    #     # Create shape for each dimension
    #     shape = []
    #     seen_dims = []
    #     self_seen_dims = []
    #     for k in self._coords:
    #         if k in other_coords._coords:
    #             shape.append(other_coords._coords[k].size)
    #             # Remove stacked duplicates
    #             if other_coords.dims_map[k] in seen_dims:
    #                 shape.pop()
    #             else:
    #                 seen_dims.append(other_coords.dims_map[k])
    #         else:
    #             shape.append(self._coords[k].size)
    #             # Remove stacked duplicates
    #             if self.dims_map[k] in self_seen_dims:
    #                 shape.pop()
    #             else:
    #                 self_seen_dims.append(self.dims_map[k])
                

    #     return tuple(shape)
        
    # @property
    # def shape(self):
    #     """Coordinates shape, corresponding to dims attribute. 
        
    #     Returns
    #     -------
    #     TYPE
    #         Description
    #     """

    #     return self.get_shape()
    
    # @property
    # def delta(self):
    #     """to be deprecated 
        
    #     Returns
    #     -------
    #     TYPE
    #         Description
    #     """

    #     try:
    #         return np.array([c.delta for c in self._coords.values()]).squeeze()
    #     except ValueError as e:
    #         return np.array([c.delta for c in self._coords.values()], 
    #                 object).squeeze()
    
    # @property
    # def dims(self):
    #     """Coordinates dimensions. 
        
    #     Returns
    #     -------
    #     TYPE
    #         Description
    #     """

    #     dims = []
    #     for v in self.dims_map.values():
    #         if v not in dims:
    #             dims.append(v)
    #     return dims
    
    # @property
    # def coords(self):
    #     """TODO 
        
    #     Returns
    #     -------
    #     TYPE
    #         Description
    #     """

    #     crds = OrderedDict()
    #     for k in self.dims:
    #         if k in self.dims_map:  # not stacked
    #             crds[k] = self._coords[k].coordinates
    #         else:
    #             coordinates = [self._coords[kk].coordinates
    #                            for kk in k.split('_')]
    #             dtype = [(str(kk), coordinates[i].dtype) 
    #                      for i, kk in enumerate(k.split('_'))]
    #             n_coords = len(coordinates)
    #             s_coords = len(coordinates[0])
    #             crds[k] = np.atleast_1d(np.array([[tuple([coordinates[j][i]
    #                                  for j in range(n_coords)])]
    #                                for i in range(s_coords)],
    #                 dtype=dtype).squeeze())
    #     return crds
    
    # #@property
    # #def gdal_transform(self):
    #     #if self['lon'].regularity == 'regular' \
    #            #and self['lat'].regularity == 'regular':
    #         #lon_bounds = self['lon'].area_bounds
    #         #lat_bounds = self['lat'].area_bounds
        
    #         #transform = [lon_bounds[0], self['lon'].delta, 0,
    #                      #lat_bounds[0], 0, -self['lat'].delta]
    #     #else:
    #         #raise NotImplementedError
    #     #return transform
    
    @property
    def gdal_crs(self):
        """GDAL coordinate reference system.
        
        Returns
        -------
        TYPE
            Description
        """

        # TODO enforce all have the same coord ref sys, possibly make that read-only and always passed from here
        return GDAL_CRS[self.coord_ref_sys]
    
    # def add_unique(self, other):
    #     """
    #     Concatenate coordinates, skipping duplicates.
        
    #     Parameters
    #     ----------
    #     other : Coordinates
    #         Coordinates to concatenate.
        
    #     Returns
    #     -------
    #     coord : Coordinates
    #         New Coordinates object with concatenated coordinates.
    #     """

    #     return self._add(other, unique=True)
    
    # def __add__(self, other):
    #     """
    #     Concatenate coordinates.
        
    #     Parameters
    #     ----------
    #     other : Coordinates
    #         Coordinates to concatenate.
        
    #     Returns
    #     -------
    #     coord : Coordinates
    #         New Coordinates object with concatenated coordinates.
    #     """

    #     return self._add(other)
    
    # def _add(self, other, unique=False):
    #     if not isinstance(other, Coordinates):
    #         raise TypeError(
    #             "Unsupported type '%s', can only add Coordinates object" % (
    #                 other.__class__.__name__))
    #     new_coords = copy.deepcopy(self._coords)
    #     dims_map = self.dims_map
    #     for key in other._coords:
    #         if key in self._coords:
    #             if dims_map[key] != other.dims_map[key]:
    #                 raise ValueError(
    #                     "Cannot add coordinates with different stacking. "
    #                     "%s != %s." % (dims_map[key], other.dims_map[key])
    #                 )
    #             if np.all(np.array(self._coords[key].coords) !=
    #                     np.array(other._coords[key].coords)) or not unique:
    #                 new_coords[key] = self._coords[key] + other._coords[key]
    #         else:
    #             dims_map[key] = other.dims_map[key]
    #             new_coords[key] = copy.deepcopy(other._coords[key])
    #     return self.__class__(coords=self.stack_dict(new_coords, dims_map))

    # def iterchunks(self, shape, return_slice=False):
    #     """
    #     TODO
        
    #     Parameters
    #     ----------
    #     shape : tuple
    #         TODO
    #     return_slice : boolean, optional
    #         Return slice in addition to Coordinates chunk.
        
    #     Yields
    #     ------
    #     l : slice
    #         If return_slice=True, slice for this Coordinates chunk.
    #     coords : Coordinates
    #         A Coordinates object with one chunk of the coordinates.
    #     """
        
    #     # TODO assumes the input shape dimension and order matches
    #     # TODO replace self[k].coords[slc] with self[k][slc] (and implement the slice)

    #     slices = [
    #         [slice(i, i+n) for i in range(0, m, n)]
    #         for m, n
    #         in zip(self.shape, shape)]

    #     for l in itertools.product(*slices):
    #         kwargs = {k:self.coords[k][slc] for k, slc in zip(self.dims, l)}
    #         kwargs['order'] = self.dims
    #         coords = Coordinates(**kwargs)
    #         if return_slice:
    #             yield l, coords
    #         else:
    #             yield coords

    # def transpose(self, *dims, **kwargs):
    #     """
    #     Transpose (re-order) the Coordinates dimensions.
              
    #     Parameters
    #     ----------
    #     in_place : boolean, optional
    #         If False, return a new, transposed Coordinates object (default).
    #         If True, transpose the dimensions in-place.
    #     *dims : str, optional
    #         Reorder dims to this order. By default, reverse the dims.
    #     **kwargs
    #         Description

    #     Returns
    #     -------
    #     transposed : Coordinates
    #         The transposed Coordinates object.
        
    #     See Also
    #     --------
    #     xarray.DataArray.transpose : return a transposed DataArray
        
    #     """

    #     if len(dims) == 0:
    #         dims = list(self._coords.keys())[::-1]

    #     coords = OrderedDict((dim, self._coords[dim]) for dim in dims)

    #     if kwargs.get('in_place', False):
    #         self._coords = coords
    #         return self

    #     else:
    #         kwargs = coords
    #         kwargs.update(self.kwargs)
    #         return Coordinates(order=dims, **kwargs)

    @property
    def latlon_bounds_str(self):
        if 'lat' in self.dims and 'lon' in self.dims:
            # Where is this really used? Shouldn't this be area_bounds?
            return '%s_%s_x_%s_%s' % (
                self['lat'].bounds[0],
                self['lon'].bounds[0],
                self['lat'].bounds[1],
                self['lon'].bounds[1])
        else:
            return 'NA'

# TODO spec here is uncertain, not in use yet
# class CoordinateGroup(BaseCoordinates):
#     """CoordinateGroup Summary
#     """
    
#     # TODO list or array?
#     _items = tl.List(trait=tl.Instance(Coordinates))

#     @tl.validate('_items')
#     def _validate_items(self, d):
#         items = d['value']
#         if not items:
#             return items

#         # unstacked dims must match, but not necessarily in order
#         dims = set(items[0].dims_map)
#         for g in items:
#             if set(g.dims_map) != dims:
#                 raise ValueError(
#                     "Mismatching dims: '%s != %s" % (dims, set(g.dims)))

#         return items

#     def __init__(self, items=[], **kwargs):
#         return super(CoordinateGroup, self).__init__(_items=items, **kwargs)

#     def __repr__(self):
#         rep = self.__class__.__name__
#         rep += '\n' + '\n'.join([repr(g) for g in self._items])
#         return rep
    
#     def __getitem__(self, key):
#         if isinstance(key, (int, slice)):
#             return self._items[key]
        
#         elif isinstance(key, tuple):
#             if len(key) != 2:
#                 raise IndexError("Too many indices for CoordinateGroup")
            
#             k, dim = key
#             # TODO list or array?
#             return [item[dim] for item in self._items[k]]
        
#         else:
#             raise IndexError(
#                 "invalid CoordinateGroup index type '%s'" % type(key))

#     def __len__(self):
#         return len(self._items)

#     def __iter__(self):
#         return self._items.__iter__()

#     def append(self, c):
#         if not isinstance(c, Coordinates):
#             raise TypeError(
#                 "Can only append Coordinates objects, not '%s'" % type(c))
        
#         self._items.append(c)
   
#     def stack(self, stack_dims, copy=True):
#         """ stack all """

#         if copy:
#             return CoordinateGroup(
#                 [c.stack(stack_dims, copy=True) for c in self._items])
#         else:
#             for c in self._items:
#                 c.stack(stack_dims)
#             return self

#     def unstack(self, copy=True):
#         """ unstack all"""
#         if copy:
#             return CoordinateGroup(
#                 [c.unstack(stack_dims, copy=True) for c in self._items])
#         else:
#             for c in self._items:
#                 c.unstack(stack_dims)
#             return self            

#     def intersect(self, other, coord_ref_sys=None, pad=1, ind=False):
#         return CoordinateGroup([c.intersect(other) for c in self._items])
    
#     @property
#     def dims(self):
#         """ unordered (set) and unstacked """
#         if len(self._items) == 0:
#             return {}
#         return set(self._items[0].dims_map)

#     def add_unique(self, other):
#         return self._add(other, unique=True)
    
#     def __add__(self, other):
#         return self._add(other)
    
#     def _add(self, other, unique=False):
#         if unique:
#             raise NotImplementedError("TODO")

#         if isinstance(other, Coordinates):
#             # TODO should this concat, fail, or do something else?
#             # items = self._items + [other]
#             raise NotImplementedError("TODO")
#         elif isinstance(other, CoordinateGroup):
#             items = self._items + g._items
#         else:
#             raise TypeError("Cannot add '%s', only BaseCoordinates" % type(c))
        
#         return CoordinateGroup(self._items + [other])

#     def __iadd__(self, other):
#         if isinstance(other, Coordinates):
#             # TODO should this append, fail, or do something else?
#             # TypeError("Cannot add individual Coordinates, use 'append'")
#             # self._items.append(other)
#             raise NotImplementedError("TODO")
#         elif isinstance(other, CoordinateGroup):
#             self._items += g._items
#         else:
#             raise TypeError("Cannot add '%s' to CoordinateGroup" % type(c))

#         return self

#     def iterchunks(self, shape, return_slice=False):
#         raise NotImplementedError("TODO")

#     @property
#     def latlon_bounds_str(self):
#         # TODO should this be a single latlon bounds or a list of bounds?
#         raise NotImplementedError("TODO")