"""
Coordinate Summary

.. testsetup:: podpac.core.coordinate.coordinate
    
    from podpac.core.coordinate.coordinate import *
"""


from __future__ import division, unicode_literals, print_function, absolute_import

import copy
import sys
import itertools
from collections import OrderedDict
import warnings

import numpy as np
import traitlets as tl
from xarray.core.coordinates import DataArrayCoordinates

from podpac.core.coordinate.coord import BaseCoord, Coord, MonotonicCoord, UniformCoord, coord_linspace
from podpac.core.coordinate.util import CRS2GDAL

def _unstack_dims(stacked_dim):
    return stacked_dim.split('_')

def _stack_dims(unstacked_dims):
    return '_'.join(unstacked_dims)

class BaseCoordinate(tl.HasTraits):
    """
    Base class for multidimensional coordinates.
    """

    @property
    def _valid_dims(self):
        return ('time', 'lat', 'lon', 'alt')

    def stack(self, stack_dims, copy=True):
        raise NotImplementedError

    def unstack(self, copy=True):
        raise NotImplementedError

    def intersect(self, other, coord_ref_sys=None, pad=1, ind=False):
        raise NotImplementedError

class Coordinate(BaseCoordinate):
    """
    Multidimensional Coordinates.

    Attributes
    ----------
    ctype : str
        Coordinates type (default 'segment'). Options::
         - 'segment': whole segment between this coordinate and the next.
         - 'point': single point
    segment_position : float
        For segment coordinates, where along the segment the coordinate is
        specified, between 0 and 1 (default 0.5). Unused for point.
    coord_ref_sys : unicode
        Coordinate reference system.
    dims_map
    stacked_coords
    is_stacked
    shape
    dims
    coords
    gdal_crs
    kwargs


    """
    
    _coords = tl.Instance(OrderedDict) # unstacked coord dictionary
    _dims_map = tl.Dict()
    
    ctype = tl.Enum(['segment', 'point'], default_value='segment')
    segment_position = tl.Float(default_value=0.5)
    coord_ref_sys = tl.CUnicode(default_value='WGS84')

    def __init__(self, coords=OrderedDict(), **kwargs):
        """
        Initialize a multidimensional coords object.

        Parameters
        ----------
        coords : OrderedDict
            TODO
        ctype : str
            Coordinate type, passed to individual coord objects during initialization.
        segment_position : float
            Segment position, passed to individual coord objects during initialization.
        coord_ref_sys : str
            Coordinate reference system, passed to individual coord objects during initialization.

        """

        if sys.version < '3.6' and not isinstance(coords, OrderedDict):
            raise TypeError("'coords' must be an OrderedDict in Python <3.6")

        if not isinstance(coords, dict):
            raise TypeError("'coords' must be a dict")

        dims_map = OrderedDict()
        _coords = OrderedDict()

        for key, val in coords.items():
            dims = _unstack_dims(key)
            if len(dims) == 1: vals = [val]
            else: vals = val
            if len(dims) != len(vals):
                raise ValueError("Invalid stacking for '%s' (expected %d coordinate values, got %d)" % (
                    key, len(dims), len(vals)))
            for dim, val in zip(dims, vals):
                if dim in dims_map:
                    raise ValueError("The dimension '%s' cannot be repeated." % dim)
                dims_map[dim] = key
                _coords[dim] = val

        with self.hold_trait_notifications():
            self._dims_map = dims_map
            self._coords = _coords

        super(Coordinate, self).__init__(**kwargs)
    
    def _validate_coords_dims_map(self, coords, dims_map):
        # validate each dim and value
        for dim, val in coords.items():
            if dim not in self._valid_dims:
                raise ValueError("Invalid dimension '%s', expected one of %s" % (dim, self._valid_dims))

            if not isinstance(val, BaseCoord):
                raise TypeError("Invalid coord type '%s'" % val.__class__.__name__)

        # check stacking (by attempting to create the stacked_coords dictionary)
        stack_coords(coords, dims_map)

    @tl.validate('_dims_map')
    def _coords_validate(self, proposal):
        self._validate_coords_dims_map(self._coords, proposal['value'])
        return proposal['value']

    @tl.validate('_coords')
    def _coords_validate(self, proposal):
        self._validate_coords_dims_map(proposal['value'], self.dims_map)
        return proposal['value']

    def __repr__(self):
        rep = str(self.__class__.__name__)
        for d in self._coords:
            d2 = self.dims_map[d]
            if d2 != d:
                d2 = d2 + '[%s]' % d
            rep += '\n\t{}: '.format(d2) + str(self._coords[d])
        return rep
    
    def __getitem__(self, dim):
        """
        Get the coordinates for the given dimension.
        
        Parameters
        ----------
        dim : str

        Returns
        -------
        coord : BaseCoord
            single-dimensional coordinates for dim

        Raises
        ------
        KeyError
            If the dimension does not exist in the coordinate
        """

        if dim not in self._coords:
            return KeyError("dimension '%s' does not exist in Coordinate" % dim)

        return self._coords[dim]

    @property
    def dims_map(self):
        """ Maps unstacked dimensions to the correct (potentially stacked) dimension """
        return self._dims_map

    @property
    def stacked_coords(self):
        """ Stacked coordinates dictionary. """
        return stack_coords(self._coords, self.dims_map)
   
    @property
    def is_stacked(self):
        """ True if any of the coordinates are stacked; False if none are stacked. """

        for k, v in self.dims_map.items():
            if k != v:
                return True
        return False
        
    @property
    def shape(self):
        """Coordinates shape, corresponding to dims attribute. """

        return tuple(self[_unstack_dims(dim)[0]].size for dim in self.dims)
    
    @property
    def delta(self):
        """ used in interpolation, to be deprecated """

        warnings.warn("delta will be removed before v0.0.1", DeprecationWarning)

        try:
            return np.array([c.delta for c in self._coords.values()]).squeeze()
        except ValueError as e:
            return np.array([c.delta for c in self._coords.values()], object).squeeze()
    
    @property
    def dims(self):
        """Coordinate dimensions."""

        dims = []
        for v in self.dims_map.values():
            if v not in dims:
                dims.append(v)
        return dims
    
    @property
    def coords(self):
        """ Coordinate values for each dimension """

        crds = OrderedDict()
        for k in self.dims:
            if k in self.dims_map:  # not stacked
                crds[k] = self._coords[k].coordinates
            else:
                unstacked_dims = _unstack_dims(k)
                coordinates = [self._coords[kk].coordinates for kk in unstacked_dims]
                dtype = [(str(kk), coordinates[i].dtype) for i, kk in enumerate(unstacked_dims)]
                n_coords = len(coordinates)
                s_coords = len(coordinates[0])
                c = [[tuple([coordinates[j][i] for j in range(n_coords)])] for i in range(s_coords)]
                crds[k] = np.array(c, dtype=dtype).squeeze()
        return crds
    
    # @property
    # def gdal_transform(self):
    #     if self['lon'].regularity == 'regular' and self['lat'].regularity == 'regular':
    #         lon_bounds = self['lon'].area_bounds
    #         lat_bounds = self['lat'].area_bounds
        
    #         transform = [lon_bounds[0], self['lon'].delta, 0, lat_bounds[0], 0, -self['lat'].delta]
    #     else:
    #         raise NotImplementedError
    #     return transform
    
    @property
    def gdal_crs(self):
        """GDAL coordinate reference system. """

        return CRS2GDAL[self.coord_ref_sys.upper()]

    @property
    def kwargs(self):
        """ Dictionary of coordinate properties. """

        return {
            'coord_ref_sys': self.coord_ref_sys,
            'segment_position': self.segment_position,
            'ctype': self.ctype
            }

    @property
    def latlon_bounds_str(self):
        """ TODO """

        if 'lat' in self._coords and 'lon' in self._coords:
            return '%s_%s_x_%s_%s' % (
                self['lat'].bounds[0],
                self['lon'].bounds[0],
                self['lat'].bounds[1],
                self['lon'].bounds[1])
        else:
            return 'NA'
   
    def stack(self, stack_dims, copy=True):
        """
        Stack the coordinates in of given dimensions.
        
        Parameters
        ----------
        stack_dims : list
            dimensions to stack
        copy : boolean, optional
            If True, stack dimensions in-place.
        
        Returns
        -------
        coord : Coordinate
            If copy=False, a new coordinate object with stacked dimensions.
            If copy=True, this object with its dimensions stacked.
        """

        sizes = [self[dim].size for dim in stack_dims]
        if any(size != sizes[0] for size in sizes):
            raise ValueError("Stacked dimensions size mismatch (%s must all match)" % (sizes))


        key = _stack_dims(stack_dims)
        stacked_dims_map = self.dims_map.copy()
        for k in stack_dims:
            stacked_dims_map[k] = key
        
        if copy:
            stacked_coords = stack_coords(self._coords.copy(), dims_map=stacked_dims_map)
            return self.__class__(coords=stacked_coords, **self.kwargs)
        else:
            self._dims_map = stacked_dims_map
            return self

    def unstack(self, copy=True):
        """
        Unstack the coordinates of all of the dimensions.
        
        Parameters
        ----------
        copy : boolean, optional
            If True, unstack dimensions in-place.
        
        Returns
        -------
        coord : Coordinate
            If copy=False, a new coordinate object with unstacked dimensions.
            If copy=True, this object with its dimensions unstacked.
        """

        if copy:
            return self.__class__(coords=self._coords.copy())
        else:
            self._dims_map = {v:v for v in self.dims_map}
            return self

    def intersect(self, other, coord_ref_sys=None, pad=1, ind=False):
        """
        TODO
        
        Parameters
        ----------
        other : Coordinate (TODO or BaseCoordinate?)
            Coordinates to intersect with.
        coord_ref_sys : str, optional
            unused
        pad : int, optional
            delta padding (default 1); to be deprecated
        ind : boolean, optional
            Return slice or indices instead of Coordinate objects.
        
        Returns
        -------
        intersection : Coordinate
            If ind=False, Coordinate object with coordinates within other.bounds.
        I : list
            If ind=True, a list of slices/indices for the intersected coordinates
            in each dimension
        """

        if ind or self.is_stacked:
            I = []
        else:
            d = OrderedDict()

        for i, dim in enumerate(self._coords):
            if isinstance(pad, (list, tuple)):
                spad = pad[j]
            elif isinstance(pad, dict):
                spad = pad[d]
            else:
                spad = pad

            if dim not in other._coords:
                if ind or self.is_stacked:
                    I.append(slice(None, None))
                else:
                    d[dim] = self._coords[dim]
                continue
            
            intersect = self._coords[dim].intersect(other._coords[dim], coord_ref_sys, ind=ind or self.is_stacked, pad=spad)
            
            if ind or self.is_stacked:
                I.append(intersect)
            else:
                d[dim] = intersect
        
        if ind and not self.is_stacked:
            return I
        
        if ind or self.is_stacked:
            # Need to handle the stacking
            I2 = [np.ones(s, bool) for s in self.shape]
            for i, d in enumerate(self.dims):
                parts = _unstack_dims(d)
                It = np.zeros_like(I2[i])
                for j, p in enumerate(parts):
                    k = list(self._coords.keys()).index(p)
                    It[I[k]] = True
                    I2[i] = I2[i] & It
                    It[:] = False
                I2[i] = np.where(I2[i])[0]
                
            if ind:
                return I2
            
            coords =  OrderedDict()
            for k in self._coords.keys():
                i = self.dims.index(self.dims_map[k])
                try:
                    coords[k] = MonotonicCoord(self._coords[k].coordinates[I2[i]])
                except:
                    coords[k] = Coord(self._coords[k].coordinates[I2[i]])
            coords = stack_coords(coords, self.dims_map)
            return Coordinate(coords, **self.kwargs)
        else:
            coords = stack_coords(d, self.dims_map)
            return Coordinate(coords, **self.kwargs)
    
    def replace_coords(self, other, copy=True):
        '''
        TODO
        
        Parameters
        ----------
        other : Coordinate
            Replacement Coordinates.
        copy : boolean, optional
            If True (default), make a new Coordinate object. If False, replace
            coordinates in-place.
        
        Returns
        -------
        coord : Coordinate
            If copy=True, a new Coordinate with replaced coordinates.
            If copy=False, this object with its coordinates replaced.
        '''

        def _replace(coords, dims_map, other):
            for c in coords:
                if c in other._coords:
                    coords[c] = other._coords[c]
                    dims_map[c] = other.dims_map[c]

        if copy:
            coords = self._coords.copy()
            dims_map = self._dims_map.copy()
            _replace(coords, dims_map, other)
            stacked_coords = stack_coords(coords, dims_map)
            return self.__class__(coords=stacked_coords)
        else:
            with self.hold_trait_notifications():
                _replace(self._coords, self._dims_map, other)
            return self
   
    def drop_dims(self, *args):
        """
        Remove the given dimensions from the Coordinate.
        
        Parameters
        ----------
        *args
            Description
        """

        unstacked_dims = []
        for arg in args:
            if arg not in self._coords:
                continue
            del self._coords[arg]
            if self._dims_map[arg] == arg:
                del self._dims_map[arg]
            else:
                unstacked_dims += _unstack_dims(self.dims_map[arg])

        if unstacked_dims:
            self.drop_dims(*unstacked_dims)
    
    def add_unique(self, other):
        """
        Concatenate coordinates, skipping duplicates.
        
        Parameters
        ----------
        other : Coordinate
            Coordinates to concatenate.
        
        Returns
        -------
        coord : Coordinate
            New Coordinate object with concatenated coordinates.
        """

        return self._add(other, unique=True)
    
    def __add__(self, other):
        """
        Concatenate coordinates.
        
        Parameters
        ----------
        other : Coordinate
            Coordinates to concatenate.
        
        Returns
        -------
        coord : Coordinate
            New Coordinate object with concatenated coordinates.
        """

        return self._add(other)
    
    def _add(self, other, unique=False):
        if not isinstance(other, Coordinate):
            raise TypeError("Unsupported type '%s', can only add Coordinate object" % (other.__class__.__name__))
        new_coords = copy.deepcopy(self._coords)
        dims_map = self.dims_map
        for key in other._coords:
            if key in self._coords:
                if dims_map[key] != other.dims_map[key]:
                    raise ValueError("Cannot add coordinates with different stacking. %s != %s." % (
                        dims_map[key], other.dims_map[key]))
                if np.all(np.array(self._coords[key].coords) != np.array(other._coords[key].coords)) or not unique:
                    new_coords[key] = self._coords[key] + other._coords[key]
            else:
                dims_map[key] = other.dims_map[key]
                new_coords[key] = copy.deepcopy(other._coords[key])
        return self.__class__(coords=stack_coords(new_coords, dims_map))

    def iterchunks(self, shape, return_slice=False):
        """
        TODO
        
        Parameters
        ----------
        shape : tuple
            TODO
        return_slice : boolean, optional
            Return slice in addition to Coordinate chunk.
        
        Yields
        ------
        l : slice
            If return_slice=True, slice for this Coordinate chunk.
        chunk : Coordinate
            A Coordinate object with one chunk of the coordinates.
        """
        
        # TODO assumes the input shape dimension and order matches
        # TODO replace self[k].coords[slc] with self[k][slc] (and implement the slice)

        slices = [map(lambda i: slice(i, i+n), range(0, m, n)) for m, n in zip(self.shape, shape)]

        for l in itertools.product(*slices):
            coords = OrderedDict((k, self.coords[k][slc]) for k, slc in zip(self.dims, l))
            chunk = Coordinate(coords, **self.kwargs)
            if return_slice:
                yield l, chunk
            else:
                yield chunk

    def transpose(self, *dims, **kwargs):
        """
        Transpose (re-order) the Coordinate dimensions.
              
        Parameters
        ----------
        in_place : boolean, optional
            If False, return a new, transposed Coordinate object (default).
            If True, transpose the dimensions in-place.
        *dims : str, optional
            Reorder dims to this order. By default, reverse the dims.
        **kwargs
            Description

        Returns
        -------
        transposed : Coordinate
            The transposed Coordinate object.
        
        See Also
        --------
        xarray.DataArray.transpose : return a transposed DataArray
        
        """

        if len(dims) == 0:
            dims = self._coords.keys()[::-1]

        coords = OrderedDict((dim, self._coords[dim]) for dim in dims)

        if kwargs.get('in_place', False):
            self._coords = coords
            return self

        else:
            return Coordinate(coords, **self.kwargs)

class CoordinateGroup(BaseCoordinate):
    """CoordinateGroup Summary
    """
    
    # TODO list or array?
    _items = tl.List(trait=tl.Instance(Coordinate))

    @tl.validate('_items')
    def _validate_items(self, d):
        items = d['value']
        if not items:
            return items

        # unstacked dims must match, but not necessarily in order
        dims = set(items[0].dims_map)
        for g in items:
            if set(g.dims_map) != dims:
                raise ValueError("Mismatching dims: '%s != %s" % (dims, set(g.dims)))

        return items

    def __init__(self, items=[], **kwargs):
        return super(CoordinateGroup, self).__init__(_items=items, **kwargs)

    def __repr__(self):
        rep = self.__class__.__name__
        rep += '\n' + '\n'.join([repr(g) for g in self._items])
        return rep
    
    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return self._items[key]
        
        elif isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Too many indices for CoordinateGroup")
            
            k, dim = key
            # TODO list or array?
            return [item[dim] for item in self._items[k]]
        
        else:
            raise IndexError("invalid CoordinateGroup index type '%s'" % type(key))

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return self._items.__iter__()

    def append(self, c):
        if not isinstance(c, Coordinate):
            raise TypeError("Can only append Coordinate objects, not '%s'" % type(c))
        
        self._items.append(c)
   
    def stack(self, stack_dims, copy=True):
        """ stack all """

        if copy:
            return CoordinateGroup([c.stack(stack_dims, copy=True) for c in self._items])
        else:
            for c in self._items:
                c.stack(stack_dims)
            return self

    def unstack(self, copy=True):
        """ unstack all"""
        if copy:
            return CoordinateGroup([c.unstack(stack_dims, copy=True) for c in self._items])
        else:
            for c in self._items:
                c.unstack(stack_dims)
            return self

    def intersect(self, other, coord_ref_sys=None, pad=1, ind=False):
        return CoordinateGroup([c.intersect(other) for c in self._items])
    
    @property
    def dims(self):
        """ unordered (set) and unstacked """
        if len(self._items) == 0:
            return {}
        return set(self._items[0].dims_map)

    def add_unique(self, other):
        return self._add(other, unique=True)
    
    def __add__(self, other):
        return self._add(other)
    
    def _add(self, other, unique=False):
        if unique:
            raise NotImplementedError("TODO")

        if isinstance(other, Coordinate):
            # TODO should this concat, fail, or do something else?
            # items = self._items + [other]
            raise NotImplementedError("TODO")
        elif isinstance(other, CoordinateGroup):
            items = self._items + g._items
        else:
            raise TypeError("Cannot add '%s', only BaseCoordinate" % type(c))
        
        return CoordinateGroup(self._items + [other])

    def __iadd__(self, other):
        if isinstance(other, Coordinate):
            # TODO should this append, fail, or do something else?
            # TypeError("Cannot add individual Coordinate, use 'append'")
            # self._items.append(other)
            raise NotImplementedError("TODO")
        elif isinstance(other, CoordinateGroup):
            self._items += g._items
        else:
            raise TypeError("Cannot add '%s' to CoordinateGroup" % type(c))

        return self

    def iterchunks(self, shape, return_slice=False):
        raise NotImplementedError("TODO")

    @property
    def latlon_bounds_str(self):
        # TODO should this be a single latlon bounds or a list of bounds?
        raise NotImplementedError("TODO")
    
# =============================================================================
# helper functions
# =============================================================================

def stack_coords(coords, dims_map):
    """
    Stack the coordinates according to the given dims_map.

    Arguments
    ---------
    coords : dict
        unstacked coords dictionary in the form `dim:BaseCoord` # TODO commondoc
    dims_map : dict
        maps single dimensions to potentially stacked dimension keys #TODO commondoc

    Returns
    -------
    stacked_coords : OrderedDict
        stacked coords dictionary # TODO commondoc

    Raises
    ------
    ValueError
        If the stacked dimensions do not have a matching size.
    """

    
    #check keys
    for k in coords:
        if k not in dims_map:
            raise ValueError("Missing dimension '%s' in 'dims_map'" % k)

    for k in dims_map:
        if k not in coords:
            raise ValueError("Unexpected dimension '%s' in 'dims_map'" % k)

    # make stacked coords dict
    stacked_coords = OrderedDict()
    for key in dims_map.values():
        if key in stacked_coords:
            continue

        if key in dims_map:
            stacked_coords[key] = coords[key]

        else:
            cs = [coords[dim] for dim in _unstack_dims(key)]
            
            # validate stacked dimensions sizes
            sizes = [c.size for c in cs]
            if any(size != sizes[0] for size in sizes):
                raise ValueError("Stacked dimensions size mismatch in '%s' (%s must all match)" % (key, sizes))
                
            stacked_coords[key] = cs

    return stacked_coords

def convert_xarray_to_podpac(xcoord):
    """
    Convert an xarray coord to podpac Coordinate.
    
    Parameters
    ----------
    xcoord : DataArrayCoordinates
        xarray coord attribute to convert
    
    Returns
    -------
    coord : Coordinate
        podpact Coordinate object
    
    Raises
    ------
    TypeError
        Description
    """

    if not isinstance(xcoord, DataArrayCoordinates):
        raise TypeError("input must be an xarray DataArrayCoordinate, not '%s'" % type(xcoord))

    d = OrderedDict()
    for dim in xcoord.dims:
        c = xcoord[dim].data
        if c.dtype.names:
            # extract/transpose stacked coordinates from structured array
            d[dim] = [c[_dim] for _dim in c.dtype.names]
        else:
            d[dim] = c

    return Coordinate(d)

def _make_coord(val, **kw):
    if isinstance(val, BaseCoord):
        return val
    
    elif isinstance(val, tuple) and len(val) == 3:
        if isinstance(val[2], (int, np.long, np.integer)) and not isinstance(val[2], (np.timedelta64)):
            return coord_linspace(*val, **kw)
        else:
            return UniformCoord(*val, **kw)
    else:
        try:
            return MonotonicCoord(val, **kw)
        except:
            return Coord(val, **kw)

def coordinate(order=None, coord_ref_sys="WGS84", segment_position=0.5, ctype='segment', **kwargs):
    """
    Create multidimensional coordinates in a Coordinate object.

    Arguments
    ----------
    order : list
        The order of the dimensions. Ignored if coords is provided and in
        Python >=3.6 where the order of kwargs is preserved. Required in
        Python <3.6 if providing more than one dimension via keyword arguments.
    ctype : str
        Coordinate type, passed to individual coord objects during
        initialization.
    segment_position : float
        Segment position, passed to individual coord objects during
        initialization.
    coord_ref_sys : str
        Coordinate reference system, passed to individual coord objects
        during initialization.
    **kwargs
        Coordinate initialization values.

        <dim>
            Coordinate initialization values for the given unstacked dimension.
            
            Valid dimensions are::
             * 'time'
             * 'lat'
             * 'lon'
             * 'alt'
            
            Valid values are::
             * an explicit BaseCoord object (Coord, UniformCoord, etc) # TODO JXM
             * a single number/datetime value
             * an array-like object of numbers/datetimes
             * tuple in the form (start, stop, size) where size is an integer
             * tuple in the form (start, stop, step) where step is a float or timedelta
        
        <dims>
            A tuple of coordinate initialization values corresponding to the
            given stacked dimensions. The keyword is split on underscores
            into dimensions and may contain any number of dimensions (e.g.
            lat_lon is split into lat and lon). The value must be a list/tuple
            with elements corresponding to the split dimensions in the keyword.
            Valid values for the elements are the same as for single unstacked
            dimensions except in the case of uniform coordinates.

            Stacked uniform coordinates are in the form

              ``dim1_dim2=((start1, start2), (stop1, stop2), size)``
              ``dim1_..._dimN=((start1, ..., startN), (stop1, ..., stopN), size)``

            where size is an integer defining the common size.

    Returns
    -------
    c : Coordinate
        Multidimensional coordinates

    Raises
    ------
    TODO

    >>> coordinate(lat=1)                                       # doctest: +SKIP
    >>> coordinate(lat_lon=((1, 2)))                            # doctest: +SKIP
    >>> coordinate(lat=(49.1, 50.2, 100))                       # doctest: +SKIP
    >>> coordinate(lat_lon=((49.1, -120), (50.2, -122), 100))   # doctest: +SKIP
    >>> coordinate(lat=(49.1, 50.1, 0.1))                       # doctest: +SKIP
    >>> coordinate(lat=np.array([50, 50.1, 50.4, 50.8, 50.9]))  # doctest: +SKIP
    >>> coordinate(lat_lon=([50, 50.1, 50.4, 50.8, 50.9], [-120, -125, -126, -127, -130])) # doctest: +SKIP
    """

    if sys.version < '3.6' and order is None and len(kwargs) > 1:
        raise TypeError("Need to specify the order of the using 'order'.")

    # get ordered kwargs
    if order is not None:
        for k in order:
            if k not in kwargs:
                raise ValueError("Unexpected dimension '%s' in 'order'" % k)
        for k in kwargs:
            if k not in order:
                raise ValueError("Unexpected keyword argument '%s' (this dimension not found in 'order')" % k)
        
        kwargs = OrderedDict([(k, kwargs[k]) for k in order])
        
    # properties
    kw = {'ctype': ctype, 'coord_ref_sys': coord_ref_sys, 'segment_position': segment_position}

    # parse coords
    coords = OrderedDict()
    for key, val in kwargs.items():
        dims = _unstack_dims(key)
        if len(dims) == 1:
            coords[key] = _make_coord(val, **kw)
        elif isinstance(val, tuple) and len(val) == len(dims) + 1:
            coords[key] = tuple(coord_linspace(start, stop, val[-1], **kw) for (start, stop) in val[:-1])
        else:
            coords[key] = tuple(_make_coord(v, **kw) for v in val)

    return Coordinate(coords, **kw)

# =============================================================================
# TODO convert to unit testing
# =============================================================================

if __name__ == '__main__':
    
    coord = coord_linspace(1, 10, 10)
    coord_cent = coord_linspace(4, 7, 4)
    
    c = coordinate(lat=coord, lon=coord, order=('lat', 'lon'))
    c_s = coordinate(lat_lon=(coord, coord))
    c_cent = coordinate(lat=coord_cent, lon=coord_cent, order=('lat', 'lon'))
    c_cent_s = coordinate(lon_lat=(coord_cent, coord_cent))

    print(c.intersect(c_cent))
    print(c.intersect(c_cent_s))
    print(c_s.intersect(c_cent))
    print(c_s.intersect(c_cent_s))
    
    try:
        c = coordinate(lat_lon=((0, 1, 10), (0, 1, 11)))
    except ValueError as e:
        print(e)
    else:
        raise Exception('expected exception')
    
    c = coordinate(lat_lon=((0, 1, 10), (0, 1, 10)), time=(0, 1, 2), order=('lat_lon', 'time'))
    c2 = coordinate(lat_lon=((0.5, 1.5, 15), (0.1, 1.1, 15)))
    print (c.shape)
    print (c.unstack().shape)
    print (c.get_shape(c2))
    print (c.get_shape(c2.unstack()))
    print (c.unstack().get_shape(c2))
    print (c.unstack().get_shape(c2.unstack()))
    
    # c = coordinate(lat=(0, 1, 10), lon=(0, 1, 10), time=(0, 1, 2), order=('lat', 'lon', 'time'))
    # print(c.stack(['lat', 'lon']))
    # try:
    #     c.stack(['lat','time'])
    # except Exception as e:
    #     print(e)
    # else:
    #     raise Exception('expected exception')

    # try:
    #     c.stack(['lat','time'], copy=False)
    # except Exception as e:
    #     print(e)
    # else:
    #     raise Exception('expected exception')

    # c = coordinate(lat_lon=((0, 1, 10), (0, 1, 10)), time=(0, 1, 2), order=('lat_lon', 'time'))
    # c2 = coordinate(lat_lon=((0.5, 1.5, 15), (0.1, 1.1, 15)))

    # print (c.replace_coords(c2))
    # print (c.replace_coords(c2.unstack()))
    # print (c.unstack().replace_coords(c2))
    # print (c.unstack().replace_coords(c2.unstack()))
    
    # c = UniformCoord(1, 10, 2)
    # np.testing.assert_equal(c.coordinates, np.arange(1., 10, 2))
    
    # c = UniformCoord(10, 1, -2)
    # np.testing.assert_equal(c.coordinates, np.arange(10., 1, -2))

    # try:
    #     c = UniformCoord(10, 1, 2)
    #     raise Exception
    # except ValueError as e:
    #     print(e)
    
    # try:
    #     c = UniformCoord(1, 10, -2)
    #     raise Exception
    # except ValueError as e:
    #     print(e)
    
    # c = UniformCoord('2015-01-01', '2015-01-04', '1,D')
    # c2 = UniformCoord('2015-01-01', '2015-01-04', '2,D')
    
    # print('Done')
