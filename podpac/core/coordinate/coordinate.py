from __future__ import division, unicode_literals, print_function, absolute_import

import copy
import sys
import itertools

import numpy as np
import traitlets as tl
from collections import OrderedDict
from xarray.core.coordinates import DataArrayCoordinates

from podpac.core.coordinate.coord import (
    BaseCoord, Coord, MonotonicCoord, UniformCoord, coord_linspace)

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
    dims_map : dict
        Maps unstacked dimensions to the correct (potentiall stacked) dimension'
    kwargs
    is_stacked
    shape
    dims
    coords
    gdal_crs

    >>> Coordinate(lat=1)
    Coordinate
            lat: MonotonicCoord: Bounds[1.0, 1.0], N[1], ctype["segment"]

    >>> Coordinate(lat_lon=((1, 2)))
    Coordinate
            lat_lon[lat]: MonotonicCoord: Bounds[1.0, 1.0], N[1], ctype["segment"]
            lat_lon[lon]: MonotonicCoord: Bounds[2.0, 2.0], N[1], ctype["segment"]

    >>> Coordinate(lat=(49.1, 50.2, 100))
    Coordinate
            lat: UniformCoord: Bounds[49.1, 50.2], N[100], ctype["segment"]

    >>> Coordinate(lat_lon=((49.1, -120), (50.2, -122), 100))
    Coordinate
            lat_lon[lat]: UniformCoord: Bounds[-120.0, 49.1], N[100], ctype["segment"]
            lat_lon[lon]: UniformCoord: Bounds[-121.99999999999999, 50.2], N[100], ctype["segment"]

    >>> Coordinate(lat=(49.1, 50.1, 0.1))
    Coordinate
            lat: UniformCoord: Bounds[49.1, 50.1], N[11], ctype["segment"]
    
    >>> Coordinate(lat=np.array([50, 50.1, 50.4, 50.8, 50.9]))
    Coordinate
            lat: MonotonicCoord: Bounds[50.0, 50.9], N[5], ctype["segment"]
    
    >>> Coordinate(lat_lon=([50, 50.1, 50.4, 50.8, 50.9],
                            [-120, -125, -126, -127, -130]))
    Coordinate
            lat_lon[lat]: MonotonicCoord: Bounds[50.0, 50.9], N[5], ctype["segment"]
            lat_lon[lon]: MonotonicCoord: Bounds[-130.0, -120.0], N[5], ctype["segment"]
    """

    # default val set in constructor
    ctype = tl.Enum(['segment', 'point'])  
    segment_position = tl.Float()  # default val set in constructor
    coord_ref_sys = tl.CUnicode()
    _coords = tl.Instance(OrderedDict)
    dims_map = tl.Dict()

    def __init__(self, coords=None, order=None, coord_ref_sys="WGS84", 
                       segment_position=0.5, ctype='segment', **kwargs):
        """
        Initialize a multidimensional coords object.

        Keyword Arguments
        -----------------
        coords : OrderedDict
            explicit coords mapping; if provided, keyword arguments for
            individual dimensions will be ignored.
        <dim>
            Coordinate initialization values for the given unstacked dimension.
            Valid dimensions are::
             * 'time'
             * 'lat'
             * 'lon'
             * 'alt'
            Valid values are::
             * an explicit BaseCoord object (Coord, UniformCoord, etc)
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
            dimensions except in the case of uniform coordinates. Stacked
            uniform coordinates are in the form

              ``dim1_dim2=((start1, stop1), (start2, stop2), size)

            where size is an integer defining the common size.
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

        """

        if coords is None:
            if sys.version < '3.6':
                if order is None:
                    if len(kwargs) > 1:
                        raise TypeError(
                            "Need to specify the order of the coordinates "
                            "using 'order'.")
                    else:
                        order = kwargs.keys()
                
                coords = OrderedDict()
                for k in order:
                    if k not in kwargs:
                        raise ValueError("Unexpected dimension '%s' in 'order'" % k)
                    coords[k] = kwargs[k]

                for k in kwargs:
                    if k not in order:
                        raise ValueError("Missing dimension '%s' in 'order'" % k)

            else:
                coords = OrderedDict(kwargs)
        elif not isinstance(coords, OrderedDict):
            raise TypeError(
                "coords must be an OrderedDict, not %s" % type(coords))

        self.dims_map = self.get_dims_map(coords)
        _coords = self.unstack_dict(coords)
        
        kw = {
            'ctype': ctype,
            'coord_ref_sys': coord_ref_sys,
            'segment_position':segment_position}

        for key, val in _coords.items():
            if isinstance(val, BaseCoord):
                continue

            # make coord helper
            if isinstance(val, tuple):
                if (isinstance(val[2], (int, np.long, np.integer)) and
                    not isinstance(val[2], (np.timedelta64))):
                    _coords[key] = coord_linspace(*val, **kw)
                else:
                    _coords[key] = UniformCoord(*val, **kw)
            else:
                try:
                    _coords[key] = MonotonicCoord(val, **kw)
                except:
                    _coords[key] = Coord(val, **kw)

        super(Coordinate, self).__init__(_coords=_coords, **kw)
    
    def __repr__(self):
        rep = str(self.__class__.__name__)
        for d in self._coords:
            d2 = self.dims_map[d]
            if d2 != d:
                d2 = d2 + '[%s]' % d
            rep += '\n\t{}: '.format(d2) + str(self._coords[d])
        return rep
    
    def __getitem__(self, item):
        """
        TODO
        
        Arguments
        ---------
        item : str

        Returns
        -------
        TODO
        """

        return self._coords[item]
    
    @tl.validate('_coords')
    def _coords_validate(self, proposal):
        seen_dims = []
        stack_dims = {}
        for key in proposal['value']:
            self._validate_dim(key, seen_dims)
            val = proposal['value'][key]
            self._validate_val(val, key, proposal['value'].keys())
            if key not in self.dims_map.values():  # stacked dim
                if self.dims_map[key] not in stack_dims:
                    stack_dims[self.dims_map[key]] = val.size
                elif val.size != stack_dims[self.dims_map[key]]:
                    raise ValueError(
                        "Stacked dimensions size mismatch for '%s' in '%s' "
                        "(%d != %d)" % (key, self.dims_map[key], val.size,
                                        stack_dims[self.dims_map[key]]))
        return proposal['value']
        
    def _validate_dim(self, dim, seen_dims):
        if dim not in self._valid_dims:
            raise ValueError("Invalid dimension '%s', expected one of %s" % (
                dim, self._valid_dims))
        if dim in seen_dims:
            raise ValueError("The dimension '%s' cannot be repeated." % dim)
        seen_dims.append(dim)
    
    def _validate_val(self, val, dim='', dims=[]):
        if not isinstance(val, BaseCoord):
            raise TypeError("Invalid coord type '%s'" % val.__class__.__name__)
   
    def get_dims_map(self, coords=None):
        """
        TODO

        Arguments
        ---------
        coords : dict, optional
            TODO

        Returns
        -------
        stacked_coords : OrderedDict
            TODO
        """

        if coords is None:
            coords = self.coords

        stacked_coords = OrderedDict()
        for c in coords:
            if '_' in c:
                for cc in c.split('_'):
                    stacked_coords[cc] = c       
            else:
                stacked_coords[c] = c 
        return stacked_coords        
    
    def unstack_dict(self, coords=None, check_dim_repeat=False):
        """
        TODO

        Arguments
        ---------
        coords : dict, optional
            TODO
        check_dim_repeat : boolean
            TODO

        Returns
        -------
        new_crds : OrderedDict
            TODO
        """

        if coords is None: 
            coords = self._coords

        dims_map = self.get_dims_map(coords)
       
        new_crds = OrderedDict()
        seen_dims = []
        for key, val in coords.items():
            if key not in self.dims_map:  # stacked
                keys = key.split('_')
                for i, k in enumerate(keys):
                    new_crds[k] = val[i]

                    # parse uniform coords tuple and append size
                    if isinstance(val[i], tuple):
                        if len(val) != len(keys) + 1:
                            raise ValueError("missing size for stacked uniform coordinates")
                        if (not isinstance(val[-1], (int, np.long, np.integer)) or
                            isinstance(val[-1], (np.timedelta64))):
                            raise TypeError("invalid size for stacked uniform coordinates \
                                             (expected integer, not '%s')" % type(val[-1]))
                        new_crds[k] += (val[-1],)
                    
                    if check_dim_repeat and k in seen_dims:
                        raise ValueError(
                            "The dimension '%s' cannot be repeated." % dim)
                    seen_dims.append(k)
            else:
                new_crds[key] = val
                if check_dim_repeat and key in seen_dims:
                    raise ValueError(
                        "The dimension '%s' cannot be repeated." % key)
                seen_dims.append(key)

        return new_crds

    def stack_dict(self, coords=None, dims_map=None):
        """
        TODO

        Arguments
        ---------
        coords : OrderedDict, optional
            TODO
        dims_map: OrderedDict, optional
            TODO

        Returns
        -------
        stacked_coords : OrderedDict
            TODO
        """

        if coords is None: 
            coords = self._coords
        if dims_map is None:
            dims_map = self.dims_map

        stacked_coords = OrderedDict()
        for key, val in dims_map.items():
            if val in stacked_coords:
                temp = stacked_coords[val]
                if not isinstance(temp, list):
                    temp = [temp]
                temp.append(coords[key])
                stacked_coords[val] = temp
            else:
                stacked_coords[val] = coords[key]
        return stacked_coords
   
    @property
    def is_stacked(self):
        """
        True if any of the coordinates are stacked; False if none are stacked.
        """

        for k, v in self.dims_map.items():
            if k != v:
                return True
        return False
   
    def stack(self, stack_dims, copy=True):
        """
        Stack the coordinates in of given dimensions.

        Arguments
        ---------
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

        stack_dim = '_'.join(stack_dims)
        dims_map = {k:v for k,v in self.dims_map.items()}
        for k in stack_dims:
            dims_map[k] = stack_dim
        stack_dict = self.stack_dict(self._coords.copy(), dims_map=dims_map)
        if copy:
            return self.__class__(coords=stack_dict, **self.kwargs)
        else:
            # Check for correct dimensions
            tmp = self.dims_map
            self.dims_map = dims_map
            try:
                self._coords_validate({'value': self._coords})
            except Exception as e:
                self.dims_map = tmp
                raise(e)
            
            return self

    def unstack(self, copy=True):
        """
        Unstack the coordinates of all of the dimensions.

        Arguments
        ---------
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
            self.dims_map = {v:v for v in self.dims_map}
            return self

    def intersect(self, other, coord_ref_sys=None, pad=1, ind=False):
        """
        TODO

        Arguments
        ---------
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
            
            intersect = self._coords[dim].intersect(
                other._coords[dim], coord_ref_sys, ind=ind or self.is_stacked,
                pad=spad)
            
            if ind or self.is_stacked:
                I.append(intersect)
            else:
                d[dim] = intersect
        
        if ind or self.is_stacked:
            if not self.is_stacked:
                return I

            # Need to handle the stacking
            I2 = [np.ones(s, bool) for s in self.shape]
            for i, d in enumerate(self.dims):
                parts = d.split('_')
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
                coords[k] = self._coords[k].coordinates[I2[i]]
            coords = self.stack_dict(coords)
            return Coordinate(coords, **self.kwargs)
        else:
            coords = self.stack_dict(d)
            return Coordinate(coords, **self.kwargs)
    
    @property
    def kwargs(self):
        '''
        Dictionary specifying the coordinate properties.
        '''

        return {
                'coord_ref_sys': self.coord_ref_sys,
                'segment_position': self.segment_position,
                'ctype': self.ctype
                }
    
    def replace_coords(self, other, copy=True):
        '''
        TODO

        Arguments
        ---------
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

        if copy:
            coords = self._coords.copy()
            dims_map = self.dims_map.copy()
        else:
            coords = self._coords
            dims_map = self.dims_map
            
        for c in coords:
            if c in other._coords:
                coords[c] = other._coords[c]
                dims_map[c] = other.dims_map[c]
        
        if copy:
            stack_dict = self.stack_dict(coords, dims_map=dims_map)
            return self.__class__(coords=stack_dict)
        else:
            return self   
   
    def drop_dims(self, *args):
        """
        Remove the given dimensions from the Coordinate.

        Arguments
        ---------
        <dim> : str
            Dimension to drop.
        """

        split_dims = []
        for arg in args:
            if arg not in self._coords:
                continue
            del self._coords[arg]
            if self.dims_map[arg] == arg:
                del self.dims_map[arg]
            else:
                split_dims += self.dims_map[arg].split('_')
        if split_dims:
            self.drop_dims(*split_dims) 
 
    def get_shape(self, other_coords=None):
        """
        Coordinates shape, corresponding to dims attribute.

        Arguments
        ---------
        other_coords : Coordinate, optional
            TODO

        Returns
        -------
        shape : tuple
            Number of coordinates in each dimension.
        """

        if other_coords is None:
            other_coords = self
        # Create shape for each dimension
        shape = []
        seen_dims = []
        self_seen_dims = []
        for k in self._coords:
            if k in other_coords._coords:
                shape.append(other_coords._coords[k].size)
                # Remove stacked duplicates
                if other_coords.dims_map[k] in seen_dims:
                    shape.pop()
                else:
                    seen_dims.append(other_coords.dims_map[k])
            else:
                shape.append(self._coords[k].size)
                # Remove stacked duplicates
                if self.dims_map[k] in self_seen_dims:
                    shape.pop()
                else:
                    self_seen_dims.append(self.dims_map[k])
                

        return tuple(shape)
        
    @property
    def shape(self):
        """ Coordinates shape, corresponding to dims attribute. """

        return self.get_shape()
    
    @property
    def delta(self):
        """ to be deprecated """

        try:
            return np.array([c.delta for c in self._coords.values()]).squeeze()
        except ValueError as e:
            return np.array([c.delta for c in self._coords.values()], 
                    object).squeeze()
    
    @property
    def dims(self):
        """ Coordinates dimensions. """

        dims = []
        for v in self.dims_map.values():
            if v not in dims:
                dims.append(v)
        return dims
    
    @property
    def coords(self):
        """ TODO """

        crds = OrderedDict()
        for k in self.dims:
            if k in self.dims_map:  # not stacked
                crds[k] = self._coords[k].coordinates
            else:
                coordinates = [self._coords[kk].coordinates 
                               for kk in k.split('_')]
                dtype = [(str(kk), coordinates[i].dtype) 
                         for i, kk in enumerate(k.split('_'))]
                n_coords = len(coordinates)
                s_coords = len(coordinates[0])
                crds[k] = np.array([[tuple([coordinates[j][i]
                                     for j in range(n_coords)])] 
                                   for i in range(s_coords)],
                    dtype=dtype).squeeze()
        return crds
    
    #@property
    #def gdal_transform(self):
        #if self['lon'].regularity == 'regular' \
               #and self['lat'].regularity == 'regular':
            #lon_bounds = self['lon'].area_bounds
            #lat_bounds = self['lat'].area_bounds
        
            #transform = [lon_bounds[0], self['lon'].delta, 0,
                         #lat_bounds[0], 0, -self['lat'].delta]
        #else:
            #raise NotImplementedError
        #return transform
    
    @property
    def gdal_crs(self):
        """ GDAL coordinate reference system. """

        crs = {'WGS84': 'EPSG:4326',
               'SPHER_MERC': 'EPSG:3857'}
        return crs[self.coord_ref_sys.upper()]
    
    def add_unique(self, other):
        """
        Concatenate coordinates, skipping duplicates.

        Arguments
        ---------
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

        Arguments
        ---------
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
            raise TypeError(
                "Unsupported type '%s', can only add Coordinate object" % (
                    other.__class__.__name__))
        new_coords = copy.deepcopy(self._coords)
        dims_map = self.dims_map
        for key in other._coords:
            if key in self._coords:
                if dims_map[key] != other.dims_map[key]:
                    raise ValueError(
                        "Cannot add coordinates with different stacking. "
                        "%s != %s." % (dims_map[key], other.dims_map[key])
                    )
                if np.all(np.array(self._coords[key].coords) !=
                        np.array(other._coords[key].coords)) or not unique:
                    new_coords[key] = self._coords[key] + other._coords[key]
            else:
                dims_map[key] = other.dims_map[key]
                new_coords[key] = copy.deepcopy(other._coords[key])
        return self.__class__(coords=self.stack_dict(new_coords, dims_map))

    def iterchunks(self, shape, return_slice=False):
        """
        TODO

        Arguments
        ---------
        shape : tuple
            TODO
        return_slice : boolean, optional
            Return slice in addition to Coordinate chunk.

        Yields
        ------
        l : slice
            If return_slice=True, slice for this Coordinate chunk.
        coords : Coordinate
            A Coordinate object with one chunk of the coordinates.
        """
        
        # TODO assumes the input shape dimension and order matches
        # TODO replace self[k].coords[slc] with self[k][slc] (and implement the slice)

        slices = [
            map(lambda i: slice(i, i+n), range(0, m, n))
            for m, n
            in zip(self.shape, shape)]

        for l in itertools.product(*slices):
            kwargs = {k:self.coords[k][slc] for k, slc in zip(self.dims, l)}
            kwargs['order'] = self.dims
            coords = Coordinate(**kwargs)
            if return_slice:
                yield l, coords
            else:
                yield coords

    def transpose(self, *dims, **kwargs):
        """
        Transpose (re-order) the Coordinate dimensions.

        Arguments
        ---------
        *dims : str, optional
            Reorder dims to this order. By default, reverse the dims.
        
        Keyword Arguments
        -----------------
        in_place : boolean, optional
            If False, return a new, transposed Coordinate object (default).
            If True, transpose the dimensions in-place.

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
            kwargs = coords
            kwargs.update(self.kwargs)
            return Coordinate(order=dims, **kwargs)

    @property
    def latlon_bounds_str(self):
        if 'lat' in self._coords and 'lon' in self._coords:
            return '%s_%s_x_%s_%s' % (
                self['lat'].bounds[0],
                self['lon'].bounds[0],
                self['lat'].bounds[1],
                self['lon'].bounds[1]
            )
        else:
            return 'NA'

class CoordinateGroup(BaseCoordinate):
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
                raise ValueError(
                    "Mismatching dims: '%s != %s" % (dims, set(g.dims)))

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
            raise IndexError(
                "invalid CoordinateGroup index type '%s'" % type(key))

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return self._items.__iter__()

    def append(self, c):
        if not isinstance(c, Coordinate):
            raise TypeError(
                "Can only append Coordinate objects, not '%s'" % type(c))
        
        self._items.append(c)
   
    def stack(self, stack_dims, copy=True):
        """ stack all """

        if copy:
            return CoordinateGroup(
                [c.stack(stack_dims, copy=True) for c in self._items])
        else:
            for c in self._items:
                c.stack(stack_dims)
            return self

    def unstack(self, copy=True):
        """ unstack all"""
        if copy:
            return CoordinateGroup(
                [c.unstack(stack_dims, copy=True) for c in self._items])
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

def convert_xarray_to_podpac(xcoord):
    """
    Convert an xarray coord to podpac Coordinate.

    Arguments
    ---------
    xcoord : DataArrayCoordinates
        xarray coord attribute to convert

    Returns
    -------
    coord : Coordinate
        podpact Coordinate object
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

# =============================================================================
# TODO convert to unit testing
# =============================================================================

if __name__ == '__main__': 
    
    coord = coord_linspace(1, 10, 10)
    coord_cent = coord_linspace(4, 7, 4)
    
    c = Coordinate(lat=coord, lon=coord, order=('lat', 'lon'))
    c_s = Coordinate(lat_lon=(coord, coord))
    c_cent = Coordinate(lat=coord_cent, lon=coord_cent, order=('lat', 'lon'))
    c_cent_s = Coordinate(lon_lat=(coord_cent, coord_cent))

    print(c.intersect(c_cent))
    print(c.intersect(c_cent_s))
    print(c_s.intersect(c_cent))
    print(c_s.intersect(c_cent_s))
    
    try:
        c = Coordinate(lat_lon=((0, 1, 10), (0, 1, 11)))
    except ValueError as e:
        print(e)
    else:
        raise Exception('expceted exception')
    
    c = Coordinate(lat_lon=((0, 1, 10), (0, 1, 10)), time=(0, 1, 2), order=('lat_lon', 'time'))
    c2 = Coordinate(lat_lon=((0.5, 1.5, 15), (0.1, 1.1, 15)))
    print (c.shape)
    print (c.unstack().shape)
    print (c.get_shape(c2))
    print (c.get_shape(c2.unstack()))
    print (c.unstack().get_shape(c2))
    print (c.unstack().get_shape(c2.unstack()))
    
    c = Coordinate(lat=(0, 1, 10), lon=(0, 1, 10), time=(0, 1, 2), order=('lat', 'lon', 'time'))
    print(c.stack(['lat', 'lon']))
    try:
        c.stack(['lat','time'])
    except Exception as e:
        print(e)
    else:
        raise Exception('expected exception')

    try:
        c.stack(['lat','time'], copy=False)
    except Exception as e:
        print(e)
    else:
        raise Exception('expected exception')

    c = Coordinate(lat_lon=((0, 1, 10), (0, 1, 10)), time=(0, 1, 2), order=('lat_lon', 'time'))
    c2 = Coordinate(lat_lon=((0.5, 1.5, 15), (0.1, 1.1, 15)))

    print (c.replace_coords(c2))
    print (c.replace_coords(c2.unstack()))
    print (c.unstack().replace_coords(c2))
    print (c.unstack().replace_coords(c2.unstack()))  
    
    c = UniformCoord(1, 10, 2)
    np.testing.assert_equal(c.coordinates, np.arange(1., 10, 2))
    
    c = UniformCoord(10, 1, -2)
    np.testing.assert_equal(c.coordinates, np.arange(10., 1, -2))    

    try:
        c = UniformCoord(10, 1, 2)
        raise Exception
    except ValueError as e:
        print(e)
    
    try:
        c = UniformCoord(1, 10, -2)
        raise Exception
    except ValueError as e:
        print(e)
    
    c = UniformCoord('2015-01-01', '2015-01-04', '1,D')
    c2 = UniformCoord('2015-01-01', '2015-01-04', '2,D')
    
    print('Done')
