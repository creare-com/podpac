from __future__ import division, unicode_literals, print_function, absolute_import

import numbers
import copy
import sys
import itertools

from six import string_types

import xarray as xr
import numpy as np
import traitlets as tl
from collections import OrderedDict


from podpac.core.units import Units
from podpac.core.utils import cached_property, clear_cache

class CoordinateException(Exception):
    pass

class Coord(tl.HasTraits):
    """
    Regular, specified
    stacked, unstacked
    independent, dependent
    """
    units = Units(allow_none=True, default_value=None)
    
    ctype = tl.Enum(['segment', 'point', 'fence', 'post'], default_value='segment',
                   help="Default is 'segment'."
                   "Indication of what coordinates type. "
                   "This is either a single point ('point' or 'post'), or it"
                   " is the whole segment between this coordinate and the next"
                   " ('segment', 'fence'). ")
    
    segment_position = tl.Float(default_value=0.5,
                                help="Default is 0.5. Where along a segment is"
                                "the coordinate specified. 0 <= segment <= ."
                                " For example, if segment=0, the coordinate is"
                                " specified at the left-most point of the line"
                                " segement connecting the current coordinate"
                                " and the next coordinate. If segment=0.5, "
                                " then the coordinate is specified at the "
                                " center of this line segement.")
    
    extents = tl.List(allow_none=True, default_value=None, 
                      help="When specifying irregular coordinates, set the "
                      "bounding box (extents) of the grid in case ctype is "
                      " 'segment' or 'fence'")
    
    @tl.validate('segment_position')
    def _segment_position_validate(self, proposal):
        if proposal['value'] <= 1 and proposal['value'] >= 0:
            return proposal["value"]
        else:
            raise CoordinateException("Coordinate dimension '" + self.dim + \
            "' must be in the segment position of [0, 1]")
    
    def __repr__(self):
        rep = 'Coord: Bounds[{min}, {max}],' + \
            ' N[{}], ctype["{}"]'.format(self.size, self.ctype)
        if self.regularity in ['single', 'regular', 'irregular']:
            rep = rep.format(min=self.bounds[0], max=self.bounds[1])       
        return rep
    
    def __add__(self, other):
        """ Should be able to add two coords together in some situations
        Although I'm really not sure about this function... may be a mistake
        """
        raise NotImplementedError()

    def __init__(self, *args, **kwargs):
        """
        bounds is for fence-specification with non-uniform coordinates
        """
        if kwargs.get('coords') is None:
            kwargs['coords'] = args
        super(Coord, self).__init__(**kwargs)

    @property
    def area_bounds(self):
        extents = copy.deepcopy(self.bounds)
        if self.ctype in ['fence', 'segment']:
            if self.regularity in ['dependent', 'irregular'] \
                    and self.extents:
                extents = self.extents
            elif self.regularity != 'single':
                p = self.segment_position
                expands = np.array([-p, 1 - p])[:, None] * self.delta[None, :]
                # for stacked coodinates
                extents += expands.reshape(extents.shape)
        return extents

    @tl.observe('extents', 'ctype', 'segment_position')
    def _clear_bounds_cache(self, change):
        clear_cache(self, change, ['bounds'])
        
    @tl.observe('coords')
    def _clear_cache(self, change):
        clear_cache(self, change, ['coords', 'bounds', 'delta'])
        
    @property
    def kwargs(self):
        kwargs = {'units': self.units,
                  'ctype': self.ctype,
                  'segment_position': self.segment_position,
                  'extents': self.extents
        }
        return kwargs
    
    coords = tl.Any()
    @tl.validate("coords")
    def _coords_validate(self, proposal):
         raise NotImplementedError()

    @property
    def regularity(self):
        raise NotImplementedError() 

    _cached_bounds = tl.Instance(np.ndarray, allow_none=True)    
    @property
    def bounds(self):
        raise NotImplementedError()
        
    _cached_delta = tl.Instance(np.ndarray, allow_none=True) 
    @property
    def delta(self):
        raise NotImplementedError()
 
    _cached_coords = tl.Any(default_value=None, allow_none=True)
    @property
    def coordinates(self):
        raise NotImplementedError()

    @property
    def size(self):
        raise NotImplementedError()

    def intersect_check(self, other_coords, ind):
        if self.units != other_coord.units:
            raise NotImplementedError("Still need to implement handling of different units")
        if np.all(self.bounds == other_coord.bounds):
            if ind:
                return [0, self.size]
            return self
        
        ibounds = [
            np.maximum(self.bounds[0], other_coord.bounds[0]),
            np.minimum(self.bounds[1], other_coord.bounds[1])        
            ]
        if np.any(ibounds[0] > ibounds[1]):
            if ind:
                return [0, 0]
            else:
                return self.__class__(coords=(self.bounds[0], self.bounds[1], 0)) 
        

    def intersect(self, other_coord, coord_ref_sys=None, pad=1, ind=False):
        """
        Returns an Coord object if ind==False
        Returns a list of start, stop coordinates if ind==True
        """
        raise NotImplementedError()

    @property
    def is_max_to_min(self):
        raise NotImplementedError()


class SingleCoord(Coord):
    coords = tl.Any()
    @tl.validate("coords")
    def _coords_validate(self, proposal):
        if not isinstance(proposal['value'], 
                          (tuple, list, np.ndarray, xr.DataArray, 
                           numbers.Number, string_types, np.datetime64)):
            raise CoordinateException("Coords type not recognized")
        val = proposal['value']
        if hasattr(val, '__len__'):
            if len(val) == 1:
                val = val[0]
            else:
                raise CoordinateException("SingleCoord cannot have multiple values")

        if isinstance(val, string_types):
            val = np.datetime64(val)
        
        return val

    @property
    def regularity(self):
        return 'single' 

    _cached_bounds = tl.Instance(np.ndarray, allow_none=True)    
    @cached_property
    def bounds(self):
        bounds = np.array(
                [self.coords - self.delta, self.coords + self.delta])
        return bounds

    _cached_delta = tl.Instance(np.ndarray, allow_none=True) 
    @cached_property
    def delta(self):
        # Arbitrary
        if isinstance(self.coords, np.datetime64):
            dtype = self.coords - self.coords
            delta = np.array([1], dtype=dtype.dtype)
        else:
            delta = np.atleast_1d(np.sqrt(np.finfo(np.float32).eps))
        return delta
            
             
    _cached_coords = tl.Any(default_value=None, allow_none=True)
    @cached_property
    def coordinates(self):
        return np.atleast1d(self.coords)

    @property
    def size(self):
        return 1 

    def intersect(self, other_coord, coord_ref_sys=None, pad=1, ind=False):
        """
        Returns an Coord object if ind==False
        Returns a list of start, stop coordinates if ind==True
        """
        check = self.intersect_checks(other_coord, ind)
        if check:
            return check

        if ind:
            return [0, 1]
        else:
            return self

    @property
    def is_max_to_min(self):
        return False


class RegularCoord(Coord):
    coords = tl.Any()
    @tl.validate("coords")
    def _coords_validate(self, proposal):
        if not isinstance(proposal['value'], 
                          (tuple, list)):
            raise CoordinateException("Coords type not recognized")
        val = proposal['value']
        if len(val) < 3:
            raise CoordinateException("Need to supply at least three entries "
                "to define a 'regular' coordinate in the form "
                "(start, stop, step) or (start, stop, number).")

        if isinstance(val[0], (int, np.ndarray, np.long)):
            val = (float(val[0]),) + tuple(val[1:])
        elif isinstance(val[0], string_types):
            val = (np.datetime64(val[0]),) + tuple(val[1:])
        if isinstance(val[1], (int, np.ndarray, np.long)):
            val = (val[0], float(val[1])) + tuple(val[2:])
        elif isinstance(val[1], string_types):
            val = (val[0], np.datetime64(val[1])) + tuple(val[2:])
        if isinstance(val[2], string_types):
            a, b = val[2].split(',')
            val = (val[0], val[1], np.timedelta64(int(a), b))

        return val

    @property
    def regularity(self):
       return 'regular'

    _cached_bounds = tl.Instance(np.ndarray, allow_none=True)    
    @cached_property
    def bounds(self):
        return np.array([np.min(self.coords[:2]),
                         np.max(self.coords[:2])]).squeeze()
        
    _cached_delta = tl.Instance(np.ndarray, allow_none=True) 
    @cached_property
    def delta(self):
        if isinstance(self.coords[2], (int, np.integer, np.long)) \
                and not isinstance(self.coords[2], np.timedelta64):
            return np.atleast_1d(((np.array(self.coords[1]) - np.array(self.coords[0]))\
               / (self.coords[2] - 1.) * (1 - 2 * self.is_max_to_min)).squeeze())
        else:
            return np.atleast_1d(np.array(self.coords[2:3]).squeeze())
 
    _cached_coords = tl.Any(default_value=None, allow_none=True)
    @cached_property
    def coordinates(self):
        N = self.size
        if isinstance(self.coords[0], np.datetime64):
            return self.coords[0] + np.arange(0, N) * self.delta
        else:
            return np.linspace(self.coords[0], self.coords[1], N)

    @property
    def size(self):
        if not isinstance(self.coords[2], (int, np.integer, np.long)) or \
                isinstance(self.coords[2], np.timedelta64):
            return  np.round((1 - 2 * self.is_max_to_min) * 
                    (self.coords[1] - self.coords[0]) / self.coords[2]) + 1
        else: #number specified
            return self.coords[2]

    def intersect(self, other_coord, coord_ref_sys=None, pad=1, ind=False):
        """
        Returns an Coord object if ind==False
        Returns a list of start, stop coordinates if ind==True
        """
        check = self.intersect_checks(other_coord, ind)
        if check:
            return check
        
        min_max_i = [np.floor((ibounds[0] - self.bounds[0]) / self.delta),
                     np.ceil((self.bounds[1] - ibounds[1]) / self.delta)]
        if ind:
            if self.is_max_to_min:
                min_max_i = min_max_i[::-1]
            I = [int(min(self.size, max(0, min_max_i[0] - pad))),
                 int(min(self.size, max(0, self.size - min_max_i[1] + pad)))]
            return I
        min_max = [np.maximum(self.bounds[0],
                       self.bounds[0] + max(0, min_max_i[0] - pad) * self.delta),
                   np.minimum(self.bounds[1],
                    self.bounds[1] - max(0, min_max_i[1] - pad) * self.delta)]
        if self.is_max_to_min:
            min_max = min_max[::-1]
        
        coords = min_max + [float(self.delta)]
        new_crd = self.__class__(coords=coords, **self.kwargs)
        return new_crd
        
    @property
    def is_max_to_min(self):
        return self.coords[0] > self.coords[1]


class IrregularCoord(Coord):
    coords = tl.Any()
    @tl.validate("coords")
    def _coords_validate(self, proposal):
        if not isinstance(proposal['value'],
                          (tuple, list, np.ndarray)):
            raise CoordinateException("Coords type not recognized: " + 
                                      str(type(proposal['value'])))

        val = np.array(proposal['value']).squeeze()
        if len(val.shape) > 1:
            raise CoordinateException("Irregular coordinates can only"
                                      " have 1 dimension, not " +
                                      len(val.shape))
        return val
    
    @property
    def regularity(self):
        return 'irregular'

    _cached_bounds = tl.Instance(np.ndarray, allow_none=True)    
    @cached_property
    def bounds(self):
        if isinstance(self.coords[0], np.datetime64):
            return np.array([np.min(self.coords), np.max(self.coords)])
        else:
            return np.array([np.nanmin(self.coords), np.nanmax(self.coords)])
  
        
    _cached_delta = tl.Instance(np.ndarray, allow_none=True) 
    @cached_property
    def delta(self):
        #print("Warning: delta is not representative for irregular coords")
        return np.atleast_1d(np.array(
            (self.coords[-1] - self.coords[0]) / float(self.coords.size) \
            * (1 - 2 * self.is_max_to_min)).squeeze())
                    
 
    _cached_coords = tl.Any(default_value=None, allow_none=True)
    @cached_property
    def coordinates(self):
        return self.coords
        
    @property
    def size(self):
        return self.coords.size

    def intersect(self, other_coord, coord_ref_sys=None, pad=1, ind=False):
        """
        Returns an Coord object if ind==False
        Returns a list of start, stop coordinates if ind==True
        """
        check = self.intersect_checks(other_coord, ind)
        if check:
            return check
        
        b = other_coord.bounds
        inds = np.where((self.coordinates >= (b[0] - self.delta))\
                        & (self.coordinates <= (b[1] + self.delta)))[0]
        if inds.size == 0:
            if ind:
                return [0, 0]
            else:
                return self.__class__(coords=(self.bounds[0], self.bounds[1], 0))                 
        min_max_i = [min(inds), max(inds)]
        #if self.is_max_to_min:
            #min_max_i = min_max_i[::-1]
        lefti = np.maximum(0, min_max_i[0] - pad)
        righti = np.minimum(min_max_i[1] + pad + 1, self.size)
        if ind:
            return [int(lefti), int(righti)]
        new_crd = self.__class__(coords=self.coordinates[lefti:righti], **self.kwargs)
        
        return new_crd

    @property
    def is_max_to_min(self):
        if isinstance(self.coords[0], np.datetime64):
            return self.coords[0] > self.coords[-1]
        else:
            non_nan_coords = self.coords[np.isfinite(self.coords)]
            return non_nan_coords[0] > non_nan_coords[-1]
        
class DependentCoord(Coord):
    coords = tl.Instance(xr.DataArray)
    coord_name = tl.Unicode('')
    @tl.validate("coords")
    def _coords_validate(self, proposal):
        if not isinstance(proposal['value'],
                          (xr.DataArray)):
            raise CoordinateException("Coords must be of type xr.DataArray"
                                      " not " + str(type(proposal['value'])))

        val = proposal['value']
        if len(val.shape) < 2:
            raise CoordinateException("Dependent coordinates need at least "
                                      "2 dimensions.")
        return val

    @property
    def regularity(self):
        return 'dependent'

    _cached_bounds = tl.Instance(np.ndarray, allow_none=True)    
    @cached_property
    def bounds(self):
        dims = [d for d in self.coords.dims]
        return np.array([self.coords.min(dims), self.coords.max(dims)]) 
        
    _cached_delta = tl.Instance(np.ndarray, allow_none=True) 
    @cached_property
    def delta(self):
         return np.array([
                self.coords[1] - self.coords[0]
            ]) * (1 - 2 * self.is_max_to_min).squeeze()
 
    @property
    def coordinates(self):
        return self.coords

    @property
    def size(self):
        return self.coords[self.coord_name].size

    def intersect(self, other_coord, coord_ref_sys=None, pad=1, ind=False):
        """
        Returns an Coord object if ind==False
        Returns a list of start, stop coordinates if ind==True
        """
        check = self.intersect_checks(other_coord, ind)
        if check:
            return check
        
        raise NotImplementedError()

    @property
    def is_max_to_min(self):
        return np.array(self.coords).ravel()[0] > np.array(self.coords).ravel()[-1] 

class GroupCoord(Coord):
    coords = tl.Any()
    @tl.validate("coords")
    def _coords_validate(self, proposal):
         raise NotImplementedError()

    @property
    def regularity(self):
        raise NotImplementedError() 

    _cached_bounds = tl.Instance(np.ndarray, allow_none=True)    
    @property
    def bounds(self):
        raise NotImplementedError()

    _cached_delta = tl.Instance(np.ndarray, allow_none=True) 
    @property
    def delta(self):
        raise NotImplementedError()
 
    _cached_coords = tl.Any(default_value=None, allow_none=True)
    @property
    def coordinates(self):
        raise NotImplementedError()

    @property
    def size(self):
        raise NotImplementedError()

    def intersect(self, other_coord, coord_ref_sys=None, pad=1, ind=False):
        """
        Returns an Coord object if ind==False
        Returns a list of start, stop coordinates if ind==True
        """
        raise NotImplementedError()

    @property
    def is_max_to_min(self):
        raise NotImplementedError()

def make_coord(coords, **kwargs): 
    available_coords = [SingleCoord, RegularCoord, IrregularCoord, 
                        DependentCoord, GroupCoord]
    for ac in available_coords:
        try:
            coord = ac(coords=coords, **kwargs)
        except CoordinateException:
            continue
        except Exception as e:
            print ("Unknown exception: " + str(e))
            continue
        break
    return coord

class Coordinate(tl.HasTraits):
    """
    You can initialize a coordinate like this: 
    # Single number
    c = Coordinate(lat=1) 
    # Single number for stacked coordinate
    c = Coordinate(lat_lon=((1, 2))) 
    # uniformly spaced range (start, stop, number)
    c = Coordinate(lat=(49.1, 50.2, 100) 
    # uniform range for stacked coordinate
    c = Coordinate(lat_lon=((49.1, -120), (50.2, -122), 100) 
    # uniformly spaced steps (start, stop, step)
    c = Coordinate(lat=(49.1, 50.1, 0.1)) 
    # uniform steps for stacked coordinate
    c = Coordinate(lat_lon=((49.1, -120), (50.2, -122), (0.1, 0.2)) 
    # specified coordinates
    c = Coordinate(lat=np.array([50, 50.1, 50.4, 50.8, 50.9])) 
    # specified stacked coordinates
    c = Coordinate(lat_lon=(np.array([50, 50.1, 50.4, 50.8, 50.9]), 
                            np.array([-120, -125, -126, -127, -130]) 
    # Depended specified coordinates
    c = Coordinate(lat=xr.DataArray([[50.1, 50.2, 50.3], [50.2, 50.3, 50.4]],
                   dims=['lat', 'lon']), lon=... )) 
    # Dependent from 3 points
    c = Coordinate(lat=((50.1, 51.4, 51.2), 100),
                   lon=((120, 120.1, 121.1), 50)) 
    """

    @property
    def _valid_dims(self):
        return ('time', 'lat', 'lon', 'alt')
    
    # default val set in constructor
    ctype = tl.Enum(['segment', 'point', 'fence', 'post'])  
    segment_position = tl.Float()  # default val set in constructor
    coord_ref_sys = tl.CUnicode
    _coords = tl.Instance(OrderedDict)
    
    def __init__(self, coords=None, coord_ref_sys="WGS84", order=None,
            segment_position=0.5, ctype='segment', **kwargs):
        """
        bounds is for fence-specification with non-uniform coordinates
        
        order is required for Python 2.x where the order of kwargs is not
        preserved.
        """
        if coords is None:
            if sys.version_info.major < 3:
                if order is None:
                    if len(kwargs) > 1:
                        raise CoordinateException(
                            "Need to specify the order of the coordinates "
                            "using 'order'.")
                    else:
                        order = kwargs.keys()
                
                coords = OrderedDict()
                for k in order:
                    coords[k] = kwargs[k]
            else:
                coords = OrderedDict(kwargs)

        for key, val in coords.items():
            if not isinstance(val, Coord):
                coords[key] = Coord(coords=val, ctype=ctype,
                                    coord_ref_sys=coord_ref_sys, 
                                    segment_position=segment_position,
                                    )
        super(Coordinate, self).__init__(_coords=coords,
                                         coord_ref_sys=coord_ref_sys,
                                         segment_position=segment_position,
                                         ctype=ctype)
    
    def __repr__(self):
        rep = str(self.__class__)
        for d in self._coords:
            rep += '\n\t{}: '.format(d) + str(self._coords[d])
        return rep
    
    def __getitem__(self, item):
        return self._coords[item]
    
    @tl.validate('_coords')
    def _coords_validate(self, proposal):
        seen_dims = []
        for key in proposal['value']:
            self._validate_dim(key, seen_dims)
            val = proposal['value'][key]
            self._validate_val(val, key, proposal['value'].keys())
        return proposal['value']
        
    def _validate_dim(self, dim, seen_dims):
        parts = dim.split('_')
        for part in parts:
            if part not in self._valid_dims:
                raise CoordinateException(
                    "The '%s' dimension of '%s' is not a valid dimension %s" % (
                        part, parts, self._valid_dims))
            if part in seen_dims:
                raise CoordinateException("The dimensions '" + part + \
                "' cannot be repeated.")
            seen_dims.append(part)
    
    def _validate_val(self, val, dim='', dims=[]):
        # Dependent array, needs to be an xarray.DataArray
        if isinstance(val, xr.DataArray):
            for key in val._coords: 
                if key not in dims:
                    raise CoordinateException("Dimensions of dependent" 
                    " coordinate DatArray needs to be in " + str(dims))
                #if val._coords.get_axis_num(dim) != 0:
                    #raise CoordinateException(
                        #"When specifying dependent coordinates, the first " 
                        #" dimension need to be equal to  " + str(dims))                    
   
    def intersect(self, other, coord_ref_sys=None, pad=1):
        # TODO: FIXME (probably should be left for the re-write)
        # This function doesn't handle stacking at all. If other is stacked, 
        # self._coords has no idea and will do the wrong thing        
        new_crds = OrderedDict()
        for i, d in enumerate(self._coords):
            if isinstance(pad, (list, tuple)):
                spad = pad[i]
            elif isinstance(pad, dict):
                spad = pad[d]
            else:
                spad = pad
            
            if d not in other._coords and self._coords[d].stacked > 1:
                parts = d.split('_')
                inds = None
                for i, p in enumerate(parts):
                    if p not in other._coords:
                        slc.append(slice(None, None))
                        continue
                    if self._coords[d].regularity != 'irregular':
                        raise NotImplementedError()
                    pcoord = Coord(coords=self._coords[d].coords[i])
                    ind  = pcoord.intersect(other._coords[p], 
                                                 coord_ref_sys, ind=True,
                                                 pad=spad)
                    if not ind:
                        inds = None
                        continue
                    elif inds:
                        inds[0] = max(inds[0], ind[0])
                        inds[1] = min(inds[1], ind[1])
                    else:
                        inds = ind
                if inds and inds[0] < inds[1]:
                    new_crds[d] = Coord(coords=[c[inds[0]: inds[1]] for c in 
                                         self.coords[d].coords])
                else:
                    new_crds[d] = Coord(coords=(self._coords[d].bound[0], 
                                                self._coords[d].bound[1], 0))
            elif  d not in other._coords:
                new_crds[d] = self._coords[d]
                continue
            else:
                new_crds[d] = self._coords[d].intersect(other._coords[d],
                                                        coord_ref_sys, pad=spad)
            
        return self.__class__(new_crds, **self.kwargs)
    
    def intersect_ind_slice(self, other, coord_ref_sys=None, pad=1):
        # TODO: FIXME (probably should be left for the re-write)
        # This function doesn't handle stacking at all. If other is stacked, 
        # self._coords has no idea and will do the wrong thing
        slc = []
        for j, d in enumerate(self._coords):
            if isinstance(pad, (list, tuple)):
                spad = pad[j]
            elif isinstance(pad, dict):
                spad = pad[d]
            else:
                spad = pad
                
            if d not in other._coords and self._coords[d].stacked > 1:
                parts = d.split('_')
                inds = None
                for i, p in enumerate(parts):
                    if p not in other._coords:
                        slc.append(slice(None, None))
                        continue
                    if self._coords[d].regularity != 'irregular':
                        raise NotImplementedError()
                    pcoord = Coord(coords=self._coords[d].coords[i])
                    ind  = pcoord.intersect(other._coords[p], 
                                                 coord_ref_sys, ind=True,
                                                 pad=spad)
                    #if pcoord.is_max_to_min:
                        #ind = ind[::-1]
                    if not ind:
                        slc.append(slice(0, 0))
                        inds = None
                        continue
                    elif inds:
                        inds[0] = max(inds[0], ind[0])
                        inds[1] = min(inds[1], ind[1])
                    else:
                        inds = ind
                if inds and inds[0] < inds[1]:
                    slc.append(slice(inds[0], inds[1]))
                else:
                    slc.append(slice(0, 0))                    
            elif d not in other._coords:
                    slc.append(slice(None, None))
                    continue
            else:    
                ind = self._coords[d].intersect(other._coords[d], 
                                                coord_ref_sys, ind=True, pad=spad)
                if self._coords[d].regularity == 'dependent':  # untested
                    i = self.coordinates.dims.index(d)
                    ind = [inds[i] for inds in ind]
                if ind:
                    slc.append(slice(ind[0], ind[1]))
                else:
                    slc.append(slice(0, 0))
        return slc
    
    @property
    def kwargs(self):
        return {
                'coord_ref_sys': self.coord_ref_sys,
                'segment_position': self.segment_position,
                'ctype': self.ctype
                }
    
    @property
    def shape(self):
        return [c.size for c in self._coords.values()]
    
    @property
    def delta(self):
        return np.array([c.delta for c in self._coords.values()]).squeeze()
    
    @property
    def dims(self):
        return self._coords.keys()
    
    @property
    def coords(self):
        crds = OrderedDict()
        for k, v in self._coords.items():
            if v.stacked == 1:
                crds[k] = v.coordinates
            else:
                dtype = [(str(kk), np.float64) for kk in k.split('_')]
                crds[k] = np.column_stack(v.coordinates).astype(np.float64)
                crds[k] = crds[k].view(dtype=dtype).squeeze()
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
        crs = {'WGS84': 'EPSG:4326',
               'SPHER_MERC': 'EPSG:3857'}
        return crs[self.coord_ref_sys.upper()]
    
    def unstack(self):
        new_crds = OrderedDict()
        for k, v in self._coords.items():
            if v.stacked == 1:
                new_crds[k] = v
            else:
                for i, kk in enumerate(k.split('_')):
                    new_crds[kk] = self._coords[k].coordinates[i]

        return self.__class__(new_crds, **self.kwargs) 
    
    @staticmethod
    def get_stacked_coord_dict(coords):
        stacked_coords = {}
        for c in coords:
            if '_' in c:
                for cc in c.split('_'):
                    stacked_coords[cc] = c        
        return stacked_coords        
    
    @property
    def stacked_coords(self):
        return Coordinate.get_stacked_coord_dict(self._coords)
    
    def __add__(self, other):
        if not isinstance(other, Coordinate):
            raise CoordinateException("Can only add Coordinate objects"
                   " together.")
        new_coords = copy.deepcopy(self._coords)
        for key in other._coords:
            if key in self._coords:
                if np.all(np.array(self._coords[key].coords) !=
                        np.array(other._coords[key].coords)):
                    new_coords[key] = self._coords[key] + other._coords[key]
            else:
                new_coords[key] = copy.deepcopy(other._coords[key])
        return self.__class__(coords=new_coords)

    def iterchunks(self, shape, return_slice=False):
        # TODO assumes the input shape dimension and order matches
        # TODO replace self[k].coords[slc] with self[k][slc] (and implement the slice)

        slices = [
            map(lambda i: slice(i, i+n), range(0, m, n))
            for m, n
            in zip(self.shape, shape)]

        for l in itertools.product(*slices):
            kwargs = {k:self[k].coordinates[slc] for k, slc in zip(self.dims, l)}
            kwargs['order'] = self.dims
            coords = Coordinate(**kwargs)
            if return_slice:
                yield l, coords
            else:
                yield coords

    @property
    def latlon_bounds_str(self):
        if 'lat' in self.dims and 'lon' in self.dims:
            return '%s_%s_x_%s_%s' % (
                self['lat'].bounds[0],
                self['lon'].bounds[0],
                self['lat'].bounds[1],
                self['lon'].bounds[1]
            )
        elif 'lat_lon' in self.dims:
            return 'TODO'
        else:
            return 'NA'

if __name__ == '__main__':
    #coord = Coordinate(lat=xr.DataArray(
        #np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0], 
                  #dims=['lat', 'lon']),
                       #lon=xr.DataArray(
        #np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0], 
                #dims=['lat', 'lon'])  )
    #c = coord.intersect(coord)    
    
    #coord = Coord(coords=(1, 10, 10))
    #coord_left = Coord(coords=(-2, 7, 10))
    #coord_right = Coord(coords=(4, 13, 10))
    #coord_cent = Coord(coords=(4, 7, 4))
    #coord_cover = Coord(coords=(-2, 13, 15))
    
    #c = coord.intersect(coord_left)
    #c = coord.intersect(coord_right)
    #c = coord.intersect(coord_cent)
    
    cus = Coord(0, 2, 5)
    cus.area_bounds  
    c = Coord((0, 1, 2), (0, -1, -2), 5)
    c.area_bounds
    c.coordinates
    
    ci = Coord(coords=(np.linspace(0, 1, 5), np.linspace(0, 2, 5), np.linspace(1, 3, 5)))
    ci.area_bounds
    cc = Coordinate(lat_lon_alt=ci)
    d = xr.DataArray(np.random.rand(5), dims=cc.dims, coords=cc.coords)
    cc2 = Coordinate(lat_lon_alt=c)
    d2 = xr.DataArray(np.random.rand(5), dims=cc2.dims, coords=cc2.coords)
    
    ccus = cc.unstack()
    cc2us = cc2.unstack()
    
    print('Done')
