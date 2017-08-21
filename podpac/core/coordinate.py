from __future__ import division, unicode_literals, print_function, absolute_import

import numbers

import xarray as xr
import numpy as np
import traitlets as tl
from collections import OrderedDict
from pint import UnitRegistry
ureg = UnitRegistry()
import node

# TODO: What to do about coord that is not monotonic? Decreases instead of increases?

class CoordinateException(Exception):
    pass

class Coord(tl.HasTraits):
    """
    Regular, specified
    stacked, unstacked
    independent, dependent
    """
    units = node.Units()
    coord_ref_sys = tl.Unicode(default_value='WGS84',
                              help="Coordinate reference system for coordinate.")
    
    ctype = tl.Enum(['segment', 'point', 'fence', 'post'], default_value='point',
                   help="Default is 'point'."
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
    
    coords = tl.Any()
    @tl.validate("coords")
    def _coords_validate(self, proposal):
        if not isinstance(proposal['value'],
                          (tuple, list, np.ndarray, xr.DataArray, numbers.Number)):
            raise CoordinateException("Coords must be of type tuple, list, " 
                                      "np.ndarray, or xr.DataArray")

        val = proposal['value']
        try:
            stacked = self._stacked(val)
            regularity = self._regularity(val)
        except Exception, e:
            raise CoordinateException("Unhandled error:" + str(e))
        
        if isinstance(val, (list, tuple)):
            # Regular, gridded equal value
            if regularity == 'single' and len(val) != 1:
                raise CoordinateException("Single stacked coordinates need"
                                          " to be specified as a tuple of "
                                          "tuples or list of lists.")
            elif len(val) != 3 and regularity == 'regular':
                raise CoordinateException("When specifying uniformly spaced" 
                "coordinates, provide it in the format (start, stop, number)"
                "or (start, stop, step)")
            elif regularity in ['irregular', 'dependent'] and \
                     np.any([v.shape != val[0].shape for v in val]):
                raise CoordinateException("When specifying irregularly-spaced "
                                          "or dependent stacked dimensions, " 
                                          "all of input array need to be the "
                                          "same size.")                
        # Dependent array, needs to be an xarray.DataArray
        elif isinstance(val, xr.DataArray):
            # These have to be checked in the coordinate object because the
            # dimension names are important.
            pass
        # Irregular spacing independent coordinates
        else:
            # No checks yet
            pass 
        return proposal['value']

    @property
    def stacked(self):
        return self._stacked(self.coords)

    def _stacked(self, coords):
        if isinstance(coords, numbers.Number):
            return 1
        elif isinstance(coords, (list, tuple)):
            if len(coords) == 1:  # single stacked coordinate
                return len(coords[0])
            elif np.all([isinstance(c, np.ndarray) for c in coords]) or \
                    np.all([isinstance(c, xr.DataArray) for c in coords]):
                return len(coords)
            elif len(coords) == 3:
                if np.all([isinstance(c, (list, tuple, np.ndarray))
                           for c in coords[:2]]):
                    return len(coords[0])
                return 1
        elif isinstance(coords, np.ndarray):
            if len(coords.shape) == 2:
                return coords.shape[1]
            return 1
        elif isinstance(coords, xr.DataArray):
            if 'stack' in coords.dims:
                return coords.shape[coords.get_axis_num('stack')]
            else: return 1

        raise CoordinateException("Coord stacking '{}'".format(coords) + \
                                  " not understood")
        
    @property
    def regularity(self):
        return self._regularity(self.coords)
    
    def _regularity(self, coords):
        if isinstance(coords, (list, tuple)):
            if len(coords) == 1:  # Single stacked coordinate
                return 'single'
            if np.all([isinstance(c, np.ndarray) for c in coords]):
                return 'irregular'
            elif np.all([isinstance(c, xr.DataArray) for c in coords]):
                return 'dependent'
            elif len(coords) == 3:
                return 'regular'
            else:
                return 'error'
        elif isinstance(coords, np.ndarray):
            return 'irregular'
        elif isinstance(coords, xr.DataArray):
            return 'dependent'
        elif isinstance(coords, numbers.Number):
            return 'single'
        
        raise CoordinateException("Coord regularity '{}'".format(coords) + \
                                  " not understood")
            
    _cached_bounds = tl.Instance(np.ndarray, allow_none=True)    
    @property
    def bounds(self):
        if self._cached_bounds is not None:
            return self._cached_bounds
        if self.regularity == 'single':
            self._cached_bounds = np.array(
                [self.coords, self.coords], float).squeeze()
        if self.regularity == 'regular':
            self._cached_bounds = np.array(self.coords[:2], float)
        elif self.regularity == 'irregular':
            if isinstance(self.coords, (list, tuple)):
                self._cached_bounds = np.array([
                    [c.min(), c.max()] for c in self.coords], float).T
            else:
                self._cached_bounds = np.array([
                    np.min(self.coords, axis=0), 
                    np.max(self.coords, axis=0)], float)
        elif self.regularity == 'dependent':
            if isinstance(self.coords, (list, tuple)):
                self._cached_bounds = np.array([
                    [c.min(), c.max()] for c in self.coords], float).T
            else:
                dims = [d for d in self.coords.dims if 'stack' not in d]
                self._cached_bounds = np.array([
                    self.coords.min(dims), 
                    self.coords.max(dims)], float)            

        return self._cached_bounds
    
    @property
    def area_bounds(self):
        extents = self.bounds  
        if self.ctype in ['fence', 'segment'] and self.regularity != 'single':
            p = self.segment_position
            extents += np.array([-p, 1 - p]) * self.delta
        return extents
        
    _cached_delta = tl.Instance(np.ndarray, allow_none=True) 
    
    @property
    def delta(self):
        if self._cached_delta is not None:
            return self._cached_delta        
        if self.regularity == 'single':
            self._cached_delta = np.array([0])
        if self.regularity == 'regular':
            if isinstance(self.coords[2], int):
                self._cached_delta = np.array(\
                    (np.array(self.coords[1]) - np.array(self.coords[0]))\
                    / (self.coords[2] - 1.))
            else:
                self._cached_delta = np.array(self.coords[2:3])
        elif self.extents is not None and len(self.extents) == 2:
            self._cached_delta = np.array(self.extents)
        else:
            print("Warning: delta probably doesn't work for stacked dependent coords")
            self._cached_delta = np.array([
            self.coords[1] - self.coords[0],
            self.coords[-1] - self.coords[-2]
        ])
        return self._cached_delta

    _cached_coords = tl.Any(default_value=None, allow_none=True)
    @property
    def coordinates(self):
        coords = self.coords
        if self._cached_coords is not None:
            return self._cached_coords
        regularity = self.regularity
            
        if regularity == 'single':
            self._cached_coords = coords
        elif regularity == 'regular':
            N = self.size
            self._cached_coords = np.linspace(self.coords[0], self.coords[1], N)
        elif regularity in ['irregular', 'dependent']:
            self._cached_coords = coords
            
        return self._cached_coords
        
    @property
    def size(self):
        if not isinstance(self.coords[2], int):  # delta specified
            N = np.round(
                (self.coords[0] - self.coords[1]) / self.coords[2]) + 1
        else:
            N = self.coords[2]
        return N
        
    @tl.observe('extents', 'ctype', 'segment_position')
    def _clear_bounds_cache(self, change):
        if (change['old'] is None and change['new'] is not None) or \
               np.any(np.array(change['old']) != np.array(change['new'])):
            self._cached_bounds = None
        
    @tl.observe('coords')
    def _clear_cache(self, change):
        if (change['old'] is None and change['new'] is not None) or \
               np.any(change['old'] != change['new']):
            self._cached_coords = None
            self._cached_bounds = None
            self._cached_delta = None
        
    def intersect(self, other_coord, coord_ref_sys=None, pad=1):
        if coord_ref_sys is not None and coord_ref_sys != self.coord_ref_sys:
            raise NotImplementedError("Still need to implement handling of "
                                      "different coordinate reference systems")        
        if self.coord_ref_sys != other_coord.coord_ref_sys:
            raise NotImplementedError("Still need to implement handling of "
                                      "different coordinate reference systems")
        if self.units != other_coord.units:
            raise NotImplementedError("Still need to implement handling of "
                                              "different units")            
        
        if np.all(self.bounds == other_coord.bounds):
            return self
        ibounds = [
            np.maximum(self.bounds[0], other_coord.bounds[0]),
            np.minimum(self.bounds[1], other_coord.bounds[1])        
            ]
        if np.any(ibounds[0] > ibounds[1]):
            return []
        if self.regularity == 'single':
            return self
        elif self.regularity == 'regular':
            lefti = np.floor((ibounds[0] - self.bounds[0]) / self.delta)
            righti =  np.ceil((self.bounds[1] - ibounds[1]) / self.delta)
            left = self.bounds[0] + max(0, lefti - pad) * self.delta
            right = min(self.bounds[1],
                        self.bounds[1] - (righti + pad) *self.delta)   
            new_crd = self.__class__((left, right, self.delta), **self.kwargs)
        elif self.regularity == 'irregular':
            lefti = np.argmin(np.abs(self.coordinates - other_coords.bounds[0]))
            righti = np.argmax(np.abs(self.coordinates - other_coords.bounds[0]))
            lefti = np.max(0, lefti - 1)
            righti = np.min(righti, self.coordinates.shape[0])
            new_crd = self.__class__(self.coordinates[lefti:righti], **self.kwargs)
        elif self.regularity == 'dependent':
            b = other_coord.bounds
            mini = [np.min(np.argmin(np.abs(self.coordinates.data - b[0]),
                                     axis=d)) \
                for d in range(len(self.coordinates.dims))]
            maxi = [np.min(np.argmin(np.abs(self.coordinates.data - b[1]),
                                     axis=d)) \
                    for d in range(len(self.coordinates.dims))]
            slc = [slice(max(0, ss[0] - pad),
                         min(self.coordinates.shape[i], ss[1] + pad)) \
                   for i, ss in enumerate(zip(mini, maxi))]
            crds = self.coordinates
            for d, s in zip(self.coordinates.dims, slc):
                crds = crds.isel(**{d: s})
                       
            new_crd = self.__class__(crds, **self.kwargs)
            
        return new_crd
            
    @property
    def kwargs(self):
        kwargs = {'units': self.units,
                  'coord_ref_sys': self.coord_ref_sys,
                  'ctype': self.ctype,
                  'segment_position': self.segment_position,
                  'extents': self.extents
        }
        return kwargs
    
    def __repr__(self):
        rep = str(self.__class__) + ' Bounds: [{min}, {max}],' + \
            ' segment position: {}, ctype: "{}"'.format(self.segment_position, 
                                                       self.ctype)
        if self.regularity in ['single', 'regular', 'irregular']:
            rep = rep.format(min=self.bounds[0], max=self.bounds[1])       
        return rep
    
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
    
    def __init__(self, coords=None, coord_ref_sys="WGS84", 
            segment_position=0.5, ctype='point', **kwargs):
        """
        bounds is for fence-specification with non-uniform coordinates
        """
        if coords is None:
            coords = OrderedDict(kwargs)
        for key, val in coords.iteritems():
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
    
    @tl.validate('_coords')
    def _coords_validate(self, proposal):
        seen_dims = []
        for key in proposal['value']:
            self._validate_dim(key, seen_dims)
            val = proposal['value'][key]
            self._validate_val(val, key, proposal['value'].keys())
        return proposal['value']
        
    def _validate_dim(self, dim, seen_dims):
        parts = dim.replace('-', '_').split('_')
        for part in parts:
            if part not in self._valid_dims:
                raise CoordinateException("The '" + part + "' dimension of '"\
                        + key + "' is not a valid dimension " \
                        + str(self.valid_dims)
                )
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
   
    def intersect(self, other, coord_ref_sys=None):
        new_crds = OrderedDict()
        for d in self._coords:
            if d not in other._coords:
                new_crds[d] = self._coords[d]
                continue
            new_crds[d] = self._coords[d].intersect(other._coords[d], coord_ref_sys)
            
        return self.__class__(new_crds, **self.kwargs)
    
    @property
    def kwargs(self):
        return {
                'coord_ref_sys': self.coord_ref_sys,
                'segment_position': self.segment_position,
                'ctype': self.ctype
                }
    
    @property
    def shape(self):
        return [c.size for c in self._coords.values]
    
    @property
    def dims(self):
        return self._coords.keys()
    
    @property
    def coords(self):
        return {k: v.coordinates for k, v in self._coords.iteritems()}
            
