from __future__ import division, unicode_literals, print_function, absolute_import

import numbers

import xarray as xr
import numpy as np
import traitlets as tl
from collections import OrderedDict
from pint import UnitRegistry
ureg = UnitRegistry()

# TODO: Implement intersections for Coordinate, returning new Coordinate object
# TODO: Initialize dataset from Coordinate object
# TODO: test Coordinate

# What to do about coord that is not monotonic? Decreases instead of increases?

class CoordinateException(Exception):
    pass

class Coord(tl.HasTraits):
    """
    Regular, specified
    stacked, unstacked
    independent, dependent
    """
    units = tl.Instance(ureg.Quantity,
                        default_value=ureg.arcdegree, 
                        help="Units for the coordinates.")
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
            if not isinstance(self.coords[2], int):  # delta specified
                N = (self.coords[0] - self.coords[1]) // self.coords[2] + 1
            else:
                N = self.coords[2]
            self._cached_coords = np.linspace(self.coords[0], self.coords[1], N)
        elif regularity in ['irregular', 'dependent']:
            self._cached_coords = coords
            
        return self._cached_coords
        
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
        
    def intersect(self, other_coord):
        ibounds = [
            np.maximum(self.bounds[0], other_coord.bounds[0]),
            np.minimum(self.bounds[1], other_coord.bounds[1])        
            ]
        if np.any(ibounds[0] > ibounds[1]):
            return []
        else:
            return ibounds
    
    def __repr__(self):
        rep = str(self.__class__) + ' Bounds: [{min}, {max}],' + \
            ' segment position: {}, ctype: "{}"'.format(self.segment_position, 
                                                       self.ctype)
        if isinstance(self.coords, tuple):
            return rep.format(min=self.bounds[0], max=self.bounds[1])       
    
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
    @staticmethod
    def _valid_dims():
        return ('time', 'lat', 'lon', 'alt')
    
    # default val set in constructor
    ctype = tl.Enum(['segment', 'point', 'fence', 'post'])  
    segment_position = tl.Integer()  # default val set in constructor
    coord_ref_sys = tl.CUnicode
    coords = tl.Instance(OrderedDict)
    
    def __init__(self, coords=None, coord_ref_sys="WGS84", 
            segment_position=0.5, ctype='point', **kwargs):
        """
        bounds is for fence-specification with non-uniform coordinates
        """
        if coords is None:
            coords = OrderedDict(kwargs)
        for key, val in coords.iteritems():
            if not isinstance(val, Coord):
                coords[key] = Coord(val, ctype=ctype,
                                    coord_ref_sys=coord_ref_sys, 
                                    segment_position=segment_position,
                                    )
        super(Coordinate, self).__init__(coords=coords,
                                         coord_ref_sys=coord_ref_sys,
                                         segment_position=segment_position,
                                         ctype=ctype)
    
    @tl.validate('coords')
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
            for key in val.coords: 
                    if key not in dims:
                        raise CoordinateException("Dimensions of dependent" 
                        " coordinate DatArray needs to be in " + str(dims))
  
    def initialize_dataset(self, initial_value=0, dtype=np.float):
        # TODO
        pass
    
    def intersect(self, coords):
        # TODO
        pass


if __name__ == "__main__":
    # Unstacked
    # Regular
    coord = Coord(coords=(0, 1, 4))
    coord.intersect(coord)
    coord = Coord(coords=[0, 1, 4])
    coord = Coord(coords=(0, 1, 1/4))    
    coord = Coord(coords=[0, 1, 1/4])    
    # Irregular
    coord = Coord(coords=np.linspace(0, 1, 4))
    coord.intersect(coord)
    # Dependent, Irregular
    coord = Coord(coords=xr.DataArray(
        np.meshgrid(np.linspace(0, 1, 4), np.linspace(-1, 0, 5))[0], 
                  dims=['lat', 'lon']))
    coord.intersect(coord)
    # Stacked
    # Regular
    coord = Coord(coords=((0, -1), (1, 0), 4))
    coord.intersect(coord)
    coord = Coord(coords=[(0, 0), (1, -1), 4])
    coord = Coord(coords=((0, 0), (1, -1), 1/4))
    coord = Coord(coords=[(0, 0), (1, -1), 1/4])
    # Irregular
    coord = Coord(coords=np.column_stack((np.linspace(0, 1, 4), 
                                          np.linspace(0, -1, 4))))
    # Dependent, Irregular
    coord = Coord(coords=[
        xr.DataArray(
                     np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0],
                     dims=['lat-lon', 'time']), 
        xr.DataArray(
                np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[1],
                         dims=['lat-lon', 'time'])        
    ])           
    
    
    # These should fail
 
    
    print("Done" )