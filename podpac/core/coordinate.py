from __future__ import division, unicode_literals, print_function, absolute_import

import numbers
import copy

import xarray as xr
import numpy as np
import traitlets as tl
from collections import OrderedDict
from pint import UnitRegistry
ureg = UnitRegistry()
import podpac

# TODO: What to do about coord that is not monotonic? Decreases instead of increases?

class CoordinateException(Exception):
    pass

class Coord(tl.HasTraits):
    """
    Regular, specified
    stacked, unstacked
    independent, dependent
    """
    units = podpac.Units(allow_none=True, default_value=None)
    
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
        """
        This returns the (min, max) value of the coordinate
        """
        if self._cached_bounds is not None:
            return self._cached_bounds
        if self.regularity == 'single':
            self._cached_bounds = np.array(
                [self.coords - self.delta, self.coords + self.delta], float).squeeze()
        if self.regularity == 'regular':
            self._cached_bounds = np.array([min(self.coords[:2]),
                                           max(self.coords[:2])], float)
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
        extents = copy.deepcopy(self.bounds)
        if self.ctype in ['fence', 'segment']:
            if self.regularity in ['dependent', 'irregular'] \
                    and self.extents is not None:
                extents = self.extents
            elif self.regularity != 'single':
                p = self.segment_position
                extents += np.array([-p, 1 - p]) * self.delta
        return extents
        
    _cached_delta = tl.Instance(np.ndarray, allow_none=True) 
    
    @property
    def delta(self):
        if self._cached_delta is not None:
            return self._cached_delta        
        if self.regularity == 'single':
            self._cached_delta = np.array(np.sqrt(np.finfo(np.float32).eps))  # Arbitrary
        elif self.regularity == 'regular':
            if isinstance(self.coords[2], int):
                self._cached_delta = np.array(\
                    (np.array(self.coords[1]) - np.array(self.coords[0]))\
                    / (self.coords[2] - 1.) * (1 - 2 * self.is_max_to_min))
            else:
                self._cached_delta = np.array(self.coords[2:3])
        else:
            print("Warning: delta probably doesn't work for stacked dependent coords")
            self._cached_delta = np.array([
            self.coords[1] - self.coords[0],
            self.coords[-1] - self.coords[-2]
        ]) * (1 - 2 * self.is_max_to_min)
        return self._cached_delta

    _cached_coords = tl.Any(default_value=None, allow_none=True)
    @property
    def coordinates(self):
        coords = self.coords
        if self._cached_coords is not None:
            return self._cached_coords
        regularity = self.regularity
            
        if regularity == 'single':
            self._cached_coords = np.atleast_1d(coords)
        elif regularity == 'regular':
            N = self.size
            self._cached_coords = np.linspace(self.coords[0], self.coords[1], N)
        elif regularity in ['irregular', 'dependent']:
            self._cached_coords = coords
            
        return self._cached_coords
        
    @property
    def size(self):
        if self.regularity == 'single':
            return 1
        if not isinstance(self.coords[2], int):  # delta specified
            N = np.round((1 - 2 * self.is_max_to_min) * 
                (self.coords[1] - self.coords[0]) / self.coords[2]) + 1
        else:
            N = self.coords[2]
        return int(N)
        
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
        
    def intersect(self, other_coord, coord_ref_sys=None, pad=1, ind=False):
        if self.units != other_coord.units:
            raise NotImplementedError("Still need to implement handling of "
                                              "different units")            
        
        if np.all(self.bounds == other_coord.bounds):
            if ind:
                return [0, self.size]
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
            min_max_i = [np.floor((ibounds[0] - self.bounds[0]) / self.delta),
                         np.ceil((self.bounds[1] - ibounds[1]) / self.delta)]
            if ind:
                if self.is_max_to_min:
                    min_max_i = min_max_i[::-1]
                I = [int(min(self.size, max(0, min_max_i[0] - pad))),
                     int(min(self.size, max(0, self.size - min_max_i[1] + pad)))]
                return I
            min_max = [self.bounds[0] + max(0, min_max_i[0] - pad) * self.delta,
                       min(self.bounds[1],
                        self.bounds[1] - (min_max_i[1] - pad) * self.delta)]
            if self.is_max_to_min:
                min_max = min_max[::-1]
            
            coords = min_max + [float(self.delta)]
            new_crd = self.__class__(coords=coords, **self.kwargs)
            
        elif self.regularity == 'irregular':
            min_max_i = [np.argmin(np.abs(self.coordinates - other_coords.bounds[0])),
                         np.argmin(np.abs(self.coordinates - other_coords.bounds[1]))]
            if self.is_max_to_min:
                min_max_i = min_max_i[::-1]
            lefti = np.max(0, min_max_i[0] - pad)
            righti = np.min(min_max_i[1] + pad, self.coordinates.shape[0])
            if ind:
                return [int(lefti), int(righti)]
            new_crd = self.__class__(self.coordinates[lefti:righti], **self.kwargs)
        elif self.regularity == 'dependent':
            b = other_coord.bounds
            mini = [np.min(np.argmin(np.abs(self.coordinates.data - b[0]),
                                     axis=d)) \
                for d in range(len(self.coordinates.dims))]
            maxi = [np.min(np.argmin(np.abs(self.coordinates.data - b[1]),
                                     axis=d)) \
                    for d in range(len(self.coordinates.dims))]
            if self.is_max_to_min:
                mini, maxi = maxi, mini
            if ind:
                return [(int(max(0, ss[0] - pad)),
                         int(min(self.coordinates.shape[i], ss[1] + pad))) \
                               for i, ss in enumerate(zip(mini, maxi))]
            slc = [slice(max(0, ss[0] - pad),
                         min(self.coordinates.shape[i], ss[1] + pad)) \
                   for i, ss in enumerate(zip(mini, maxi))]
            crds = self.coordinates
            for d, s in zip(self.coordinates.dims, slc):
                crds = crds.isel(**{d: s})
                       
            new_crd = self.__class__(coords=crds, **self.kwargs)
            
        return new_crd
            
    @property
    def kwargs(self):
        kwargs = {'units': self.units,
                  'ctype': self.ctype,
                  'segment_position': self.segment_position,
                  'extents': self.extents
        }
        return kwargs
    
    @property
    def is_max_to_min(self):
        if self.regularity == 'regular':
            return self.coords[0] > self.coords[1]
        elif self.regularity == 'irregular':
            return self.coords[0] > self.coords[-1]
        elif self.regularity == 'dependent':
            return np.array(self.coords).ravel()[0] > np.array(self.coords).ravel()[-1] 
        else:
            return False
        
    
    def __repr__(self):
        rep = 'Coord: Bounds[{min}, {max}],' + \
            ' N[{}], ctype["{}"]'.format(self.size, self.ctype)
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
            segment_position=0.5, ctype='segment', **kwargs):
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
   
    def intersect(self, other, coord_ref_sys=None, pad=1):
        new_crds = OrderedDict()
        for d in self._coords:
            if d not in other._coords:
                new_crds[d] = self._coords[d]
                continue
            new_crds[d] = self._coords[d].intersect(other._coords[d],
                                                    coord_ref_sys, pad=pad)
            
        return self.__class__(new_crds, **self.kwargs)
    
    def intersect_ind_slice(self, other, coord_ref_sys=None, pad=1):
        slc = []
        for d in self._coords:
            if d not in other._coords:
                slc.append(slice(None, None))
                continue
            ind = self._coords[d].intersect(other._coords[d], 
                                            coord_ref_sys, ind=True, pad=pad)
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
    def dims(self):
        return self._coords.keys()
    
    @property
    def coords(self):
        return {k: v.coordinates for k, v in self._coords.iteritems()}
    
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
    
            
if __name__ == '__main__':
    coord = Coordinate(lat=xr.DataArray(
        np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0], 
                  dims=['lat', 'lon']),
                       lon=xr.DataArray(
        np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0], 
                dims=['lat', 'lon'])  )
    c = coord.intersect(coord)    
    
    coord = Coord(coords=(1, 10, 10))
    coord_left = Coord(coords=(-2, 7, 10))
    coord_right = Coord(coords=(4, 13, 10))
    coord_cent = Coord(coords=(4, 7, 4))
    coord_cover = Coord(coords=(-2, 13, 15))
    
    c = coord.intersect(coord_left)
    c = coord.intersect(coord_right)
    c = coord.intersect(coord_cent)
    print('Done')