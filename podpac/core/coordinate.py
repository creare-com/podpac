from __future__ import division, unicode_literals, print_function, absolute_import

import numbers
import copy
import sys

from six import string_types

import xarray as xr
import numpy as np
import traitlets as tl
from collections import OrderedDict
from pint import UnitRegistry
ureg = UnitRegistry()
import podpac

# TODO: Perhaps, Coord should not deal with stacking, and leave that 
#       functionality to Coordinate instead

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
                          (tuple, list, np.ndarray, xr.DataArray, 
                           numbers.Number, string_types, np.datetime64)):
            raise CoordinateException("Coords must be of type tuple, list, " 
                                      "np.ndarray, xr.DataArray, str, or "
                                      "np.datetime64, not " + 
                                      str(type(proposal['value'])))

        val = proposal['value']
        try:
            stacked = self._stacked(val)
            regularity = self._regularity(val)
        except Exception as e:
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
        elif isinstance(val, string_types):
            val = np.datetime64(val)
        # Irregular spacing independent coordinates
        else:
            # No checks yet
            pass 

        # enforce floating point coordinates in some cases
        if regularity == 'regular':
            if isinstance(val[0], (int, np.ndarray)):
                val = (float(val[0]),) + tuple(val[1:])
            if isinstance(val[1], (int, np.ndarray)):
                    val = (val[0], float(val[1])) + tuple(val[2:])
            if stacked > 1:
                newval0 = []
                for v in val[0]:
                    newval0.append(float(v))
                newval1 = []
                for v in val[1]:
                    newval1.append(float(v))                
                val = (tuple(newval0), tuple(newval1), val[2])
                
        
        if regularity == 'irregular':
            if len(val) == 1:  # This should actually be single
                val = np.atleast_1d(val)[0]
        return val

    def __init__(self, *args, **kwargs):
        """
        bounds is for fence-specification with non-uniform coordinates
        """
        if kwargs.get('coords') is None:
            kwargs['coords'] = args
        super(Coord, self).__init__(**kwargs)

    @property
    def stacked(self):
        return self._stacked(self.coords)

    def _stacked(self, coords):
        if isinstance(coords, (numbers.Number, string_types, np.datetime64)):
            return 1
        elif isinstance(coords, (list, tuple)):
            if len(coords) == 1:  # single stacked coordinate
                return len(coords[0])
            elif len(coords) == 3 and \
                    np.all([isinstance(c, np.ndarray) for c in coords]) and \
                    len(coords[2]) == 1 and len(coords[0]) == len(coords[1]):
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
            elif len(coords) == 3 and \
                     np.all([isinstance(c, np.ndarray) for c in coords]) and \
                     len(coords[2]) == 1 and len(coords[0]) == len(coords[1]):
                return 'regular'
            elif np.all([isinstance(c, np.ndarray) for c in coords]):
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
        elif isinstance(coords, (numbers.Number, np.datetime64, string_types)):
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
                [self.coords - self.delta, self.coords + self.delta]).squeeze()
        if self.regularity == 'regular':
            self._cached_bounds = np.array([np.min(self.coords[:2]),
                                            np.max(self.coords[:2])]).squeeze()
        elif self.regularity == 'irregular':
            if isinstance(self.coords, (list, tuple)):
                self._cached_bounds = np.array([
                    [np.nanmin(c), np.nanmax(c)] for c in self.coords]).T
            else:
                if isinstance(self.coords[0], np.datetime64):
                    self._cached_bounds = np.array([
                        np.min(self.coords, axis=0), 
                        np.max(self.coords, axis=0)])
                else:
                    self._cached_bounds = np.array([
                        np.nanmin(self.coords, axis=0), 
                        np.nanmax(self.coords, axis=0)])
        elif self.regularity == 'dependent':
            if isinstance(self.coords, (list, tuple)):
                self._cached_bounds = np.array([
                    [c.min(), c.max()] for c in self.coords]).T
            else:
                dims = [d for d in self.coords.dims if 'stack' not in d]
                self._cached_bounds = np.array([
                    self.coords.min(dims), 
                    self.coords.max(dims)])            

        return self._cached_bounds
    
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
        
    _cached_delta = tl.Instance(np.ndarray, allow_none=True) 
    
    @property
    def delta(self):
        if self._cached_delta is not None:
            return self._cached_delta        
        if self.regularity == 'single':
            # Arbitrary
            if isinstance(self.coords, np.datetime64):
                dtype = self.coords - self.coords
                self._cached_delta = np.array([1], dtype=dtype.dtype)
            else:
                self._cached_delta = np.atleast_1d(np.sqrt(np.finfo(np.float32).eps))  
        elif self.regularity == 'regular':
            if isinstance(self.coords[2], int):
                self._cached_delta = np.atleast_1d((\
                    (np.array(self.coords[1]) - np.array(self.coords[0]))\
                    / (self.coords[2] - 1.) * (1 - 2 * self.is_max_to_min)).squeeze())
            else:
                self._cached_delta = np.atleast_1d(np.array(self.coords[2:3]).squeeze())
        elif self.regularity == 'irregular':
            print("Warning: delta is not representative for irregular coords")
            if self.stacked == 1:
                self._cached_delta = np.atleast_1d(np.array(
                    (self.coords[1] - self.coords[0])*(1 - 2 * self.is_max_to_min)).squeeze())
            else:
                self._cached_delta = np.atleast_1d([
                    (c[1] - c[0])*(1 - 2 * m2m) 
                    for c, m2m in zip(self.coords, self.is_max_to_min)]).squeeze()
                    
        else:
            print("Warning: delta probably doesn't work for stacked dependent coords")
            self._cached_delta = np.array([
                self.coords[1] - self.coords[0],
                self.coords[-1] - self.coords[-2]
            ]) * (1 - 2 * self.is_max_to_min).squeeze()
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
            if self.stacked == 1:
                self._cached_coords = np.linspace(self.coords[0], self.coords[1], N)
            else:
                self._cached_coords = \
                    tuple([np.linspace(cs, ce, N) \
                     for cs, ce in zip(self.coords[0], self.coords[1])])
        elif regularity in ['irregular', 'dependent']:
            self._cached_coords = coords
            
        return self._cached_coords
        
    @property
    def size(self):
        if self.regularity == 'single':
            return 1
        elif self.regularity == 'regular':
            if not isinstance(self.coords[2], int):  # delta specified
                N = np.round((1 - 2 * self.is_max_to_min) * 
                    (self.coords[1] - self.coords[0]) / self.coords[2]) + 1
            else:
                N = self.coords[2]
        elif self.regularity == 'irregular':
            if self.stacked == 1:
                N = self.coords.size
            else:
                N = self.coords[0].size
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
        """
        Returns an Coord object if ind==False
        Returns a list of start, stop coordinates if ind==True
        """
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
            if ind:
                return [0, 0]
            else:
                return self.__class__(coords=(self.bounds[0], self.bounds[1], 0)) 
        if self.regularity == 'single':
            if ind:
                return [0, 1]
            else:
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
            min_max = [np.maximum(self.bounds[0],
                           self.bounds[0] + max(0, min_max_i[0] - pad) * self.delta),
                       np.minimum(self.bounds[1],
                        self.bounds[1] - max(0, min_max_i[1] - pad) * self.delta)]
            if self.is_max_to_min:
                min_max = min_max[::-1]
            
            coords = min_max + [float(self.delta)]
            new_crd = self.__class__(coords=coords, **self.kwargs)
            
        elif self.regularity == 'irregular':
            b = other_coord.bounds
            min_max_i = [np.nanargmin(np.abs(self.coordinates - b[0])),
                         np.nanargmin(np.abs(self.coordinates - b[1]))]
            if self.is_max_to_min:
                min_max_i = min_max_i[::-1]
            lefti = np.maximum(0, min_max_i[0] - pad)
            righti = np.minimum(min_max_i[1] + pad + 1, self.size)
            if ind:
                return [int(lefti), int(righti)]
            new_crd = self.__class__(coords=self.coordinates[lefti:righti], **self.kwargs)
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
                         min(self.coordinates.shape[i], ss[1] + pad + 1)) \
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
            if self.stacked == 1:
                if isinstance(self.coords[0], np.datetime64):
                    return self.coords[0] > self.coords[-1]
                else:
                    non_nan_coords = self.coords[np.isfinite(self.coords)]
                    return non_nan_coords[0] > non_nan_coords[-1]
            else:
                m2m = []
                for c in self.coords:
                    if isinstance(c, np.datetime64):
                        m2m.append(c[0] > c[-1])
                    else:
                        non_nan_coords = c[np.isfinite(c)]
                        m2m.append(non_nan_coords[0] > non_nan_coords[-1])
                return np.array(m2m) 
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
    
    def __add__(self, other):
        """ Should be able to add two coords together in some situations
        Although I'm really not sure about this function... may be a mistake
        """
        if not isinstance(other, Coord):
            raise CoordinateException("Can only add two Coord object together")
        if self.regularity == 'dependent' or other.regularity == 'dependent'\
                or self.stacked != other.stacked:
            raise NotImplementedError
        c1 = self.coordinates
        c2 = other.coordinates
        if self.stacked == 1:
            return self.__class__(coords=np.concatenate((c1, c2)))
        else:
            return self.__class__(coords=[np.concatenate((cc1, cc2)) \
                    for cc1, cc2 in zip(c1, c2)])
    
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
            if order is None and sys.version_info.major < 3 \
                   and len(kwargs) > 1:
                raise CoordinateException("Need to specify the order of the"
                                          " coordinates 'using order'.")
            if sys.version_info.major < 3:
                coords = OrderedDict()
                if len(kwargs) == 1 and order is None:
                    order = kwargs.keys()
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
        for i, d in enumerate(self._coords):
            if d not in other._coords:
                new_crds[d] = self._coords[d]
                continue
            if isinstance(pad, (list, tuple)):
                spad = pad[i]
            elif isinstance(pad, dict):
                spad = pad[d]
            else:
                spad = pad
            new_crds[d] = self._coords[d].intersect(other._coords[d],
                                                    coord_ref_sys, pad=spad)
            
        return self.__class__(new_crds, **self.kwargs)
    
    def intersect_ind_slice(self, other, coord_ref_sys=None, pad=1):
        slc = []
        for d in self._coords:
            if d not in other._coords:
                slc.append(slice(None, None))
                continue
            if isinstance(pad, (list, tuple)):
                spad = pad[i]
            elif isinstance(pad, dict):
                spad = pad[d]
            else:
                spad = pad
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
