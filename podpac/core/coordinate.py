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

def get_timedelta(s):
    a, b = s.split(',')
    return np.timedelta64(int(a), b)

class BaseCoord(tl.HasTraits):
    """
    Base class for a single dimension of coordinates.
    
    Attributes
    ----------
    units
    ctype
    segment_position
    extents
    coord_ref_system

    Properties
    ----------
    bounds
    area_bounds
    delta (signed)
    coordinates
    size
    is_descending
    is_monotonic
    is_regular
    is_datetime

    Methods
    -------
    intersect
    intersect_bounds


    """
    
    units = Units(allow_none=True, default_value=None)
    
    ctype = tl.Enum(['segment', 'point', 'fence', 'post'], default_value='segment',
                    help="Default is 'segment'. Indication of the coordinates "
                         "type. This is either a single point ('point' or "
                         "'post') or the whole segment between this "
                         "coordinate and the next ('segment', 'fence').")
    
    segment_position = tl.Float(default_value=0.5,
                                help="Default is 0.5. Where along a segment "
                                     "is the coordinate specified. 0 <= "
                                     "segment <= 1. For example, if segment=0, "
                                     "the coordinate is specified at the "
                                     "left-most point of the line segement "
                                     "connecting the current coordinate and "
                                     "the next coordinate. If segment=0.5, "
                                     "then the coordinate is specified at the "
                                     "center of this line segement.")
    
    extents = tl.List(allow_none=True, default_value=None, 
                      help="When specifying non-uniform coordinates, set the "
                           "bounding box (extents) of the grid in case ctype "
                           "is 'segment' or 'fence'")
    coord_ref_sys = tl.Unicode()
    
    @tl.validate('segment_position')
    def _segment_position_validate(self, proposal):
        val = proposal['value']
        if not 0 <= val <= 1:
            raise ValueError(
                "Invalid segment_position '%s', must be in [0, 1]" % val)
        
        return val

    @property
    def kwargs(self):
        kwargs = {'units': self.units,
                  'ctype': self.ctype,
                  'segment_position': self.segment_position,
                  'extents': self.extents
        }
        return kwargs

    @tl.observe('extents', 'ctype', 'segment_position')
    def _clear_bounds_cache(self, change):
        clear_cache(self, change, ['bounds'])
        
    _cached_coordinates = tl.Instance(np.ndarray, default_value=None, allow_none=True)
    @property
    def coordinates(self):
        raise NotImplementedError()
    
    @property
    def area_bounds(self):
        raise NotImplementedError
    
    _cached_bounds = tl.Instance(np.ndarray, allow_none=True)    
    @property
    def bounds(self):
        raise NotImplementedError()

    @property
    def size(self):
        raise NotImplementedError()

    @property
    def is_datetime(self):
        raise NotImplementedError

    @property
    def is_monotonic(self):
        raise NotImplementedError

    @property
    def rasterio_regularity(self):
        raise NotImplementedError

    @property
    def scipy_regularity(self):
        raise NotImplementedError

    def intersect(self, other, coord_ref_sys=None, ind=False, **kwargs):
        """ Wraps intersect_bounds using the other coordinates bounds. """

        if self.units != other.units:
            raise NotImplementedError("Still need to implement handling different units")

        # no valid other bounds, empty
        if other.size == 0:
            if ind:
                return slice(0, 0)
            else:
                return Coord([], **self.kwargs)

        return self.select(other.bounds, ind=ind, **kwargs)

    def select(self, bounds, ind=False, **kwargs):
        """
        Get the coordinates within the given bounds.

        Arguments
        ---------
        bounds : min, max
            selection bounds
        ind : bool
            return slice or indices for selection instead of coordinates

        Returns
        -------
        intersection : BaseCoord
            coord object with selected coordinates (if ind=False)
        I : slice or list
            index or slice for the selected coordinates (if ind=True)
        """

        # empty
        if self.size == 0: 
            if ind:
                return slice(0, 0)
            else:
                return self

        # full
        if self.bounds[0] >= bounds[0] and self.bounds[1] <= bounds[1]:
            if ind:
                return slice(None, None)
            else:
                return self

        # none
        if self.bounds[0] > bounds[1] or self.bounds[1] < bounds[0]:
            if ind:
                return slice(0, 0)
            else:
                return Coord([], **self.kwargs)

        # partial, implemented in child classes
        return self._select(bounds, ind=ind, **kwargs)

    def _select(self, bounds, ind=False):
        """ Partial selection, implemented in child classes. """
        raise NotImplementedError()
    
    def __sub__(self, other):
        if isinstance(other, string_types):
            other = get_timedelta(other)
        
        if isinstance(other, np.timedelta64):
            if not self.is_datetime:
                raise TypeError("Cannot add timedelta to numerical coord")
            return self._add(-other)

        elif isinstance(other, numbers.Number):
            if self.is_datetime:
                raise TypeError("Cannot add '%s' to datetime coord" % type(other))
            return self._add(-other)

        elif isinstance(other, BaseCoord):
            if self.is_datetime != other.is_datetime:
                raise TypeError("Mismatching coordinates types")
            return self.intersect(other)

        else:
            raise TypeError("Cannot subtract '%s' to '%s'" % (
                other.__class__.__name__, self.__class__.__name__))

    def __add__(self, other):
        if isinstance(other, string_types):
            other = get_timedelta(other)
        
        if isinstance(other, np.timedelta64):
            if not self.is_datetime:
                raise TypeError("Cannot add timedelta to numerical coord")
            return self._add(other)

        elif isinstance(other, numbers.Number):
            if self.is_datetime:
                raise TypeError("Cannot add '%s' to datetime coord" % type(other))
            return self._add(other)

        elif isinstance(other, BaseCoord):
            if self.is_datetime != other.is_datetime:
                raise TypeError("Mismatching coordinates types")
            return self._concat(other)

        else:
            raise TypeError("Cannot add '%s' to '%s'" % (
                other.__class__.__name__, self.__class__.__name__))

    def __iadd__(self, other):
        if isinstance(other, string_types):
            other = get_timedelta(other)
        
        if isinstance(other, numbers.Number):
            if self.is_datetime:
                raise TypeError("Cannot add '%s' to datetime coord" % type(other))
            return self._add_equal(other)

        elif isinstance(other, np.timedelta64):
            if not self.is_datetime:
                raise TypeError("Cannot add timedelta to numerical coord")
            return self._add_equal(other)

        elif isinstance(other, BaseCoord):
            return self._concat_equal(other)

        else:
            raise TypeError("Cannot add '%s' to '%s'" % (
                other.__class__.__name__, self.__class__.__name__))

    def _add(self, other):
        raise NotImplementedError

    def _add_equal(self, other):
        raise NotImplementedError

    def _concat(self, other):
        raise NotImplementedError

    def _concat_equal(self, other):
        raise NotImplementedError

    def __repr__(self):
        return '%s: Bounds[%s, %s], N[%d], ctype["%s"]' % (
            self.__class__.__name__,
            self.bounds[0], self.bounds[1],
            self.size,
            self.ctype)

class Coord(BaseCoord):
    """
    A basic array of coordinates.

    Attributes
    ----------
    coords : np.ndarray
        input coordinate array, must all be the same type.
        string datetime coordinates are converted to datetime64

    Properties
    ----------
    coordinates : np.ndarray
        the input coords array

    """

    def __init__(self, coords, **kwargs):
        super(Coord, self).__init__(coords=coords, **kwargs)

    coords = tl.Any()
    @tl.validate('coords')
    def _coords_validate(self, proposal):
        # squeeze and check dimensions
        val = np.atleast_1d(np.array(proposal['value']).squeeze())
        if val.ndim != 1:
            raise ValueError(
                "Invalid coords (ndim=%d, must be ndim=1)" % val.ndim)
        
        if val.size == 0:
            return val
        
        # convert strings to datetime
        if isinstance(val[0], string_types):
            val = np.array(val, dtype=np.datetime64)

        # check dtype
        if (not np.issubdtype(val.dtype, np.number) and
            not np.issubdtype(val.dtype, np.datetime64)):
            raise TypeError(
                "coords must all be a number or datetime and must all match")

        return val

    @tl.observe('coords')
    def _clear_cache(self, change):
        clear_cache(self, change, ['bounds'])

    @property
    def coordinates(self):
        return self.coords

    @cached_property
    def bounds(self):
        if self.size == 0:
            # TODO or is it better to do something arbitrary like -1, 1?
            lo, hi = np.nan, np.nan
        elif self.is_datetime:
            lo, hi = np.min(self.coords), np.max(self.coords)
        else:
            lo, hi = [np.nanmin(self.coords), np.nanmax(self.coords)]

        return np.array([lo, hi])

    @property
    def area_bounds(self):
        if self.ctype in ['fence', 'segment'] and self.extents:
            extents = self.extents
        else:
            extents = copy.deepcopy(self.bounds)
        return extents

    @property
    def size(self):
        return self.coords.size

    @property
    def is_datetime(self):
        return np.issubdtype(self.coords.dtype, np.datetime64)

    @property
    def is_monotonic(self):
        return False

    @property
    def rasterio_regularity(self):
        return self.size == 1

    @property
    def scipy_regularity(self):
        return True

    def _select(self, bounds, ind=False, pad=None):
        # returns a list of indices rather than a slice
        if pad is not None:
            # Should we ignore and raise a warning? We could also implement the
            # padding by finding the necessary bounds increase required to
            # include the appropriate coordinates wherever they are in the
            # list. It may not matter or get factored out depending on how
            # interpolations need to happen.
            raise NotImplementedError("TODO unsorted Coord pad not implemented")

        gt = self.coordinates >= bounds[0]
        lt = self.coordinates <= bounds[1]
        I = np.where(gt & lt)[0]
        
        if ind:
            return I
        
        return Coord(self.coordinates[I], **self.kwargs)

    def _add(self, other):
        return Coord(self.coords + other, **self.kwargs)

    def _add_equal(self, other):
        self.coords += other
        return self

    def _concat(self, other):
        # always returns a Coord object

        # TODO kwargs (units, etc)
        coords = np.concatenate([self.coordinates, other.coordinates])
        return Coord(coords, **self.kwargs)

    def _concat_equal(self, other):
        self.coords = np.concatenate([self.coordinates, other.coordinates])
        return self

class MonotonicCoord(Coord):
    """
    An array of monotonically increasing or decreasing coordinates.
    
    Attributes
    ----------
    coords : np.ndarray
        input coordinate array; must all be the same type and sorted.
        string datetime coordinates are converted to datetime64.

    Properties
    ----------
    is_monotonic : bool
        True
    is_descending : bool
        Is the coordinates array in monotonitacally decreasing order.
    """

    @tl.validate("coords")
    def _coords_validate(self, proposal):
        # basic validation
        val = super(MonotonicCoord, self)._coords_validate(proposal)

        # TODO nan?
        d = (val[1:] - val[:-1]) * (val[1] - val[0])
        if np.any(d <= 0):
            raise ValueError("Invalid coords, must be ascending or descending")

        return val

    @cached_property
    def bounds(self):
        # TODO nan?
        if self.size == 0:
            lo, hi = np.nan, np.nan # TODO something arbitrary like -1, 1?
        elif self.is_descending:
            lo, hi = self.coords[-1], self.coords[0]
        else:
            lo, hi = self.coords[0], self.coords[-1]
        return np.array([lo, hi])
    
    @property
    def is_monotonic(self):
        return True

    @property
    def is_descending(self):
        # TODO nan?
        if self.size == 0:
            return False
        return self.coords[0] > self.coords[-1]

    def _select(self, bounds, ind=False, pad=1):
        gt = self.coordinates >= bounds[0]
        lt = self.coordinates <= bounds[1]
        I = np.where(gt & lt)[0]

        if I.size == 0:
            if ind:
                return slice(0, 0)
            else:
                return Coord([], **self.kwargs)
        
        imin = max(0, I.min() - pad)
        imax = min(self.size, I.max() + pad + 1)
        slc = slice(imin, imax)
        
        if ind:
            return slc

        return MonotonicCoord(self.coordinates[slc], **self.kwargs)

    def _concat(self, other):
        # returns a MonotonicCoord if possible, else Coord

        # TODO kwargs (units, etc)

        if other.size == 0:
            return MonotonicCoord(self.coord, **self.kwargs)

        if other.is_monotonic:
            other_coords = other.coordinates
            
            # Let's match self.is_descending for the output
            if self.is_descending != other.is_descending:
                other_coords = other_coords[::-1]
            
            concat_list = [self.coordinates, other_coords]
            overlap = False
            if self.is_descending:
                if concat_list[0][-1] > concat_list[1][0]: # then we're good!
                    coords = np.concatenate(concat_list)
                elif concat_list[1][-1] > concat_list[0][0]: # need to reverse
                    coords = np.concatenate(concat_list[::-1])
                else: 
                    overlap = True
            else:
                if concat_list[0][-1] < concat_list[1][0]: # then we're good!
                    coords = np.concatenate(concat_list)
                elif concat_list[1][-1] < concat_list[0][0]: # need to reverse
                    coords = np.concatenate(concat_list[::-1])
                else: 
                    overlap = True
                
            if not overlap:
                return MonotonicCoord(coords, **self.kwargs)
            
        # otherwise return a plain Coord object
        coords = np.concatenate([self.coordinates, other.coordinates])
        return Coord(coords, **self.kwargs)

    def _concat_equal(self, other):
        if other.size == 0:
            return self

        if other.is_monotonic:
            other_coords = other.coordinates
            
            # Let's match self.is_descending for the output
            if self.is_descending != other.is_descending:
                other_coords = other_coords[::-1]
            
            concat_list = [self.coordinates, other_coords]
            overlap = False
            if self.is_descending:
                if concat_list[0][-1] > conact_list[1][0]: # then we're good!
                    coords = np.concatenate(concat_list)
                elif concat_list[1][-1] > cancat_list[0][0]: # need to reverse
                    coords = np.concatenate(concat_list[::-1])
                else: 
                    overlap = True
            else:
                if concat_list[0][-1] < conact_list[1][0]: # then we're good!
                    coords = np.concatenate(concat_list)
                elif concat_list[1][-1] < concat_list[0][0]: # need to reverse
                    coords = np.concatenate(concat_list[::-1])
                else: 
                    overlap = True
                
            if not overlap:
                self.coords = coords
                return self
        
        # otherwise
        raise TypeError("Cannot concatenate '%s' to '%s' in-place" % (
            other.__class__.__name__, self.__class__.__name__))

class UniformCoord(BaseCoord):
    """
    A uniformly-spaced coordinates defined by a start, stop, and delta.

    Attributes
    ----------
    start : float or np.datetime64
        start coordinate
    stop : float or np.datetime64
        stop coordinate; will be included if aligned with the start and step
    delta : float or np.timedelta64
        signed delta between coordinates; timedelta string input is parsed
    epsg : TODO, str, unicode, or possibly enum
        <not yet in use>
    
    Properties
    ----------
    coords : tuple
        (start, stop, size) or (start, stop, size, epsg)
    delta : float or np.timedelta64
        delta between coordinates; this is either the step or calculated from
        the start, stop, and size. It is signed. 
    """

    start = tl.Any()
    stop = tl.Any()
    delta = tl.Any()
    epsg = tl.Any() # TODO

    def __init__(self, start, stop, delta, epsg=None, **kwargs):
        super(UniformCoord, self).__init__(
            start=start, stop=stop, delta=delta, epsg=epsg, **kwargs)

    @tl.validate('delta')
    def _validate_delta(self, d):
        val = d['value']

        # type checking and conversion
        if isinstance(val, string_types):
            val = get_timedelta(val)
        elif isinstance(val, np.timedelta64):
            pass
        elif isinstance(val, numbers.Number):
            val = float(val)
        else:
            raise TypeError(
                "delta must be number or timedelta, not '%s'" % type(val))

        # check nonzero
        if val == val*0:
            raise ValueError("UniformCoord delta must be nonzero")
        
        # check sign
        if self.is_descending:
            if np.array(val).astype(float) > 0:
                raise ValueError("UniformCoord delta must be less than zero"
                                 " if start > stop.")
        elif np.array(val).astype(float) < 0:
            raise ValueError("UniformCoord delta must be greater than zero"
                             " if start < stop.")            
        
        return val

    @tl.validate('start', 'stop')
    def _validate_start_stop(self, d):
        val = d['value']
        
        # type checking and conversion
        if isinstance(val, string_types):
            val = np.datetime64(val)
        elif isinstance(val, np.datetime64):
            pass
        elif isinstance(val, numbers.Number):
            val = float(val)
        else:
            raise TypeError(
                "start/stop must be number or datetime, not '%s'" % type(val))

        return val

    @tl.observe('start', 'stop', 'delta')
    def _clear_cache(self, change):
        clear_cache(self, change, ['coordinates', 'bounds'])

    @cached_property
    def coordinates(self):
        # Syntax a little odd, but works for datetime64 as well as floats
        return self.start + np.arange(0, self.size) * self.delta
    
    @cached_property
    def bounds(self):
        lo = self.start
        hi = self.start + self.delta * (self.size - 1)
        if self.is_descending:
            lo, hi = hi, lo

        return np.array([lo, hi])

    @property
    def area_bounds(self):
        extents = copy.deepcopy(self.bounds)
        if self.ctype in ['fence', 'segment']:
            p = self.segment_position
            extents[0] -= p * np.abs(self.delta)
            extents[1] += (1-p) * np.abs(self.delta)
        return extents

    @property
    def size(self):
        return max(0, int(np.floor((self.stop-self.start) / self.delta) + 1))

    @property
    def is_datetime(self):
        return isinstance(self.start, np.datetime64)
    
    @property
    def is_monotonic(self):
        return True

    @property
    def is_descending(self):
        return self.stop < self.start

    @property
    def rasterio_regularity(self):
        return True

    @property
    def scipy_regularity(self):
        return True

    def _select(self, bounds, ind=False, pad=1):
        lo = max(bounds[0], self.bounds[0])
        hi = min(bounds[1], self.bounds[1])
        
        imin = int(np.ceil((lo - self.bounds[0]) / np.abs(self.delta)))
        imax = int(np.ceil((hi - self.bounds[0]) / np.abs(self.delta)))

        imin = np.clip(imin-pad, 0, self.size)
        imax = np.clip(imax+pad+1, 0, self.size)

        if self.is_descending:
            imax, imin = self.size - imin, self.size - imax

        if ind:
            return slice(imin, imax)
            
        start = self.start + imin*self.delta
        stop = self.start + (imax-1)*self.delta
        return UniformCoord(start, stop, self.delta, **self.kwargs)

    def _add(self, other):
        return UniformCoord(self.start + other,
                            self.stop + other, self.delta, **self.kwargs)

    def _add_equal(self, other):
        self.start += other
        self.stop += other
        return self

    def _concat(self, other):
        # tries to return UniformCoord first, then MonotonicCoord, then Coord
        TOL = 1e-12
        if other.size == 0:
            return UniformCoord(
                self.start, self.stop, self.delta, **self.kwargs)

        if isinstance(other, UniformCoord) \
                and np.abs(np.abs(self.delta) - np.abs(other.delta)).astype(float) < TOL:
            # WARNING: This code is duplicated below in _concat_equal
            delta = self.delta
            ostart, ostop = other.start, other.stop
            if self.is_descending != other.is_descending:
                ostart, ostop = ostop, ostart
                overlap = False

            new_start, new_stop = self.start, ostop
            if self.is_descending:
                if (self.stop > ostart):
                    new_start, new_stop = self.start, ostop
                elif (ostop > self.start):
                    new_start, new_stop = ostart, self.stop
            else:
                if (self.stop < ostart):
                    new_start, new_stop = self.start, ostop
                elif (ostop < self.start):
                    new_start, new_stop = ostart, self.stop

            # use the size trick to see if these align
            size = (np.floor((new_stop - new_start) / self.delta) + 1)
            if (self.size + other.size ) == size:
                return UniformCoord(
                    new_start, new_stop, delta, **self.kwargs)
            elif (self.size + other.size) < size:  # No overlap, but separated
                return MonotonicCoord._concat(self, other)
            #else: # overlapping

        if isinstance(other, MonotonicCoord):
            return MonotonicCoord._concat(self, other)
        
        # otherwise
        coords = np.concatenate([self.coordinates, other.coordinates])
        return Coord(coords, **self.kwargs)

    def _concat_equal(self, other):
        if other.size == 0:
            return self

        if isinstance(other, UniformCoord) \
                and np.abs(np.abs(self.delta) - np.abs(other.delta)).astype(float) < TOL:
            # WARNING: This code is copied above in _concat
            delta = self.delta
            ostart, ostop = other.start, other.stop
            if self.is_descending != other.is_descending:
                ostart, ostop = ostop, ostart
                overlap = False

            if self.is_descending:
                if (self.stop > ostart):
                    new_start, new_stop = self.start, ostop
                elif (ostop > self.start):
                    new_start, new_stop = ostart, self.stop
            else:
                if (self.stop < ostart):
                    new_start, new_stop = self.start, ostop
                elif (ostop < self.start):
                    new_start, new_stop = ostart, self.stop

            # use the size trick to see if these align
            size = (np.floor((new_stop - new_start) / self.delta) + 1)
            if (self.size + other.size ) == size:
                self.stop, self.start, self.delta = new_stop, new_start, delta
                return self

        raise TypeError("Cannot concatenate '%s' to '%s' in-place" % (
            other.__class__.__name__, self.__class__.__name__))

def coord_linspace(start, stop, num, **kwargs):
    """
    Convencence wrapper to get a UniformCoord with the given bounds and size.
    
    WARNING: the number of points may not be respected for time coordinates
    because this function calculates a 'delta' which is quantized. If the 
    start, stop, and num do not divide nicely you may end up with more or
    fewer points than specified. 
    """

    if not isinstance(num, (int, np.long, np.integer)):
        raise TypeError("num must be an integer, not '%s'" % type(num))
    start = UniformCoord._validate_start_stop(None, {'value':start})
    stop = UniformCoord._validate_start_stop(None, {'value':stop})
    delta = (stop - start) / (num-1)
    
    return UniformCoord(start, stop, delta, **kwargs)

def _make_coord(arg, **kwargs):
    if isinstance(arg, BaseCoord):
        return arg
    elif isinstance(arg, tuple):
        if isinstance(arg[2], (int, np.long, np.integer)):
            return coord_linspace(*arg, **kwargs)
        else:
            return UniformCoord(*arg, **kwargs)
    else:
        try:
            return MonotonicCoord(arg, **kwargs)
        except:
            return Coord(arg, **kwargs)

class BaseCoordinate(tl.HasTraits):
    pass

class Coordinate(BaseCoordinate):
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
    coord_ref_sys = tl.CUnicode()
    _coords = tl.Instance(OrderedDict)
    dims_map = tl.Dict()

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
                        raise TypeError(
                            "Need to specify the order of the coordinates "
                            "using 'order'.")
                    else:
                        order = kwargs.keys()
                
                coords = OrderedDict()
                for k in order:
                    coords[k] = kwargs[k]
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
            _coords[key] = _make_coord(val, **kw)

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
        if coords is None:
            coords = self._coords
        stacked_coords = OrderedDict()
        for c in coords:
            if '_' in c:
                for cc in c.split('_'):
                    stacked_coords[cc] = c       
            else:
                stacked_coords[c] = c 
        return stacked_coords        
    
    def unstack_dict(self, coords=None, check_dim_repeat=False):
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
   
    def stack(self, stack_dims, copy=True):
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
        if copy:
            return self.__class__(coords=self._coords.copy())
        else:
            self.dims_map = {v:v for v in self.dims_map}
            return self

    def intersect(self, other, coord_ref_sys=None, pad=1, ind=False):
        if ind:
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
                if ind:
                    I.append(slice(None, None))
                else:
                    d[dim] = self._coords[dim]
                continue
            
            intersection = self._coords[dim].intersect(
                other._coords[dim], coord_ref_sys, ind=ind, pad=spad)
            
            if ind:
                I.append(intersection)
            else:
                d[dim] = intersection
        
        if ind:
            return I
        else:
            coords = self.stack_dict(d)
            return Coordinate(coords, **self.kwargs)
    
    @property
    def kwargs(self):
        return {
                'coord_ref_sys': self.coord_ref_sys,
                'segment_position': self.segment_position,
                'ctype': self.ctype
                }
    
    def replace_coords(self, other, copy=True):
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
    
    def get_shape(self, other_coords=None):
        if other_coords is None:
            other_coords = self
        # Create shape for each dimension
        shape = []
        seen_dims = []
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

        return shape
        
    @property
    def shape(self):
        return self.get_shape()
    
    @property
    def delta(self):
        return np.array([c.delta for c in self._coords.values()]).squeeze()
    
    @property
    def dims(self):
        dims = []
        for v in self.dims_map.values():
            if v not in dims:
                dims.append(v)
        return dims
    
    @property
    def coords(self):
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
        crs = {'WGS84': 'EPSG:4326',
               'SPHER_MERC': 'EPSG:3857'}
        return crs[self.coord_ref_sys.upper()]
    
    def __add__(self, other):
        if not isinstance(other, Coordinate):
            raise TypeError(
                "Unsupported type '%s', can only add Coordinate object" % (
                    other.__class__.__name__))
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
            kwargs = {k:self.coords[k][slc] for k, slc in zip(self.dims, l)}
            kwargs['order'] = self.dims
            coords = Coordinate(**kwargs)
            if return_slice:
                yield l, coords
            else:
                yield coords

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
    groups = tl.List(trait=tl.Instance(Coordinate), minlen=1)

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

if __name__ == '__main__': 
    
    coord = coord_linspace(1, 10, 10)
    coord_left = coord_linspace(-2, 7, 10)
    coord_left_r = coord_linspace(7, -2, 10)
    coord_right = coord_linspace(4, 13, 10)
    coord_right_r = coord_linspace(13, 4, 10)
    coord_cent = coord_linspace(4, 7, 4)
    coord_cover = coord_linspace(-2, 13, 15)
    
    print(coord.intersect(coord_left))
    print(coord.intersect(coord_right))
    print(coord.intersect(coord_right_r))
    print(coord.intersect(coord_cent))
    print(coord_right_r.intersect(coord))
    print(coord_left_r.intersect(coord))
    
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

    coord_left = coord_linspace(-2, 7, 3)
    coord_right = coord_linspace(8, 13, 3)
    coord_right2 = coord_linspace(13, 8, 3)
    coord_cent = coord_linspace(4, 11, 4)
    coord_pts = Coord(15)
    coord_irr = Coord(np.random.rand(5))
    
    print ((coord_left + coord_right).coordinates)
    print ((coord_right + coord_left).coordinates)
    print ((coord_left + coord_right2).coordinates)
    print ((coord_right2 + coord_left).coordinates)
    print ((coord_left + coord_pts).coordinates)
    print (coord_irr + coord_pts + coord_cent)

    c = Coordinate(lat_lon=((0, 1, 10), (0, 1, 10)), time=(0, 1, 2), order=('lat_lon', 'time'))
    c2 = Coordinate(lat_lon=((0.5, 1.5, 15), (0.1, 1.1, 15)))

    print (c.replace_coords(c2))
    print (c.replace_coords(c2.unstack()))
    print (c.unstack().replace_coords(c2))
    print (c.unstack().replace_coords(c2.unstack()))
    
    
    coord_left = coord_linspace(0, 2, 3)
    coord_right = coord_linspace(3, 5, 3)
    coord_right_r = coord_linspace(5, 3, 3)
    coord_right_g = coord_linspace(4, 6, 3)
    coord_right_g_r = coord_linspace(6, 4, 3)
    
    coord = coord_left + coord_right
    assert(isinstance(coord, UniformCoord))
    print (coord.coordinates)
    
    coord = coord_left + coord_right_r
    assert(isinstance(coord, UniformCoord))
    print (coord.coordinates)    
    
    coord = coord_right_r + coord_left 
    assert(isinstance(coord, UniformCoord))
    print (coord.coordinates)        
    
    coord = coord_right_g + coord_left 
    assert(isinstance(coord, MonotonicCoord))
    print (coord.coordinates)            
    
    coord = coord_left + coord_right_g
    assert(isinstance(coord, MonotonicCoord))
    print (coord.coordinates)    

    coord = coord_right_g_r + coord_left 
    assert(isinstance(coord, MonotonicCoord))
    print (coord.coordinates)            
    
    coord = coord_left + coord_right_g_r
    assert(isinstance(coord, MonotonicCoord))
    print (coord.coordinates)    
    
    coord = coord_left + coord_left
    assert(isinstance(coord, Coord))
    print (coord.coordinates)    
    
    coord = coord_right_r + coord_right_r
    assert(isinstance(coord, Coord))
    print (coord.coordinates)    
    
    coord = coord_right_g_r + coord_right_g_r
    assert(isinstance(coord, Coord))
    print (coord.coordinates)        
    
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
    
    np.testing.assert_array_equal(coord_right.area_bounds, coord_right_r.area_bounds)
    
    c = UniformCoord('2015-01-01', '2015-01-04', '1,D')
    c2 = UniformCoord('2015-01-01', '2015-01-04', '2,D')
    
    print('Done')
