from __future__ import division, unicode_literals, print_function, absolute_import

import numbers
import copy

from six import string_types
import numpy as np
import traitlets as tl

from podpac.core.units import Units
from podpac.core.utils import cached_property, clear_cache
from podpac.core.coordinate.util import get_timedelta

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
    
    delta = tl.Any(allow_none=False)
    
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
    def is_descending(self):
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

    @tl.default('delta')
    def _delta_default(self):  # average delta
        return (self.bounds[1] - self.bounds[0]) / (self.size - 1)

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
    def is_descending(self):
        return None  # No way of telling so None (which evaluates as False)

    @property
    def rasterio_regularity(self):
        return self.size == 1

    @property
    def scipy_regularity(self):
        return True

    def _select(self, bounds, ind=False, pad=None):
        # returns a list of indices rather than a slice
        if pad is None:
            pad = 0

        gt = self.coordinates >= (bounds[0] - pad * self.delta)
        lt = self.coordinates <= (bounds[1] + pad * self.delta)
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
    
    @tl.default('delta')
    def _delta_default(self):  # average delta
        return (self.coordinates[-1] - self.coordinates[0]) / (self.size - 1)

    @tl.validate("coords")
    def _coords_validate(self, proposal):
        # basic validation
        val = super(MonotonicCoord, self)._coords_validate(proposal)

        # TODO nan?
        if isinstance(val[0], np.datetime64):
            d = (val[1:] - val[:-1]).astype(float) * (val[1] - val[0]).astype(float)
        else:
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
        if self.is_descending:
            gt, lt = lt, gt
        imin = max(0, np.where(gt)[0].min() - pad)
        imax = min(self.size, np.where(lt)[0].max() + pad + 1)

        if imin == imax:
            if ind:
                return slice(0, 0)
            else:
                return Coord([], **self.kwargs)
        
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
        
    @property
    def coords(self):  # For backwards compatibility
        return [self.start, self.stop]

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
                return MonotonicCoord(
                    np.concatenate((self.coordinates, other.coordinates)))
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

# =============================================================================
# helper functions
# =============================================================================

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

# =============================================================================
# TODO convert to unit testing
# =============================================================================

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
    
    np.testing.assert_array_equal(coord_right.area_bounds, coord_right_r.area_bounds)
    
    print('Done')
