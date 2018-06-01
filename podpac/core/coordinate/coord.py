"""
Coord summary

Attributes
----------
TOL : float
    Description
"""


from __future__ import division, unicode_literals, print_function, absolute_import

import numbers
import copy
import datetime

from six import string_types
import numpy as np
import traitlets as tl

from podpac.core.units import Units
from podpac.core.utils import cached_property, clear_cache
from podpac.core.coordinate.util import (
    make_coord_value, make_coord_delta, get_timedelta, add_coord)

TOL = 1e-12

class BaseCoord(tl.HasTraits):
    """
    Base class for a single dimension of coordinates.
    
    Attributes
    ----------
    coord_ref_sys : unicode
        Coordinate reference system.
    ctype : str
        Coordinates type (default 'segment'). Options::
         - 'segment': whole segment between this coordinate and the next.
         - 'point': single point
    delta : TYPE
        Description
    extents : ndarray
        Custom bounding box (if ctype='segment').
    segment_position : float
        For segment coordinates, where along the segment the coordinate is
        specified, between 0 and 1 (default 0.5). Unused for point.
    units : Units
        TODO
    kwargs : TYPE
        TODO
    coordinates : TYPE
        TODO
    bounds : TYPE
        TODO
    area_bounds : TYPE
        TODO
    delta : TYPE
        TODO
    size : TYPE
        TODO
    is_datetime : TYPE
        TODO
    is_monotonic : TYPE
        TODO
    is_descending : TYPE
        TODO
    rasterio_regularity : TYPE
        TODO
    scipy_regularity : TYPE
        TODO
    
    """
    
    units = Units(allow_none=True, default_value=None)
    
    ctype = tl.Enum(['segment', 'point'], default_value='segment',
        help="Coordinates type. Options are 'segment' (default) for the whole "
             "segment between this coordinate and the next or 'point' for a "
             "single location.")
    
    segment_position = tl.Float(default_value=0.5,
        help="Default is 0.5. Where along a segment is the coordinate "
             "specified. 0 <= segment <= 1. For example, if segment=0, the "
             "coordinate is specified at the left-most point of the line "
             "segement connecting the current coordinate and the next "
             "coordinate. If segment=0.5, then the coordinate is specified at "
             "the center of this line segement.")
    
    extents = tl.Any(allow_none=True, default_value=None,
        help="Custom bounding box of the grid when ctype='segment'. Useful "
             "when specifying non-uniform segment coordinates")
    
    coord_ref_sys = tl.Unicode()
    
    delta = tl.Any(allow_none=False)
    
    @tl.validate('segment_position')
    def _segment_position_validate(self, proposal):
        val = proposal['value']
        if not 0 <= val <= 1:
            raise ValueError(
                "Invalid segment_position '%s', must be in [0, 1]" % val)
        
        return val

    @tl.validate('extents')
    def _extents_validate(self, proposal):
        # TODO test this
        val = proposal['value']
        if val is None:
            return None

        # check shape
        val = np.array(val)
        if val.shape != (2,):
            raise ValueError("Invalid extents shape %s, must be (2,)" % val.shape)

        # check and cast values
        val = np.array([make_coord_value(val[0]), make_coord_value(val[1])])
        return val

    @property
    def kwargs(self):
        '''
        Dictionary specifying the coordinate properties.
        
        Returns
        -------
        TYPE
            Description
        '''

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
        '''
        Full coordinates array. See subclasses for specific implementations.
        
        Returns
        -------
        TYPE
            Description
        '''

        # get coordinates and ensure read-only array with correct dtype
        coordinates = np.array(self._coordinates(), dtype=self.dtype)
        coordinates.setflags(write=False)
        return coordinates

    def _coordinates(self):
        raise NotImplementedError

    @property
    def dtype(self):
        '''
        Coordinate array dtype.
        
        Returns
        -------
        TYPE
            Description
        '''

        if self.is_datetime:
            return np.datetime64
        else:
            return np.float64
    
    @property
    def area_bounds(self):
        '''
        Low and high bounds. If ctype is 'segment', the bounds will include
        the portions of the segments beyond the low and high coordinate bounds.
        
        Returns
        -------
        TYPE
            Description
        '''
        
        # get area_bounds and ensure read-only array with the correct dtype
        area_bounds = np.array(self._area_bounds(), dtype=self.dtype)
        area_bounds.setflags(write=False)
        return area_bounds

    def _area_bounds(self):
        # point: same as bounds
        if self.ctype == 'point':
            return self.bounds

        # segment: explicit extents
        elif self.extents is not None:
            return self.extents

        # segment: calculate extents from segment_position and delta
        else:
            p = self.segment_position
            b = self.bounds
            d = np.abs(self.delta)
            return [add_coord(b[0], -p * d), add_coord(b[1], (1-p) * d)]

    _cached_bounds = tl.Instance(np.ndarray, allow_none=True)
    @property
    def bounds(self):
        '''
        Low and high bounds.
        
        Returns
        -------
        TYPE
            Description
        '''

        # get bounds and ensure read-only array with the correct dtype
        bounds = np.array(self._bounds(), dtype=self.dtype)
        bounds.setflags(write=False)
        return bounds

    def _bounds(self):
        return NotImplementedError

    @property
    def size(self):
        '''
        Number of coordinates.
        
        Raises
        ------
        NotImplementedError
            Description
        '''

        raise NotImplementedError()

    @property
    def is_datetime(self):
        '''
        True for datetime coordinates, False for numerical coordinates.
        
        Raises
        ------
        NotImplementedError
            Description
        '''

        raise NotImplementedError

    @property
    def is_monotonic(self):
        '''
        True if the coordinates are guaranteed to be in-order, else False.
        
        Raises
        ------
        NotImplementedError
            Description
        '''

        raise NotImplementedError
    
    @property
    def is_descending(self):
        '''
        True if the coordinates are monotonically descending, False if
        monotonically ascending, and None for non-monotonic coordinates.
        
        Raises
        ------
        NotImplementedError
            Description
        '''

        raise NotImplementedError

    @property
    def rasterio_regularity(self):
        """
        TODO
        
        Raises
        ------
        NotImplementedError
            Description
        """

        raise NotImplementedError

    @property
    def scipy_regularity(self):
        """
        TODO
        
        Raises
        ------
        NotImplementedError
            Description
        """

        raise NotImplementedError

    def intersect(self, other, coord_ref_sys=None, ind=False, **kwargs):
        """
        Get the coordinates within the bounds the given coordinates object.
        
        Parameters
        ----------
        other : BaseCoord
            coordinates to intersect with
        coord_ref_sys : str, optional
            TODO
        ind : bool, optional
            If True, return slice or indices for the selection instead of
            coordinates. Default False.
        **kwargs
            Description
        
        Returns
        -------
        intersection : BaseCoord
            coord object with coordinates with other.bounds (if ind=False)
        I : slice or list
            index or slice for the intersected coordinates (if ind=True)
        
        See Also
        --------
        select : Get the coordinates within the given bounds.
        
        Raises
        ------
        NotImplementedError
            Description
        """

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
        
        Parameters
        ----------
        bounds : min, max
            selection bounds
        ind : bool, optional
            return slice or indices for selection instead of coordinates
        **kwargs
            Description
        
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
                return copy.deepcopy(self)

        # full
        if self.bounds[0] >= bounds[0] and self.bounds[1] <= bounds[1]:
            if ind:
                return slice(None, None)
            else:
                return copy.deepcopy(self)

        # none
        if self.area_bounds[0] > bounds[1] or self.area_bounds[1] < bounds[0]:
            if ind:
                return slice(0, 0)
            else:
                return Coord([], **self.kwargs)

        # partial, implemented in child classes
        return self._select(bounds, ind=ind, **kwargs)

    def add(self, delta, inplace=False):
        """
        Add a delta value to each coordinate.
        
        Parameters
        ----------
        delta : TYPE
            Description
        inplace : bool (optional)
            If True, update coordinates in-place. Default False.
        
        Returns
        -------
        result : BaseCoord
            If inplace, this object with resulting coordinates.
            Otherwise, new BaseCoord object with resulting coordinates.
        
        Raises
        ------
        TypeError
            Description
        """

        delta = make_coord_delta(delta)
        
        if self.is_datetime is True and not isinstance(delta, np.timedelta64):
            raise TypeError("Cannot add '%s' to datetime coord" % type(delta))
        
        if self.is_datetime is False and isinstance(delta, np.timedelta64):
            raise TypeError("Cannot add timedelta to numerical coord")

        # empty case
        if self.size == 0:
            if inplace:
                return self
            else:
                return copy.deepcopy(self)
        
        # standard case
        if inplace:
            return self._add_equal(delta)
        else:
            return self._add(delta)
    
    def concat(self, other, inplace=False):
        """
        Concatenate coordinates.
        
        Parameters
        ----------
        other : BaseCoord
            coords object to concatenate
        inplace : bool (optional)
            If True, update coordinates in-place. Default False.
        
        Returns
        -------
        result : BaseCoord
            If inplace, this object with concatenated coordinates.
            New coords object with concatenated coordinates.
        
        Raises
        ------
        TypeError
            Description
        """

        if not isinstance(other, BaseCoord):
            raise TypeError("Cannot concatenate '%s' to '%s'" % (
                other.__class__.__name__, self.__class__.__name__))

        if self.is_datetime is True and other.is_datetime is False:
            raise TypeError("Cannot concatenate numerical coords to datetime coords")
        
        if self.is_datetime is False and other.is_datetime is True:
            raise TypeError("Cannot concatenate datetime coords to numerical coords")

        # empty cases
        if other.size == 0:
            if inplace:
                return self
            else:
                return copy.deepcopy(self)

        # standard case
        if inplace:
            return self._concat_equal(other)
        else:
            return self._concat(other)

    def _select(self, bounds, ind=False):
        raise NotImplementedError()

    def _add(self, other):
        raise NotImplementedError

    def _add_equal(self, other):
        raise NotImplementedError

    def _concat(self, other):
        raise NotImplementedError

    def _concat_equal(self, other):
        raise NotImplementedError
    
    # ----------------------------------
    # standard operators / magic methods
    # ----------------------------------

    def __len__(self):
        """ number of coordinate values """
        return self.size

    def __contains__(self, value):
        """ is the value within the coordinate area """
        if self.size == 0:
            return False

        return self.area_bounds[0] <= value <= self.area_bounds[1]

    def __getitem__(self, index):
        """ indexes coordinates """
        return self.coordinates[index]

    def __add__(self, other):
        """ add a delta or concatenate """

        if isinstance(other, BaseCoord):
            return self.concat(other)
        else:
            return self.add(other)

    def __iadd__(self, other):
        """ add a delta or concatenate in-place """
        
        if isinstance(other, BaseCoord):
            return self.concat(other, inplace=True)

        else:
            other = make_coord_delta(other)
            return self.add(other, inplace=True)

    def __sub__(self, other):
        """ subtract a delta """
        _other = -make_coord_delta(other)
        return self.add(_other)

    def __isub__(self, other):
        """ subtract a delta in place """
        _other = -make_coord_delta(other)
        return self.add(_other, inplace=True)

    def __and__(self, other):
        """ intersect """
        return self.intersect(other)

    def __iand__(self, other):
        """ intersect in-place """
        raise NotImplementedError

    def __repr__(self):
        return '%s: Bounds[%s, %s], N[%d], ctype["%s"]' % (
            self.__class__.__name__,
            self.bounds[0], self.bounds[1],
            self.size,
            self.ctype)

class Coord(BaseCoord):
    """
    A basic array of coordinates. Not guaranteed to be sorted, unique, or
    uniformly-spaced.
    
    Attributes
    ----------
    coords : ndarray
        Input coordinate array, must all be the same type. Numerical values
        are converted to floats, datetime values are converted to datetime64.
    
    See Also
    --------
    MonotonicCoord : An array of sorted coordinates.
    UniformCoord : An array of sorted, uniformly-spaced coordinates.
    
    """

    def __init__(self, coords=[], **kwargs):
        """
        Initialize coords from an array.
        
        Parameters
        ----------
        coords : array-like
            coordinate values.
        **kwargs
            Description
        """

        super(Coord, self).__init__(coords=coords, **kwargs)

    coords = tl.Any()
    @tl.validate('coords')
    def _coords_validate(self, proposal):
        val = np.array(proposal['value'])
        
        # protects numbers from being cast to string
        if np.issubdtype(val.dtype, np.dtype(str).type):
            val = np.array(proposal['value'], object)

        # squeeze and check dimensions
        val = np.atleast_1d(val.squeeze())
        if val.ndim != 1:
            raise ValueError(
                "Invalid coords (ndim=%d, must be ndim=1)" % val.ndim)
        
        if val.size == 0:
            return val
        
        # convert strings and datetimes to datetime64
        if isinstance(val[0], (string_types, datetime.date)):
            if any(isinstance(v, numbers.Number) for v in val):
                raise TypeError("coords must be all numbers or all datetimes")

            val = np.array(val, dtype=np.datetime64)

        if np.issubdtype(val.dtype, np.number):
            val = val.astype(float)
        elif np.issubdtype(val.dtype, np.datetime64):
            pass
        else:
            raise TypeError("coords must be all numbers or all datetimes")

        return val

    @tl.default('delta')
    def _delta_default(self):
        if self.size == 1 and self.is_datetime:
            time_delta = self.bounds[0] - self.bounds[0]
            delta = np.ones_like(time_delta) * 2
            return time_delta + delta
        else:
            return (self.bounds[1] - self.bounds[0]) / (self.size - 1)

    @tl.observe('coords')
    def _clear_cache(self, change):
        clear_cache(self, change, ['bounds'])

    def _coordinates(self):
        return self.coords

    def _bounds(self):
        if self.size == 0:
            # TODO or is it better to do something arbitrary like -1, 1?
            lo, hi = np.nan, np.nan
        elif self.is_datetime:
            lo, hi = np.min(self.coords), np.max(self.coords)
        else:
            lo, hi = np.nanmin(self.coords), np.nanmax(self.coords)

        return lo, hi

    @property
    def size(self):
        '''
        Number of coordinates.
        
        Returns
        -------
        TYPE
            Description
        '''

        return self.coords.size

    @property
    def is_datetime(self):
        '''
        True for datetime coordinates, False for numerical coordinates. None if empty.
        
        Returns
        -------
        TYPE
            Description
        '''

        if self.size == 0:
            return None
        
        return np.issubdtype(self.coords.dtype, np.datetime64)

    @property
    def is_monotonic(self):
        '''
        False; the coordinates are not guaranteed to be in-order.
        
        Returns
        -------
        TYPE
            Description
        '''

        return False

    @property
    def is_descending(self):
        '''
        None; n/a.
        
        Returns
        -------
        TYPE
            Description
        '''

        return None  # No way of telling so None (which evaluates as False)

    @property
    def rasterio_regularity(self):
        """True if size is 1, else False
        
        Returns
        -------
        TYPE
            Description
        """

        return self.size == 1

    @property
    def scipy_regularity(self):
        """True
        
        Returns
        -------
        TYPE
            Description
        """

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
        if self.size == 0:
            coords = other.coordinates
        else:
            coords = np.concatenate([self.coordinates, other.coordinates])

        return Coord(coords, **self.kwargs)

    def _concat_equal(self, other):
        if self.size == 0:
            self.coords = other.coordinates
        else:
            self.coords = np.concatenate([self.coordinates, other.coordinates])

        return self

class MonotonicCoord(Coord):
    """
    An array of monotonically increasing or decreasing coordinates. The
    coordinates are guaranteed to be sorted and unique, but not guaranteed to
    be uniformly spaced.
    
    Attributes
    ----------
    coords : ndarray
        Input coordinate array, must all be the same type, unique, and sorted.
        Numerical values are converted to floats, datetime values are
        converted to datetime64.
    
    See Also
    --------
    Coord : A basic array of coordinates.
    UniformCoord : An array of sorted, uniformly-spaced coordinates.
    """
    
    @tl.default('delta')
    def _delta_default(self):  # average delta
        return (self.coordinates[-1] - self.coordinates[0]) / (self.size - 1)

    @tl.validate("coords")
    def _coords_validate(self, proposal):
        # basic validation
        val = super(MonotonicCoord, self)._coords_validate(proposal)

        if val.size > 1:
            d = (val[1:] - val[:-1]).astype(float) * (val[1] - val[0]).astype(float)
            if np.any(d <= 0):
                raise ValueError("Invalid coords, must be ascending or descending")

        return val

    @cached_property
    def bounds(self):
        if self.size == 0:
            lo, hi = np.nan, np.nan # TODO something arbitrary like -1, 1?
        elif self.is_descending:
            lo, hi = self.coords[-1], self.coords[0]
        else:
            lo, hi = self.coords[0], self.coords[-1]
        return np.array([lo, hi])
    
    @property
    def is_monotonic(self):
        '''
        True; the coordinates are guaranteed to be in-order.
        
        Returns
        -------
        TYPE
            Description
        '''

        return True

    @property
    def is_descending(self):
        '''
        True if the coordinates are descending, False if ascending (or empty).
        
        Returns
        -------
        TYPE
            Description
        '''

        if self.size == 0:
            return None
        return self.coords[0] > self.coords[-1]

    def _select(self, bounds, ind=False, pad=1):
        gt = self.coordinates >= bounds[0]
        lt = self.coordinates <= bounds[1]
        if self.is_descending:
            gt, lt = lt, gt
        gtw = np.where(gt)[0]
        ltw = np.where(lt)[0]
        if gtw.size == 0 or ltw.size == 0:
            if ind:
                return slice(0, 0)
            else:
                return Coord([], **self.kwargs)
        imin = max(0, gtw.min() - pad)
        imax = min(self.size, ltw.max() + pad + 1)

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

        if self.size == 0:
            return copy.deepcopy(other)

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
        if not other.is_monotonic:
            raise TypeError("Cannot concatenate '%s' to '%s' in-place" % (
                other.__class__.__name__, self.__class__.__name__))
        
        if self.size == 0:
            self.coords = other.coordinates
            return self

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
            
        if overlap:
            raise ValueError("Cannot concatenate overlapping monotonic coords")

        self.coords = coords
        return self

    def _add(self, other):
        return MonotonicCoord(self.coords + other, **self.kwargs)
        

class UniformCoord(BaseCoord):
    """
    An array of sorted, uniformly-spaced coordinates defined by a start, stop,
    and delta.
    
    Attributes
    ----------
    delta : float or timedelta64
        signed, non-zero delta between coordinates. numerical inputs are cast
        as floats and non-numerical inputs are parsed as timedelta64
    epsg : str
        <not yet in use>
    start, stop : float or datetime64
        start and stop coordinates. numerical inputs are cast as floats and
        non-numerical inputs are parsed as datetime64.
    coords :
        TODO
    
    See Also
    --------
    Coord : A basic array of coordinates.
    MonotonicCoord: An array of sorted coordinates.
    """

    start = tl.Any()
    stop = tl.Any()
    delta = tl.Any()
    epsg = tl.Any() # TODO

    def __init__(self, start, stop, delta, epsg=None, **kwargs):
        """
        Initialize uniformly-spaced coordinates.

        Parameters
        ----------
        start : float or datetime64
            coordinates start value
        stop : float or datetime64
            coordinates stop value
        delta : float or timedelta64
            step
        """

        self.start = start
        self.stop = stop
        self.delta = delta
        super(UniformCoord, self).__init__(epsg=epsg, **kwargs)

        if self.is_datetime and not isinstance(self.delta, np.timedelta64):
            raise TypeError("delta must a timedelta for datetime coords")

        if not self.is_datetime and isinstance(self.delta, np.timedelta64):
            raise TypeError("delta must a number for numerical coords")
        
    @property
    def coords(self):
        """
        Coordinates start and stop, for backwards compatibility.
        
        Returns
        -------
        start : float, datetime64
        stop : float, datetime64
        """

        return self.start, self.stop

    @tl.validate('delta')
    def _validate_delta(self, d):
        # type checking and conversion
        val = make_coord_delta(d['value'])

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
        val = make_coord_value(d['value'])
        return val

    @tl.observe('start', 'stop', 'delta')
    def _clear_cache(self, change):
        clear_cache(self, change, ['coordinates', 'bounds'])

    def _coordinates(self):
        return add_coord(self.start, np.arange(0, self.size) * self.delta)
    
    @cached_property
    def bounds(self):
        lo = self.start
        hi = add_coord(self.start, self.delta * (self.size - 1))
        if self.is_descending:
            lo, hi = hi, lo

        return np.array([lo, hi])

    @property
    def size(self):
        '''
        Number of coordinates.
        
        Returns
        -------
        TYPE
            Description
        '''

        dname = np.array(self.delta).dtype.name

        if dname == 'timedelta64[Y]':
            dyear = self.stop.item().year - self.start.item().year
            if dyear > 0 and self.stop.item().month < self.start.item().month:
                dyear -= 1
            range_ = dyear
            step = self.delta.item()

        elif dname == 'timedelta64[M]':
            dyear = self.stop.item().year - self.start.item().year
            dmonth = self.stop.item().month - self.start.item().month
            range_ = 12*dyear + dmonth
            step = self.delta.item()

        else:
            range_ = self.stop - self.start
            step = self.delta

        eps = 1e-12  # To avoid floating point errors when calculating delta
        return max(0, int(np.floor(range_/step + eps) + 1))

    @property
    def is_datetime(self):
        '''
        True for datetime coordinates, False for numerical coordinates.
        
        Returns
        -------
        TYPE
            Description
        '''
        return isinstance(self.start, np.datetime64)
    
    @property
    def is_monotonic(self):
        '''
        True; the coordinates are guaranteed to be in-order.
        
        Returns
        -------
        TYPE
            Description
        '''

        return True

    @property
    def is_descending(self):
        '''
        True if the coordinates are descending, False if ascending.
        
        Returns
        -------
        TYPE
            Description
        '''

        return self.stop < self.start

    @property
    def rasterio_regularity(self):
        """True
        
        Returns
        -------
        TYPE
            Description
        """
        
        return True

    @property
    def scipy_regularity(self):
        """True
        
        Returns
        -------
        TYPE
            Description
        """

        return True

    def _select(self, bounds, ind=False, pad=1):
        lo = max(bounds[0], self.bounds[0])
        hi = min(bounds[1], self.bounds[1])
        
        imin = int(np.ceil((lo - self.bounds[0]) / np.abs(self.delta)))
        imax = int(np.floor((hi - self.bounds[0]) / np.abs(self.delta)))

        imax = np.clip(imax+pad+1, 0, self.size)
        imin = np.clip(imin-pad, 0, self.size)
        
        # empty case
        if imin >= imax:
            if ind:
                return slice(0, 0)
            else:
                return Coord()

        if self.is_descending:
            imax, imin = self.size - imin, self.size - imax

        if ind:
            return slice(imin, imax)
            
        start = self.start + imin*self.delta
        stop = self.start + (imax-1)*self.delta
        return UniformCoord(start, stop, self.delta, **self.kwargs)

    def _add(self, other):
        start = add_coord(self.start, other)
        stop = add_coord(self.stop, other)
        return UniformCoord(start, stop, self.delta, **self.kwargs)

    def _add_equal(self, other):
        self.start = add_coord(self.start, other)
        self.stop = add_coord(self.stop, other)
        return self

    def _concat(self, other):
        # empty other
        if other.size == 0:
            return copy.deepcopy(self)
        
        # MonotonicCoord other
        if isinstance(other, MonotonicCoord):
            return MonotonicCoord(self.coordinates).concat(other)
        
        # Coord other
        if isinstance(other, Coord):
            return Coord(self.coordinates).concat(other)

        if isinstance(other, UniformCoord):
            # mismatched delta
            if np.abs(np.abs(self.delta) - np.abs(other.delta)).astype(float) > TOL:
                return MonotonicCoord(self.coordinates).concat(other)

            start, stop, last, size = self._get_concat_values(other)

            # aligned -> return UniformCoord
            if self.size + other.size == size:
                return UniformCoord(start, stop, self.delta, **self.kwargs)
            
            # separated (no overlap) -> return MonotonicCoord
            elif self.size + other.size < size:
                return MonotonicCoord(self.coordinates).concat(other)

            # overlapping -> return MonotonicCoord or Coord
            else:
                return MonotonicCoord(self.coordinates).concat(other)

    def _concat_equal(self, other):
        if other.size == 0:
            return self

        if not isinstance(other, UniformCoord):
            raise TypeError("Cannot concatenate '%s' to '%s' in-place" % (
                other.__class__.__name__, self.__class__.__name__))

        if np.abs(np.abs(self.delta) - np.abs(other.delta)).astype(float) > TOL:
            raise ValueError("Cannot concatenate UniformCoord, delta mismatch (%f != %f)" % (
                self.delta, other.delta))

        start, stop, last, size = self._get_concat_values(other)

        # aligned
        if self.size + other.size == size:
            self.stop, self.start = stop, start
            return self

        # separated (no overlap)
        if self.size + other.size < size:
            raise ValueError("Cannot concatenate UniformCoord, ranges are separated")

        # overlapping
        else:
            raise ValueError("Cannot concatenate UniformCoord, ranges are overlapping")

    def _get_concat_values(self, other):
        ostart, ostop = other.start, other.stop
        ofirst, olast = other.bounds
        
        if other.is_descending:
            ofirst, olast = olast, ofirst

        if self.is_descending != other.is_descending:
            ostart, ostop = ostop, ostart
            ofirst, olast = olast, ofirst

        if self.is_descending == (self.stop > ostart):
            start, stop, last = self.start, ostop, olast
        elif self.is_descending:
            start, stop, last = ostart, self.stop, self.bounds[0]
        else:
            start, stop, last = ostart, self.stop, self.bounds[1]

        size = np.floor((last - start) / self.delta) + 1

        return start, stop, last, size

# =============================================================================
# helper functions
# =============================================================================

def coord_linspace(start, stop, num, **kwargs):
    """
    Get a UniformCoord with the given bounds and size.
    
    Notes
    -----
    The number of points may not be respected for time coordinates because
    this function calculates a 'delta' which is quantized. If the start, stop,
    and num do not divide nicely you may end up with more or fewer points than
    specified.
    
    Parameters
    ----------
    start : TYPE
        Description
    stop : TYPE
        Description
    num : TYPE
        Description
    **kwargs
        Description
    
    Returns
    -------
    TYPE
        Description
    
    Raises
    ------
    TypeError
        Description
    """

    if not isinstance(num, (int, np.long, np.integer)) or isinstance(num, np.timedelta64):
        raise TypeError("num must be an integer, not '%s'" % type(num))
    start = make_coord_value(start)
    stop = make_coord_value(stop)
    delta = (stop - start) / (num - 1)
    
    return UniformCoord(start, stop, delta, **kwargs)
