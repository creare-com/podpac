"""
One-Dimensional Coordinates
"""


from __future__ import division, unicode_literals, print_function, absolute_import

import copy

import numpy as np
import traitlets as tl

from podpac.core.units import Units
from podpac.core.utils import cached_property, clear_cache
from podpac.core.coordinates.utils import make_coord_value, make_coord_delta, make_coord_array, add_coord

TOL = 1e-12

class BaseCoordinates1d(tl.HasTraits):
    pass

class Coordinates1d(BaseCoordinates1d):
    """
    Base class for one-dimensional single coordinates.
    
    Attributes
    ----------
    name : str
        Dimension name, one of 'lat', 'lon', 'time', 'alt'.
    units : Units
        TODO
    coord_ref_sys : unicode
        Coordinate reference system.
    ctype : str
        Coordinates type: 'point', 'left', 'right', or 'midpoint'.
         - 'point': each coordinate represents a single location
         - 'left': segments; each coordinate is the left endpoint of its segment
         - 'right': segments; each coordinate is the right endpoint of its endpoint
         - 'midpoint': segments; segment endpoints are at the midpoints between coordinate values
    extents : ndarray
        When ctype != 'point', defines a custom bounding box of the grid.
        Useful when specifying non-uniform segment coordinates.
    properties : dict
        Coordinate properties (units, coord_ref_sys, ctype, extents)
    coordinates : np.ndarray
        Full array of coordinate values.
    dtype : type
        Coordinates dtype, either np.datetime64 or np.float64.
    size : int
        Number of coordinates.
    bounds : np.ndarray
        Coordinate bounds, np.array(low, high).
    area_bounds : np.ndarray
        Area bounds, np.array(low, high). When ctype != 'point', this including the portions of the segments beyond the
        coordinate bounds.
    is_monotonic : bool
        True if the coordinates are guaranteed to be sorted.
    is_descending : bool
        True if the coordinates are monotonically descending, False if the coordinates are monotonically ascending, and
        None if the coordinates are potentially unordered.

    Methods
    -------

    """

    name = tl.Enum(['lat', 'lon', 'time', 'alt'])
    units = Units(allow_none=True, default_value=None)
    coord_ref_sys = tl.Unicode()
    ctype = tl.Enum(['point', 'left', 'right', 'midpoint'], default_value='midpoint')
    extents = tl.Instance(np.ndarray, allow_none=True, default_value=None)

    def __init__(self, name=None, extents=None, **kwargs):
        if name is None:
            raise TypeError("missing argument 'name'")

        if extents is not None:
            kwargs['extents'] = make_coord_array(extents)

        super(Coordinates1d, self).__init__(name=name, **kwargs)

    @tl.validate('extents')
    def _validate_extents(self, d):
        val = d['value']
        if self.ctype == 'point' and val is not None:
            raise TypeError("extents must be None when ctype='point'")
        if val.shape != (2,):
            raise ValueError("Invalid extents shape, %s != (2,)" % val.shape)
        if self.dtype == float and val.dtype != float:
            raise ValueError("Invalid extents dtype, coordinates are numerical but extents are '%s'" % val.dtype)
        if self.dtype == np.datetime64 and not np.issubdtype(val.dtype, np.datetime64):
            raise ValueError("Invalid extents dtype, coordinates are datetime but extents are '%s'" % val.dtype)
        return val

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def properties(self):
        '''
        Coordinate properties (units, coord_ref_sys, ctype, extents)
        
        Returns
        -------
        properties : dict
            Coordinate properties (units, coord_ref_sys, ctype, extents)
        '''

        d = {}
        d['units'] = self.units
        d['coord_ref_sys'] = self.coord_ref_sys
        d['ctype'] = self.ctype
        if self.ctype != 'point':
            d['extents'] = self.extents
        return d

    @property
    def coordinates(self):
        ''' Full coordinates array. '''

        # get coordinates and ensure read-only array with correct dtype
        coordinates = np.array(self._coordinates(), dtype=self.dtype)
        coordinates.setflags(write=False)
        return coordinates

    def _coordinates(self):
        raise NotImplementedError

    @property
    def dtype(self):
        ''' Coordinate array dtype, either np.datetime64 or np.float64. '''

        raise NotImplementedError

    @property
    def size(self):
        '''Number of coordinates. '''

        raise NotImplementedError

    @property
    def bounds(self):
        ''' Coordinate bounds. '''

        # get bounds and ensure read-only array with the correct dtype
        bounds = np.array(self._bounds(), dtype=self.dtype)
        bounds.setflags(write=False)
        return bounds

    def _bounds(self):
        return NotImplementedError
    
    @property
    def area_bounds(self):
        '''
        Low and high area bounds. When ctype != 'point', this includes the portions of the segments beyond the
        coordinate bounds.
        '''

        if self.ctype == 'point':
            area_bounds = self._bounds()
        elif self.extents is not None:
            area_bounds = self.extents
        else:
            area_bounds = self._segment_area_bounds()

        # get area bounds and ensure read-only array with the correct dtype
        area_bounds = np.array(area_bounds, dtype=self.dtype)
        area_bounds.setflags(write=False)
        return area_bounds

    def _segment_area_bounds(self):
        raise NotImplementedError

    @property
    def is_monotonic(self):
        ''' True if the coordinates are guaranteed to be in-order, else False. '''

        raise NotImplementedError
    
    @property
    def is_descending(self):
        ''' 
        True if the coordinates are monotonically descending, False if monotonically ascending, and None for
        potentially unordered coordinates.
        '''

        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def intersect(self, other, coord_ref_sys=None, ind=False, **kwargs):
        """
        Get the coordinates within the bounds the given coordinates object.
        
        Parameters
        ----------
        other : Coordinates1d
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
        intersection : Coordinates1d
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
                return Coord([], **self.properties)

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
        intersection : Coordinates1d
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
                return Coord([], **self.properties)

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
        result : Coordinates1d
            If inplace, this object with resulting coordinates.
            Otherwise, new Coordinates1d object with resulting coordinates.
        
        Raises
        ------
        TypeError
            Description
        """

        delta = make_coord_delta(delta)
        
        if self.dtype is np.datetime64 and not isinstance(delta, np.timedelta64):
            raise TypeError("Cannot add '%s' to datetime coord" % type(delta))
        
        if self.dtype is np.float64 and isinstance(delta, np.timedelta64):
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
        other : Coordinates1d
            coords object to concatenate
        inplace : bool (optional)
            If True, update coordinates in-place. Default False.
        
        Returns
        -------
        result : Coordinates1d
            If inplace, this object with concatenated coordinates.
            New coords object with concatenated coordinates.
        
        Raises
        ------
        TypeError
            Description
        """

        if not isinstance(other, Coordinates1d):
            raise TypeError("Cannot concatenate '%s' to '%s'" % (other.__class__.__name__, self.__class__.__name__))

        if self.dtype is np.datetime64 and other.dtype is np.float64:
            raise TypeError("Cannot concatenate numerical coords to datetime coords")
        
        if self.dtype is np.float64 and other.dtype is np.datetime64:
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
    
    # ------------------------------------------------------------------------------------------------------------------
    # Operators ("magic methods")
    # ------------------------------------------------------------------------------------------------------------------

    def __len__(self):
        """ number of coordinate values """
        return self.size

    def __getitem__(self, index):
        """ indexes coordinates """
        raise NotImplementedError

    def __add__(self, other):
        """ add a delta or concatenate """

        if isinstance(other, Coordinates1d):
            return self.concat(other)
        else:
            return self.add(other)

    def __iadd__(self, other):
        """ add a delta or concatenate in-place """
        
        if isinstance(other, Coordinates1d):
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

    def __repr__(self):
        return "%s: '%s', Bounds[%s, %s], N[%d], ctype['%s']" % (
            self.__class__.__name__,
            self.name,
            self.bounds[0], self.bounds[1],
            self.size,
            self.ctype)

class ArrayCoordinates1d(Coordinates1d):
    """
    A basic array of coordinates. Not guaranteed to be sorted, unique, or uniformly-spaced.
    
    Attributes
    ----------
    coords : np.ndarray
        Input coordinate array. Values must all be the same type.
        Numerical values are converted to floats, and datetime values are converted to np.datetime64.
    
    See Also
    --------
    MonotonicCoordinates1d : An array of sorted coordinates.
    UniformCoordinates1d : An array of sorted, uniformly-spaced coordinates.
    
    """

    coords = tl.Instance(np.ndarray)
    ctype = tl.Enum(['point'], default_value='point')

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

        coords = make_coord_array(coords)
        super(ArrayCoordinates1d, self).__init__(coords=coords, **kwargs)

    @tl.validate('coords')
    def _validate_coords(self, d):
        val = d['value']
        if val.ndim != 1:
            raise ValueError("Invalid coords (ndim='%d', must be ndim=1" % val.ndim)
        if val.dtype != float and not np.issubdtype(val.dtype, np.datetime64):
            raise ValueError("Invalid coords (dtype='%s', must be np.float64 or np.datetime64)")
        return val

    @tl.observe('coords')
    def _clear_cache(self, change):
        clear_cache(self, change, ['bounds', 'coordinates'])
    
    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    def _coordinates(self):
        return self.coords

    def _bounds(self):
        if self.size == 0:
            lo, hi = np.nan, np.nan
        elif self.dtype is np.datetime64:
            lo, hi = np.min(self.coords), np.max(self.coords)
        else:
            lo, hi = np.nanmin(self.coords), np.nanmax(self.coords)
        return lo, hi

    @property
    def size(self):
        ''' Number of coordinates. '''

        return self.coords.size

    @property
    def dtype(self):
        if self.size == 0:
            return None
        elif self.coords.dtype == float:
            return float
        elif np.issubdtype(self.coords.dtype, np.datetime64):
            return np.datetime64
        else:
            raise ValueError("Invalid coords dtype '%s'" % self.coords.dtype)

    @property
    def is_monotonic(self):
        ''' False (the coordinates are not guaranteed to be in-order). '''

        return False

    @property
    def is_descending(self):
        ''' None (n/a) '''

        return None

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def _select(self, bounds, ind=False, pad=None):
        # returns a list of indices rather than a slice
        if pad is None:
            pad = 0

        gt = self.coordinates >= (bounds[0] - pad * self.delta) # TODO JXM
        lt = self.coordinates <= (bounds[1] + pad * self.delta) # TODO JXM
        I = np.where(gt & lt)[0]
        
        if ind:
            return I
        
        return ArrayCoordinates1d(self.coordinates[I], **self.properties)

    def _add(self, other):
        return ArrayCoordinates1d(self.coords + other, **self.properties)

    def _add_equal(self, other):
        self.coords += other
        return self

    def _concat(self, other):
        # always returns a ArrayCoordinates1d object
        if self.size == 0:
            coords = other.coordinates
        else:
            coords = np.concatenate([self.coordinates, other.coordinates])

        return ArrayCoordinates1d(coords, **self.properties)

    def _concat_equal(self, other):
        if self.size == 0:
            self.coords = other.coordinates
        else:
            self.coords = np.concatenate([self.coordinates, other.coordinates])

        return self

    def __getitem__(self, index):
        coords = self.coords[index]
        return ArrayCoordinates1d(coords, name=self.name, **self.properties)

class MonotonicCoordinates1d(ArrayCoordinates1d):
    """
    An array of monotonically increasing or decreasing coordinates. The coordinates are guaranteed to be sorted and
    unique, but not guaranteed to be uniformly spaced.
    
    Attributes
    ----------
    coords : ndarray
        Input coordinate array. Values must all be the same type, and must be unique and sorted.
        Numerical values are converted to floats, and datetime values are converted to np.datetime64.
    
    See Also
    --------
    ArrayCoordinates1d : A basic array of coordinates.
    UniformCoordinates1d : An array of sorted, uniformly-spaced coordinates.
    """

    ctype = tl.Enum(['point', 'left', 'right', 'midpoint'], default_value='midpoint')

    @tl.validate('coords')
    def _validate_coords(self, d):
        val = super(MonotonicCoordinates1d, self)._validate_coords(d)
        # check sorted
        if val.size > 1:
            d = (val[1:] - val[:-1]).astype(float) * (val[1] - val[0]).astype(float)
            if np.any(d <= 0):
                raise ValueError("Invalid coords (must be monotonically ascending or descending)")
        return val

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    def _bounds(self):
        if self.size == 0:
            lo, hi = np.nan, np.nan
        elif self.is_descending:
            lo, hi = self.coords[-1], self.coords[0]
        else:
            lo, hi = self.coords[0], self.coords[-1]
        return lo, hi

    def _segment_area_bounds(self):
        if self.size == 0:
            return np.nan, np.nan

        elif self.size == 1:
            return self.coords[0], self.coords[0]

        if self.ctype == 'midpoint':
            lo = add_coord(self.coords[0], -(self.coords[ 1] - self.coords[ 0]) / 2.)
            hi = add_coord(self.coords[-1], (self.coords[-1] - self.coords[-2]) / 2.)
        elif (self.ctype == 'left' and not self.is_descending) or (self.ctype == 'right' and self.is_descending):
            lo = self.coords[0]
            hi = add_coord(self.coords[-1], (self.coords[-1] - self.coords[-2]))
        else:
            lo = add_coord(self.coords[0], -(self.coords[ 1] - self.coords[ 0]))
            hi = self.coords[-1]

        if self.is_descending:
            lo, hi = hi, lo

        return lo, hi
    
    @property
    def is_monotonic(self):
        ''' True (the coordinates are guaranteed to be in-order.) '''

        return True

    @property
    def is_descending(self):
        ''' True if the coordinates are descending, False if ascending, and None if empty. '''

        if self.size == 0:
            return None
        elif self.size == 1:
            return None
        return self.coords[0] > self.coords[-1]

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

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
                return ArrayCoordinates1d([], **self.properties)
        imin = max(0, gtw.min() - pad)
        imax = min(self.size, ltw.max() + pad + 1)

        if imin == imax:
            if ind:
                return slice(0, 0)
            else:
                return ArrayCoordinates1d([], **self.properties)
        
        slc = slice(imin, imax)
        
        if ind:
            return slc

        return MonotonicCoordinates1d(self.coordinates[slc], **self.properties)

    def _concat(self, other):
        # returns a MonotonicCoordinates1d if possible, else ArrayCoordinates1d

        # TODO check matching properties (units, ctype, etc)

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
                return MonotonicCoordinates1d(coords, **self.properties)
            
        # otherwise return a plain ArrayCoordinates1d object
        coords = np.concatenate([self.coordinates, other.coordinates])
        return ArrayCoordinates1d(coords, **self.properties)

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
        return MonotonicCoordinates1d(self.coords + other, **self.properties)

    def __getitem__(self, index):
        coords = self.coords[index]
        try:
            return MonotonicCoordinates1d(coords, name=self.name, **self.properties)
        except ValueError:
            return ArrayCoordinates1d(coords, name=self.name, **self.properties)
        

class UniformCoordinates1d(Coordinates1d):
    """
    An array of sorted, uniformly-spaced coordinates defined by a start, stop, and step.
    
    Attributes
    ----------
    start, stop : float or datetime64
        Start and stop coordinates.
        Numerical inputs are cast as floats and non-numerical inputs are parsed as datetime64.
    step : float or timedelta64
        Signed, non-zero step between coordinates.
        Numerical inputs are cast as floats and non-numerical inputs are parsed as timedelta64.
    
    See Also
    --------
    ArrayCoordinates1d : A basic array of coordinates.
    MonotonicCoordinates1d: An array of sorted coordinates.
    """

    start = tl.Union([tl.Float(), tl.Instance(np.datetime64)])
    stop = tl.Union([tl.Float(), tl.Instance(np.datetime64)])
    step = tl.Union([tl.Float(), tl.Instance(np.timedelta64)])

    def __init__(self, start, stop, step=None, size=None, **kwargs):
        """
        Initialize uniformly-spaced coordinates.

        Parameters
        ----------
        start : float or datetime64
            start value
        stop : float or datetime64
            stop value
        step : float or timedelta64
            step between coordinates (either step or size required)
        size : int
            number of coordinates (either step or size required)
        """

        if step is not None and size is not None:
            raise TypeError("only one of 'step' and 'size' is allowed")
        elif step is None and size is None:
            raise TypeError("'step' or 'size' is required")

        start = make_coord_value(start)
        stop = make_coord_value(stop)
        if step is None:
            if not isinstance(size, (int, np.long, np.integer) or isinstance(size, np.timedelta64)):
                raise TypeError("size must be an integer, not '%s'" % type(size))
            step = (stop - start) / (size - 1)
        else:
            step = make_coord_delta(step)

        super(UniformCoordinates1d, self).__init__(start=start, stop=stop, step=step, **kwargs)

    @tl.validate('start')
    def _validate_start(self, d):
        self._validate_start_stop_step(d['value'], self.stop, self.step)
        return d['value']

    @tl.validate('stop')
    def _validate_stop(self, d):
        self._validate_start_stop_step(self.start, d['value'], self.step)
        return d['value']

    @tl.validate('step')
    def _validate_step(self, d):
        self._validate_start_stop_step(self.start, self.stop, d['value'])
        if d['value'] == 0 * d['value']:
            raise ValueError("UniformCoordinates1d step must be nonzero")
        return d['value']

    def _validate_start_stop_step(self, start, stop, step):
        if isinstance(start, float) and isinstance(stop, float) and isinstance(step, float):
            fstep = step
        elif isinstance(start, np.datetime64) and isinstance(stop, np.datetime64) and isinstance(step, np.timedelta64):
            fstep = step.astype(float)
        else:
            raise TypeError("UniformCoordinates1d mismatching types (start '%s', stop '%s', step '%s')." % (
                type(start), type(stop), type(step)))

        if fstep < 0 and start < stop:
            raise ValueError("UniformCoordinates1d step must be less than zero if start > stop.")

        if fstep > 0 and start > stop:
            raise ValueError("UniformCoordinates1d step must be greater than zero if start < stop.")

    @tl.observe('start', 'stop', 'step')
    def _clear_cache(self, change):
        clear_cache(self, change, ['coordinates', 'bounds'])

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    def _coordinates(self):
        return add_coord(self.start, np.arange(0, self.size) * self.step)
    
    @property
    def dtype(self):
        ''' Coordinate dtype, datetime or float '''
        
        if isinstance(self.start, np.datetime64):
            return np.datetime64
        else:
            return float

    @property
    def size(self):
        ''' Number of coordinates. '''

        dname = np.array(self.step).dtype.name

        if dname == 'timedelta64[Y]':
            dyear = self.stop.item().year - self.start.item().year
            if dyear > 0 and self.stop.item().month < self.start.item().month:
                dyear -= 1
            range_ = dyear
            step = self.step.item()

        elif dname == 'timedelta64[M]':
            dyear = self.stop.item().year - self.start.item().year
            dmonth = self.stop.item().month - self.start.item().month
            range_ = 12*dyear + dmonth
            step = self.step.item()

        else:
            range_ = self.stop - self.start
            step = self.step

        return max(0, int(np.floor(range_/step + 1e-12) + 1))

    def _bounds(self):
        lo = self.start
        hi = add_coord(self.start, self.step * (self.size - 1))
        if self.is_descending:
            lo, hi = hi, lo
        return lo, hi

    def _segment_area_bounds(self):
        lo, hi = self._bounds()
        if self.ctype == 'left':
            hi = add_coord(hi, np.abs(self.step))
        elif self.ctype == 'right':
            lo = add_coord(lo, -np.abs(self.step))
        elif self.ctype == 'midpoint':
            lo = add_coord(lo, -0.5*np.abs(self.step)) # TODO datetimes
            hi = add_coord(hi,  0.5*np.abs(self.step)) # TODO datetimes
        return lo, hi
    
    @property
    def is_monotonic(self):
        ''' True (the coordinates are guaranteed to be in-order). '''

        return True

    @property
    def is_descending(self):
        ''' True if the coordinates are descending, False if ascending. '''

        if self.start == self.stop:
            return None
        
        return self.stop < self.start

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def _select(self, bounds, ind=False, pad=1):
        lo = max(bounds[0], self.bounds[0])
        hi = min(bounds[1], self.bounds[1])
        
        imin = int(np.ceil((lo - self.bounds[0]) / np.abs(self.step)))
        imax = int(np.floor((hi - self.bounds[0]) / np.abs(self.step)))

        imax = np.clip(imax+pad+1, 0, self.size)
        imin = np.clip(imin-pad, 0, self.size)
        
        # empty case
        if imin >= imax:
            if ind:
                return slice(0, 0)
            else:
                return ArrayCoordinates1d()

        if self.is_descending:
            imax, imin = self.size - imin, self.size - imax

        if ind:
            return slice(imin, imax)
            
        start = self.start + imin*self.step
        stop = self.start + (imax-1)*self.step
        return UniformCoordinates1d(start, stop, self.step, **self.properties)

    def _add(self, other):
        start = add_coord(self.start, other)
        stop = add_coord(self.stop, other)
        return UniformCoordinates1d(start, stop, self.step, **self.properties)

    def _add_equal(self, other):
        self.start = add_coord(self.start, other)
        self.stop = add_coord(self.stop, other)
        return self

    def _concat(self, other):
        # empty other
        if other.size == 0:
            return copy.deepcopy(self)
        
        # MonotonicCoordinates1d other
        if isinstance(other, MonotonicCoordinates1d):
            return MonotonicCoordinates1d(self.coordinates).concat(other)
        
        # ArrayCoordinates1d other
        if isinstance(other, ArrayCoordinates1d):
            return ArrayCoordinates1d(self.coordinates).concat(other)

        if isinstance(other, UniformCoordinates1d):
            # mismatched step
            if np.abs(np.abs(self.step) - np.abs(other.step)).astype(float) > TOL:
                return MonotonicCoordinates1d(self.coordinates).concat(other)

            start, stop, last, size = self._get_concat_values(other)

            # aligned -> return UniformCoordinates1d
            if self.size + other.size == size:
                return UniformCoordinates1d(start, stop, self.step, **self.properties)
            
            # separated (no overlap) -> return MonotonicCoordinates1d
            elif self.size + other.size < size:
                return MonotonicCoordinates1d(self.coordinates).concat(other)

            # overlapping -> return MonotonicCoordinates1d or ArrayCoordinates1d
            else:
                return MonotonicCoordinates1d(self.coordinates).concat(other)

    def _concat_equal(self, other):
        if other.size == 0:
            return self

        if not isinstance(other, UniformCoordinates1d):
            raise TypeError("Cannot concatenate '%s' to '%s' in-place" % (
                other.__class__.__name__, self.__class__.__name__))

        if np.abs(np.abs(self.step) - np.abs(other.step)).astype(float) > TOL:
            raise ValueError("Cannot concatenate UniformCoordinates1d, step mismatch (%f != %f)" % (
                self.step, other.step))

        start, stop, last, size = self._get_concat_values(other)

        # aligned
        if self.size + other.size == size:
            self.stop, self.start = stop, start
            return self

        # separated (no overlap)
        if self.size + other.size < size:
            raise ValueError("Cannot concatenate UniformCoordinates1d, ranges are separated")

        # overlapping
        else:
            raise ValueError("Cannot concatenate UniformCoordinates1d, ranges are overlapping")

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

        size = np.floor((last - start) / self.step) + 1

        return start, stop, last, size

    def __getitem__(self, index):
        if isinstance(index, int):
            if index >= self.size or index < -self.size:
                raise IndexError('index %d is out of bounds for coordinates with size %d' % (index, self.size))
            if index > 0:
                value = add_coord(self.start, self.step * index)
            else:
                value = add_coord(self.start, self.step * (self.size+index))

            return UniformCoordinates1d(value, value, self.step, name=self.name, **self.properties)

        elif isinstance(index, slice):
            if index.start is None:
                start = self.start
            elif index.start >= 0:
                start = add_coord(self.start, self.step * min(index.start, self.size-1))
            else:
                start = add_coord(self.start, self.step * max(0, self.size+index.start))

            if index.stop is None:
                stop = self.stop
            elif index.stop >= 0:
                stop = add_coord(self.start, self.step * (min(index.stop, self.size)-1))
            else:
                stop = add_coord(self.start, self.step * max(0, self.size+index.stop-1))

            if index.step is None:
                step = self.step
            else:
                step = index.step * self.step
                if index.step < 0:
                    start, stop = stop, start

            return UniformCoordinates1d(start, stop, step, name=self.name, **self.properties)

        else:
            coords = self.coordinates[index]
            try:
                return MonotonicCoordinates1d(coords, name=self.name, **self.properties)
            except ValueError:
                return ArrayCoordinates1d(coords, name=self.name, **self.properties)

# ---------------------------------------------------------------------------------------------------------------------
# Shorthand
# ---------------------------------------------------------------------------------------------------------------------

def _ca(coords=[], **kwargs):
    return ArrayCoordinates1d(coords, **kwargs)

def _cm(coords=[], **kwargs):
    return MonotonicCoordinates1d(coords, **kwargs)

def _cu(start, stop, step, **kwargs):
    return UniformCoordinates1d(start, stop, step, **kwargs)

def _cl(start, stop, size, **kwargs):
    return ArrayCoordinates1d(start, stop, size=size, **kwargs)

def _ca_lat(self, coords=[], **kwargs):
    return ArrayCoordinates1d(coords, name='lat', **kwargs)

def _ca_lon(self, coords=[], **kwargs):
    return ArrayCoordinates1d(coords, name='lon', **kwargs)

def _ca_alt(self, coords=[], **kwargs):
    return ArrayCoordinates1d(coords, name='alt', **kwargs)

def _ca_time(self, coords=[], **kwargs):
    return ArrayCoordinates1d(coords, name='time', **kwargs)

def _cm_lat(self, coords=[], **kwargs):
    return MonotonicCoordinates1d(coords, name='lat', **kwargs)
    
def _cm_lon(self, coords=[], **kwargs):
    return MonotonicCoordinates1d(coords, name='lon', **kwargs)

def _cm_alt(self, coords=[], **kwargs):
    return MonotonicCoordinates1d(coords, name='alt', **kwargs)

def _cm_time(self, coords=[], **kwargs):
    return MonotonicCoordinates1d(coords, name='time', **kwargs)

def _cu_lat(self, start, stop, step, **kwargs):
    return MonotonicCoordinates1d(start, stop, step, name='lat', **kwargs)
    
def _cu_lon(self, start, stop, step, **kwargs):
    return MonotonicCoordinates1d(start, stop, step, name='lon', **kwargs)

def _cu_alt(self, start, stop, step, **kwargs):
    return MonotonicCoordinates1d(start, stop, step, name='alt', **kwargs)

def _cu_time(self, start, stop, step, **kwargs):
    return MonotonicCoordinates1d(start, stop, step, name='time', **kwargs)

def _cl_lat(self, start, stop, size, **kwargs):
    return MonotonicCoordinates1d(start, stop, size=size, name='lat', **kwargs)
    
def _cl_lon(self, start, stop, size, **kwargs):
    return MonotonicCoordinates1d(start, stop, size=size, name='lon', **kwargs)

def _cl_alt(self, start, stop, size, **kwargs):
    return MonotonicCoordinates1d(start, stop, size=size, name='alt', **kwargs)

def _cl_time(self, start, stop, size, **kwargs):
    return MonotonicCoordinates1d(start, stop, size=size, name='time', **kwargs)