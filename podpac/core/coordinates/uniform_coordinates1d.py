
from __future__ import division, unicode_literals, print_function, absolute_import

import copy
from collections import OrderedDict

import numpy as np
import traitlets as tl

# from podpac.core.utils import cached_property, clear_cache
from podpac.core.coordinates.utils import make_coord_value, make_coord_delta, add_coord
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d

class UniformCoordinates1d(Coordinates1d):
    """
    An array of sorted, uniformly-spaced coordinates defined by a start, stop, and step.
    
    Attributes
    ----------
    start : float or datetime64
        Start coordinate.
        Numerical inputs are cast as floats and non-numerical inputs are parsed as datetime64.
    stop : float or datetime64
        Stop coordinate.
        Numerical inputs are cast as floats and non-numerical inputs are parsed as datetime64.
    step : float or timedelta64
        Signed, non-zero step between coordinates.
        Numerical inputs are cast as floats and non-numerical inputs are parsed as timedelta64.
    
    See Also
    --------
    ArrayCoordinates1d : An array of coordinates.
    """

    start = tl.Union([tl.Float(), tl.Instance(np.datetime64)])
    stop = tl.Union([tl.Float(), tl.Instance(np.datetime64)])
    step = tl.Union([tl.Float(), tl.Instance(np.timedelta64)])
    
    is_monotonic = tl.CBool(True, readonly=True)
    is_uniform = tl.CBool(True, readonly=True)

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
    def _observe_coords(self, change):
        # clear_cache(self, change, ['coordinates', 'bounds'])

        if self.start == self.stop:
            self.set_trait('is_descending', None)
        else:
            self.set_trait('is_descending', bool(self.stop < self.start))

    # ------------------------------------------------------------------------------------------------------------------
    # Alternate Constructors
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_tuple(cls, items, **kwargs):
        if len(items) != 3:
            raise ValueError("Cannot parse, todo better message")
        elif isinstance(items[2], int):
            return cls(items[0], items[1], size=items[2], **kwargs)
        else:
            step = make_coord_delta(items[2])
            return cls(items[0], items[1], step, **kwargs)

    @classmethod
    def from_json(self, d):
        start = d.pop('start')
        stop = d.pop('stop')
        step = d.pop('step')
        return cls(start, stop, step, **d)

    def copy(self, **kwargs):
        properties = self.properties
        properties.update(kwargs)
        return UniformCoordinates1d(self.start, self.stop, self.step, **properties)

    # -----------------------------------------------------------------------------------------------------------------
    # Standard methods, array-like
    # -----------------------------------------------------------------------------------------------------------------

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if isinstance(index, int):
            if index >= self.size or index < -self.size:
                raise IndexError('index %d is out of bounds for coordinates with size %d' % (index, self.size))
            if index > 0:
                value = add_coord(self.start, self.step * index)
            else:
                value = add_coord(self.start, self.step * (self.size+index))

            return UniformCoordinates1d(value, value, self.step, **self.properties)

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

            return UniformCoordinates1d(start, stop, step, **self.properties)

        else:
            return ArrayCoordinates1d(self.coordinates[index], **self.properties)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def coordinates(self):
        """ Coordinate values """
        coordinates = add_coord(self.start, np.arange(0, self.size) * self.step)
        coordinates.setflags(write=False)
        return coordinates
    
    @property
    def dtype(self):
        ''' Coordinate dtype, datetime or float '''
        
        return type(self.start)

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

    @property
    def bounds(self):
        lo = self.start
        hi = add_coord(self.start, self.step * (self.size - 1))
        if self.is_descending:
            lo, hi = hi, lo
        
        # read-only array with the correct dtype
        bounds = np.array([lo, hi], dtype=self.dtype)
        bounds.setflags(write=False)
        return bounds

    @property
    def area_bounds(self):
        # point ctypes, just use bounds
        if self.ctype == 'point':
            return self.bounds
        
        # segment ctypes, with explicit extents
        if self.extents is not None:
            return self.extents

        # segment ctypes, calculated            
        lo, hi = self.bounds
        if self.ctype == 'left':
            hi = add_coord(hi, np.abs(self.step))
        elif self.ctype == 'right':
            lo = add_coord(lo, -np.abs(self.step))
        elif self.ctype == 'midpoint':
            # TODO datetimes, need a dived_coord method
            lo = add_coord(lo, -np.abs(self.step)/2.0)
            hi = add_coord(hi,  np.abs(self.step)/2.0)

        # read-only array with the correct dtype
        area_bounds = np.array([lo, hi], dtype=self.dtype)
        area_bounds.setflags(write=False)
        return area_bounds

    def json(self):
        d = OrderedDict()
        if self.dtype == float:
            d['start'] = self.start
            d['stop'] = self.stop
            d['step'] = self.step
        else:
            d['start'] = str(self.start)
            d['stop'] = str(self.stop)
            d['step'] = str(self.step)
        d.update(self.properties)
        return d

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def select(self, bounds, outer=False, return_indices=False):
        bounds = make_coord_value(bounds[0]), make_coord_value(bounds[1])

        # full
        if self.bounds[0] >= bounds[0] and self.bounds[1] <= bounds[1]:
            return self._select_full(return_indices)

        # none
        if self.area_bounds[0] > bounds[1] or self.area_bounds[1] < bounds[0]:
            return self._select_empty(return_indices)

        # TODO is there an easier way to do this with the new outer flag?

        lo = max(bounds[0], self.bounds[0])
        hi = min(bounds[1], self.bounds[1])
        
        imin = int(np.ceil((lo - self.bounds[0]) / np.abs(self.step)))
        imax = int(np.floor((hi - self.bounds[0]) / np.abs(self.step)))

        if outer:
            imin -= 1
            imax += 1

        imax = np.clip(imax+1, 0, self.size)
        imin = np.clip(imin, 0, self.size)
        
        # empty case
        if imin >= imax:
            return self._select_empty(return_indices)

        if self.is_descending:
            imax, imin = self.size - imin, self.size - imax

        start = self.start + imin*self.step
        stop = self.start + (imax-1)*self.step
        c = UniformCoordinates1d(start, stop, self.step, **self.properties)
        
        if return_indices:
            return c, slice(imin, imax)
        else:
            return c