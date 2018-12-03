
from __future__ import division, unicode_literals, print_function, absolute_import

import copy
from collections import OrderedDict

import numpy as np
import traitlets as tl
from collections import OrderedDict

# from podpac.core.utils import cached_property, clear_cache
from podpac.core.units import Units
from podpac.core.coordinates.utils import make_coord_value, make_coord_delta, add_coord
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d

class UniformCoordinates1d(Coordinates1d):
    """
    1-dimensional array of uniformly-spaced coordinates defined by a start, stop, and step.

    UniformCoordinates1d efficiently stores a uniformly-spaced coordinate array; the full coordinates array is only
    calculated when needed. For numerical coordinates, the start, stop, and step are converted to ``float``. For time
    coordinates, the start and stop are converted to numpy ``datetime64``, and the step is converted to numpy
    ``timedelta64``. For convenience, podpac automatically converts datetime strings such as ``'2018-01-01'`` to
    ``datetime64`` and timedelta strings such as ``'1,D'`` to ``timedelta64``.

    UniformCoordinates1d can also be created by specifying the size instead of the step.
    
    Parameters
    ----------
    start : float or datetime64
        Start coordinate.
    stop : float or datetime64
        Stop coordinate.
    step : float or timedelta64
        Signed, non-zero step between coordinates.
    name : str
        Dimension name, one of 'lat', 'lon', 'time', 'alt'.
    coordinates : array, read-only
        Full array of coordinate values.
    units : podpac.Units
        Coordinate units.
    coord_ref_sys : str
        Coordinate reference system.
    ctype : str
        Coordinates type, one of'point', 'left', 'right', or 'midpoint'.
    extents : ndarray
        When ctype != 'point', defines custom area bounds for the coordinates.
        *Note: To be replaced with segment_lengths.*

    See Also
    --------
    :class:`Coordinates1d`, :class:`ArrayCoordinates1d`, :class:`crange`, :class:`clinspace`
    """

    start = tl.Union([tl.Float(), tl.Instance(np.datetime64)])
    start.__doc__ = ":float, datetime64: Start coordinate."
    
    stop = tl.Union([tl.Float(), tl.Instance(np.datetime64)])
    stop.__doc__ = ":float, datetime64: Stop coordinate."

    step = tl.Union([tl.Float(), tl.Instance(np.timedelta64)])
    step.__doc__ = ":float, timedelta64: Signed, non-zero step between coordinates."

    is_monotonic = tl.CBool(True, readonly=True)
    is_monotonic.__doc__ = ":bool: Are the coordinate values unique and sorted (always True)."
    
    is_uniform = tl.CBool(True, readonly=True)
    is_uniform.__doc__ = ":bool: Are the coordinate values uniformly-spaced (always True)."

    def __init__(self, start, stop, step=None, size=None, name=None, ctype=None, units=None, coord_ref_sys=None, extents=None):
        """
        Create uniformly-spaced 1d coordinates from a `start`, `stop`, and `step` or `size`.

        Parameters
        ----------
        start : float or datetime64
            Start coordinate.
        stop : float or datetime64
            Stop coordinate.
        step : float or timedelta64
            Signed, nonzero step between coordinates (either step or size required).
        size : int
            Number of coordinates (either step or size required).
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

        kwargs = {}
        if name is not None:
            kwargs['name'] = name
        if ctype is not None:
            kwargs['ctype'] = ctype
        if units is not None:
            kwargs['units'] = units
        if coord_ref_sys is not None:
            kwargs['coord_ref_sys'] = coord_ref_sys
        if extents is not None:
            kwargs['extents'] = extents

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

    @tl.validate('extents')
    def _validate_extents(self, d):
        return super(UniformCoordinates1d, self)._validate_extents(d)

    @tl.default('coord_ref_sys')
    def _default_coord_ref_sys(self):
        return super(UniformCoordinates1d, self)._default_coord_ref_sys()
    
    @tl.default('ctype')
    def _default_ctype(self):
        return super(UniformCoordinates1d, self)._default_ctype()

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
    def from_definition(cls, d):
        """
        Create uniformly-spaced 1d Coordinates from a coordinates definition.

        The definition must contain the coordinate start, stop, and step or size::

            c = UniformCoordinates1d.from_definition({
                "start": 1,
                "stop": 10,
                "step": 0.5
            })

            c = UniformCoordinates1d.from_definition({
                "start": 1,
                "stop": 10,
                "size": 21
            })

        The definition may also contain any of the 1d Coordinates properties::

            c = UniformCoordinates1d.from_definition({
                "start": 1,
                "stop": 10,
                "step": 0.5,
                "name": "lat",
                "ctype": "points"
            })

        Arguments
        ---------
        d : dict
            uniform 1d coordinates definition

        Returns
        -------
        :class:`UniformCoordinates1d`
            uniformly-spaced 1d Coordinates

        See Also
        --------
        definition
        """

        start = d.pop('start')
        stop = d.pop('stop')
        return cls(start, stop, **d)

    def copy(self, **kwargs):
        """
        Make a deep copy of the uniform 1d Coordinates.

        The coordinates properties will be copied. Any provided keyword arguments will override these properties.

        Arguments
        ---------
        name : str, optional
            Dimension name. One of 'lat', 'lon', 'alt', and 'time'.
        units : str, optional
            Coordinates units.
        coord_ref_sys : str, optional
            Coordinates reference system.
        ctype : str, optional
            Coordinates type. One of 'point', 'midpoint', 'left', 'right'.

        Returns
        -------
        :class:`UniformCoordinates1d`
            Copy of the coordinates, with provided properties.
        """

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
        """:array, read-only: Coordinate values. """

        coordinates = add_coord(self.start, np.arange(0, self.size) * self.step)
        coordinates.setflags(write=False)
        return coordinates

    @property
    def size(self):
        """ Number of coordinates. """

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
    def dtype(self):
        """ :type: Coordinates dtype.

        ``float`` for numerical coordinates and numpy ``datetime64`` for datetime coordinates.
        """

        return type(self.start)

    @property
    def bounds(self):
        """ Low and high coordinate bounds. """

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
        """
        Low and high coordinate area bounds.

        When ctype != 'point', this includes the portions of the segments beyond the coordinate bounds.
        """

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

    @property
    def definition(self):
        """:dict: Serializable uniform 1d coordinates definition.

        The ``definition`` can be used to create new UniformCoordinates1d::

            c = podpac.UniformCoordinates1d(0, 10, step=1)
            c2 = podpac.UniformCoordinates1d.from_definition(c.definition)

        See Also
        --------
        from_definition
        """

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
        """
        Get the coordinate values that are within the given bounds.

        The default selection returns coordinates that are within the other coordinates bounds::

            In [1]: c = UniformCoordinates1d(0, 3, step=1, name='lat')

            In [2]: c.select([1.5, 2.5]).coordinates
            Out[2]: array([2.])

        The *outer* selection returns the minimal set of coordinates that contain the other coordinates::
        
            In [3]: c.intersect([1.5, 2.5], outer=True).coordinates
            Out[3]: array([1., 2., 3.])

        The *outer* selection also returns a boundary coordinate if the other coordinates are outside this
        coordinates bounds but *inside* its area bounds::
        
            In [4]: c.intersect([3.25, 3.35], outer=True).coordinates
            Out[4]: array([3.0], dtype=float64)

            In [5]: c.intersect([10.0, 11.0], outer=True).coordinates
            Out[5]: array([], dtype=float64)
        
        Arguments
        ---------
        bounds : low, high
            selection bounds
        outer : bool, optional
            If True, do an *outer* selection. Default False.
        return_indices : bool, optional
            If True, return slice or indices for the selection in addition to coordinates. Default False.

        Returns
        -------
        selection : :class:`UniformCoordinates`
            UniformCoordinates1d object with coordinates within the other coordinates bounds.
        I : slice or list
            index or slice for the intersected coordinates (only if return_indices=True)
        """

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
