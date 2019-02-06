
from __future__ import division, unicode_literals, print_function, absolute_import

import copy
from collections import OrderedDict

import numpy as np
import traitlets as tl
from collections import OrderedDict

# from podpac.core.utils import cached_property, clear_cache
from podpac.core.units import Units
from podpac.core.coordinates.utils import make_coord_value, make_coord_delta, add_coord, divide_delta
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
    segment_lengths : array, float, timedelta
        When ctype is a segment type, the segment lengths for the coordinates.

    See Also
    --------
    :class:`Coordinates1d`, :class:`ArrayCoordinates1d`, :class:`crange`, :class:`clinspace`
    """

    start = tl.Union([tl.Float(), tl.Instance(np.datetime64)], read_only=True)
    start.__doc__ = ":float, datetime64: Start coordinate."
    
    stop = tl.Union([tl.Float(), tl.Instance(np.datetime64)], read_only=True)
    stop.__doc__ = ":float, datetime64: Stop coordinate."

    step = tl.Union([tl.Float(), tl.Instance(np.timedelta64)], read_only=True)
    step.__doc__ = ":float, timedelta64: Signed, non-zero step between coordinates."

    def __init__(self, start, stop, step=None, size=None,
                       name=None, ctype=None, units=None, coord_ref_sys=None, segment_lengths=None):
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
        segment_lengths: array, float, timedelta, optional
            When ctype is a segment type, the segment lengths for the coordinates. By defaul, the segment_lengths are
            equal the step.
        """

        if step is not None and size is not None:
            raise TypeError("only one of 'step' and 'size' is allowed")
        elif step is None and size is None:
            raise TypeError("'step' or 'size' is required")

        # validate and set start, stop, and step
        start = make_coord_value(start)
        stop = make_coord_value(stop)
        if step == 0:
            raise ValueError("step must be nonzero")
        elif step is not None:
            step = make_coord_delta(step)
        elif isinstance(size, (int, np.long, np.integer)) and not isinstance(size, np.timedelta64):
            step = divide_delta(stop - start, size - 1)
        else:
            raise TypeError("size must be an integer, not '%s'" % type(size))

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

        self.set_trait('start', start)
        self.set_trait('stop', stop)
        self.set_trait('step', step)

        # set common properties
        super(UniformCoordinates1d, self).__init__(
            name=name, ctype=ctype, units=units, segment_lengths=segment_lengths, coord_ref_sys=coord_ref_sys)
        
    @tl.default('ctype')
    def _default_ctype(self):
        return 'midpoint'

    @tl.default('segment_lengths')
    def _default_segment_lengths(self):
        if self.ctype == 'point':
            return None

        return np.abs(self.step)

    def __eq__(self, other):
        if not super(UniformCoordinates1d, self).__eq__(other):
            return False

        # not necessary
        # if isinstance(other, UniformCoordinates1d):
        #     if self.start != other.start or self.stop != other.stop or self.step != other.step:
        #         return False

        if isinstance(other, ArrayCoordinates1d):
            if not np.array_equal(self.coordinates, other.coordinates):
                return False
        
        return True

    # ------------------------------------------------------------------------------------------------------------------
    # Alternate Constructors
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_tuple(cls, items, **kwargs):
        if not isinstance(items, tuple) or len(items) != 3:
            raise ValueError(
                "UniformCoordinates1d.from_tuple expects a tuple of (start, stop, step/size), got %s" % (items,))
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

        if 'start' not in d:
            raise ValueError('UniformCoordinates1d definition requires "start" property')
        if 'stop' not in d:
            raise ValueError('UniformCoordinates1d definition requires "stop" property')

        start = d.pop('start')
        stop = d.pop('stop')
        return cls(start, stop, **d)

    def copy(self):
        """
        Make a deep copy of the uniform 1d Coordinates.

        Returns
        -------
        :class:`UniformCoordinates1d`
            Copy of the coordinates.
        """

        kwargs = self.properties
        if self._segment_lengths:
            kwargs['segment_lengths'] = self.segment_lengths
        return UniformCoordinates1d(self.start, self.stop, self.step, **kwargs)

    # -----------------------------------------------------------------------------------------------------------------
    # Standard methods, array-like
    # -----------------------------------------------------------------------------------------------------------------

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if isinstance(index, slice):
            # start, stop, step
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

            # properties and segment_lengths
            kwargs = self.properties
            
            if self.ctype != 'point':
                if isinstance(self.segment_lengths, np.ndarray):
                    kwargs['segment_lengths'] = self.segment_lengths[index]
                elif self.segment_lengths != step:
                    kwargs['segment_lengths'] = self.segment_lengths

            # reroute empty slices to the else clause
            if start > stop and step > 0:
                return self[[]]

            return UniformCoordinates1d(start, stop, step, **kwargs)

        else:
            # coordinates
            coords = self.coordinates[index]

            # properties and segment_lengths
            kwargs = self.properties
            
            if self.ctype != 'point':
                if isinstance(self.segment_lengths, np.ndarray):
                    kwargs['segment_lengths'] = self.segment_lengths[index]
                else:
                    kwargs['segment_lengths'] = self.segment_lengths

            kwargs['ctype'] = self.ctype

            return ArrayCoordinates1d(coords, **kwargs)

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
    def is_monotonic(self):
        return True

    @property
    def is_descending(self):
        if self.start == self.stop:
            return None
        
        return self.stop < self.start

    @property
    def is_uniform(self):
        return True

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
    def argbounds(self):
        if self.is_descending:
            return -1, 0
        else:
            return 0, -1

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
        d['start'] = self.start
        d['stop'] = self.stop
        d['step'] = self.step
        if self._segment_lengths:
            d['segment_lengths'] = self.segment_lengths
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

        fmin = (lo - self.bounds[0]) / np.abs(self.step)
        fmax = (hi - self.bounds[0]) / np.abs(self.step)
        imin = int(np.ceil(fmin))
        imax = int(np.floor(fmax))

        if outer:
            if imin != fmin:
                imin -= 1
            if imax != fmax:
                imax += 1

        imax = np.clip(imax+1, 0, self.size)
        imin = np.clip(imin, 0, self.size)

        # empty case
        if imin >= imax:
            return self._select_empty(return_indices)

        if self.is_descending:
            imax, imin = self.size - imin, self.size - imax

        I = slice(imin, imax)
        if return_indices:
            return self[I], I
        else:
            return self[I]
