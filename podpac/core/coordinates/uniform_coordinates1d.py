from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy
from collections import OrderedDict

import numpy as np
import traitlets as tl

from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.utils import (add_coord, divide_delta,
                                           lower_precision_time_bounds,
                                           make_coord_delta, make_coord_value,
                                           timedelta_divisible)
from podpac.core.utils import cached_property


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
        Stop coordinate. Unless fix_stop_val == True at creation, this may not always be
        exactly equal to what the user specified. Internally we ensure that stop = start + step * (size - 1)
    step : float or timedelta64
        Signed, non-zero step between coordinates. Note, the specified step my be changed internally to satisfy floating point consistency.
        That is, the consistent step will ensure that step = (stop - start)  / (size - 1)
    name : str
        Dimension name, one of 'lat', 'lon', 'time', 'alt'.
    coordinates : array, read-only
        Full array of coordinate values.

    See Also
    --------
    :class:`Coordinates1d`, :class:`ArrayCoordinates1d`, :class:`crange`, :class:`clinspace`
    """

    start = tl.Union([tl.Float(), tl.Instance(np.datetime64), tl.Instance(np.timedelta64)], read_only=True)
    start.__doc__ = ":float, datetime64: Start coordinate."

    stop = tl.Union([tl.Float(), tl.Instance(np.datetime64), tl.Instance(np.timedelta64)], read_only=True)
    stop.__doc__ = ":float, datetime64: Stop coordinate."

    step = tl.Union([tl.Float(), tl.Instance(np.timedelta64)], read_only=True)
    step.__doc__ = ":float, timedelta64: Signed, non-zero step between coordinates."

    def __init__(self, start, stop, step=None, size=None, name=None, fix_stop_val=False):
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
        name : str, optional
            Dimension name, one of 'lat', 'lon', 'time', or 'alt'.
        fix_stop_val : bool, optional
            Default is False. If True, the constructor will modify the step to be consistent
            instead of the stop value. Otherwise, the stop value *may* be modified to ensure that
            stop = start + step * size

        Notes
        ------
        When the user specifies fix_stop_val, then `stop` will always be exact as specified by the user.

        For floating point coordinates, the specified `step` my be changed internally to satisfy floating point consistency.
        That is, for consistency `step = (stop - start)  / (size - 1)`
        """

        if step is not None and size is not None:
            raise TypeError("only one of 'step' and 'size' is allowed")
        elif step is None and size is None:
            raise TypeError("'step' or 'size' is required")

        # validate and set start, stop, and step
        start = make_coord_value(start)
        stop = make_coord_value(stop)

        if step is not None:
            step = make_coord_delta(step)
        elif isinstance(size, (int, np.int64, np.integer)) and not isinstance(size, np.timedelta64):
            step = divide_delta(stop - start, size - 1)
        else:
            raise TypeError("size must be an integer, not '%s'" % type(size))

        if isinstance(start, float) and isinstance(stop, float) and isinstance(step, float):
            fstep = step
        elif isinstance(start, np.datetime64) and isinstance(stop, np.datetime64) and isinstance(step, np.timedelta64):
            fstep = step.astype(float)
        elif (
            isinstance(start, np.timedelta64) and isinstance(stop, np.timedelta64) and isinstance(step, np.timedelta64)
        ):
            fstep = step.astype(float)
        else:
            raise TypeError(
                "UniformCoordinates1d mismatching types (start '%s', stop '%s', step '%s')."
                % (type(start), type(stop), type(step))
            )

        if fstep == 0:
            raise ValueError("Uniformcoordinates1d step cannot be zero")

        if fstep <= 0 and start < stop:
            raise ValueError("UniformCoordinates1d step must be greater than zero if start < stop.")

        if fstep >= 0 and start > stop:
            raise ValueError("UniformCoordinates1d step must be less than zero if start > stop.")

        self.set_trait("start", start)
        self.set_trait("stop", stop)
        self.set_trait("step", step)

        if not fix_stop_val:  # Need to make sure that 'stop' is consistent with self.coordinates[-1]
            self.set_trait("stop", add_coord(self.start, (self.size - 1) * self.step))

        # Make sure step is floating-point error consistent in all cases
        # This is only needed when the type is float
        if fstep == step and self.size > 1:
            step = divide_delta(self.stop - self.start, self.size - 1)
            self.set_trait("step", step)

        # set common properties
        super(UniformCoordinates1d, self).__init__(name=name)

    def __eq__(self, other):
        if not self._eq_base(other):
            return False

        if isinstance(other, UniformCoordinates1d):
            if self.dtype == float:
                if not np.allclose([self.start, self.stop, self.step], [other.start, other.stop, other.step]):
                    return False
            elif self.start != other.start or self.stop != other.stop or self.step != other.step:
                return False

        if isinstance(other, ArrayCoordinates1d):
            if self.dtype == float:
                if not np.allclose(self.coordinates, other.coordinates):
                    return False
            else:
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
                "UniformCoordinates1d.from_tuple expects a tuple of (start, stop, step/size), got %s" % (items,)
            )
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
                "name": "lat"
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

        if "start" not in d:
            raise ValueError('UniformCoordinates1d definition requires "start" property')
        if "stop" not in d:
            raise ValueError('UniformCoordinates1d definition requires "stop" property')

        start = d["start"]
        stop = d["stop"]
        kwargs = {k: v for k, v in d.items() if k not in ["start", "stop"]}
        return cls(start, stop, **kwargs)

    # -----------------------------------------------------------------------------------------------------------------
    # Standard methods, array-like
    # -----------------------------------------------------------------------------------------------------------------

    def __getitem__(self, index):
        # fallback for non-slices
        if not isinstance(index, slice):
            # The following 3 lines is copied from ArrayCoordinates1d.__getitem__
            if self.ndim == 1 and np.ndim(index) > 1 and np.array(index).dtype == int:
                index = np.array(index).flatten().tolist()
            return ArrayCoordinates1d(self.coordinates[index], **self.properties)

        # start, stop, step
        if index.start is None:
            start = self.start
        elif index.start >= 0:
            start = add_coord(self.start, self.step * min(index.start, self.size - 1))
        else:
            start = add_coord(self.start, self.step * max(0, self.size + index.start))

        if index.stop is None:
            stop = self.stop
        elif index.stop >= 0:
            stop = add_coord(self.start, self.step * (min(index.stop, self.size) - 1))
        else:
            stop = add_coord(self.start, self.step * max(0, self.size + index.stop - 1))

        if index.step is None:
            step = self.step
        else:
            step = index.step * self.step
            if index.step < 0:
                start, stop = stop, start

        # empty slice
        if ((start > stop) and np.array(step).astype(float) > 0) or (
            (start < stop) and np.array(step).astype(float) < 0
        ):
            return ArrayCoordinates1d([], **self.properties)
        return UniformCoordinates1d(start, stop, step, **self.properties)

    def __contains__(self, item):
        # overrides the Coordinates1d.__contains__ method with optimizations for uniform coordinates.

        try:
            item = make_coord_value(item)
        except:
            return False

        if type(item) != self.dtype:
            return False

        if item < self.bounds[0] or item > self.bounds[1]:
            return False

        if (self.dtype == np.datetime64) or (self.dtype == np.timedelta64):
            return timedelta_divisible(item - self.start, self.step)
        else:
            return (item - self.start) % self.step == 0

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @cached_property
    def coordinates(self):
        """:array, read-only: Coordinate values."""

        coordinates = add_coord(self.start, np.arange(0, self.size) * self.step)
        # coordinates.setflags(write=False)  # This breaks the 002-open-point-file example
        return coordinates

    @property
    def ndim(self):
        return 1

    @property
    def shape(self):
        return (self.size,)

    @property
    def size(self):
        """Number of coordinates."""

        dname = np.array(self.step).dtype.name

        if dname == "timedelta64[Y]":
            dyear = self.stop.item().year - self.start.item().year
            if dyear > 0 and self.stop.item().month < self.start.item().month:
                dyear -= 1
            range_ = dyear
            step = self.step.item()

        elif dname == "timedelta64[M]":
            dyear = self.stop.item().year - self.start.item().year
            dmonth = self.stop.item().month - self.start.item().month
            range_ = 12 * dyear + dmonth
            step = self.step.item()

        else:
            range_ = self.stop - self.start
            step = self.step

        return max(0, int(np.floor(range_ / step + 1e-10) + 1))

    @property
    def dtype(self):
        """:type: Coordinates dtype.

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
        """Low and high coordinate bounds."""

        lo = self.start
        hi = add_coord(self.start, self.step * (self.size - 1))
        if self.is_descending:
            lo, hi = hi, lo
        return lo, hi

    @property
    def argbounds(self):
        if self.is_descending:
            return -1, 0
        else:
            return 0, -1

    def _get_definition(self, full=True):
        d = OrderedDict()
        d["start"] = self.start
        d["stop"] = self.stop
        d["step"] = self.step
        d.update(self._full_properties if full else self.properties)
        return d

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def copy(self):
        """
        Make a deep copy of the uniform 1d Coordinates.

        Returns
        -------
        :class:`UniformCoordinates1d`
            Copy of the coordinates.
        """

        kwargs = self.properties
        return UniformCoordinates1d(self.start, self.stop, self.step, **kwargs)

    def unique(self, return_index=False):
        """
        Return the coordinates (uniform coordinates are already unique).

        Arguments
        ---------
        return_index : bool, optional
            If True, return index for the unique coordinates in addition to the coordinates. Default False.

        Returns
        -------
        unique : :class:`ArrayCoordinates1d`
            New ArrayCoordinates1d object with unique, sorted coordinate values.
        unique_index : list of indices
            index
        """

        if return_index:
            return self.copy(), np.arange(self.size).tolist()
        else:
            return self.copy()

    def simplify(self):
        """Get the simplified/optimized representation of these coordinates.

        Returns
        -------
        simplified : UniformCoordinates1d
            These coordinates (the coordinates are already simplified).
        """

        return self.copy()

    def flatten(self):
        """
        Return a copy of the uniform coordinates, for consistency.

        Returns
        -------
        :class:`UniformCoordinates1d`
            Flattened coordinates.
        """

        return self.copy()

    def reshape(self, newshape):
        return ArrayCoordinates1d(self.coordinates, **self.properties).reshape(newshape)

    def issubset(self, other):
        """Report whether other coordinates contains these coordinates.

        Arguments
        ---------
        other : Coordinates, Coordinates1d
            Other coordinates to check

        Returns
        -------
        issubset : bool
            True if these coordinates are a subset of the other coordinates.

        Notes
        -----
        This overrides the Coordinates1d.issubset method with optimizations for uniform coordinates.
        """

        from podpac.core.coordinates import Coordinates

        if isinstance(other, Coordinates):
            if self.name not in other.dims:
                return False
            other = other[self.name]

        # use Coordinates1d implementation when the other coordinates are not uniform
        if not other.is_uniform:
            return super(UniformCoordinates1d, self).issubset(other)

        # use Coordinates1d implementation when the steps cannot be compared (e.g. months and days)
        try:
            self.step / other.step
        except TypeError:
            return super(UniformCoordinates1d, self).issubset(other)

        # short-cuts that don't require checking coordinates
        if self.dtype != other.dtype:
            return False

        if self.bounds[0] < other.bounds[0] or self.bounds[1] > other.bounds[1]:
            return False

        # check start and step
        if self.start not in other:
            return False

        if self.size == 1:
            return True

        if (self.dtype == np.datetime64) or (self.dtype == np.timedelta64):
            return timedelta_divisible(self.step, other.step)
        else:
            return self.step % other.step == 0

    def _select(self, bounds, return_index, outer):
        # TODO is there an easier way to do this with the new outer flag?
        my_bounds = self.bounds

        # If the bounds are of instance datetime64, then the comparison should happen at the lowest precision
        if self.dtype == np.datetime64:
            my_bounds, bounds = lower_precision_time_bounds(my_bounds, bounds, outer)

        lo = max(bounds[0], my_bounds[0])
        hi = min(bounds[1], my_bounds[1])

        fmin = (lo - my_bounds[0]) / np.abs(self.step)
        fmax = (hi - my_bounds[0]) / np.abs(self.step)
        imin = int(np.ceil(fmin))
        imax = int(np.floor(fmax))

        if outer:
            if imin != fmin:
                imin -= 1
            if imax != fmax:
                imax += 1

        imax = np.clip(imax + 1, 0, self.size)
        imin = np.clip(imin, 0, self.size)

        # empty case
        if imin > imax:
            return self._select_empty(return_index)
        if imax == imin:
            # could have been selected between two existing coordinates
            imin = int(np.round(fmin))
            imax = int(np.round(fmax)) + 1
            if imin >= (self.size - 1) | imin < 0:
                return self._select_empty(return_index)

        if self.is_descending:
            imax, imin = self.size - imin, self.size - imax

        I = slice(imin, imax)
        if return_index:
            return self[I], I
        else:
            return self[I]
