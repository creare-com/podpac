"""
One-Dimensional Coordinates
"""


from __future__ import division, unicode_literals, print_function, absolute_import

import copy

import numpy as np
import traitlets as tl

from podpac.core.utils import ArrayTrait
from podpac.core.coordinates.utils import make_coord_value, make_coord_delta, make_coord_delta_array
from podpac.core.coordinates.utils import add_coord, divide_delta, lower_precision_time_bounds
from podpac.core.coordinates.utils import Dimension, CoordinateType
from podpac.core.coordinates.base_coordinates import BaseCoordinates


class Coordinates1d(BaseCoordinates):
    """
    Base class for 1-dimensional coordinates.

    Coordinates1d objects contain values and metadata for a single dimension of coordinates. :class:`Coordinates` and
    :class:`StackedCoordinates` use Coordinate1d objects.

    The following coordinates types (``ctype``) are supported:

     * 'point': each coordinate represents a single location
     * 'left': each coordinate is the left endpoint of its segment
     * 'right': each coordinate is the right endpoint of its endpoint
     * 'midpoint': segment endpoints are at the midpoints between coordinate values.

    The ``bounds`` are always the low and high coordinate value. For *point* coordinates, the ``area_bounds`` are the
    same as the ``bounds``. For *segment* coordinates (left, right, and midpoint), the ``area_bounds`` include the
    portion of the segments above and below the ``bounds`.
    
    Parameters
    ----------
    name : str
        Dimension name, one of 'lat', 'lon', 'time', or 'alt'.
    coordinates : array, read-only
        Full array of coordinate values.
    ctype : str
        Coordinates type: 'point', 'left', 'right', or 'midpoint'.
    segment_lengths : array, float, timedelta
        When ctype is a segment type, the segment lengths for the coordinates. This may be single coordinate delta for
        uniform segment lengths or an array of coordinate deltas corresponding to the coordinates for variable lengths.

    See Also
    --------
    :class:`ArrayCoordinates1d`, :class:`UniformCoordinates1d`
    """

    name = Dimension(allow_none=True)
    ctype = CoordinateType(read_only=True)
    segment_lengths = tl.Any(read_only=True)

    _properties = tl.Set()

    def __init__(self, name=None, ctype=None, segment_lengths=None):
        """*Do not use.*"""

        if name is not None:
            self.name = name

        if ctype is not None:
            self.set_trait("ctype", ctype)

        if segment_lengths is not None:
            if np.array(segment_lengths).ndim == 0:
                segment_lengths = make_coord_delta(segment_lengths)
            else:
                segment_lengths = make_coord_delta_array(segment_lengths)
                segment_lengths.setflags(write=False)

            self.set_trait("segment_lengths", segment_lengths)

        super(Coordinates1d, self).__init__()

    @tl.observe("name", "ctype", "segment_lengths")
    def _set_property(self, d):
        self._properties.add(d["name"])

    @tl.validate("segment_lengths")
    def _validate_segment_lengths(self, d):
        val = d["value"]

        if self.ctype == "point":
            if val is not None:
                raise TypeError("segment_lengths must be None when ctype='point'")
            return None

        if isinstance(val, np.ndarray):
            if val.size != self.size:
                raise ValueError("coordinates and segment_lengths size mismatch, %d != %d" % (self.size, val.size))
            if not np.issubdtype(val.dtype, np.dtype(self.deltatype).type):
                raise ValueError(
                    "coordinates and segment_lengths dtype mismatch, %s != %s" % (self.dtype, self.deltatype)
                )

        else:
            if self.size > 0 and not isinstance(val, self.deltatype):
                raise TypeError("coordinates and segment_lengths type mismatch, %s != %s" % (self.deltatype, type(val)))

        if np.any(np.array(val).astype(float) <= 0.0):
            raise ValueError("segment_lengths must be positive")

        return val

    def _set_name(self, value):
        # set name if it is not set already, otherwise check that it matches
        if "name" not in self._properties:
            self.name = value
        elif self.name != value:
            raise ValueError("Dimension mismatch, %s != %s" % (value, self.name))

    def _set_ctype(self, value):
        # only set ctype if it is not set already
        if "ctype" not in self._properties:
            self.set_trait("ctype", value)

    # ------------------------------------------------------------------------------------------------------------------
    # standard methods
    # ------------------------------------------------------------------------------------------------------------------

    def __repr__(self):
        return "%s(%s): Bounds[%s, %s], N[%d], ctype['%s']" % (
            self.__class__.__name__,
            self.name or "?",
            self.bounds[0],
            self.bounds[1],
            self.size,
            self.ctype,
        )

    def __eq__(self, other):
        if not isinstance(other, Coordinates1d):
            return False

        # defined coordinate properties should match
        for name in self._properties.union(other._properties):
            if name == "segment_lengths":
                if not np.all(self.segment_lengths == other.segment_lengths):
                    return False

            elif getattr(self, name) != getattr(other, name):
                return False

        # shortcuts (not strictly necessary)
        for name in ["size", "is_monotonic", "is_descending", "is_uniform"]:
            if getattr(self, name) != getattr(other, name):
                return False

        return True

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def dims(self):
        if self.name is None:
            raise TypeError("cannot access dims property of unnamed Coordinates1d")
        return (self.name,)

    @property
    def udims(self):
        return self.dims

    @property
    def idims(self):
        return self.dims

    @property
    def shape(self):
        return (self.size,)

    @property
    def coords(self):
        """:dict-like: xarray coordinates (container of coordinate arrays)"""

        return {self.name: self.coordinates}

    @property
    def dtype(self):
        """:type: Coordinates dtype.

        ``float`` for numerical coordinates and numpy ``datetime64`` for datetime coordinates.
        """

        raise NotImplementedError

    @property
    def deltatype(self):
        if self.dtype is np.datetime64:
            return np.timedelta64
        else:
            return self.dtype

    @property
    def is_monotonic(self):
        raise NotImplementedError

    @property
    def is_descending(self):
        raise NotImplementedError

    @property
    def is_uniform(self):
        raise NotImplementedError

    @property
    def bounds(self):
        """ Low and high coordinate bounds. """

        raise NotImplementedError

    @property
    def area_bounds(self):
        """
        Low and high coordinate area bounds.

        When ctype != 'point', this includes the portions of the segments beyond the coordinate bounds.
        """

        # point ctypes, just use bounds
        if self.ctype == "point":
            return self.bounds

        # empty coordinates [np.nan, np.nan]
        if self.size == 0:
            return self.bounds

        # segment ctypes, calculated
        L, H = self.argbounds
        lo, hi = self.bounds

        if not isinstance(self.segment_lengths, np.ndarray):
            lo_length = hi_length = self.segment_lengths  # uniform segment_lengths
        else:
            lo_length, hi_length = self.segment_lengths[L], self.segment_lengths[H]

        if self.ctype == "left":
            hi = add_coord(hi, hi_length)
        elif self.ctype == "right":
            lo = add_coord(lo, -lo_length)
        elif self.ctype == "midpoint":
            lo = add_coord(lo, -divide_delta(lo_length, 2.0))
            hi = add_coord(hi, divide_delta(hi_length, 2.0))

        # read-only array with the correct dtype
        area_bounds = np.array([lo, hi], dtype=self.dtype)
        area_bounds.setflags(write=False)
        return area_bounds

    @property
    def properties(self):
        """:dict: Dictionary of the coordinate properties. """

        return {key: getattr(self, key) for key in self._properties}

    @property
    def definition(self):
        """:dict: Serializable 1d coordinates definition."""
        return self._get_definition(full=False)

    @property
    def full_definition(self):
        """:dict: Serializable 1d coordinates definition, containing all properties. For internal use."""
        return self._get_definition(full=True)

    def _get_definition(self, full=True):
        raise NotImplementedError

    @property
    def _full_properties(self):
        return {"name": self.name, "ctype": self.ctype, "segment_lengths": self.segment_lengths}

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def copy(self):
        """
        Make a deep copy of the 1d Coordinates.

        Returns
        -------
        :class:`Coordinates1d`
            Copy of the coordinates.
        """

        raise NotImplementedError

    def _select_empty(self, return_indices):
        I = []
        if return_indices:
            return self[I], I
        else:
            return self[I]

    def _select_full(self, return_indices):
        I = slice(None)
        if return_indices:
            return self[I], I
        else:
            return self[I]

    def select(self, bounds, return_indices=False, outer=False):
        """
        Get the coordinate values that are within the given bounds.

        The default selection returns coordinates that are within the bounds::

            In [1]: c = ArrayCoordinates1d([0, 1, 2, 3], name='lat')

            In [2]: c.select([1.5, 2.5]).coordinates
            Out[2]: array([2.])

        The *outer* selection returns the minimal set of coordinates that contain the bounds::
        
            In [3]: c.select([1.5, 2.5], outer=True).coordinates
            Out[3]: array([1., 2., 3.])

        The *outer* selection also returns a boundary coordinate if a bound is outside this coordinates bounds but
        *inside* its area bounds::
        
            In [4]: c.select([3.25, 3.35], outer=True).coordinates
            Out[4]: array([3.0], dtype=float64)

            In [5]: c.select([10.0, 11.0], outer=True).coordinates
            Out[5]: array([], dtype=float64)
        
        Parameters
        ----------
        bounds : (low, high) or dict
            Selection bounds. If a dictionary of dim -> (low, high) bounds is supplied, the bounds matching these
            coordinates will be selected if available, otherwise the full coordinates will be returned.
        outer : bool, optional
            If True, do an *outer* selection. Default False.
        return_indices : bool, optional
            If True, return slice or indices for the selection in addition to coordinates. Default False.

        Returns
        -------
        selection : :class:`Coordinates1d`
            Coordinates1d object with coordinates within the bounds.
        I : slice or list
            index or slice for the selected coordinates (only if return_indices=True)
        """

        # empty case
        if self.dtype is None:
            return self._select_empty(return_indices)

        if isinstance(bounds, dict):
            bounds = bounds.get(self.name)
            if bounds is None:
                return self._select_full(return_indices)

        bounds = make_coord_value(bounds[0]), make_coord_value(bounds[1])

        # check type
        if not isinstance(bounds[0], self.dtype):
            raise TypeError(
                "Input bounds do match the coordinates dtype (%s != %s)" % (type(self.bounds[0]), self.dtype)
            )
        if not isinstance(bounds[1], self.dtype):
            raise TypeError(
                "Input bounds do match the coordinates dtype (%s != %s)" % (type(self.bounds[1]), self.dtype)
            )

        my_bounds = self.area_bounds.copy()

        # If the bounds are of instance datetime64, then the comparison should happen at the lowest precision
        if self.dtype == np.datetime64:
            my_bounds, bounds = lower_precision_time_bounds(my_bounds, bounds, outer)

        # full
        if my_bounds[0] >= bounds[0] and my_bounds[1] <= bounds[1]:
            return self._select_full(return_indices)

        # none
        if my_bounds[0] > bounds[1] or my_bounds[1] < bounds[0]:
            return self._select_empty(return_indices)

        # partial, implemented in child classes
        return self._select(bounds, return_indices, outer)

    def _select(self, bounds, return_indices, outer):
        raise NotImplementedError

    def _transform(self, transformer):
        from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d

        if self.name == "alt":
            # coordinates
            _, _, tcoordinates = transformer.transform(np.zeros(self.size), np.zeros(self.size), self.coordinates)

            # segment lengths
            properties = self.properties
            if self.ctype != "point" and "segment_lengths" in self.properties:
                _ = np.zeros_like(self.segment_lengths)
                _, _, tsl = transformer.transform(_, _, self.segment_lengths)
                properties["segment_lengths"] = tsl

            t = ArrayCoordinates1d(tcoordinates, **properties)

        else:
            # this assumes that the transformer has been checked and that if this is a lat or lon dimension, the
            # transformer must not have a spatial transform
            t = self.copy()

        return t
