"""
One-Dimensional Coordinates
"""


from __future__ import division, unicode_literals, print_function, absolute_import

import copy

import numpy as np
import traitlets as tl

from podpac.core.units import Units
from podpac.core.utils import ArrayTrait
from podpac.core.coordinates.utils import make_coord_delta, make_coord_delta_array, add_coord, divide_delta
from podpac.core.coordinates.base_coordinates import BaseCoordinates

DEFAULT_CRS = 'EPSG:4326'
""":str: Default coordinate reference system """

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
    units : podpac.Units
        Coordinate units.
    coord_ref_sys : str
        Coordinate reference system. Supports any PROJ4 compliant string (https://proj4.org/index.html).
        If not defined, set to :str:`DEFAULT_CRS`
    ctype : str
        Coordinates type: 'point', 'left', 'right', or 'midpoint'.
    segment_lengths : array, float, timedelta
        When ctype is a segment type, the segment lengths for the coordinates. This may be single coordinate delta for
        uniform segment lengths or an array of coordinate deltas corresponding to the coordinates for variable lengths.

    See Also
    --------
    :class:`ArrayCoordinates1d`, :class:`UniformCoordinates1d`
    """

    name = tl.Enum(['lat', 'lon', 'time', 'alt'], allow_none=True)
    units = tl.Instance(Units, allow_none=True, read_only=True)
    coord_ref_sys = tl.Unicode(default_value=None, allow_none=True)
    ctype = tl.Enum(['point', 'left', 'right', 'midpoint'], read_only=True)
    segment_lengths = tl.Any(read_only=True)

    _properties = tl.Set()
    _segment_lengths = tl.Bool()

    def __init__(self, name=None, ctype=None, units=None, segment_lengths=None, coord_ref_sys=None):
        """*Do not use.*"""

        if name is not None:
            self.name = name

        if ctype is not None:
            self.set_trait('ctype', ctype)

        if units is not None:
            self.set_trait('units', units)

        if coord_ref_sys is not None:
            self.set_trait('coord_ref_sys', coord_ref_sys)

        if segment_lengths is not None:
            if np.array(segment_lengths).ndim == 0:
                segment_lengths = make_coord_delta(segment_lengths)
            else:
                segment_lengths = make_coord_delta_array(segment_lengths)
                segment_lengths.setflags(write=False)
            
            self.set_trait('segment_lengths', segment_lengths)

        super(Coordinates1d, self).__init__()

    @tl.observe('name', 'units', 'coord_ref_sys', 'ctype')
    def _set_property(self, d):
        self._properties.add(d['name'])

    @tl.observe('segment_lengths')
    def _set_segment_lengths(self, d):
        self._segment_lengths = True

    @tl.validate('segment_lengths')
    def _validate_segment_lengths(self, d):
        val = d['value']
        
        if self.ctype == 'point':
            if val is not None:
                raise TypeError("segment_lengths must be None when ctype='point'")
            return None
        
        if isinstance(val, np.ndarray):
            if val.size != self.size:
                raise ValueError("coordinates and segment_lengths size mismatch, %d != %d" % (self.size, val.size))
            if not np.issubdtype(val.dtype, self.deltatype):
                raise ValueError("coordinates and segment_lengths dtype mismatch, %s != %s" % (self.dtype, self.deltatype))

        else:
            if self.size > 0 and not isinstance(val, self.deltatype):
                raise TypeError("coordinates and segment_lengths type mismatch, %s != %s" % (self.deltatype, type(val)))

        if np.any(np.array(val).astype(float) <= 0.0):
            raise ValueError("segment_lengths must be positive")

        return val

    @tl.default('coord_ref_sys')
    def _default_coord_ref_sys(self):
        return DEFAULT_CRS

    def __repr__(self):
        return "%s(%s): Bounds[%s, %s], N[%d], ctype['%s']" % (
            self.__class__.__name__, self.name or '?', self.bounds[0], self.bounds[1], self.size, self.ctype)

    def __eq__(self, other):
        if not isinstance(other, Coordinates1d):
            return False

        # defined coordinate properties should match
        for name in self._properties.union(other._properties):
            if getattr(self, name) != getattr(other, name):
                return False
        
        # shortcuts (not strictly necessary)
        for name in ['size', 'is_monotonic', 'is_descending', 'is_uniform']:
            if getattr(self, name) != getattr(other, name):
                return False

        # only check if one of the coordinates has custom segment lengths
        if self._segment_lengths or other._segment_lengths:
            if not np.all(self.segment_lengths == other.segment_lengths):
                return False

        return True

    def from_definition(self, d):
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def dims(self):
        if self.name is None:
            raise TypeError("cannot access dims property of unnamed Coordinates1d")
        return [self.name]

    @property
    def udims(self):
        return self.dims

    @property
    def coordinates(self):
        """:array, read-only: Full array of coordinates values."""

        raise NotImplementedError

    @property
    def values(self):
        """:array, read-only: Full array of coordinates values."""

        return self.coordinates
        
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
    def size(self):
        """Number of coordinates. """

        raise NotImplementedError

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
        if self.ctype == 'point':
            return self.bounds

        # empty coordinates [np.nan, np.nan]
        if self.size == 0:
            return self.bounds

        # segment ctypes, calculated
        L, H = self.argbounds
        lo, hi = self.bounds
        
        if not isinstance(self.segment_lengths, np.ndarray):
            lo_length = hi_length = self.segment_lengths # uniform segment_lengths
        else:
            lo_length, hi_length = self.segment_lengths[L], self.segment_lengths[H]

        if self.ctype == 'left':
            hi = add_coord(hi, hi_length)
        elif self.ctype == 'right':
            lo = add_coord(lo, -lo_length)
        elif self.ctype == 'midpoint':
            lo = add_coord(lo, -divide_delta(lo_length, 2.0))
            hi = add_coord(hi, divide_delta(hi_length, 2.0))

        # read-only array with the correct dtype
        area_bounds = np.array([lo, hi], dtype=self.dtype)
        area_bounds.setflags(write=False)
        return area_bounds

    @property
    def properties(self):
        """:dict: Dictionary of the coordinate properties. """

        return {key:getattr(self, key) for key in self._properties}

    @property
    def definition(self):
        """ Serializable 1d coordinates definition."""

        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def copy(self, **kwargs):
        """
        Make a deep copy of the 1d Coordinates.

        The coordinates properties will be copied. Any provided keyword arguments will override these properties.

        *Note: Defined in child classes.*

        Arguments
        ---------
        name : str, optional
            Dimension name. One of 'lat', 'lon', 'alt', and 'time'.
        coord_ref_sys : str, optional
            Coordinates reference system
        ctype : str, optional
            Coordinates type. One of 'point', 'midpoint', 'left', 'right'.
        units : podpac.Units, optional
            Coordinates units.

        Returns
        -------
        :class:`Coordinates1d`
            Copy of the coordinates, with provided properties.
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

    def intersect(self, other, return_indices=False, outer=False):
        """
        Get the coordinate values that are within the bounds of a given coordinates object.

        If a Coordinates1d ``other`` is provided, then this dimension must match the other dimension
        (``self.name == other.name``). If a multidimensional :class:`Coordinates` ``other`` is provided, then the
        corresponding 1d coordinates are used for the intersection if available, and otherwise the entire coordinates
        are returned.

        The default intersection selects coordinates that are within the other coordinates bounds::

            In [1]: c = ArrayCoordinates1d([0, 1, 2, 3], name='lat')

            In [2]: other = ArrayCoordinates1d([1.5, 2.5], name='lat')

            In [3]: c.intersect(other).coordinates
            Out[3]: array([2.])

        The *outer* intersection selects the minimal set of coordinates that contain the other coordinates::
        
            In [4]: c.intersect(other, outer=True).coordinates
            Out[4]: array([1., 2., 3.])

        The *outer* intersection also selects a boundary coordinate if the other coordinates are outside this
        coordinates bounds but *inside* its area bounds::
        
            In [5]: c.area_bounds
            Out[5]: array([-0.5,  3.5])

            In [6]: other1 = podpac.coordinates.ArrayCoordinates1d([3.25], name='lat')
            
            In [7]: other2 = podpac.coordinates.ArrayCoordinates1d([3.75], name='lat')

            In [8]: c.intersect(o2, outer=True).coordinates
            Out[8]: array([3.0], dtype=float64)

            In [9]: c.intersect(o2, outer=True).coordinates
            Out[9]: array([], dtype=float64)
        
        Parameters
        ----------
        other : :class:`Coordinates1d`, :class:`StackedCoordinates`, :class:`Coordinates`
            Coordinates to intersect with.
        outer : bool, optional
            If True, do an *outer* intersection. Default False.
        return_indices : bool, optional
            If True, return slice or indices for the selection in addition to coordinates. Default False.
        
        Returns
        -------
        intersection : :class:`Coordinates1d`
            Coordinates1d object with coordinates within the other coordinates bounds.
        I : slice or list
            index or slice for the intersected coordinates (only if return_indices=True)
        
        Raises
        ------
        ValueError
            If the coordinates names do not match, when intersecting with a Coordinates1d other.

        See Also
        --------
        select : Get the coordinates within the given bounds.
        """

        from podpac.core.coordinates import Coordinates, StackedCoordinates

        if not isinstance(other, (BaseCoordinates, Coordinates)):
            raise TypeError("Cannot intersect with type '%s'" % type(other))

        if isinstance(other, (Coordinates, StackedCoordinates)):
            # short-circuit
            if self.name not in other.udims:
                return self._select_full(return_indices)
            
            other = other[self.name]

        if self.name != other.name:
            raise ValueError("Cannot intersect mismatched dimensions ('%s' != '%s')" % (self.name, other.name))

        if self.dtype is not None and other.dtype is not None and self.dtype != other.dtype:
            raise ValueError("Cannot intersect mismatched dtypes ('%s' != '%s')" % (self.dtype, other.dtype))

        if self.units != other.units:
            raise NotImplementedError("Still need to implement handling different units")

        # no valid other bounds, empty
        if other.size == 0:
            return self._select_empty(return_indices)

        return self.select(other.bounds, return_indices=return_indices, outer=outer)

    def select(self, bounds, return_indices=False, outer=False):
        """
        Get the coordinate values that are within the given bounds.

        The default selection returns coordinates that are within the other coordinates bounds::

            In [1]: c = ArrayCoordinates1d([0, 1, 2, 3], name='lat')

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

        *Note: Defined in child classes.*
        
        Parameters
        ----------
        bounds : low, high
            selection bounds
        outer : bool, optional
            If True, do an *outer* selection. Default False.
        return_indices : bool, optional
            If True, return slice or indices for the selection in addition to coordinates. Default False.

        Returns
        -------
        selection : :class:`Coordinates1d`
            Coordinates1d object with coordinates within the other coordinates bounds.
        I : slice or list
            index or slice for the intersected coordinates (only if return_indices=True)
        """

        raise NotImplementedError
