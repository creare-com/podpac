"""
One-Dimensional Coordinates
"""


from __future__ import division, unicode_literals, print_function, absolute_import

import copy

import numpy as np
import traitlets as tl

from podpac.core.units import Units
# from podpac.core.utils import cached_property, clear_cache
from podpac.core.coordinates.utils import make_coord_value, make_coord_delta, make_coord_array, add_coord
from podpac.core.coordinates.base_coordinates import BaseCoordinates

DEFAULT_COORD_REF_SYS = 'WGS84'

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
        Coordinate reference system.
    ctype : str
        Coordinates type: 'point', 'left', 'right', or 'midpoint'.
    extents : (low, high)
        When ctype != 'point', defines custom (low, high) area bounds for the coordinates.
        *Note: To be replaced with segment_lengths.*

    See Also
    --------
    :class:`ArrayCoordinates1d`, :class:`UniformCoordinates1d`
    """

    name = tl.Enum(['lat', 'lon', 'time', 'alt'], allow_none=True)
    name.__doc__ = ":str: Dimension name, one of 'lat', 'lon', 'time', or 'alt'"

    units = tl.Instance(Units, allow_none=True)
    units.__doc__ = ":Units: Coordinate units."

    coord_ref_sys = tl.Enum(['WGS84', 'SPHER_MERC'], allow_none=True)
    coord_ref_sys.__doc__ = ":str: Coordinate reference system."

    ctype = tl.Enum(['point', 'left', 'right', 'midpoint'])
    ctype.__doc__ = ":str: Coordinates type, one of 'point', 'left', 'right', or 'midpoint'."

    extents = tl.Instance(np.ndarray, allow_none=True, default_value=None)
    extents.__doc__ = ":: *To be replaced.*"

    is_monotonic = tl.CBool(allow_none=True, readonly=True)
    is_monotonic.__doc__ = ":bool: Are the coordinate values unique and sorted."
    
    is_descending = tl.CBool(allow_none=True, readonly=True)
    is_descending.__doc__ = ":bool: Are the coordinate values sorted in descending order."

    is_uniform = tl.CBool(allow_none=True, readonly=True)
    is_uniform.__doc__ = ":bool: Are the coordinate values uniformly-spaced."

    def __init__(self, name=None, ctype=None, units=None, extents=None, coord_ref_sys=None, **kwargs):
        """*Do not use.*"""

        if name is not None:
            kwargs['name'] = name
        if ctype is not None:
            kwargs['ctype'] = ctype
        if units is not None:
            kwargs['units'] = units
        if coord_ref_sys is not None:
            kwargs['coord_ref_sys'] = coord_ref_sys
        if extents is not None:
            extents = make_coord_array(extents)
            extents.setflags(write=False)
            kwargs['extents'] = extents

        super(Coordinates1d, self).__init__(**kwargs)

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

    @tl.default('coord_ref_sys')
    def _default_coord_ref_sys(self):
        return DEFAULT_COORD_REF_SYS
    
    @tl.default('ctype')
    def _default_ctype(self):
        return 'midpoint'

    def __repr__(self):
        return "%s(%s): Bounds[%s, %s], N[%d], ctype['%s']" % (
            self.__class__.__name__, self.name or '?', self.bounds[0], self.bounds[1], self.size, self.ctype)

    def __eq__(self, other):
        if super(Coordinates1d, self).__eq__(other):
            return True

        # special case for ArrayCoordinates1d and UniformCoordinates1d with the same properties and coordinates
        if (isinstance(other, Coordinates1d) and
            type(other) != type(self) and
            self.is_uniform and other.is_uniform and 
            self.properties == other.properties and
            np.array_equal(self.coordinates, other.coordinates)):
            return True

        return False

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
    def properties(self):
        """:dict: Dictionary of the coordinate properties. """

        d = {}
        if self.name is not None:
            d['name'] = self.name
        if self.units is not None:
            d['units'] = self.units
        if self.coord_ref_sys is not None:
            d['coord_ref_sys'] = self.coord_ref_sys
        d['ctype'] = self.ctype
        if self.extents is not None:
            d['extents'] = self.extents
        return d

    @property
    def coordinates(self):
        """:array, read-only: Full array of coordinates values."""

        raise NotImplementedError

    @property
    def dtype(self):
        """:type: Coordinates dtype.

        ``float`` for numerical coordinates and numpy ``datetime64`` for datetime coordinates.
        """

        raise NotImplementedError

    @property
    def size(self):
        """Number of coordinates. """

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

        raise NotImplementedError

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
        from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
        c = ArrayCoordinates1d([], **self.properties)
        if return_indices:
            return c, slice(0, 0)
        else:
            return c

    def _select_full(self, return_indices):
        c = copy.deepcopy(self)
        if return_indices:
            return c, slice(None, None)
        else:
            return c        

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