"""
One-Dimensional Coordinates: Array
"""


from __future__ import division, unicode_literals, print_function, absolute_import

import copy
from collections import OrderedDict

import numpy as np
import traitlets as tl
from collections import OrderedDict

# from podpac.core.utils import cached_property, clear_cache
from podpac.core.units import Units
from podpac.core.coordinates.utils import make_coord_value, make_coord_array, add_coord
from podpac.core.coordinates.coordinates1d import Coordinates1d

class ArrayCoordinates1d(Coordinates1d):
    """
    1-dimensional array of coordinates.

    ArrayCoordinates1d is a basic array of 1d coordinates created from an array of coordinate values. Numerical
    coordinates values are converted to ``float``, and time coordinate values are converted to numpy ``datetime64``.
    For convenience, podpac automatically converts datetime strings such as ``'2018-01-01'`` to ``datetime64``. The
    coordinate values must all be of the same type.

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
    extents : ndarray
        When ctype != 'point', defines a custom acea bounds for the coordinates.
        *Note: To be replaced with segment_lengths.*

    See Also
    --------
    :class:`Coordinates1d`, :class:`UniformCoordinates1d`
    """

    #: array : User-defined coordinate values
    coords = tl.Instance(np.ndarray)

    # inherited traits, duplicated here for the docstrings

    #:str: Dimension name, one of 'lat', 'lon', 'time', or 'alt'.
    name = Coordinates1d.name

    #: Units : Coordinate units.
    units = Coordinates1d.units

    #: str : Coordinate reference system.
    coord_ref_sys = Coordinates1d.coord_ref_sys

    #: str : Coordinates type, on of 'point', 'left', 'right', or 'midpoint'.
    ctype = Coordinates1d.ctype

    #: : *To be replaced.*
    extents = Coordinates1d.extents

    #: bool : Are the coordinate values unique and sorted.
    is_monotonic = Coordinates1d.is_monotonic

    #: bool : Are the coordinate values sorted in descending order.
    is_descending = Coordinates1d.is_descending

    #: bool : Are the coordinate values uniformly-spaced.
    is_uniform = Coordinates1d.is_uniform

    def __init__(self, coords, name=None, ctype=None, units=None, extents=None, coord_ref_sys=None):
        """
        Create 1d coordinates from an array.

        Arguments
        ---------
        coords : array-like
            coordinate values.
        name : str, optional
            Dimension name, one of 'lat', 'lon', 'time', or 'alt'.
        units : Units, optional
            Coordinate units.
        coord_ref_sys : str, optional
            Coordinate reference system.
        ctype : str, optional
            Coordinates type: 'point', 'left', 'right', or 'midpoint'.
        extents : (low, high), optional
            When ctype != 'point', defines custom (low, high) area bounds for the coordinates.
            *Note: To be replaced with segment_lengths.*
        """

        coords = make_coord_array(coords)

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
    def _observe_coords(self, change):
        # clear_cache(self, change, ['bounds', 'coordinates'])

        val = self.coords
        if val.size == 0:
            self.set_trait('is_monotonic', None)
            self.set_trait('is_descending', None)
            self.set_trait('is_uniform', None)
        elif val.size == 1:
            self.set_trait('is_monotonic', True)
            self.set_trait('is_descending', None)
            self.set_trait('is_uniform', True)
        else:
            deltas = (val[1:] - val[:-1]).astype(float) * (val[1] - val[0]).astype(float)
            if np.any(deltas <= 0):
                self.set_trait('is_monotonic', False)
                self.set_trait('is_descending', None)
                self.set_trait('is_uniform', False)
            else:
                self.set_trait('is_monotonic', True)
                self.set_trait('is_descending', self.coords[1] < self.coords[0])
                self.set_trait('is_uniform', np.allclose(deltas, deltas[0]))

    @tl.validate('extents')
    def _validate_extents(self, d):
        return super(ArrayCoordinates1d, self)._validate_extents(d)

    @tl.default('coord_ref_sys')
    def _default_coord_ref_sys(self):
        return super(ArrayCoordinates1d, self)._default_coord_ref_sys()
    
    @tl.default('ctype')
    def _default_ctype(self):
        return super(ArrayCoordinates1d, self)._default_ctype()

    # ------------------------------------------------------------------------------------------------------------------
    # Alternate Constructors
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_xarray(cls, x, ctype=None, units=None, extents=None, coord_ref_sys=None):
        """
        Create 1d Coordinates from named xarray coords.

        Arguments
        ---------
        x : xarray.DataArray
            Nade DataArray of the coordinate values
        units : Units, optional
            Coordinate units.
        coord_ref_sys : str, optional
            Coordinate reference system.
        ctype : str, optional
            Coordinates type: 'point', 'left', 'right', or 'midpoint'.
        extents : (low, high), optional
            When ctype != 'point', defines custom (low, high) area bounds for the coordinates.
            *Note: To be replaced with segment_lengths.*

        Returns
        -------
        ArrayCoordinates1d
            1d coordinates
        """

        return cls(x.data, name=x.name)

    @classmethod
    def from_definition(cls, d):
        """
        Create 1d coordinates from a coordinates definition.

        The definition must contain the coordinate values::

            c = ArrayCoordinates1d.from_definition({
                "values": [0, 1, 2, 3]
            })

        The definition may also contain any of the 1d Coordinates properties::

            c = ArrayCoordinates1d.from_definition({
                "values": [0, 1, 2, 3],
                "name": "lat",
                "ctype": "points"
            })

        Arguments
        ---------
        d : dict
            1d coordinates array definition

        Returns
        -------
        ArrayCoordinates1d
            1d Coordinates

        See Also
        --------
        definition
        """

        coords = d.pop('values')
        return cls(coords, **d)

    def copy(self, **kwargs):
        """
        Make a deep copy of the 1d Coordinates array.

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
        ArrayCoordinates1d
            Copy of the coordinates, with provided properties.
        """

        properties = self.properties
        properties.update(kwargs)
        return ArrayCoordinates1d(self.coords, **properties)

    # ------------------------------------------------------------------------------------------------------------------
    # standard methods, array-like
    # ------------------------------------------------------------------------------------------------------------------

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        coords = self.coords[index]
        return ArrayCoordinates1d(coords, **self.properties)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def coordinates(self):
        """ Coordinate values.

        :type: array, read-only
        """

        # get coordinates and ensure read-only array with correct dtype
        coordinates = self.coords.copy()
        coordinates.setflags(write=False)
        return coordinates

    @property
    def size(self):
        """ Number of coordinates. """
        return self.coords.size

    @property
    def dtype(self):
        """ Coordinates dtype.

        ``float`` for numerical coordinates and numpy ``datetime64`` for datetime coordinates.

        :type: type
        """

        if self.size == 0:
            return None
        elif self.coords.dtype == float:
            return float
        elif np.issubdtype(self.coords.dtype, np.datetime64):
            return np.datetime64
        else:
            raise ValueError("Invalid coords dtype '%s'" % self.coords.dtype)

    @property
    def bounds(self):
        """ Low and high coordinate bounds. """

        # TODO are we sure this can't be a tuple?

        if self.size == 0:
            lo, hi = np.nan, np.nan
        elif self.is_monotonic:
            lo, hi = sorted([self.coords[0], self.coords[-1]])
        elif self.dtype is np.datetime64:
            lo, hi = np.min(self.coords), np.max(self.coords)
        else:
            lo, hi = np.nanmin(self.coords), np.nanmax(self.coords)

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
        if self.size == 0:
            lo, hi = np.nan, np.nan
        elif self.size == 1:
            lo, hi = self.coords[0], self.coords[0]
        elif self.ctype == 'midpoint':
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

        # read-only array with the correct dtype
        area_bounds = np.array([lo, hi], dtype=self.dtype)
        area_bounds.setflags(write=False)
        return area_bounds

    @property
    def definition(self):
        """
        Serializable 1d coordinates array definition.

        The ``definition`` can be used to create new ArrayCoordinates1d::

            c = podpac.ArrayCoordinates1d([0, 1, 2, 3])
            c2 = podpac.ArrayCoordinates1d.from_definition(c.definition)

        :type: dict

        See Also
        --------
        from_definition
        """

        d = OrderedDict()
        if self.dtype == float:
            d['values'] = self.coords.tolist()
        else:
            d['values'] = self.coords.astype('str').tolist()
        d.update(self.properties)
        return d

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def select(self, bounds, outer=False, return_indices=False):
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
        selection : Coordinates1d
            Coordinates1d object with coordinates within the other coordinates bounds.
        I : slice or list
            index or slice for the intersected coordinates (only if return_indices=True)
        """

        bounds = make_coord_value(bounds[0]), make_coord_value(bounds[1])

        # empty
        if self.size == 0:
            return self._select_empty(return_indices)

        # full
        if self.bounds[0] >= bounds[0] and self.bounds[1] <= bounds[1]:
            return self._select_full(return_indices)

        # none
        if self.area_bounds[0] > bounds[1] or self.area_bounds[1] < bounds[0]:
            return self._select_empty(return_indices)

        if not outer:
            gt = self.coordinates >= bounds[0]
            lt = self.coordinates <= bounds[1]
            I = np.where(gt & lt)[0]

        elif self.is_monotonic:
            gt = np.where(self.coords >= bounds[0])[0]
            lt = np.where(self.coords <= bounds[1])[0]
            if self.is_descending:
                lt, gt = gt, lt
            start = max(0, gt[0]-1)
            stop = min(self.size-1, lt[-1]+1)
            I = slice(start, stop+1)

        else:
            raise NotImplementedError("select outer=True is not yet supported for non-monotonic coordinates")

        c = ArrayCoordinates1d(self.coords[I], **self.properties)
        if return_indices:
            return c, I
        else:
            return c
