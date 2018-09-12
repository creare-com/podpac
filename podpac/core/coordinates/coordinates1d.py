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
from podpac.core.coordinates.base_coordinates1d import BaseCoordinates1d

DEFAULT_COORD_REF_SYS = 'WGS84'

class Coordinates1d(BaseCoordinates1d):
    """
    Base class for one-dimensional single coordinates.
    
    Attributes
    ----------
    name : str
        Dimension name, one of 'lat', 'lon', 'time', 'alt'.
    dims : tuple
        The dimension name as a tuple, `(self.name,)`
    units : Units
        TODO
    coord_ref_sys : str
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
        Area bounds, np.array(low, high).
        When ctype != 'point', this including the portions of the segments beyond the coordinate bounds.
    is_monotonic : bool
        True if the coordinates are guaranteed to be sorted.
    is_descending : bool
        True if the coordinates are monotonically descending.
        False if the coordinates are monotonically ascending or not sorted.
    is_uniform : bool
        True if the coordinates are uniformly-spaced (and monotonic).
        False if the coordinates are not uniformly-spaced, unsorted, or empty.

    Methods
    -------

    """

    name = tl.Enum(['lat', 'lon', 'time', 'alt'], allow_none=True)
    units = tl.Instance(Units, allow_none=True)
    coord_ref_sys = tl.Enum(['WGS84', 'SPHER_MERC'], allow_none=True)
    ctype = tl.Enum(['point', 'left', 'right', 'midpoint'])
    extents = tl.Instance(np.ndarray, allow_none=True, default_value=None)
    is_monotonic = tl.CBool(allow_none=True, readonly=True)
    is_descending = tl.CBool(allow_none=True, readonly=True)
    is_uniform = tl.CBool(allow_none=True, readonly=True)

    def __init__(self, extents=None, **kwargs):
        if extents is not None:
            extents = make_coord_array(extents)
            extents.setflags(write=False)

        super(Coordinates1d, self).__init__(extents=extents, **kwargs)

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

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def dims(self):
        if self.name is None:
            raise TypeError("cannot access dims property of unnamed Coordinates1d")

        return [self.name]

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
        ''' Coordinate values. '''

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

        raise NotImplementedError
    
    @property
    def area_bounds(self):
        '''
        Low and high area bounds. When ctype != 'point', this includes the portions of the segments beyond the
        coordinate bounds.
        '''

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def copy(self, **kwargs):
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
        Get the coordinates within the bounds the given coordinates object.
        
        Parameters
        ----------
        other : Coordinates1d
            coordinates to intersect with
        outer : bool, optional
            If True, select minimal coordinates that contain the other coordinates bounds.
            If False, select maximal coordinates that are within the other coordinates bounds.
            Default False.
        return_indices : bool, optional
            If True, return slice or indices for the selection in addition to coordinates. Default False.
        
        Returns
        -------
        intersection : Coordinates1d
            Coordinates1d object with coordinates with other.bounds
        I : slice or list
            index or slice for the intersected coordinates (only if return_indices=True)
        
        See Also
        --------
        select : Get the coordinates within the given bounds.
        
        Raises
        ------
        NotImplementedError
            Description
        """

        from podpac.core.coordinates import Coordinates, StackedCoordinates

        if not isinstance(other, (BaseCoordinates1d, Coordinates)):
            raise TypeError("Cannot intersect with type '%s'" % type(other))

        # short-circuit
        if self.name not in other.dims:
            return self._select_full(return_indices)

        if isinstance(other, (Coordinates, StackedCoordinates)):
            other = other[self.name]

        if self.name != other.name:
            raise ValueError("Cannot intersect mismatched dimensions ('%s' != '%s')" % (selfename, other.name))

        if self.units != other.units:
            raise NotImplementedError("Still need to implement handling different units")

        # no valid other bounds, empty
        if other.size == 0:
            return self._select_empty(return_indices)

        return self.select(other.bounds, return_indices=return_indices, outer=outer)

    def select(self, bounds, return_indices=False, outer=False):
        """
        Get the coordinates within the given bounds.
        
        Parameters
        ----------
        bounds : min, max
            selection bounds
        return_indices : bool, optional
            return slice or indices for selection instead of coordinates
        **kwargs
            Description
        
        Returns
        -------
        intersection : Coordinates1d
            coord object with selected coordinates (if return_indices=False)
        I : slice or list
            index or slice for the selected coordinates (if return_indices=True)
        """

        raise NotImplementedError

    # def add(self, delta, inplace=False):
    #     """
    #     Add a delta value to each coordinate.
        
    #     Parameters
    #     ----------
    #     delta : TYPE
    #         Description
    #     inplace : bool (optional)
    #         If True, update coordinates in-place. Default False.
        
    #     Returns
    #     -------
    #     result : Coordinates1d
    #         If inplace, this object with resulting coordinates.
    #         Otherwise, new Coordinates1d object with resulting coordinates.
        
    #     Raises
    #     ------
    #     TypeError
    #         Description
    #     """

    #     delta = make_coord_delta(delta)
        
    #     if self.dtype is np.datetime64 and not isinstance(delta, np.timedelta64):
    #         raise TypeError("Cannot add '%s' to datetime coord" % type(delta))
        
    #     if self.dtype is float and isinstance(delta, np.timedelta64):
    #         raise TypeError("Cannot add timedelta to numerical coord")

    #     if inplace:
    #         return self._add_equal(delta)
    #     else:
    #         return self._add(delta)

    # def _add(self, other):
    #     raise NotImplementedError

    # def _add_equal(self, other):
    #     raise NotImplementedError
    
    # def concat(self, other, inplace=False):
    #     """
    #     Concatenate coordinates.
        
    #     Parameters
    #     ----------
    #     other : Coordinates1d
    #         coords object to concatenate
    #     inplace : bool (optional)
    #         If True, update coordinates in-place. Default False.
        
    #     Returns
    #     -------
    #     result : Coordinates1d
    #         If inplace, this object with concatenated coordinates.
    #         New coords object with concatenated coordinates.
        
    #     Raises
    #     ------
    #     TypeError
    #         Description
    #     """

    #     if not isinstance(other, Coordinates1d):
    #         raise TypeError("Cannot concatenate '%s' to '%s'" % (other.__class__.__name__, self.__class__.__name__))

    #     if self.dtype is np.datetime64 and other.dtype is np.float64:
    #         raise TypeError("Cannot concatenate numerical coords to datetime coords")
        
    #     if self.dtype is np.float64 and other.dtype is np.datetime64:
    #         raise TypeError("Cannot concatenate datetime coords to numerical coords")

    #     # empty cases
    #     if other.size == 0:
    #         if inplace:
    #             return self
    #         else:
    #             return copy.deepcopy(self)

    #     # standard case
    #     if inplace:
    #         return self._concat_equal(other)
    #     else:
    #         return self._concat(other)

    # def _concat(self, other):
    #     raise NotImplementedError

    # def _concat_equal(self, other):
    #     raise NotImplementedError
    
    # # ------------------------------------------------------------------------------------------------------------------
    # # Operators ("magic methods")
    # # ------------------------------------------------------------------------------------------------------------------

    def __add__(self, other):
        """ add a delta """

        if isinstance(other, Coordinates1d):
            raise RuntimeError("concatenating Coordinates1d with + has been removed, use 'concat' instead")

        raise Exception("JXM checking if this is used anywhere")
        return self.add(other)

    def __iadd__(self, other):
        """ add a delta in-place """

        if isinstance(other, Coordinates1d):
            raise RuntimeError("concatenating Coordinates1d with += has been removed, use 'concat' instead")
        
        raise Exception("JXM checking if this is used anywhere")
        return self.add(other, inplace=True)

    # def __sub__(self, other):
    #     """ subtract a delta """
    #     _other = -make_coord_delta(other)
    #     return self.add(_other)

    # def __isub__(self, other):
    #     """ subtract a delta in place """
    #     _other = -make_coord_delta(other)
    #     return self.add(_other, inplace=True)

    def __repr__(self):
        return "%s(%s): Bounds[%s, %s], N[%d], ctype['%s']" % (
            self.__class__.__name__, self.name or '?', self.bounds[0], self.bounds[1], self.size, self.ctype)