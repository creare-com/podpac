"""
Single-Dimensional Coordinates: Array
"""

from __future__ import division, unicode_literals, print_function, absolute_import

from typing import Tuple

import numpy as np
from collections import OrderedDict

from podpac.core.utils import ArrayTrait
from podpac.core.coordinates.utils import make_coord_array, higher_precision_time_bounds
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

    See Also
    --------
    :class:`Coordinates1d`, :class:`UniformCoordinates1d`
    """

    coordinates = ArrayTrait(read_only=True)

    _is_monotonic = None
    _is_descending = None
    _is_uniform = None
    _step = None
    _start = None
    _stop = None

    def __init__(self, coordinates, name=None, **kwargs):
        """
        Create 1d coordinates from an array.

        Arguments
        ---------
        coordinates : array-like
            coordinate values.
        name : str, optional
            Dimension name, one of 'lat', 'lon', 'time', or 'alt'.
        """

        # validate and set coordinates
        coordinates = make_coord_array(coordinates)
        self.set_trait("coordinates", coordinates)
        self.not_a_trait = coordinates

        # precalculate once
        if self.coordinates.size == 0:
            pass

        elif self.coordinates.size == 1:
            self._is_monotonic = True

        elif self.coordinates.ndim > 1:
            self._is_monotonic = None
            self._is_descending = None
            self._is_uniform = None

        else:
            deltas = self.deltas
            if np.any(deltas <= 0):
                self._is_monotonic = False
                self._is_descending = False
                self._is_uniform = False
            else:
                self._is_monotonic = True
                self._is_descending = self.coordinates[1] < self.coordinates[0]
                self._is_uniform = np.allclose(deltas, deltas[0])
                if self._is_uniform:
                    self._start = self.coordinates[0]
                    self._stop = self.coordinates[-1]
                    self._step = (self._stop - self._start) / (self.coordinates.size - 1)

        # set common properties
        super(ArrayCoordinates1d, self).__init__(name=name, **kwargs)

    def __eq__(self, other):
        if not self._eq_base(other):
            return False

        if not np.array_equal(self.coordinates, other.coordinates):
            return False

        return True

    # ------------------------------------------------------------------------------------------------------------------
    # Alternate Constructors
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_xarray(cls, x, **kwargs):
        """
        Create 1d Coordinates from named xarray coordinates.

        Arguments
        ---------
        x : xarray.DataArray
            Nade DataArray of the coordinate values

        Returns
        -------
        :class:`ArrayCoordinates1d`
            1d coordinates
        """

        return cls(x.data, name=x.name, **kwargs).simplify()

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
                "name": "lat"
            })

        Arguments
        ---------
        d : dict
            1d coordinates array definition

        Returns
        -------
        :class:`ArrayCoordinates1d`
            1d Coordinates

        See Also
        --------
        definition
        """

        if "values" not in d:
            raise ValueError('ArrayCoordinates1d definition requires "values" property')

        coordinates = d["values"]
        kwargs = {k: v for k, v in d.items() if k != "values"}
        return cls(coordinates, **kwargs)

    def copy(self):
        """
        Make a deep copy of the 1d Coordinates array.

        Returns
        -------
        :class:`ArrayCoordinates1d`
            Copy of the coordinates.
        """

        return ArrayCoordinates1d(self.coordinates, **self.properties)

    def unique(self, return_index=False):
        """
        Remove duplicate coordinate values from each dimension.

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

        # shortcut, monotonic coordinates are already unique
        if self.is_monotonic:
            if return_index:
                return self.flatten(), np.arange(self.size).tolist()
            else:
                return self.flatten()

        a, I = np.unique(self.coordinates, return_index=True)
        if return_index:
            return self.flatten()[I], I
        else:
            return self.flatten()[I]

    def simplify(self):
        """Get the simplified/optimized representation of these coordinates.

        Returns
        -------
        :class:`ArrayCoordinates1d`, :class:`UniformCoordinates1d`
            UniformCoordinates1d if the coordinates are uniform, otherwise ArrayCoordinates1d
        """

        from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d

        if self.is_uniform:
            return UniformCoordinates1d(self.start, self.stop, self.step, **self.properties)

        return self

    def flatten(self):
        """
        Get a copy of the coordinates with a flattened array (wraps numpy.flatten).

        Returns
        -------
        :class:`ArrayCoordinates1d`
            Flattened coordinates.
        """

        if self.ndim == 1:
            return self.copy()

        return ArrayCoordinates1d(self.coordinates.flatten(), **self.properties)

    def reshape(self, newshape):
        """
        Get a copy of the coordinates with a reshaped array (wraps numpy.reshape).

        Arguments
        ---------
        newshape: int, tuple
            The new shape.

        Returns
        -------
        :class:`ArrayCoordinates1d`
            Reshaped coordinates.
        """

        return ArrayCoordinates1d(self.coordinates.reshape(newshape), **self.properties)

    # ------------------------------------------------------------------------------------------------------------------
    # standard methods, array-like
    # ------------------------------------------------------------------------------------------------------------------

    def __getitem__(self, index):
        # The following 3 lines are copied by UniformCoordinates1d.__getitem__
        if self.ndim == 1 and np.ndim(index) > 1 and np.array(index).dtype == int:
            index = np.array(index).flatten().tolist()
        try:
            return ArrayCoordinates1d(self.coordinates[index], **self.properties)
        except IndexError as e:  # This happens when index is a list, but should be a tuple
            if isinstance(index, list):
                return ArrayCoordinates1d(self.coordinates[tuple(index)], **self.properties)
            raise (e)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def deltas(self):
        return (self.coordinates[1:] - self.coordinates[:-1]).astype(float) * np.sign(
            self.coordinates[1] - self.coordinates[0]
        ).astype(float)

    @property
    def ndim(self):
        return self.coordinates.ndim

    @property
    def size(self):
        """Number of coordinates."""
        return self.coordinates.size

    @property
    def shape(self):
        return self.coordinates.shape

    @property
    def dtype(self):
        """:type: Coordinates dtype.

        ``float`` for numerical coordinates and numpy ``datetime64`` for datetime coordinates.
        """

        if self.size == 0:
            return None
        elif self.coordinates.dtype == float:
            return float
        elif np.issubdtype(self.coordinates.dtype, np.datetime64):
            return np.datetime64
        elif np.issubdtype(self.coordinates.dtype, np.timedelta64):
            return np.timedelta64

    @property
    def is_monotonic(self):
        return self._is_monotonic

    @property
    def is_descending(self):
        return self._is_descending

    @property
    def is_uniform(self):
        return self._is_uniform

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def step(self):
        return self._step

    @property
    def bounds(self):
        """Low and high coordinate bounds."""

        if self.size == 0:
            lo, hi = np.nan, np.nan
        elif self.is_monotonic:
            lo, hi = sorted([self.coordinates[0], self.coordinates[-1]])
        elif (self.dtype is np.datetime64) or (self.dtype == np.timedelta64):
            lo, hi = np.min(self.coordinates), np.max(self.coordinates)
        else:
            lo, hi = np.nanmin(self.coordinates), np.nanmax(self.coordinates)

        return lo, hi

    @property
    def argbounds(self):
        if self.size == 0:
            raise RuntimeError("Cannot get argbounds for empty coordinates")

        if not self.is_monotonic:
            argbounds = np.argmin(self.coordinates), np.argmax(self.coordinates)
            return np.unravel_index(argbounds[0], self.shape), np.unravel_index(argbounds[1], self.shape)
        elif not self.is_descending:
            return 0, -1
        else:
            return -1, 0

    def _get_definition(self, full=True):
        d = OrderedDict()
        d["values"] = self.coordinates
        d.update(self._full_properties if full else self.properties)
        return d

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def _get_index_default(self, bounds: Tuple[float, float]) -> np.array:
        gt = self.coordinates >= bounds[0]
        lt = self.coordinates <= bounds[1]
        index_inside_bounds = gt & lt
        index_outside_bounds = gt | lt
        if (
            index_outside_bounds.sum() != index_outside_bounds.size
            or index_inside_bounds.sum() != 0
            or not self.is_monotonic
        ):
            return index_inside_bounds

        # bounds between data points
        indlt = np.argwhere(lt).squeeze()
        indgt = np.argwhere(gt).squeeze()
        if self._is_descending:
            indlt = indlt[0] if indlt.size > 0 else index_inside_bounds.size - 1
            indgt = indgt[-1] if indgt.size > 0 else 0
        else:
            indlt = indlt[-1] if indlt.size > 0 else 0
            indgt = indgt[0] if indgt.size > 0 else index_inside_bounds.size - 1

        ind0 = min(indlt, indgt)
        ind1 = max(indlt, indgt) + 1
        index_inside_bounds[ind0:ind1] = True
        if index_inside_bounds.sum() > 1:
            # These two coordinates are candidates, we need
            # to make sure that the bounds cross the edge between
            # the two points (selects both) or not (only selects)
            crds = self.coordinates[index_inside_bounds]
            step = np.diff(self.coordinates[index_inside_bounds])[0]
            edge = crds[0] + step / 2
            bounds_lt = bounds <= edge
            bounds_gt = bounds > edge
            keep_point = [np.any(bounds_lt), np.any(bounds_gt)]
            if self._is_descending:
                keep_point = keep_point[::-1]
            index_inside_bounds[ind0:ind1] = keep_point
        return index_inside_bounds

    def _get_index_outer_monotonic(self, bounds: Tuple[float, float]) -> np.array:
        gt = np.nonzero(self.coordinates >= bounds[0])[0]
        lt = np.nonzero(self.coordinates <= bounds[1])[0]
        lo, hi = bounds[0], bounds[1]
        if self.is_descending:
            lt, gt = gt, lt
            lo, hi = hi, lo
        if self.coordinates[gt[0]] != lo:
            gt[0] -= 1
        if self.coordinates[lt[-1]] != hi:
            lt[-1] += 1
        start = max(0, gt[0])
        stop = min(self.size - 1, lt[-1])
        b = slice(start, stop + 1)
        return b

    def _get_index_outer_nonmonotonic(self, bounds: Tuple[float, float]) -> np.array:
        try:
            gt = self.coordinates >= max(self.coordinates[self.coordinates <= bounds[0]])
        except ValueError:
            if (self.dtype == np.datetime64) or (self.dtype == np.timedelta64):
                gt = ~np.isnat(self.coordinates)
            else:
                gt = self.coordinates >= -np.inf
        try:
            lt = self.coordinates <= min(self.coordinates[self.coordinates >= bounds[1]])
        except ValueError:
            if self.dtype == np.datetime64 or (self.dtype == np.timedelta64):
                lt = ~np.isnat(self.coordinates)
            else:
                lt = self.coordinates <= np.inf

        b = gt & lt
        return b

    def _select(self, bounds: Tuple[float, float], return_index: bool, outer: bool):
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
        bounds : (low, high)
            Selection bounds.
        outer : bool, optional
            If True, do an *outer* selection. Default False.
        return_index : bool, optional
            If True, return index for the selection in addition to coordinates. Default False.

        Returns
        -------
        selection : :class:`Coordinates1d`
            Coordinates1d object with coordinates within the bounds.
        I : slice, boolean array
            index or slice for the selected coordinates (only if return_index=True)
        """
        if self.dtype == np.datetime64:
            _, bounds = higher_precision_time_bounds(self.bounds, bounds, outer)

        if not outer:
            b = self._get_index_default(bounds)
        elif self.is_monotonic:
            b = self._get_index_outer_monotonic(bounds)
        else:
            b = self._get_index_outer_nonmonotonic(bounds)

        if return_index:
            return self[b], b
        else:
            return self[b]
