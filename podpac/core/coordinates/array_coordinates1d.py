"""
One-Dimensional Coordinates: Array
"""


from __future__ import division, unicode_literals, print_function, absolute_import

import copy
from collections import OrderedDict

import numpy as np
import traitlets as tl
from collections import OrderedDict

from podpac.core.utils import ArrayTrait
from podpac.core.coordinates.utils import make_coord_array
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

    coordinates = ArrayTrait(ndim=1, read_only=True)
    # coordinates.__doc__ = ":array: User-defined coordinate values"
    # coordinates = None

    _is_monotonic = None
    _is_descending = None
    _is_uniform = None
    _step = None
    _start = None
    _stop = None

    def __init__(self, coordinates, name=None):
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

        else:
            deltas = (
                (self.coordinates[1:] - self.coordinates[:-1]).astype(float)
                / (self.coordinates[1] - self.coordinates[0]).astype(float)
                * self.coordinates.size
            )
            if np.any(deltas <= 0):
                self._is_monotonic = False
                self._is_descending = False
                self._is_uniform = False
            else:
                self._is_monotonic = True
                self._is_descending = self.coordinates[1] < self.coordinates[0]
                self._is_uniform = np.allclose(deltas, deltas[0], atol=1e-7)
                if self._is_uniform:
                    self._start = self.coordinates[0]
                    self._stop = self.coordinates[-1]
                    self._step = (self._stop - self._start) / (self.coordinates.size - 1)

        # set common properties
        super(ArrayCoordinates1d, self).__init__(name=name)

    def __eq__(self, other):
        if not super(ArrayCoordinates1d, self).__eq__(other):
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

        return cls(x.data, name=x.name, **kwargs)

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

    def simplify(self):
        """ Get the simplified/optimized representation of these coordinates.

        Returns
        -------
        simplified : ArrayCoordinates1d, UniformCoordinates1d
            UniformCoordinates1d if the coordinates are uniform, otherwise ArrayCoordinates1d
        """

        from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d

        if self.is_uniform:
            return UniformCoordinates1d(self.start, self.stop, self.step, **self.properties)

        return self

    # ------------------------------------------------------------------------------------------------------------------
    # standard methods, array-like
    # ------------------------------------------------------------------------------------------------------------------

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return ArrayCoordinates1d(self.coordinates[index], **self.properties)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def size(self):
        """ Number of coordinates. """
        return self.coordinates.size

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
        """ Low and high coordinate bounds. """

        if self.size == 0:
            lo, hi = np.nan, np.nan
        elif self.is_monotonic:
            lo, hi = sorted([self.coordinates[0], self.coordinates[-1]])
        elif self.dtype is np.datetime64:
            lo, hi = np.min(self.coordinates), np.max(self.coordinates)
        else:
            lo, hi = np.nanmin(self.coordinates), np.nanmax(self.coordinates)

        return lo, hi

    @property
    def argbounds(self):
        if not self.is_monotonic:
            return np.argmin(self.coordinates), np.argmax(self.coordinates)
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

    def _select(self, bounds, return_indices, outer):
        if not outer:
            gt = self.coordinates >= bounds[0]
            lt = self.coordinates <= bounds[1]
            I = np.where(gt & lt)[0]

        elif self.is_monotonic:
            gt = np.where(self.coordinates >= bounds[0])[0]
            lt = np.where(self.coordinates <= bounds[1])[0]
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
            I = slice(start, stop + 1)

        else:
            try:
                gt = self.coordinates >= max(self.coordinates[self.coordinates <= bounds[0]])
            except ValueError as e:
                if self.dtype == np.datetime64:
                    gt = ~np.isnat(self.coordinates)
                else:
                    gt = self.coordinates >= -np.inf
            try:
                lt = self.coordinates <= min(self.coordinates[self.coordinates >= bounds[1]])
            except ValueError as e:
                if self.dtype == np.datetime64:
                    lt = ~np.isnat(self.coordinates)
                else:
                    lt = self.coordinates <= np.inf

            I = np.where(gt & lt)[0]

        if return_indices:
            return self[I], I
        else:
            return self[I]
