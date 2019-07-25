from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict

import numpy as np
import traitlets as tl
import lazy_import

rasterio = lazy_import.lazy_module("rasterio")

from podpac.core.utils import ArrayTrait
from podpac.core.coordinates.dependent_coordinates import DependentCoordinates


class RotatedCoordinates(DependentCoordinates):
    """
    A grid of rotated latitude and longitude coordinates.

    RotatedCoordinates are dependent spatial coordinates defined by a shape, rotation angle, upper left corner, and
    step size. The lower right corner can be specified instead of the step. RotatedCoordinates can also be converted
    to/from GDAL geotransform.

    Parameters
    ----------
    shape : tuple
        shape (m, n) of the grid.
    theta : float
        rotation angle, in radians
    origin : np.ndarray(shape=(2,), dtype=float)
        origin coordinates (position [0, 0])
    corner : np.ndarray(shape=(2,), dtype=float)
        opposing corner coordinates (position [m-1, n-1])
    step : np.ndarray(shape=(2,), dtype=float)
        Rotated distance between points in the grid, in each dimension. This is equivalent to the scaling of the
        affine transformation used to calculate the coordinates.
    dims : tuple
        Tuple of dimension names.
    coords : dict-like
        xarray coordinates (container of coordinate arrays)
    coordinates : tuple
        Tuple of 2d coordinate values in each dimension.
    """

    shape = tl.Tuple(tl.Integer(), tl.Integer(), read_only=True)
    theta = tl.Float(read_only=True)
    origin = ArrayTrait(shape=(2,), dtype=float, read_only=True)
    step = ArrayTrait(shape=(2,), dtype=float, read_only=True)
    ndims = 2

    def __init__(
        self, shape=None, theta=None, origin=None, step=None, corner=None, dims=None, ctypes=None, segment_lengths=None
    ):
        """
        Create a grid of rotated coordinates from a `shape`, `theta`, `origin`, and `step` or `corner`.

        Parameters
        ----------
        shape : tuple
            shape (m, n) of the grid.
        theta : float
            rotation angle, in radians
        origin : np.ndarray(shape=(2,), dtype=float)
            origin coordinates
        corner : np.ndarray(shape=(2,), dtype=float)
            opposing corner coordinates (corner or step required)
        step : np.ndarray(shape=(2,), dtype=float)
            Scaling, ie rotated distance between points in the grid, in each dimension. (corner or step required)
        dims : tuple (required)
            tuple of dimension names ('lat', 'lon', 'time', or 'alt').
        ctype : tuple, str (optional)
            tuple of coordinates types ('point', 'left', 'right', or 'midpoint') for each dimension. A single ctype
            str can be specified for all dimensions.
        segment_lengths : tuple, float, or timedelta (optional)
            tuple of segment lengths for each dimension. A single segment length can be specified for both dimensions.
            For point coordinates, the segment_lengths must be None; omit if all dimensions are point coordinates.
        """

        self.set_trait("shape", shape)
        self.set_trait("theta", theta)
        self.set_trait("origin", origin)
        if step is None:
            deg = np.rad2deg(theta)
            a = ~rasterio.Affine.rotation(deg) * ~rasterio.Affine.translation(*origin)
            d = np.array(a * corner) - np.array(a * origin)
            step = d / np.array([shape[0] - 1, shape[1] - 1])
        self.set_trait("step", step)

        self._set_properties(dims, ctypes, segment_lengths)

    @tl.validate("dims")
    def _validate_dims(self, d):
        val = super(RotatedCoordinates, self)._validate_dims(d)
        for dim in val:
            if dim not in ["lat", "lon"]:
                raise ValueError("RotatedCoordinates dims must be 'lat' or 'lon', not '%s'" % dim)
        return val

    @tl.validate("shape")
    def _validate_shape(self, d):
        val = d["value"]
        if val[0] <= 0 or val[1] <= 0:
            raise ValueError("Invalid shape %s, shape must be positive" % (val,))
        return val

    @tl.validate("step")
    def _validate_step(self, d):
        val = d["value"]
        if val[0] == 0 or val[1] == 0:
            raise ValueError("Invalid step %s, step cannot be 0" % val)
        return val

    # ------------------------------------------------------------------------------------------------------------------
    # Alternate Constructors
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_geotransform(cls, geotransform, shape, dims=None, ctypes=None, segment_lengths=None):
        affine = rasterio.Affine.from_gdal(*geotransform)
        origin = affine.c, affine.f
        deg = affine.rotation_angle
        scale = ~affine.rotation(deg) * ~affine.translation(*origin) * affine
        step = np.array([scale.a, scale.e])
        return cls(shape, np.deg2rad(deg), origin, step, dims=dims, ctypes=ctypes, segment_lengths=segment_lengths)

    @classmethod
    def from_definition(cls, d):
        """
        Create RotatedCoordinates from a rotated coordinates definition.

        Arguments
        ---------
        d : dict
            rotated coordinates definition

        Returns
        -------
        :class:`RotatedCoordinates`
            rotated coordinates object

        See Also
        --------
        definition
        """

        if "shape" not in d:
            raise ValueError('RotatedCoordinates definition requires "shape" property')
        if "theta" not in d:
            raise ValueError('RotatedCoordinates definition requires "theta" property')
        if "origin" not in d:
            raise ValueError('RotatedCoordinates definition requires "origin" property')
        if "step" not in d and "corner" not in d:
            raise ValueError('RotatedCoordinates definition requires "step" or "corner" property')
        if "dims" not in d:
            raise ValueError('RotatedCoordinates definition requires "dims" property')

        shape = d["shape"]
        theta = d["theta"]
        origin = d["origin"]
        kwargs = {k: v for k, v in d.items() if k not in ["shape", "theta", "origin"]}
        return RotatedCoordinates(shape, theta, origin, **kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    # standard methods
    # ------------------------------------------------------------------------------------------------------------------

    def __repr__(self):
        if self.ctypes[0] == self.ctypes[1]:
            ctypes = "ctype['%s']" % self.ctypes[0]
        else:
            ctypes = "ctypes[%s]" % ", ".join(self.ctypes)

        return "%s(%s): Origin%s, Corner%s, rad[%.4f], shape%s, %s" % (
            self.__class__.__name__,
            self.dims,
            self.origin,
            self.corner,
            self.theta,
            self.shape,
            ctypes,
        )

    def __eq__(self, other):
        if not isinstance(other, RotatedCoordinates):
            return False

        if self.shape != other.shape:
            return False

        if self.affine != other.affine:
            return False

        # defined coordinate properties should match
        for name in self._properties.union(other._properties):
            if getattr(self, name) != getattr(other, name):
                return False

        return True

    def __getitem__(self, index):
        if isinstance(index, slice):
            index = index, slice(None)

        if isinstance(index, tuple) and isinstance(index[0], slice) and isinstance(index[1], slice):
            I = np.arange(self.shape[0])[index[0]]
            J = np.arange(self.shape[1])[index[1]]
            origin = self.affine * [I[0], J[0]]
            step = self.step * [index[0].step or 1, index[1].step or 1]
            shape = I.size, J.size
            return RotatedCoordinates(shape, self.theta, origin, step, **self.properties)

        else:
            return super(RotatedCoordinates, self).__getitem__(index)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def deg(self):
        """ :float: rotation angle in degrees. """
        return np.rad2deg(self.theta)

    @property
    def affine(self):
        """ :rasterio.Affine: affine transformation for computing the coordinates from indexing values. Contains the 
            tranlation, rotation, and scaling.
        """
        t = rasterio.Affine.translation(*self.origin)
        r = rasterio.Affine.rotation(self.deg)
        s = rasterio.Affine.scale(*self.step)
        return t * r * s

    @property
    def corner(self):
        """ :array: lower right corner. """
        return np.array(self.affine * np.array([self.shape[0] - 1, self.shape[1] - 1]))

    @property
    def geotransform(self):
        """ :tuple: GDAL geotransform. """
        return self.affine.to_gdal()

    @property
    def coordinates(self):
        """ :tuple: computed coordinave values for each dimension. """
        I = np.arange(self.shape[0])
        J = np.arange(self.shape[1])
        c1, c2 = self.affine * np.meshgrid(I, J)
        return c1.T, c2.T

    @property
    def properties(self):
        """:dict: Dictionary of the coordinate properties. """
        return {key: getattr(self, key) for key in self._properties}

    @property
    def area_bounds(self):
        """:dict: Dictionary of (low, high) coordinates area_bounds in each unstacked dimension"""
        # TODO this is not accurate, the segment lengths need to be rotated
        return super(RotatedCoordinates, self).area_bounds

    def _get_definition(self, full=True):
        d = OrderedDict()
        d["dims"] = self.dims
        d["shape"] = self.shape
        d["theta"] = self.theta
        d["origin"] = self.origin
        d["step"] = self.step
        d.update(self._full_properties if full else self.properties)
        return d

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def copy(self):
        """
        Make a copy of the rotated coordinates.

        Returns
        -------
        :class:`RotatedCoordinates`
            Copy of the rotated coordinates.
        """
        return RotatedCoordinates(self.shape, self.theta, self.origin, self.step, **self.properties)

    def select(self, bounds, outer=False, return_indices=False):
        """
        Get the coordinate values that are within the given bounds in all dimensions.

        *Note: you should not generally need to call this method directly.*

        Parameters
        ----------
        bounds : dict
            dictionary of dim -> (low, high) selection bounds
        outer : bool, optional
            If True, do *outer* selections. Default False.
        return_indices : bool, optional
            If True, return slice or indices for the selections in addition to coordinates. Default False.

        Returns
        -------
        selection : :class:`RotatedCoordinates`, :class:`DependentCoordinates`, :class:`StackedCoordinates`
            rotated, dependent, or stacked coordinates consisting of the selection in all dimensions.
        I : slice or list
            Slice or index for the selected coordinates, only if ``return_indices`` is True.
        """

        # TODO return RotatedCoordinates when possible
        return super(RotatedCoordinates, self).select(bounds, outer=outer, return_indices=return_indices)

    # ------------------------------------------------------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------------------------------------------------------

    # def plot(self, marker='b.', origin_marker='bo', corner_marker='bx'):
    #     from matplotlib import pyplot
    #     super(RotatedCoordinates, self).plot(marker=marker)
    #     ox, oy = self.origin
    #     cx, cy = self.corner
    #     pyplot.plot(ox, oy, origin_marker)
    #     pyplot.plot(cx, cy, corner_marker)
