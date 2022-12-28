from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict

import numpy as np
import traitlets as tl
import lazy_import
import warnings

from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d

affine = lazy_import.lazy_module("affine")

from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates
from podpac.core.coordinates.cfunctions import clinspace


class AffineCoordinates(StackedCoordinates):
    """
    A grid of latitude and longitude coordinates, defined by an affine transformation.

    Parameters
    ----------
    geotransform : tuple
        GDAL geotransform
    shape : tuple
        shape (m, n) of the grid.
    dims : tuple
        Tuple of dimension names.
    coords : dict-like
        xarray coordinates (container of coordinate arrays)
    coordinates : tuple
        Tuple of 2d coordinate values in each dimension.

    Notes
    -----

    https://gdal.org/tutorials/geotransforms_tut.html

    GT(0) x-coordinate of the upper-left corner of the upper-left pixel.
    GT(1) w-e pixel resolution / pixel width.
    GT(2) row rotation (typically zero).
    GT(3) y-coordinate of the upper-left corner of the upper-left pixel.
    GT(4) column rotation (typically zero).
    GT(5) n-s pixel resolution / pixel height (negative value for a north-up image).

    """

    geotransform = tl.Tuple(tl.Float(), tl.Float(), tl.Float(), tl.Float(), tl.Float(), tl.Float(), read_only=True)
    shape = tl.Tuple(tl.Integer(), tl.Integer(), read_only=True)

    def __init__(self, geotransform=None, shape=None):
        """
        Create a grid of coordinates from a `geotransform` and `shape`.

        Parameters
        ----------
        geotransform : tuple
            GDAL geotransform
        shape : tuple
            shape (m, n) of the grid.
        """
        if isinstance(geotransform, np.ndarray):
            geotransform = tuple(geotransform.tolist())
        self.set_trait("geotransform", geotransform)
        self.set_trait("shape", shape)

        # private traits
        self._affine = affine.Affine.from_gdal(*self.geotransform)

    @tl.validate("shape")
    def _validate_shape(self, d):
        val = d["value"]
        if val[0] <= 0 or val[1] <= 0:
            raise ValueError("Invalid shape %s, shape must be positive" % (val,))
        return val

    # ------------------------------------------------------------------------------------------------------------------
    # Alternate Constructors
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_definition(cls, d):
        """
        Create AffineCoordinates from an affine coordinates definition.

        Arguments
        ---------
        d : dict
            affine coordinates definition

        Returns
        -------
        :class:`AffineCoordinates`
            affine coordinates object

        See Also
        --------
        definition
        """

        if "geotransform" not in d:
            raise ValueError('AffineCoordinates definition requires "geotransform" property')
        if "shape" not in d:
            raise ValueError('AffineCoordinates definition requires "shape" property')
        return AffineCoordinates(geotransform=d["geotransform"], shape=d["shape"])

    # ------------------------------------------------------------------------------------------------------------------
    # standard methods
    # ------------------------------------------------------------------------------------------------------------------

    def __repr__(self):
        return "%s(%s): Bounds(lat, lon)([%g, %g], [%g, %g]), Shape%s" % (
            self.__class__.__name__,
            self.dims,
            self.bounds["lat"][0],
            self.bounds["lat"][1],
            self.bounds["lon"][0],
            self.bounds["lon"][1],
            self.shape,
        )

    def __eq__(self, other):
        if not self._eq_base(other):
            return False

        if not other.is_affine:
            return False

        if not np.allclose(self.geotransform, other.geotransform):
            return False

        return True

    def _getsubset(self, index):
        if isinstance(index, tuple) and isinstance(index[0], slice) and isinstance(index[1], slice):
            lat = self["lat"].coordinates[index]
            lon = self["lon"].coordinates[index]

            # We don't have to check every point in lat/lon for the same step
            # since the self.is_affine call did that already
            dlati = (lat[-1, 0] - lat[0, 0]) / (lat.shape[0] - 1)
            dlatj = (lat[0, -1] - lat[0, 0]) / (lat.shape[1] - 1)
            dloni = (lon[-1, 0] - lon[0, 0]) / (lon.shape[0] - 1)
            dlonj = (lon[0, -1] - lon[0, 0]) / (lon.shape[1] - 1)

            # origin point
            p0 = np.array([lat[0, 0], lon[0, 0]]) - np.array([[dlati, dlatj], [dloni, dlonj]]) @ np.ones(2) / 2

            # This is defined as x ulc, x width, x height, y ulc, y width, y height
            # x and y are defined by the CRS. Here we are assuming that it's always
            # lon and lat == x and y
            geotransform = [p0[1], dlonj, dloni, p0[0], dlatj, dlati]

            # get shape from indexed coordinates
            shape = lat.shape

            return AffineCoordinates(geotransform=geotransform, shape=lat.shape)

        else:
            return super(AffineCoordinates, self)._getsubset(index).simplify()

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def _coords(self):
        if not hasattr(self, "_coords_"):
            self._coords_ = [
                ArrayCoordinates1d(c, name=dim) for c, dim in zip(self.coordinates.transpose(2, 0, 1), self.dims)
            ]
        return self._coords_

    @property
    def ndim(self):
        return 2

    @property
    def affine(self):
        """:affine.Affine: affine transformation for computing the coordinates from indexing values."""
        return self._affine

    @property
    def dims(self):
        return ("lat", "lon")

    @property
    def is_affine(self):
        return True

    @property
    def origin(self):
        origin = self.affine * [0, 0]
        if self.dims == ("lat", "lon"):
            origin = origin[::-1]
        return origin

    @property
    def coordinates(self):
        """:tuple: computed coordinate values for each dimension."""

        I = np.arange(self.shape[1]) + 0.5
        J = np.arange(self.shape[0]) + 0.5
        x, y = self.affine * np.meshgrid(I, J)
        if self.dims == ("lat", "lon"):
            c = np.stack([y, x])
        else:
            c = np.stack([x, y])
        return c.transpose(1, 2, 0)

    @property
    def definition(self):
        d = OrderedDict()
        d["geotransform"] = self.geotransform
        d["shape"] = self.shape
        return d

    @property
    def full_definition(self):
        return self.definition

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def copy(self):
        """
        Make a copy of the affine coordinates.

        Returns
        -------
        :class:`AffineCoordinates`
            Copy of the affine coordinates.
        """
        return AffineCoordinates(self.geotransform, self.shape)

    def get_area_bounds(self, boundary):
        """Get coordinate area bounds, including boundary information, for each unstacked dimension.

        Arguments
        ---------
        boundary : dict
            dictionary of boundary offsets for each unstacked dimension. Point dimensions can be omitted.

        Returns
        -------
        area_bounds : dict
            Dictionary of (low, high) coordinates area_bounds in each unstacked dimension
        """

        # TODO the boundary offsets need to be transformed
        warnings.warn("AffineCoordinates area_bounds are not yet correctly implemented.")
        return super(AffineCoordinates, self).get_area_bounds(boundary)

    def select(self, bounds, outer=False, return_index=False):
        """
        Get the coordinate values that are within the given bounds in all dimensions.

        *Note: you should not generally need to call this method directly.*

        Parameters
        ----------
        bounds : dict
            dictionary of dim -> (low, high) selection bounds
        outer : bool, optional
            If True, do *outer* selections. Default False.
        return_index : bool, optional
            If True, return index for the selections in addition to coordinates. Default False.

        Returns
        -------
        selection : :class:`StackedCoordinates`, :class:`AffineCoordinates`
            coordinates consisting of the selection in all dimensions.
        selection_index : list
            index for the selected coordinates, only if ``return_index`` is True.
        """

        if not outer:
            # if the geotransform is rotated, the inner selection is not a grid
            # returning the general stacked coordinates is a general solution
            return super(AffineCoordinates, self).select(bounds, outer=outer, return_index=return_index)

        # same rotation and step, new origin and shape
        lat = self.coordinates[:, :, 0]
        lon = self.coordinates[:, :, 1]
        b = (
            (lat >= bounds["lat"][0])
            & (lat <= bounds["lat"][1])
            & (lon >= bounds["lon"][0])
            & (lon <= bounds["lon"][1])
        )

        I, J = np.where(b)
        imin = max(0, np.min(I) - 1)
        jmin = max(0, np.min(J) - 1)
        imax = min(self.shape[0] - 1, np.max(I) + 1)
        jmax = min(self.shape[1] - 1, np.max(J) + 1)

        origin = np.array([lat[imin, jmin], lon[imin, jmin]])
        origin -= np.array([lat[0, 0], lon[0, 0]]) - self.origin

        shape = int(imax - imin + 1), int(jmax - jmin + 1)

        geotransform = (
            origin[1],
            self.geotransform[1],
            self.geotransform[2],
            origin[0],
            self.geotransform[4],
            self.geotransform[5],
        )

        selected = AffineCoordinates(geotransform=geotransform, shape=shape)

        if return_index:
            return selected, (slice(imin, imax + 1), slice(jmin, jmax + 1))
        else:
            return selected

    def simplify(self):
        # NOTE: podpac prefers unstacked UniformCoordinates to AffineCoordinates
        #       if that changes, just return self.copy()
        if self.affine.is_rectilinear:
            tol = 1e-15  # tolerance for deciding when a number is zero
            a = self.affine
            shape = self.shape

            if np.abs(a.e) <= tol and np.abs(a.a) <= tol:
                order = -1
                step = np.array([a.d, a.b])
            else:
                order = 1
                step = np.array([a.e, a.a])

            origin = a.f + step[0] / 2, a.c + step[1] / 2
            end = origin[0] + step[0] * (shape[::order][0] - 1), origin[1] + step[1] * (shape[::order][1] - 1)
            # when the shape == 1, UniformCoordinates1d cannot infer the step from the size
            # we have have to create the UniformCoordinates1d manually
            if shape[::order][0] == 1:
                lat = UniformCoordinates1d(origin[0], end[0], step=step[0], name="lat")
            else:
                lat = clinspace(origin[0], end[0], shape[::order][0], "lat")
            if shape[::order][1] == 1:
                lon = UniformCoordinates1d(origin[1], end[1], step=step[1], name="lon")
            else:
                lon = clinspace(origin[1], end[1], shape[::order][1], "lon")
            return [lat, lon][::order]

        return self.copy()

    # ------------------------------------------------------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------------------------------------------------------

    def plot(self, marker="b.", origin_marker="bo", corner_marker="bx"):
        from matplotlib import pyplot

        x = self.coordinates[:, :, 0]
        y = self.coordinates[:, :, 1]
        pyplot.plot(x.flatten(), y.flatten(), marker)
        ox, oy = self.origin
        pyplot.plot(ox, oy, origin_marker)
        pyplot.gca().set_aspect("equal")
