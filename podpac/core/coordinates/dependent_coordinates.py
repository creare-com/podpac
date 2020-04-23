from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
import warnings

import numpy as np
import traitlets as tl
import lazy_import
from six import string_types
import numbers

from podpac.core.settings import settings
from podpac.core.utils import ArrayTrait, TupleTrait
from podpac.core.coordinates.utils import Dimension
from podpac.core.coordinates.utils import make_coord_array, make_coord_value, make_coord_delta
from podpac.core.coordinates.base_coordinates import BaseCoordinates
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates
from podpac.core.coordinates.cfunctions import clinspace


class DependentCoordinates(BaseCoordinates):
    """
    Base class for dependent/calculated coordinates.

    DependentCoordinates are coordinates from one or more different dimensions that are determined or calculated from
    indexing dimensions. The base class simply contains the dependent coordinates for each dimension. Generally, you
    should not need to create DependentCoordinates, but DependentCoordinates may be the return type when indexing,
    selecting, or intersecting its subclasses.

    DependentCoordinates map an indexing dimension to its dependent coordinate values. For example, rotated coordinates
    are a 2d grid that map index dimensions ('i', 'j') to dependent dimensions ('lat', 'lon').

        >>> import podpac
        >>> c = podpac.coordinates.RotatedCoordinates([20, 30], 0.2, [0, 0], [2, 2], dims=['lat', 'lon'])
        >>> c.dims
        ['lat', 'lon']
        >>> c.idims
        ['i', 'j']
        >>> c[2, 3].coordinates.values
        array([(5.112282296135334, -5.085722143867205)], dtype=object)

    Parameters
    ----------
    dims : tuple
        Tuple of dimension names.
    idims: tuple
        Tuple of indexing dimensions, default ('i', 'j', 'k', 'l') as needed.
    coords : dict-like
        xarray coordinates (container of coordinate arrays)
    coordinates : tuple
        Tuple of coordinate values in each dimension.
    """

    coordinates = TupleTrait(trait=ArrayTrait(), read_only=True)
    dims = TupleTrait(trait=Dimension(allow_none=True), read_only=True)
    idims = TupleTrait(trait=tl.Unicode(), read_only=True)

    _properties = tl.Set()

    def __init__(self, coordinates, dims=None):
        """
        Create dependent coordinates manually. You should not need to use this class directly.

        Parameters
        ----------
        coordinates : tuple
            tuple of coordinate values for each dimension, each the same shape.
        dims : tuple (optional)
            tuple of dimension names ('lat', 'lon', 'time', or 'alt').
        """

        coordinates = [np.array(a) for a in coordinates]
        coordinates = [make_coord_array(a.flatten()).reshape(a.shape) for a in coordinates]
        self.set_trait("coordinates", coordinates)
        if dims is not None:
            self.set_trait("dims", dims)

    @tl.default("dims")
    def _default_dims(self):
        return tuple(None for c in self.coordinates)

    @tl.default("idims")
    def _default_idims(self):
        return tuple("ijkl")[: self.ndims]

    @tl.validate("coordinates")
    def _validate_coordinates(self, d):
        val = d["value"]
        if len(val) == 0:
            raise ValueError("Dependent coordinates cannot be empty")

        for i, a in enumerate(val):
            if a.shape != val[0].shape:
                raise ValueError("coordinates shape mismatch at position %d, %s != %s" % (i, a.shape, val[0].shape))
        return val

    @tl.validate("dims")
    def _validate_dims(self, d):
        val = d["value"]
        if len(val) != self.ndims:
            raise ValueError("dims and coordinates size mismatch, %d != %d" % (len(val), self.ndims))
        for i, dim in enumerate(val):
            if dim is not None and dim in val[:i]:
                raise ValueError("Duplicate dimension '%s' in stacked coords" % dim)
        return val

    @tl.validate("idims")
    def _validate_idims(self, d):
        val = d["value"]
        if len(val) != self.ndims:
            raise ValueError("idims and coordinates size mismatch, %d != %d" % (len(val), self.ndims))
        return val

    @tl.observe("dims", "idims")
    def _set_property(self, d):
        self._properties.add(d["name"])

    def _set_name(self, value):
        # only set if the dims have not been set already
        if "dims" not in self._properties:
            dims = [dim.strip() for dim in value.split(",")]
            self.set_trait("dims", dims)
        elif self.name != value:
            raise ValueError("Dimension mismatch, %s != %s" % (value, self.name))

    # ------------------------------------------------------------------------------------------------------------------
    # Alternate Constructors
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_definition(cls, d):
        """
        Create DependentCoordinates from a dependent coordinates definition.

        Arguments
        ---------
        d : dict
            dependent coordinates definition

        Returns
        -------
        :class:`DependentCoordinates`
            dependent coordinates object

        See Also
        --------
        definition
        """

        if "values" not in d:
            raise ValueError('DependentCoordinates definition requires "values" property')

        coordinates = d["values"]
        kwargs = {k: v for k, v in d.items() if k not in ["values"]}
        return DependentCoordinates(coordinates, **kwargs)

    # -----------------------------------------------------------------------------------------------------------------
    # standard methods
    # -----------------------------------------------------------------------------------------------------------------

    def __repr__(self):
        rep = str(self.__class__.__name__)
        for i, dim in enumerate(self.dims):
            rep += "\n\t%s" % self._rep(dim, index=i)
        return rep

    def _rep(self, dim, index=None):
        if dim is not None:
            index = self.dims.index(dim)
        else:
            dim = "?"  # unnamed dimensions

        c = self.coordinates[index]
        bounds = np.min(c), np.max(c)
        return "%s(%s->%s): Bounds[%s, %s], shape%s" % (
            self.__class__.__name__,
            ",".join(self.idims),
            dim,
            bounds[0],
            bounds[1],
            self.shape,
        )

    def __eq__(self, other):
        if not isinstance(other, DependentCoordinates):
            return False

        # shortcut
        if self.shape != other.shape:
            return False

        # defined coordinate properties should match
        for name in self._properties.union(other._properties):
            if getattr(self, name) != getattr(other, name):
                return False

        # full coordinates check
        if not np.array_equal(self.coordinates, other.coordinates):
            return False

        return True

    def __iter__(self):
        return iter(self[dim] for dim in self.dims)

    def __getitem__(self, index):
        if isinstance(index, string_types):
            dim = index
            if dim not in self.dims:
                raise KeyError("Cannot get dimension '%s' in RotatedCoordinates %s" % (dim, self.dims))

            i = self.dims.index(dim)
            return ArrayCoordinatesNd(self.coordinates[i], **self._properties_at(i))

        else:
            coordinates = tuple(a[index] for a in self.coordinates)
            # return DependentCoordinates(coordinates, **self.properties)

            # NOTE: this is optional, but we can convert to StackedCoordinates if ndim is 1
            if coordinates[0].ndim == 1 or coordinates[0].size <= 1:
                cs = [ArrayCoordinates1d(a, **self._properties_at(i)) for i, a in enumerate(coordinates)]
                return StackedCoordinates(cs)
            else:
                return DependentCoordinates(coordinates, **self.properties)

    def _properties_at(self, index=None, dim=None):
        if index is None:
            index = self.dims.index(dim)
        properties = {}
        properties["name"] = self.dims[index]
        return properties

    # -----------------------------------------------------------------------------------------------------------------
    # properties
    # -----------------------------------------------------------------------------------------------------------------

    @property
    def name(self):
        """:str: combined dependent dimensions name.

        The combined dependent dimension name is the individual `dims` joined by a comma.
        """
        return "%s" % ",".join([dim or "?" for dim in self.dims])

    @property
    def udims(self):
        """:tuple: Tuple of unstacked dimension names, for compatibility. This is the same as the dims."""
        return self.dims

    @property
    def shape(self):
        """:tuple: Shape of the coordinates (in every dimension)."""
        return self.coordinates[0].shape

    @property
    def size(self):
        """:int: Number of coordinates (in every dimension)."""
        return np.prod(self.shape)

    @property
    def ndims(self):
        """:int: Number of dependent dimensions."""
        return len(self.coordinates)

    @property
    def dtypes(self):
        """:tuple: Dtype for each dependent dimension."""
        return tuple(c.dtype for c in self.coordinates)

    @property
    def bounds(self):
        """:dict: Dictionary of (low, high) coordinates bounds in each unstacked dimension"""
        if None in self.dims:
            raise ValueError("Cannot get bounds for DependentCoordinates with un-named dimensions")
        return {dim: self[dim].bounds for dim in self.dims}

    @property
    def coords(self):
        """:dict-like: xarray coordinates (container of coordinate arrays)"""
        if None in self.dims:
            raise ValueError("Cannot get coords for DependentCoordinates with un-named dimensions")
        return {dim: (self.idims, c) for dim, c in (zip(self.dims, self.coordinates))}

    @property
    def properties(self):
        """:dict: Dictionary of the coordinate properties. """

        return {key: getattr(self, key) for key in self._properties}

    @property
    def definition(self):
        """:dict: Serializable dependent coordinates definition."""

        return self._get_definition(full=False)

    @property
    def full_definition(self):
        """:dict: Serializable dependent coordinates definition, containing all properties. For internal use."""

        return self._get_definition(full=True)

    def _get_definition(self, full=True):
        d = OrderedDict()
        d["dims"] = self.dims
        d["values"] = self.coordinates
        d.update(self._full_properties if full else self.properties)
        return d

    @property
    def _full_properties(self):
        return {"dims": self.dims}

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def copy(self):
        """
        Make a copy of the dependent coordinates.

        Returns
        -------
        :class:`DependentCoordinates`
            Copy of the dependent coordinates.
        """

        return DependentCoordinates(self.coordinates, **self.properties)

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

        if None in self.dims:
            raise ValueError("Cannot get area_bounds for DependentCoordinates with un-named dimensions")
        return {dim: self[dim].get_area_bounds(boundary.get(dim)) for dim in self.dims}

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
        selection : :class:`DependentCoordinates`, :class:`StackedCoordinates`
            DependentCoordinates or StackedCoordinates object consisting of the selection in all dimensions.
        I : slice or list
            Slice or index for the selected coordinates, only if ``return_indices`` is True.
        """

        # logical AND of selection in each dimension
        Is = [self._within(a, bounds.get(dim), outer) for dim, a in zip(self.dims, self.coordinates)]
        I = np.logical_and.reduce(Is)

        if np.all(I):
            return self._select_all(return_indices)

        if return_indices:
            return self[I], np.where(I)
        else:
            return self[I]

    def _within(self, coordinates, bounds, outer):
        if bounds is None:
            return np.ones(self.shape, dtype=bool)

        lo, hi = bounds
        lo = make_coord_value(lo)
        hi = make_coord_value(hi)

        if outer:
            below = coordinates[coordinates <= lo]
            above = coordinates[coordinates >= hi]
            lo = max(below) if below.size else -np.inf
            hi = min(above) if above.size else np.inf

        gt = coordinates >= lo
        lt = coordinates <= hi
        return gt & lt

    def _select_all(self, return_indices):
        if return_indices:
            return self, slice(None)
        else:
            return self

    def _transform(self, transformer):
        coords = [c.copy() for c in self.coordinates]
        properties = self.properties

        if "lat" in self.dims and "lon" in self.dims and "alt" in self.dims:
            ilat = self.dims.index("lat")
            ilon = self.dims.index("lon")
            ialt = self.dims.index("alt")

            lat = coords[ilat].flatten()
            lon = coords[ilon].flatten()
            alt = coords[ialt].flatten()
            tlon, tlat, talt = transformer.transform(lon, lat, alt)
            coords[ilat] = tlat.reshape(self.shape)
            coords[ilon] = tlon.reshape(self.shape)
            coords[ialt] = talt.reshape(self.shape)

        elif "lat" in self.dims and "lon" in self.dims:
            ilat = self.dims.index("lat")
            ilon = self.dims.index("lon")

            lat = coords[ilat].flatten()
            lon = coords[ilon].flatten()
            tlon, tlat = transformer.transform(lon, lat)
            coords[ilat] = tlat.reshape(self.shape)
            coords[ilon] = tlon.reshape(self.shape)

        elif "alt" in self.dims:
            ialt = self.dims.index("alt")

            alt = coords[ialt].flatten()
            _, _, talt = transformer.transform(np.zeros(self.size), np.zeros(self.size), alt)
            coords[ialt] = talt.reshape(self.shape)

        return DependentCoordinates(coords, **properties).simplify()

    def simplify(self):
        coords = [c.copy() for c in self.coordinates]
        slc_start = [slice(0, 1) for d in self.dims]

        for dim in self.dims:
            i = self.dims.index(dim)
            slc = slc_start.copy()
            slc[i] = slice(None)
            if dim in ["lat", "lon"] and not np.allclose(coords[i][tuple(slc)], coords[i], atol=1e-7):
                return self
            coords[i] = ArrayCoordinates1d(coords[i][tuple(slc)].squeeze(), name=dim).simplify()

        return coords

    def transpose(self, *dims, **kwargs):
        """
        Transpose (re-order) the dimensions of the DependentCoordinates.

        Parameters
        ----------
        dim_1, dim_2, ... : str, optional
            Reorder dims to this order. By default, reverse the dims.
        in_place : boolean, optional
            If True, transpose the dimensions in-place.
            Otherwise (default), return a new, transposed Coordinates object.
        
        Returns
        -------
        transposed : :class:`DependentCoordinates`
            The transposed DependentCoordinates object.
        """

        in_place = kwargs.get("in_place", False)

        if len(dims) == 0:
            dims = list(self.dims[::-1])

        if set(dims) != set(self.dims):
            raise ValueError("Invalid transpose dimensions, input %s does match dims %s" % (dims, self.dims))

        coordinates = [self.coordinates[self.dims.index(dim)] for dim in dims]

        if in_place:
            self.set_trait("coordinates", coordinates)
            self.set_trait("dims", dims)
            return self
        else:
            properties = self.properties
            properties["dims"] = dims
            return DependentCoordinates(coordinates, **properties)

    # ------------------------------------------------------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------------------------------------------------------

    # def plot(self, marker='b.'):
    #     from matplotlib import pyplot
    #     if self.ndims != 2:
    #         raise NotImplementedError("Only 2d DependentCoordinates plots are supported")
    #     x, y = self.coordinates
    #     pyplot.plot(x, y, marker)
    #     pyplot.xlabel(self.dims[0])
    #     pyplot.ylabel(self.dims[1])
    #     pyplot.axis('equal')


class ArrayCoordinatesNd(ArrayCoordinates1d):
    """
    Partial implementation for internal use.
    
    Provides name, dtype, size, bounds (and others).
    Prohibits coords, intersect, select (and others).

    Used primarily for intersection with DependentCoordinates.
    """

    coordinates = ArrayTrait(read_only=True)

    def __init__(self, coordinates, name=None):
        """
        Create shaped array coordinates. You should not need to use this class directly.

        Parameters
        ----------
        coordinates : array
            coordinate values.
        name : str, optional
            Dimension name, one of 'lat', 'lon', 'time', or 'alt'.
        """

        self.set_trait("coordinates", coordinates)
        self._is_monotonic = None
        self._is_descending = None
        self._is_uniform = None

        Coordinates1d.__init__(self, name=name)

    def __repr__(self):
        return "%s(%s): Bounds[%s, %s], shape%s" % (
            self.__class__.__name__,
            self.name or "?",
            self.bounds[0],
            self.bounds[1],
            self.shape,
        )

    @property
    def shape(self):
        """:tuple: Shape of the coordinates."""
        return self.coordinates.shape

    # Restricted methods and properties

    @classmethod
    def from_xarray(cls, x):
        """ restricted """
        raise RuntimeError("ArrayCoordinatesNd from_xarray is unavailable.")

    @classmethod
    def from_definition(cls, d):
        """ restricted """
        raise RuntimeError("ArrayCoordinatesNd from_definition is unavailable.")

    @property
    def definition(self):
        """ restricted """
        raise RuntimeError("ArrayCoordinatesNd definition is unavailable.")

    @property
    def full_definition(self):
        """ restricted """
        raise RuntimeError("ArrayCoordinatesNd full_definition is unavailable.")

    @property
    def coords(self):
        """ restricted """
        raise RuntimeError("ArrayCoordinatesNd coords is unavailable.")

    def intersect(self, other, outer=False, return_indices=False):
        """ restricted """
        raise RuntimeError("ArrayCoordinatesNd intersect is unavailable.")

    def select(self, bounds, outer=False, return_indices=False):
        """ restricted """
        raise RuntimeError("ArrayCoordinatesNd select is unavailable.")
