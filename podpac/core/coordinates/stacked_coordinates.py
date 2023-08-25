from __future__ import division, unicode_literals, print_function, absolute_import

import copy
import warnings

import numpy as np
import xarray as xr
import pandas as pd
import traitlets as tl
from six import string_types
import lazy_import
from scipy import spatial

import podpac
from podpac.core.coordinates.base_coordinates import BaseCoordinates
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.utils import make_coord_value
from podpac.core.coordinates.utils import calculate_distance


class StackedCoordinates(BaseCoordinates):
    """
    Stacked coordinates.

    StackedCoordinates contain coordinates from two or more different dimensions that are stacked together to form a
    list of points (rather than a grid). The underlying coordinates values are :class:`Coordinates1d` objects of equal
    size. The name for the stacked coordinates combines the underlying dimensions with underscores, e.g. ``'lat_lon'``
    or ``'lat_lon_time'``.

    When creating :class:`Coordinates`, podpac automatically detects StackedCoordinates. The following Coordinates
    contain 3 stacked lat-lon coordinates and 2 time coordinates in a 3 x 2 grid::

        >>> lat = [0, 1, 2]
        >>> lon = [10, 20, 30]
        >>> time = ['2018-01-01', '2018-01-02']
        >>> podpac.Coordinates([[lat, lon], time], dims=['lat_lon', 'time'])
        Coordinates
            lat_lon[lat]: ArrayCoordinates1d(lat): Bounds[0.0, 2.0], N[3]
            lat_lon[lon]: ArrayCoordinates1d(lon): Bounds[10.0, 30.0], N[3]
            time: ArrayCoordinates1d(time): Bounds[2018-01-01, 2018-01-02], N[2]

    For convenience, you can also create uniformly-spaced stacked coordinates using :class:`clinspace`::

        >>> lat_lon = podpac.clinspace((0, 10), (2, 30), 3)
        >>> time = ['2018-01-01', '2018-01-02']
        >>> podpac.Coordinates([lat_lon, time], dims=['lat_lon', 'time'])
        Coordinates
            lat_lon[lat]: ArrayCoordinates1d(lat): Bounds[0.0, 2.0], N[3]
            lat_lon[lon]: ArrayCoordinates1d(lon): Bounds[10.0, 30.0], N[3]
            time: ArrayCoordinates1d(time): Bounds[2018-01-01, 2018-01-02], N[2]

    Parameters
    ----------
    dims : tuple
        Tuple of dimension names.
    name : str
        Stacked dimension name.
    coords : dict-like
        xarray coordinates (container of coordinate arrays)
    coordinates : pandas.MultiIndex
        MultiIndex of stacked coordinates values.

    """

    _coords = tl.List(trait=tl.Instance(Coordinates1d), read_only=True)

    def __init__(self, coords, name=None, dims=None):
        """
        Initialize a multidimensional coords bject.

        Parameters
        ----------
        coords : list, :class:`StackedCoordinates`
            Coordinate values in a list, or a StackedCoordinates object to copy.

        See Also
        --------
        clinspace, crange
        """

        if not isinstance(coords, (list, tuple)):
            raise TypeError("Unrecognized coords type '%s'" % type(coords))

        if len(coords) < 2:
            raise ValueError("Stacked coords must have at least 2 coords, got %d" % len(coords))

        # coerce
        coords = tuple(c if isinstance(c, Coordinates1d) else ArrayCoordinates1d(c) for c in coords)

        # set coords
        self.set_trait("_coords", coords)

        # propagate properties
        if dims is not None and name is not None:
            raise TypeError("StackedCoordinates expected 'dims' or 'name', not both")
        if dims is not None:
            self._set_dims(dims)
        if name is not None:
            self._set_name(name)

        # finalize
        super(StackedCoordinates, self).__init__()

    @tl.validate("_coords")
    def _validate_coords(self, d):
        val = d["value"]

        # check sizes
        shape = val[0].shape
        for c in val[1:]:
            if c.shape != shape:
                raise ValueError("Shape mismatch in stacked coords %s != %s" % (c.shape, shape))

        # check dims
        dims = [c.name for c in val]
        for i, dim in enumerate(dims):
            if dim is not None and dim in dims[:i]:
                raise ValueError("Duplicate dimension '%s' in stacked coords" % dim)

        return val

    def _set_name(self, value):
        dims = value.split("_")

        # check length
        if len(dims) != len(self._coords):
            raise ValueError("Invalid name '%s' for StackedCoordinates with length %d" % (value, len(self._coords)))

        self._set_dims(dims)

    def _set_dims(self, dims):
        # check size
        if len(dims) != len(self._coords):
            raise ValueError("Invalid dims '%s' for StackedCoordinates with length %d" % (dims, len(self._coords)))

        for i, dim in enumerate(dims):
            if dim is not None and dim in dims[:i]:
                raise ValueError("Duplicate dimension '%s' in dims" % dim)

        # set names, checking for duplicates
        for i, (c, dim) in enumerate(zip(self._coords, dims)):
            if dim is None:
                continue
            c._set_name(dim)

    # ------------------------------------------------------------------------------------------------------------------
    # Alternate constructors
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

        dims = x.dims[0].split("_")
        cs = [x[dim].data for dim in dims]
        return cls(cs, dims=dims, **kwargs)

    @classmethod
    def from_definition(cls, d):
        """
        Create StackedCoordinates from a stacked coordinates definition.

        Arguments
        ---------
        d : list
            stacked coordinates definition

        Returns
        -------
        :class:`StackedCoordinates`
            stacked coordinates object

        See Also
        --------
        definition
        """

        coords = []
        for elem in d:
            if "start" in elem and "stop" in elem and ("step" in elem or "size" in elem):
                c = UniformCoordinates1d.from_definition(elem)
            elif "values" in elem:
                c = ArrayCoordinates1d.from_definition(elem)
            else:
                raise ValueError("Could not parse coordinates definition with keys %s" % elem.keys())

            coords.append(c)

        return cls(coords)

    # ------------------------------------------------------------------------------------------------------------------
    # standard methods, list-like
    # ------------------------------------------------------------------------------------------------------------------

    def __repr__(self):
        rep = str(self.__class__.__name__)
        for c in self._coords:
            rep += "\n\t%s[%s]: %s" % (self.name, c.name or "?", c)
        return rep

    def __iter__(self):
        return iter(self._coords)

    def __len__(self):
        return len(self._coords)

    def __getitem__(self, index):
        if isinstance(index, string_types):
            if index not in self.dims:
                raise KeyError("Dimension '%s' not found in dims %s" % (index, self.dims))

            return self._coords[self.dims.index(index)]

        else:
            return self._getsubset(index)

    def _getsubset(self, index):
        return StackedCoordinates([c[index] for c in self._coords])

    def __setitem__(self, dim, c):
        if not dim in self.dims:
            raise KeyError("Cannot set dimension '%s' in StackedCoordinates %s" % (dim, self.dims))

        # try to cast to ArrayCoordinates1d
        if not isinstance(c, Coordinates1d):
            c = ArrayCoordinates1d(c)

        if c.name is None:
            c.name = dim

        # replace the element of the coords list
        idx = self.dims.index(dim)
        coords = list(self._coords)
        coords[idx] = c

        # set (and check) new coords list
        self.set_trait("_coords", coords)

    def __contains__(self, item):
        try:
            item = np.array([make_coord_value(value) for value in item])
        except:
            return False

        if len(item) != len(self._coords):
            return False

        if any(val not in c for val, c in zip(item, self._coords)):
            return False

        return (self.flatten().coordinates == item).all(axis=1).any()

    def _eq_base(self, other):
        if not isinstance(other, StackedCoordinates):
            return False

        # shortcuts
        if self.dims != other.dims:
            return False

        if self.shape != other.shape:
            return False

        return True

    def __eq__(self, other):
        if not self._eq_base(other):
            return False

        # full check of underlying coordinates
        if self._coords != other._coords:
            return False

        return True

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def dims(self):
        """:tuple: Tuple of dimension names."""
        return tuple(c.name for c in self._coords)

    @property
    def ndim(self):
        """:int: coordinates array ndim."""
        return self._coords[0].ndim

    @property
    def name(self):
        """:str: Stacked dimension name. Stacked dimension names are the individual `dims` joined by an underscore."""

        if any(self.dims):
            return "_".join(dim or "?" for dim in self.dims)

    @property
    def size(self):
        """:int: Number of stacked coordinates."""
        return self._coords[0].size

    @property
    def shape(self):
        """:tuple: Shape of the stacked coordinates."""
        return self._coords[0].shape

    @property
    def bounds(self):
        """:dict: Dictionary of (low, high) coordinates bounds in each dimension"""
        if None in self.dims:
            raise ValueError("Cannot get bounds for StackedCoordinates with un-named dimensions")
        return {dim: self[dim].bounds for dim in self.udims}

    @property
    def coordinates(self):
        dtypes = [c.dtype for c in self._coords]
        if len(set(dtypes)) == 1:
            dtype = dtypes[0]
        else:
            dtype = object
        return np.dstack([c.coordinates.astype(dtype) for c in self._coords]).squeeze()

    @property
    def xcoords(self):
        """:dict-like: xarray coordinates (container of coordinate arrays)"""
        if None in self.dims:
            raise ValueError("Cannot get xcoords for StackedCoordinates with un-named dimensions")

        if self.ndim == 1:
            # use a multi-index so that we can use DataArray.sel easily
            coords = pd.MultiIndex.from_arrays([np.array(c.coordinates) for c in self._coords], names=self.dims)
            xcoords = {self.name: coords}
        else:
            # fall-back for shaped coordinates
            xcoords = {c.name: (self.xdims, c.coordinates) for c in self._coords}
        return xcoords

    @property
    def definition(self):
        """:list: Serializable stacked coordinates definition."""

        return [c.definition for c in self._coords]

    @property
    def full_definition(self):
        """:list: Serializable stacked coordinates definition, containing all properties. For internal use."""

        return [c.full_definition for c in self._coords]

    @property
    def is_stacked(self):
        return True

    # -----------------------------------------------------------------------------------------------------------------
    # Methods
    # -----------------------------------------------------------------------------------------------------------------

    def copy(self):
        """
        Make a copy of the stacked coordinates.

        Returns
        -------
        :class:`StackedCoordinates`
            Copy of the stacked coordinates.
        """

        return StackedCoordinates(self._coords)

    def unique(self, return_index=False):
        """
        Remove duplicate stacked coordinate values.

        Arguments
        ---------
        return_index : bool, optional
            If True, return index for the unique coordinates in addition to the coordinates. Default False.

        Returns
        -------
        unique : :class:`StackedCoordinates`
            New StackedCoordinates object with unique, sorted, flattened coordinate values.
        unique_index : list of indices
            index
        """

        flat = self.flatten()
        a, I = np.unique(flat.coordinates, axis=0, return_index=True)
        if return_index:
            return flat[I], I
        else:
            return flat[I]

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
            raise ValueError("Cannot get area_bounds for StackedCoordinates with un-named dimensions")
        return {dim: self[dim].get_area_bounds(boundary.get(dim)) for dim in self.dims}

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
        selection : :class:`StackedCoordinates`
            StackedCoordinates object consisting of the selection in all dimensions.
        selection_index : slice, boolean array
            Slice or index for the selected coordinates, only if ``return_index`` is True.
        """

        # logical AND of the selection in each dimension
        indices = [c.select(bounds, outer=outer, return_index=True)[1] for c in self._coords]
        index = self._and_indices(indices)

        if return_index:
            return self[index], index
        else:
            return self[index]

    def _and_indices(self, indices):
        def _index_len(index):
            if isinstance(index, slice):
                if index.stop is None:
                    stop = self.size
                elif index.stop < 0:
                    stop = self.size - index.stop
                else:
                    stop = index.stop
                if index.start is None:
                    start = 0
                elif index.start < 0:
                    start = self.size - index.start
                else:
                    start = index.start
                return stop - start
            return len(index)

        if all(isinstance(index, slice) for index in indices):
            index = slice(max(index.start or 0 for index in indices), min(index.stop or self.size for index in indices))
            # for consistency
            if index.start == 0 and index.stop == self.size:
                if self.ndim > 1:
                    index = [slice(None, None) for dim in self.dims]
                else:
                    index = slice(None, None)
        elif any(_index_len(index) == 0 for index in indices):
            index = slice(0, 0)
        else:
            # convert any slices to boolean array
            for i, index in enumerate(indices):
                if isinstance(index, slice):
                    indices[i] = np.zeros(self.shape, dtype=bool)
                    indices[i][index] = True

            # logical and
            index = np.logical_and.reduce(indices)

            # for consistency
            if np.all(index):
                if self.ndim > 1:
                    index = [slice(None, None) for dim in self.dims]
                else:
                    index = slice(None, None)

        return index

    def _transform(self, transformer):
        if self.size == 0:
            return self.copy()

        coords = [c.copy() for c in self._coords]

        if "lat" in self.dims and "lon" in self.dims and "alt" in self.dims:
            ilat = self.dims.index("lat")
            ilon = self.dims.index("lon")
            ialt = self.dims.index("alt")

            lat = coords[ilat]
            lon = coords[ilon]
            alt = coords[ialt]
            tlon, tlat, talt = transformer.transform(lon.coordinates, lat.coordinates, alt.coordinates)

            coords[ilat] = ArrayCoordinates1d(tlat, "lat").simplify()
            coords[ilon] = ArrayCoordinates1d(tlon, "lon").simplify()
            coords[ialt] = ArrayCoordinates1d(talt, "alt").simplify()

        elif "lat" in self.dims and "lon" in self.dims:
            ilat = self.dims.index("lat")
            ilon = self.dims.index("lon")

            lat = coords[ilat]
            lon = coords[ilon]
            tlon, tlat = transformer.transform(lon.coordinates, lat.coordinates)

            if (
                self.ndim == 2
                and all(np.allclose(a, tlat[:, 0]) for a in tlat.T)
                and all(np.allclose(a, tlon[0]) for a in tlon)
            ):
                coords[ilat] = ArrayCoordinates1d(tlat[:, 0], name="lat").simplify()
                coords[ilon] = ArrayCoordinates1d(tlon[0], name="lon").simplify()
                return coords

            coords[ilat] = ArrayCoordinates1d(tlat, "lat").simplify()
            coords[ilon] = ArrayCoordinates1d(tlon, "lon").simplify()

        elif "alt" in self.dims:
            ialt = self.dims.index("alt")

            alt = coords[ialt]
            _, _, talt = transformer.transform(np.zeros(self.size), np.zeros(self.size), alt.coordinates)

            coords[ialt] = ArrayCoordinates1d(talt, "alt").simplify()

        return StackedCoordinates(coords).simplify()

    def transpose(self, *dims, **kwargs):
        """
        Transpose (re-order) the dimensions of the StackedCoordinates.

        Parameters
        ----------
        dim_1, dim_2, ... : str, optional
            Reorder dims to this order. By default, reverse the dims.
        in_place : boolean, optional
            If True, transpose the dimensions in-place.
            Otherwise (default), return a new, transposed Coordinates object.

        Returns
        -------
        transposed : :class:`StackedCoordinates`
            The transposed StackedCoordinates object.
        """

        in_place = kwargs.get("in_place", False)

        if len(dims) == 0:
            dims = list(self.dims[::-1])

        if set(dims) != set(self.dims):
            raise ValueError("Invalid transpose dimensions, input %s does match any dims in %s" % (dims, self.dims))

        coordinates = [self._coords[self.dims.index(dim)] for dim in dims]

        if in_place:
            self.set_trait("_coords", coordinates)
            return self
        else:
            return StackedCoordinates(coordinates)

    def flatten(self):
        return StackedCoordinates([c.flatten() for c in self._coords])

    def reshape(self, newshape):
        return StackedCoordinates([c.reshape(newshape) for c in self._coords])

    def issubset(self, other):
        """Report whether other coordinates contains these coordinates.

        Arguments
        ---------
        other : Coordinates, StackedCoordinates
            Other coordinates to check

        Returns
        -------
        issubset : bool
            True if these coordinates are a subset of the other coordinates.
        """

        from podpac.core.coordinates import Coordinates

        if not isinstance(other, (Coordinates, StackedCoordinates)):
            raise TypeError(
                "StackedCoordinates issubset expected Coordinates or StackedCoordinates, not '%s'" % type(other)
            )

        if isinstance(other, StackedCoordinates):
            if set(self.dims) != set(other.dims):
                return False

            mine = self.flatten().coordinates
            other = other.flatten().transpose(*self.dims).coordinates
            if len(mine.shape) > len(other.shape):
                other = other.reshape(-1, 1)
            return set(map(tuple, mine)).issubset(map(tuple, other))

        elif isinstance(other, Coordinates):
            if not all(dim in other.udims for dim in self.dims):
                return False

            acs = []
            ocs = []
            for coords in other.values():
                dims = [dim for dim in coords.dims if dim in self.dims]

                if len(dims) == 0:
                    continue

                elif len(dims) == 1:
                    acs.append(self[dims[0]])
                    if isinstance(coords, Coordinates1d):
                        ocs.append(coords)
                    elif isinstance(coords, StackedCoordinates):
                        ocs.append(coords[dims[0]])

                elif len(dims) > 1:
                    acs.append(StackedCoordinates([self[dim] for dim in dims]))
                    if isinstance(coords, StackedCoordinates):
                        ocs.append(StackedCoordinates([coords[dim] for dim in dims]))

            return all(a.issubset(o) for a, o in zip(acs, ocs))

    def simplify(self):
        if self.is_affine:
            from podpac.core.coordinates.affine_coordinates import AffineCoordinates

            # build the geotransform directly
            lat = self["lat"].coordinates
            lon = self["lon"].coordinates

            # We don't have to check every point in lat/lon for the same step
            # since the self.is_affine call did that already
            dlati = (lat[-1, 0] - lat[0, 0]) / (lat.shape[0] - 1)
            dlatj = (lat[0, -1] - lat[0, 0]) / (lat.shape[1] - 1)
            dloni = (lon[-1, 0] - lon[0, 0]) / (lon.shape[0] - 1)
            dlonj = (lon[0, -1] - lon[0, 0]) / (lon.shape[1] - 1)

            # origin point
            p0 = [lat[0, 0], lon[0, 0]] - np.array([[dlati, dlatj], [dloni, dlonj]]) @ np.ones(2) / 2

            # This is defined as x ulc, x width, x height, y ulc, y width, y height
            # x and y are defined by the CRS. Here we are assuming that it's always
            # lon and lat == x and y
            geotransform = [p0[1], dlonj, dloni, p0[0], dlatj, dlati]

            a = AffineCoordinates(geotransform=geotransform, shape=self.shape)

            # simplify in order to convert to UniformCoordinates if appropriate
            return a.simplify()

        return StackedCoordinates([c.simplify() for c in self._coords])

    @property
    def is_affine(self):
        if set(self.dims) != {"lat", "lon"}:
            return False

        if not (self.ndim == 2 and self.shape[0] > 1 and self.shape[1] > 1):
            return False

        lat = self["lat"].coordinates
        lon = self["lon"].coordinates

        d = lat[1:] - lat[:-1]
        if not np.allclose(d, d[0, 0]):
            return False

        d = lat[:, 1:] - lat[:, :-1]
        if not np.allclose(d, d[0, 0]):
            return False

        d = lon[1:] - lon[:-1]
        if not np.allclose(d, d[0, 0]):
            return False

        d = lon[:, 1:] - lon[:, :-1]
        if not np.allclose(d, d[0, 0]):
            return False

        return True

    def horizontal_resolution(self, latitude, ellipsoid_tuple, coordinate_name, restype="nominal", units="meter"):
        """Return the horizontal resolution of a Uniform 1D Coordinate

        Parameters
        ----------
        ellipsoid_tuple: tuple
            a tuple containing ellipsoid information from the the original coordinates to pass to geopy
        coordinate_name: str
            "cartesian" or "ellipsoidal", to tell calculate_distance what kind of calculation to do
        restype: str
            The kind of horizontal resolution that should be returned. Supported values are:
            - "nominal" <-- Gives average nearest distance of all points, with some error
            - "summary" <-- Gives the mean and standard deviation of nearest distance betweem points, with some error
            - "full" <-- Gives exact distance matrix
        units: str
            desired unit to return

        Returns
        -------
        float * (podpac.unit)
            If restype == "nominal", return the average nearest distance with some error
        tuple * (podpac.unit)
            If restype == "summary", return average and std.dev of nearest distances, with some error
        np.ndarray * (podpac.unit)
            if restype == "full", return exact distance matrix
        ValueError
            if unknown restype

        """
        order = tuple([self.dims.index(d) for d in ["lat", "lon"]])

        def nominal_stacked_resolution():
            """Use a KDTree to return approximate stacked resolution with some loss of accuracy.

            Returns
            -------
            The average min distance of every point

            """
            tree = spatial.KDTree(self.coordinates[:, order] + [0, 180.0], boxsize=[0.0, 360.0000000000001])
            return np.average(
                calculate_distance(
                    tree.data - [0, 180.0],
                    tree.data[tree.query(tree.data, k=2)[1][:, 1]] - [0, 180.0],
                    ellipsoid_tuple,
                    coordinate_name,
                    units,
                )
            )

        def summary_stacked_resolution():
            """Return the approximate mean resolution and std.deviation using a KDTree

            Returns
            -------
            tuple
                Average min distance of every point and standard deviation of those min distances
            """
            tree = spatial.KDTree(self.coordinates[:, order] + [0, 180.0], boxsize=[0.0, 360.0000000000001])
            distances = calculate_distance(
                tree.data - [0, 180.0],
                tree.data[tree.query(tree.data, k=2)[1][:, 1]] - [0, 180.0],
                ellipsoid_tuple,
                coordinate_name,
                units,
            )
            return (np.average(distances), np.std(distances))

        def full_stacked_resolution():
            """Returns the exact distance between every point using brute force

            Returns
            -------
            distance matrix of size (NxN), where N is the number of points in the dimension
            """
            distance_matrix = np.zeros((len(self.coordinates), len(self.coordinates)))
            for i in range(len(self.coordinates)):
                distance_matrix[i, :] = calculate_distance(
                    self.coordinates[i, order], self.coordinates[:, order], ellipsoid_tuple, coordinate_name, units
                ).magnitude
            return distance_matrix * podpac.units(units)

        if restype == "nominal":
            return nominal_stacked_resolution()
        elif restype == "summary":
            return summary_stacked_resolution()
        elif restype == "full":
            return full_stacked_resolution()
        else:
            raise ValueError("Invalid value for type: {}".format(restype))
