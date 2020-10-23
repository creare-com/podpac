"""
Generic Data Source Class

DataSource is the root class for all other podpac defined data sources,
including user defined data sources.
"""

from __future__ import division, unicode_literals, print_function, absolute_import
from collections import OrderedDict
from copy import deepcopy
import warnings
import logging

import numpy as np
import xarray as xr
import traitlets as tl

# Internal imports
from podpac.core.settings import settings
from podpac.core.units import UnitsDataArray
from podpac.core.coordinates import Coordinates, Coordinates1d, StackedCoordinates
from podpac.core.coordinates.utils import VALID_DIMENSION_NAMES, make_coord_delta, make_coord_delta_array
from podpac.core.node import Node, NodeException
from podpac.core.utils import common_doc
from podpac.core.node import COMMON_NODE_DOC
from podpac.core.interpolation.interpolation import Interpolate, InterpolationTrait

log = logging.getLogger(__name__)

DATA_DOC = {
    "coordinates": "The coordinates of the data source.",
    "get_data": """
        This method must be defined by the data source implementing the DataSource class.
        When data source nodes are evaluated, this method is called with request coordinates and coordinate indexes.
        The implementing method can choose which input provides the most efficient method of getting data
        (i.e via coordinates or via the index of the coordinates).
        
        Coordinates and coordinate indexes may be strided or subsets of the
        source data, but all coordinates and coordinate indexes will match 1:1 with the subset data.

        This method may return a numpy array, an xarray DaraArray, or a podpac UnitsDataArray.
        If a numpy array or xarray DataArray is returned, :meth:`podpac.data.DataSource.evaluate` will
        cast the data into a `UnitsDataArray` using the requested source coordinates.
        If a podpac UnitsDataArray is passed back, the :meth:`podpac.data.DataSource.evaluate`
        method will not do any further processing.
        The inherited Node method `create_output_array` can be used to generate the template UnitsDataArray
        in your DataSource.
        See :meth:`podpac.Node.create_output_array` for more details.
        
        Parameters
        ----------
        coordinates : :class:`podpac.Coordinates`
            The coordinates that need to be retrieved from the data source using the coordinate system of the data
            source
        coordinates_index : List
            A list of slices or a boolean array that give the indices of the data that needs to be retrieved from
            the data source. The values in the coordinate_index will vary depending on the `coordinate_index_type`
            defined for the data source.
            
        Returns
        --------
        np.ndarray, xr.DataArray, :class:`podpac.UnitsDataArray`
            A subset of the returned data. If a numpy array or xarray DataArray is returned,
            the data will be cast into  UnitsDataArray using the returned data to fill values
            at the requested source coordinates.
        """,
    "get_coordinates": """
        Returns a Coordinates object that describes the coordinates of the data source.

        In most cases, this method is defined by the data source implementing the DataSource class.
        If method is not implemented by the data source, it will try to return ``self.coordinates``
        if ``self.coordinates`` is not None.

        Otherwise, this method will raise a NotImplementedError.

        Returns
        --------
        :class:`podpac.Coordinates`
           The coordinates describing the data source array.

        Notes
        ------
        Need to pay attention to:
        - the order of the dimensions
        - the stacking of the dimension
        - the type of coordinates

        Coordinates should be non-nan and non-repeating for best compatibility
        """,
    "interpolation": """
        Interpolation definition for the data source.
        By default, the interpolation method is set to ``'nearest'`` for all dimensions.
        """,
    "interpolation_long": """
        {interpolation}

        If input is a string, it must match one of the interpolation shortcuts defined in
        :attr:`podpac.data.INTERPOLATION_SHORTCUTS`. The interpolation method associated
        with this string will be applied to all dimensions at the same time.

        If input is a dict or list of dict, the dict or dict elements must adhere to the following format:

        The key ``'method'`` defining the interpolation method name.
        If the interpolation method is not one of :attr:`podpac.data.INTERPOLATION_SHORTCUTS`, a
        second key ``'interpolators'`` must be defined with a list of
        :class:`podpac.interpolators.Interpolator` classes to use in order of uages.
        The dictionary may contain an option ``'params'`` key which contains a dict of parameters to pass along to
        the :class:`podpac.interpolators.Interpolator` classes associated with the interpolation method.
        
        The dict may contain the key ``'dims'`` which specifies dimension names (i.e. ``'time'`` or ``('lat', 'lon')`` ). 
        If the dictionary does not contain a key for all unstacked dimensions of the source coordinates, the
        :attr:`podpac.data.INTERPOLATION_DEFAULT` value will be used.
        All dimension keys must be unstacked even if the underlying coordinate dimensions are stacked.
        Any extra dimensions included but not found in the source coordinates will be ignored.

        The dict may contain a key ``'params'`` that can be used to configure the :class:`podpac.interpolators.Interpolator` classes associated with the interpolation method.

        If input is a :class:`podpac.data.Interpolation` class, this Interpolation
        class will be used without modification.
        """,
}

COMMON_DATA_DOC = COMMON_NODE_DOC.copy()
COMMON_DATA_DOC.update(DATA_DOC)  # inherit and overwrite with DATA_DOC


@common_doc(COMMON_DATA_DOC)
class DataSource(Node):
    """Base node for any data obtained directly from a single source.

    Parameters
    ----------
    source : Any
        The location of the source. Depending on the child node this can be a filepath,
        numpy array, or dictionary as a few examples.
    coordinates : :class:`podpac.Coordinates`
        {coordinates}
    nan_vals : List, optional
        List of values from source data that should be interpreted as 'no data' or 'nans'
    coordinate_index_type : str, optional
        Type of index to use for data source. Possible values are ``['slice', 'numpy']``
        Default is 'numpy', which allows a tuple of integer indices.
    cache_coordinates : bool
        Whether to cache coordinates using the podpac ``cache_ctrl``. Default False.
    cache_output : bool
        Should the node's output be cached? If not provided or None, uses default based on
        settings["CACHE_DATASOURCE_OUTPUT_DEFAULT"]. If True, outputs will be cached and retrieved from cache. If False,
        outputs will not be cached OR retrieved from cache (even if they exist in cache).

    Notes
    -----
    Custom DataSource Nodes must implement the :meth:`get_data` and :meth:`get_coordinates` methods.
    """

    interpolation = InterpolationTrait().tag(attr=True)
    nan_vals = tl.List().tag(attr=True)
    boundary = tl.Dict().tag(attr=True)

    coordinate_index_type = tl.Enum(["slice", "numpy"], default_value="numpy")  # ,"list", "xarray", "pandas"],
    cache_coordinates = tl.Bool(False)
    cache_output = tl.Bool()

    # privates
    _coordinates = tl.Instance(Coordinates, allow_none=True, default_value=None, read_only=True)

    if settings["DEBUG"]:
        _original_requested_coordinates = tl.Instance(Coordinates, allow_none=True)
        _requested_source_coordinates = tl.Instance(Coordinates)
        _requested_source_coordinates_index = tl.Tuple()
        _requested_source_boundary = tl.Dict()
        _requested_source_data = tl.Instance(UnitsDataArray)
        _evaluated_coordinates = tl.Instance(Coordinates)

    @tl.validate("boundary")
    def _validate_boundary(self, d):
        val = d["value"]
        for dim, boundary in val.items():
            if dim not in VALID_DIMENSION_NAMES:
                raise ValueError("Invalid dimension '%s' in boundary" % dim)
            if np.array(boundary).ndim == 0:
                try:
                    delta = make_coord_delta(boundary)
                except ValueError:
                    raise ValueError(
                        "Invalid boundary for dimension '%s' ('%s' is not a valid coordinate delta)" % (dim, boundary)
                    )

                if np.array(delta).astype(float) < 0:
                    raise ValueError("Invalid boundary for dimension '%s' (%s < 0)" % (dim, delta))

            if np.array(boundary).ndim == 1:
                make_coord_delta_array(boundary)
                raise NotImplementedError("Non-centered boundary not yet supported for dimension '%s'" % dim)

            if np.array(boundary).ndim == 2:
                for elem in boundary:
                    make_coord_delta_array(elem)
                raise NotImplementedError("Non-uniform boundary not yet supported for dimension '%s'" % dim)

        return val

    @tl.default("cache_output")
    def _cache_output_default(self):
        return settings["CACHE_DATASOURCE_OUTPUT_DEFAULT"]

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def coordinates(self):
        """{coordinates}"""

        if self._coordinates is not None:
            nc = self._coordinates
        elif self.cache_coordinates and self.has_cache("coordinates"):
            nc = self.get_cache("coordinates")
            self.set_trait("_coordinates", nc)
        else:
            nc = self.get_coordinates()
            self.set_trait("_coordinates", nc)
            if self.cache_coordinates:
                self.put_cache(nc, "coordinates")
        return nc

    # ------------------------------------------------------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------------------------------------------------------

    def _get_data(self, rc, rci):
        """Wrapper for `self.get_data` with pre and post processing

        Returns
        -------
        podpac.core.units.UnitsDataArray
            Returns UnitsDataArray with coordinates defined by _requested_source_coordinates

        Raises
        ------
        ValueError
            Raised if unknown data is passed by from self.get_data
        NotImplementedError
            Raised if get_data is not implemented by data source subclass

        """
        # get data from data source at requested source coordinates and requested source coordinates index
        data = self.get_data(rc, rci)

        # convert data into UnitsDataArray depending on format
        # TODO: what other processing needs to happen here?
        if isinstance(data, UnitsDataArray):
            udata_array = data
        elif isinstance(data, xr.DataArray):
            # TODO: check order of coordinates here
            udata_array = self.create_output_array(rc, data=data.data)
        elif isinstance(data, np.ndarray):
            udata_array = self.create_output_array(rc, data=data)
        else:
            raise ValueError(
                "Unknown data type passed back from "
                + "{}.get_data(): {}. ".format(type(self).__name__, type(data))
                + "Must be one of numpy.ndarray, xarray.DataArray, or podpac.UnitsDataArray"
            )

        # extract single output, if necessary
        # subclasses should extract single outputs themselves if possible, but this provides a backup
        if "output" in udata_array.dims and self.output is not None:
            udata_array = udata_array.sel(output=self.output)

        # fill nan_vals in data array
        for nan_val in self.nan_vals:
            udata_array.data[udata_array.data == nan_val] = np.nan

        return udata_array

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    @common_doc(COMMON_DATA_DOC)
    def _eval(self, coordinates, output=None, _selector=None):
        """Evaluates this node using the supplied coordinates.

        The coordinates are mapped to the requested coordinates, interpolated if necessary, and set to
        `_requested_source_coordinates` with associated index `_requested_source_coordinates_index`. The requested
        source coordinates and index are passed to `get_data()` returning the source data at the
        coordinatesset to `_requested_source_data`. Finally `_requested_source_data` is interpolated
        using the `interpolate` method and set to the `output` attribute of the node.


        Parameters
        ----------
        coordinates : :class:`podpac.Coordinates`
            {requested_coordinates}

            An exception is raised if the requested coordinates are missing dimensions in the DataSource.
            Extra dimensions in the requested coordinates are dropped.
        output : :class:`podpac.UnitsDataArray`, optional
            {eval_output}
        _selector: callable(coordinates, request_coordinates)
            {eval_selector}

        Returns
        -------
        {eval_return}

        Raises
        ------
        ValueError
            Cannot evaluate these coordinates
        """
        log.debug("Evaluating {} data source".format(self.__class__.__name__))

        if self.coordinate_index_type not in ["slice", "numpy"]:
            warnings.warn(
                "Coordinates index type {} is not yet supported.".format(self.coordinate_index_type)
                + "`coordinate_index_type` is set to `numpy`",
                UserWarning,
            )

        # store requested coordinates for debugging
        if settings["DEBUG"]:
            self._original_requested_coordinates = coordinates

        # check for missing dimensions
        for c in self.coordinates.values():
            if isinstance(c, Coordinates1d):
                if c.name not in coordinates.udims:
                    raise ValueError("Cannot evaluate these coordinates, missing dim '%s'" % c.name)
            elif isinstance(c, StackedCoordinates):
                if any(s.name not in coordinates.udims for s in c):
                    raise ValueError("Cannot evaluate these coordinates, missing at least one dim in '%s'" % c.name)

        # remove extra dimensions
        extra = [
            c.name
            for c in coordinates.values()
            if (isinstance(c, Coordinates1d) and c.name not in self.coordinates.udims)
            or (isinstance(c, StackedCoordinates) and all(dim not in self.coordinates.udims for dim in c.dims))
        ]
        coordinates = coordinates.drop(extra)

        # transform coordinates into native crs if different
        if self.coordinates.crs.lower() != coordinates.crs.lower():
            coordinates = coordinates.transform(self.coordinates.crs)

        # get source coordinates that are within the requested coordinates bounds
        (rsc, rsci) = self.coordinates.intersect(coordinates, outer=True, return_indices=True)

        # if requested coordinates and coordinates do not intersect, shortcut with nan UnitsDataArary
        if rsc.size == 0:
            if output is None:
                output = self.create_output_array(coordinates)
                if "output" in output.dims and self.output is not None:
                    output = output.sel(output=self.output)
            else:
                output[:] = np.nan

            if settings["DEBUG"]:
                self._evaluated_coordinates = deepcopy(coordinates)
                self._requested_source_coordinates = rsc
                self._requested_source_coordinates_index = rsci
                self._requested_source_boundary = None
                self._requested_source_data = None
                self._output = output

            return output

        # Use the selector
        if _selector is not None:
            (rsc, rsci) = _selector(rsc, rsci, coordinates)

        # Check the coordinate_index_type
        if self.coordinate_index_type == "slice":  # Most restrictive
            new_rsci = []
            for index in rsci:
                if isinstance(index, slice):
                    new_rsci.append(index)
                    continue

                if len(index) > 1:
                    mx, mn = np.max(index), np.min(index)
                    df = np.diff(index)
                    if np.all(df == df[0]):
                        step = df[0]
                    else:
                        step = 1
                    new_rsci.append(slice(mn, mx + 1, step))
                else:
                    new_rsci.append(slice(np.max(index), np.max(index) + 1))

            rsci = tuple(new_rsci)

        # get data from data source
        rsd = self._get_data(rsc, rsci)

        # if not provided, create output using the evaluated coordinates, or
        # if provided, set the order of coordinates to match the output dims
        if output is None:
            output = rsd.part_transpose(coordinates.dims)
        else:
            output.data[:] = rsd.part_transpose(coordinates.dims).data

        # get indexed boundary
        rsb = self._get_boundary(rsci)
        output.attrs["boundary_data"] = rsb

        # save output to private for debugging
        if settings["DEBUG"]:
            self._evaluated_coordinates = deepcopy(coordinates)
            self._requested_source_coordinates = rsc
            self._requested_source_coordinates_index = rsci
            self._requested_source_boundary = rsb
            self._requested_source_data = rsd
            self._output = output

        return output

    def find_coordinates(self):
        """
        Get the available coordinates for the Node. For a DataSource, this is just the coordinates.

        Returns
        -------
        coords_list : list
            singleton list containing the coordinates (Coordinates object)
        """

        return [self.coordinates]

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}

        Raises
        ------
        NotImplementedError
            This needs to be implemented by derived classes
        """
        raise NotImplementedError

    @common_doc(COMMON_DATA_DOC)
    def get_coordinates(self):
        """{get_coordinates}

        Raises
        ------
        NotImplementedError
            This needs to be implemented by derived classes
        """
        raise NotImplementedError

    def set_coordinates(self, coordinates, force=False):
        """Set the coordinates. Used by Compositors as an optimization.

        Arguments
        ---------
        coordinates : :class:`podpac.Coordinates`
            Coordinates to set. Usually these are coordinates that are shared across compositor sources.

        NOTE: This is only currently used by SMAPCompositor. It should potentially be moved to the SMAPSource.
        """

        if force or not self.trait_is_defined("_coordinates"):
            self.set_trait("_coordinates", coordinates)

    def _get_boundary(self, index):
        """
        Select the boundary for the given the coordinates index. Only non-uniform boundary arrays need to be indexed.

        Arguments
        ---------
        index : tuple
            Coordinates index (e.g. coordinates_index)

        Returns
        -------
        boundary : dict
            Indexed boundary. Uniform boundaries are unchanged and non-uniform boundary arrays are indexed.
        """

        boundary = {}
        for c, I in zip(self.coordinates.values(), index):
            for dim in c.dims:
                if dim not in self.boundary:
                    pass
                elif np.array(self.boundary[dim]).ndim == 2:
                    boundary[dim] = np.array(self.boundary[dim][I])
                else:
                    boundary[dim] = self.boundary[dim]
        return boundary
