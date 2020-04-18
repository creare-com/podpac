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
from podpac.core.node import Node, NodeException
from podpac.core.utils import common_doc
from podpac.core.node import COMMON_NODE_DOC
from podpac.core.node import node_eval
from podpac.core.interpolation.interpolation import Interpolation, InterpolationTrait

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
    interpolation : str, dict, optional
        {interpolation_long}
    nan_vals : List, optional
        List of values from source data that should be interpreted as 'no data' or 'nans'
    coordinate_index_type : str, optional
        Type of index to use for data source. Possible values are ``['list', 'numpy', 'xarray', 'pandas']``
        Default is 'numpy'
    cache_coordinates : bool
        Whether to cache coordinates using the podpac ``cache_ctrl``. Default False.

    
    Notes
    -----
    Custom DataSource Nodes must implement the :meth:`get_data` and :meth:`get_coordinates` methods.
    """

    interpolation = InterpolationTrait().tag(attr=True)
    boundary = tl.Dict().tag(attr=True)
    nan_vals = tl.List().tag(attr=True)

    coordinate_index_type = tl.Enum(["slice", "list", "numpy"], default_value="numpy")  # , "xarray", "pandas"],
    cache_coordinates = tl.Bool(False)

    # privates
    _interpolation = tl.Instance(Interpolation)
    _coordinates = tl.Instance(Coordinates, allow_none=True, default_value=None, read_only=True)

    _original_requested_coordinates = tl.Instance(Coordinates, allow_none=True)
    _requested_source_coordinates = tl.Instance(Coordinates)
    _requested_source_coordinates_index = tl.Tuple()
    _requested_source_data = tl.Instance(UnitsDataArray)
    _evaluated_coordinates = tl.Instance(Coordinates)

    # this adds a more helpful error message if user happens to try an inspect _interpolation before evaluate
    @tl.default("_interpolation")
    def _default_interpolation(self):
        self._set_interpolation()
        return self._interpolation

    @tl.observe("_coordinates")
    def _set_default_boundary(self, d):
        if self.trait_is_defined("boundary"):
            return

        # uniform numeric coordinates have a uniform centered boundary by default
        boundary = {}
        for dim in self.coordinates.udims:
            c = self.coordinates[dim]
            if c.step and c.dtype == float:
                boundary[dim] = c.step / 2.0
        self.set_trait("boundary", boundary)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def interpolation_class(self):
        """Get the interpolation class currently set for this data source.
        
        The DataSource ``interpolation`` property is used to define the
        :class:`podpac.data.Interpolation` class that will handle interpolation for requested coordinates.
        
        Returns
        -------
        :class:`podpac.data.Interpolation`
            Interpolation class defined by DataSource `interpolation` definition
        """

        return self._interpolation

    @property
    def interpolators(self):
        """Return the interpolators selected for the previous node evaluation interpolation.
        If the node has not been evaluated, or if interpolation was not necessary, this will return
        an empty OrderedDict
        
        Returns
        -------
        OrderedDict
            Key are tuple of unstacked dimensions, the value is the interpolator used to interpolate these dimensions
        """

        if self._interpolation._last_interpolator_queue is not None:
            return self._interpolation._last_interpolator_queue
        else:
            return OrderedDict()

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

    def _set_interpolation(self):
        """Update _interpolation property
        """

        # define interpolator with source coordinates dimensions
        if isinstance(self.interpolation, Interpolation):
            self._interpolation = self.interpolation
        else:
            self._interpolation = Interpolation(self.interpolation)

    def _get_data(self):
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
        data = self.get_data(self._requested_source_coordinates, self._requested_source_coordinates_index)

        # convert data into UnitsDataArray depending on format
        # TODO: what other processing needs to happen here?
        if isinstance(data, UnitsDataArray):
            udata_array = data
        elif isinstance(data, xr.DataArray):
            # TODO: check order of coordinates here
            udata_array = self.create_output_array(self._requested_source_coordinates, data=data.data)
        elif isinstance(data, np.ndarray):
            udata_array = self.create_output_array(self._requested_source_coordinates, data=data)
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
    @node_eval
    def eval(self, coordinates, output=None):
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
        extra = []
        for c in coordinates.values():
            if isinstance(c, Coordinates1d):
                if c.name not in self.coordinates.udims:
                    extra.append(c.name)
            elif isinstance(c, StackedCoordinates):
                if all(dim not in self.coordinates.udims for dim in c.dims):
                    extra.append(c.name)
        coordinates = coordinates.drop(extra)

        # store input coordinates to evaluated coordinates
        self._evaluated_coordinates = deepcopy(coordinates)

        # transform coordinates into native crs if different
        if self.coordinates.crs.lower() != coordinates.crs.lower():
            coordinates = coordinates.transform(self.coordinates.crs)

        # intersect the coordinates with requested coordinates to get coordinates within requested coordinates bounds
        # TODO: support coordinate_index_type parameter to define other index types
        (rsc, rsci) = self.coordinates.intersect(coordinates, boundary=self.boundary, outer=True, return_indices=True)
        self._requested_source_coordinates = rsc
        self._requested_source_coordinates_index = rsci

        # if requested coordinates and coordinates do not intersect, shortcut with nan UnitsDataArary
        if self._requested_source_coordinates.size == 0:
            if output is None:
                output = self.create_output_array(self._evaluated_coordinates)
                if "output" in output.dims and self.output is not None:
                    output = output.sel(output=self.output)
            else:
                output[:] = np.nan
            return output

        # reset interpolation
        self._set_interpolation()

        # interpolate requested coordinates before getting data
        (rsc, rsci) = self._interpolation.select_coordinates(
            self._requested_source_coordinates, self._requested_source_coordinates_index, coordinates
        )
        self._requested_source_coordinates = rsc
        self._requested_source_coordinates_index = rsci

        # Check the coordinate_index_type
        if self.coordinate_index_type == "slice":  # Most restrictive
            new_rsci = []
            for rsci in self._requested_source_coordinates_index:
                if isinstance(rsci, slice):
                    new_rsci.append(rsci)
                    continue

                if len(rsci) > 1:
                    mx, mn = np.max(rsci), np.min(rsci)
                    df = np.diff(rsci)
                    if np.all(df == df[0]):
                        step = df[0]
                    else:
                        step = 1
                    new_rsci.append(slice(mn, mx + 1, step))
                else:
                    new_rsci.append(slice(np.max(rsci), np.max(rsci) + 1))

            self._requested_source_coordinates_index = tuple(new_rsci)

        # get data from data source
        self._requested_source_data = self._get_data()

        # if not provided, create output using the evaluated coordinates, or
        # if provided, set the order of coordinates to match the output dims
        # Note that at this point the coordinates are in the same CRS as the coordinates
        if output is None:
            requested_dims = None
            output_dims = None
            output = self.create_output_array(coordinates)
            if "output" in output.dims and self.output is not None:
                output = output.sel(output=self.output)
        else:
            requested_dims = self._evaluated_coordinates.dims
            output_dims = output.dims
            o = output
            if "output" in output.dims:
                requested_dims = requested_dims + ("output",)
            output = output.transpose(*requested_dims)

            # check crs compatibility
            if output.crs != self._evaluated_coordinates.crs:
                raise ValueError(
                    "Output coordinate reference system ({}) does not match".format(output.crs)
                    + "request Coordinates coordinate reference system ({})".format(coordinates.crs)
                )

        # get indexed boundary
        self._requested_source_boundary = self.get_boundary(self._requested_source_coordinates_index)

        # interpolate data into output
        output = self._interpolation.interpolate(
            self._requested_source_coordinates, self.boundary, self._requested_source_data, coordinates, output
        )

        # Fill the output that was passed to eval with the new data
        if requested_dims is not None and requested_dims != output_dims:
            o = o.transpose(*output_dims)
            o.data[:] = output.transpose(*output_dims).data

        # if requested crs is differented than coordinates,
        # fabricate a new output with the original coordinates and new values
        if self._evaluated_coordinates.crs != coordinates.crs:
            output = self.create_output_array(self._evaluated_coordinates, data=output[:].values)

        # save output to private for debugging
        if settings["DEBUG"]:
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
        """ Set the coordinates. Used by Compositors as an optimization.

        Arguments
        ---------
        coordinates : :class:`podpac.Coordinates`
            Coordinates to set. Usually these are coordinates that are shared across compositor sources.

        NOTE: This is only currently used by SMAPCompositor. It should potentially be moved to the SMAPSource.
        """

        if not self.trait_is_defined("_coordinates"):
            self.set_trait("_coordinates", coordinates)

    def get_boundary(self, index):
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
