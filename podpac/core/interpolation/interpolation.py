from __future__ import division, unicode_literals, print_function, absolute_import

import traitlets as tl
from copy import deepcopy
from collections import OrderedDict
from six import string_types
import logging

import traitlets as tl
import numpy as np

from podpac.core.node import Node, node_eval
from podpac.core.utils import NodeTrait, common_doc
from podpac.core.units import UnitsDataArray
from podpac.core.coordinates import merge_dims, Coordinates
from podpac.core.interpolation.interpolation_manager import InterpolationManager, InterpolationTrait

_logger = logging.getLogger(__name__)


def interpolation_decorator():
    pass  ## TODO


class InterpolationMixin(object):
    source_class = None
    harmonization_class = None
    source_node = None
    harmonization_node = None

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        _logger.debug("Get {}".format(item))
        # pass get calls to THIS class
        if item in ["interpolation", "source"]:
            return super().__getattribute__(item)

        _logger.debug("get source")
        source = super().__getattribute__("source")
        _logger.debug("got source")
        if source.has_trait(item):
            return getattr(source, item)
        _logger.debug("Check source hasattr")
        if hasattr(source, item):
            return getattr(source, item)
        _logger.debug("Get interp")
        interp = super().__getattribute__("interpolation")
        _logger.debug("got interp")
        if interp.has_trait(item):
            return getattr(interp, item)
        raise AttributeError()


class Interpolation(Node):
    """Base node for any data obtained directly from a single source.
    
    Parameters
    ----------
    source : Any
        The source node which will be interpolated
    interpolation : str, dict, optional
        {interpolation_long}
    cache_output : bool
        Should the node's output be cached? If not provided or None, uses default based on 
        settings["CACHE_DATASOURCE_OUTPUT_DEFAULT"]. If True, outputs will be cached and retrieved from cache. If False,
        outputs will not be cached OR retrieved from cache (even if they exist in cache). 
    
    Notes
    -----
    Custom DataSource Nodes must implement the :meth:`get_data` and :meth:`get_coordinates` methods.
    """

    source = NodeTrait()

    interpolation = InterpolationTrait().tag(attr=True)
    cache_output = tl.Bool()

    # privates
    _interpolation = tl.Instance(InterpolationManager)
    _coordinates = tl.Instance(Coordinates, allow_none=True, default_value=None, read_only=True)

    _requested_source_coordinates = tl.Instance(Coordinates)
    _requested_source_coordinates_index = tl.Tuple()
    _requested_source_data = tl.Instance(UnitsDataArray)
    _evaluated_coordinates = tl.Instance(Coordinates)

    # this adds a more helpful error message if user happens to try an inspect _interpolation before evaluate
    @tl.default("_interpolation")
    def _default_interpolation(self):
        self._set_interpolation()
        return self._interpolation

    @tl.default("cache_output")
    def _cache_output_default(self):
        return podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"]

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def interpolation_class(self):
        """Get the interpolation class currently set for this data source.
        
        The DataSource ``interpolation`` property is used to define the
        :class:`podpac.data.InterpolationManager` class that will handle interpolation for requested coordinates.
        
        Returns
        -------
        :class:`podpac.data.InterpolationManager`
            InterpolationManager class defined by DataSource `interpolation` definition
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

    def _set_interpolation(self):
        """Update _interpolation property
        """

        # define interpolator with source coordinates dimensions
        if isinstance(self.interpolation, InterpolationManager):
            self._interpolation = self.interpolation
        else:
            self._interpolation = InterpolationManager(self.interpolation)

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

        # store input coordinates to evaluated coordinates
        self._evaluated_coordinates = deepcopy(coordinates)

        # transform coordinates into native crs if different
        if self.coordinates.crs.lower() != coordinates.crs.lower():
            coordinates = coordinates.transform(self.coordinates.crs)

        # intersect the coordinates with requested coordinates to get coordinates within requested coordinates bounds
        # TODO: support coordinate_index_type parameter to define other index types
        (rsc, rsci) = self.coordinates.intersect(coordinates, outer=True, return_indices=True)
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
        self._requested_source_boundary = self._get_boundary(self._requested_source_coordinates_index)

        # interpolate data into output
        output = self._interpolation.interpolate(
            self._requested_source_coordinates, self._requested_source_data, coordinates, output
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

        return self.source.find_coordinates()
