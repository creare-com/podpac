from __future__ import division, unicode_literals, print_function, absolute_import

import traitlets as tl
from copy import deepcopy
from collections import OrderedDict
from six import string_types
import logging

import traitlets as tl
import numpy as np

from podpac.core.settings import settings
from podpac.core.node import Node, node_eval
from podpac.core.utils import NodeTrait, common_doc
from podpac.core.units import UnitsDataArray
from podpac.core.coordinates import merge_dims, Coordinates
from podpac.core.interpolation.interpolation_manager import InterpolationManager, InterpolationTrait

_logger = logging.getLogger(__name__)


def interpolation_decorator():
    pass  ## TODO


class InterpolationMixin(tl.HasTraits):
    interpolation = InterpolationTrait().tag(attr=True)

    @node_eval
    def eval(self, coordinates, output=None):
        node = Interpolation(interpolation=self.interpolation)
        node._set_interpolation()
        coordinates.set_selector(node._interpolation.select_coordinates)
        source = super().eval(coordinates)
        node.set_trait("source", source)

        return node.eval(coordinates, output)


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

    source = tl.Union([tl.Instance(UnitsDataArray), NodeTrait()]).tag(attr=True)

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
        return settings["CACHE_NODE_OUTPUT_DEFAULT"]

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

        _logger.debug("Evaluating {} data source".format(self.__class__.__name__))

        # store requested coordinates for debugging
        if settings["DEBUG"]:
            self._original_requested_coordinates = coordinates

        # store input coordinates to evaluated coordinates
        self._evaluated_coordinates = deepcopy(coordinates)

        # reset interpolation
        self._set_interpolation()

        self._evaluated_coordinates.set_selector(self._interpolation.select_coordinates)

        source_out = self._source_eval(self._evaluated_coordinates)
        source_coords = Coordinates.from_xarray(source_out.coords)

        # Drop extra coordinates
        extra_dims = [d for d in coordinates.dims if d not in source_coords.dims]
        coordinates.drop(extra_dims)

        # Transform so that interpolation happens on the source data coordinate system
        if source_coords.crs.lower() != coordinates.crs.lower():
            coordinates = coordinates.transform(source_coords.crs)

        if output is None:
            output = self.create_output_array(coordinates)

        # interpolate data into output
        output = self._interpolation.interpolate(source_coords, source_out, coordinates, output)

        # if requested crs is differented than coordinates,
        # fabricate a new output with the original coordinates and new values
        if self._evaluated_coordinates.crs != coordinates.crs:
            output = self.create_output_array(self._evaluated_coordinates.drop(extra_dims), data=output[:].values)

        # save output to private for debugging
        if settings["DEBUG"]:
            self._output = output

        return output

    def _source_eval(self, coordinates, output=None):
        if isinstance(self.source, Node):
            return self.source.eval(coordinates, output)
        elif isinstance(self.source, UnitsDataArray):
            return self.source

    def find_coordinates(self):
        """
        Get the available coordinates for the Node. For a DataSource, this is just the coordinates.

        Returns
        -------
        coords_list : list
            singleton list containing the coordinates (Coordinates object)
        """

        return self.source.find_coordinates()
