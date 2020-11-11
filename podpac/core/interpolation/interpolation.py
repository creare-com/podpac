from __future__ import division, unicode_literals, print_function, absolute_import

import traitlets as tl
from copy import deepcopy
from collections import OrderedDict
from six import string_types
import logging

import traitlets as tl
import numpy as np

from podpac.core.settings import settings
from podpac.core.node import Node
from podpac.core.utils import NodeTrait, common_doc
from podpac.core.units import UnitsDataArray
from podpac.core.coordinates import merge_dims, Coordinates
from podpac.core.interpolation.interpolation_manager import InterpolationManager, InterpolationTrait

_logger = logging.getLogger(__name__)


def interpolation_decorator():
    pass  ## TODO


class InterpolationMixin(tl.HasTraits):
    interpolation = InterpolationTrait().tag(attr=True)
    _interp_node = None

    def _eval(self, coordinates, output=None, _selector=None):
        node = Interpolate(interpolation=self.interpolation)
        node._set_interpolation()
        node._source_xr = super()._eval(coordinates, _selector=node._interpolation.select_coordinates)
        self._interp_node = node
        return node.eval(coordinates, output=output)


class Interpolate(Node):
    """Node to used to interpolate from self.source.coordinates to the user-specified, evaluated coordinates.

    Parameters
    ----------
    source : Any
        The source node which will be interpolated
    interpolation : str, dict, optional
        Interpolation definition for the data source.
        By default, the interpolation method is set to ``'nearest'`` for all dimensions.

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
    cache_output : bool
        Should the node's output be cached? If not provided or None, uses default based on
        settings["CACHE_DATASOURCE_OUTPUT_DEFAULT"]. If True, outputs will be cached and retrieved from cache. If False,
        outputs will not be cached OR retrieved from cache (even if they exist in cache).

    Examples
    -----
    # To use bilinear interpolation for [lat,lon]  a specific interpolator for [time], and the default for [alt], use:
    >>> interp_node = Interpolation(
            source=some_node,
            interpolation=interpolation = [
                {
                'method': 'bilinear',
                'dims': ['lat', 'lon']
                },
                {
                'method': [podpac.interpolators.NearestNeighbor],
                'dims': ['time']
                }
            ]
        )

    """

    source = NodeTrait(allow_none=True).tag(attr=True)
    _source_xr = tl.Instance(UnitsDataArray, allow_none=True)  # This is needed for the Interpolation Mixin

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
        """Update _interpolation property"""

        # define interpolator with source coordinates dimensions
        if isinstance(self.interpolation, InterpolationManager):
            self._interpolation = self.interpolation
        else:
            self._interpolation = InterpolationManager(self.interpolation)

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
        _selector :
            {eval_selector}

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

        selector = self._interpolation.select_coordinates

        source_out = self._source_eval(self._evaluated_coordinates, selector)
        source_coords = Coordinates.from_xarray(source_out.coords, crs=source_out.crs)

        # Drop extra coordinates
        extra_dims = [d for d in coordinates.udims if d not in source_coords.udims]
        coordinates = coordinates.drop(extra_dims)

        # Transform so that interpolation happens on the source data coordinate system
        if source_coords.crs.lower() != coordinates.crs.lower():
            coordinates = coordinates.transform(source_coords.crs)

        if output is None:
            if "output" in source_out.dims:
                self.set_trait("outputs", source_out.coords["output"].data.tolist())
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
            self._source_xr = source_out

        return output

    def _source_eval(self, coordinates, selector, output=None):
        if isinstance(self._source_xr, UnitsDataArray):
            return self._source_xr
        else:
            return self.source.eval(coordinates, output=output, _selector=selector)

    def find_coordinates(self):
        """
        Get the available coordinates for the Node. For a DataSource, this is just the coordinates.

        Returns
        -------
        coords_list : list
            singleton list containing the coordinates (Coordinates object)
        """

        return self.source.find_coordinates()
