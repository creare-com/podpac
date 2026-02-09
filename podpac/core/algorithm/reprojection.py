"""
Reprojection Algorithm Node
"""

from __future__ import division, unicode_literals, print_function, absolute_import

from six import string_types

import numpy as np
import xarray as xr
import traitlets as tl

# Internal dependencies
from podpac.core.node import Node
from podpac.core.coordinates.coordinates import Coordinates, merge_dims
from podpac.core.interpolation.interpolation import Interpolate
from podpac.core.utils import NodeTrait, cached_property
from podpac import settings


class Reproject(Interpolate):
    """
    Create a Algorithm that evalutes a Node with one set of coordinates, and then interpolates it.
    This can be used to bilinearly interpolate an averaged dataset, for example.

    Attributes
    ----------
    source : Node
        The source node. This node will use it's own, specified interpolation scheme
    interpolation : str
        Type of interpolation method to use for the interpolation
    coordinates : Coordinates, Node, str, dict
        Coordinates used to evaluate the source. These can be specified as a dictionary, json-formatted string,
        PODPAC Coordinates, or a PODPAC Node, where the node MUST implement the 'coordinates' attribute.
    reproject_dims : list
        Dimensions to reproject. The source will be evaluated with the reprojection coordinates in these dims
        and the requested coordinates for any other dims.
    """

    coordinates = tl.Union(
        [NodeTrait(), tl.Dict(), tl.Unicode(), tl.Instance(Coordinates)],
        help="""Coordinates used to evaluate the source. These can be specified as a dictionary,
                           json-formatted string, PODPAC Coordinates, or a PODPAC Node, where the node MUST implement
                           the 'coordinates' attribute""",
    ).tag(attr=True)

    reproject_dims = tl.List(trait=tl.Unicode(), allow_none=True, default_value=None).tag(attr=True)

    @tl.validate("coordinates")
    def _validate_coordinates(self, d):
        val = d["value"]
        if isinstance(val, Node):
            if not hasattr(val, "coordinates"):
                raise ValueError(
                    "When specifying the coordinates as a PODPAC Node, this Node must have a 'coordinates' attribute"
                )
        elif isinstance(val, dict):
            Coordinates.from_definition(self.coordinates)
        elif isinstance(val, string_types):
            Coordinates.from_json(self.coordinates)
        return val

    @cached_property
    def reprojection_coordinates(self):
        # get coordinates
        if isinstance(self.coordinates, Coordinates):
            coordinates = self.coordinates
        elif isinstance(self.coordinates, Node):
            coordinates = self.coordinates.coordinates
        elif isinstance(self.coordinates, dict):
            coordinates = Coordinates.from_definition(self.coordinates)
        elif isinstance(self.coordinates, string_types):
            coordinates = Coordinates.from_json(self.coordinates)

        # drop non-reprojection dims
        if self.reproject_dims is not None:
            coordinates = coordinates.drop([dim for dim in coordinates if dim not in self.reproject_dims])

        return coordinates

    def _source_eval(self, coordinates, selector, output=None):
        coords = self.reprojection_coordinates.intersect(coordinates, outer=True)
        extra_eval_coords = coordinates.drop(self.reproject_dims or self.reprojection_coordinates.dims)
        if coords.crs != coordinates.crs:
            # Better to evaluate in reproject coordinate crs than eval crs for next step of interpolation
            extra_eval_coords = extra_eval_coords.transform(coords.crs)
        coords = merge_dims([coords, extra_eval_coords])
        if settings["MULTITHREADING"]:
            # we have to do a new node here to avoid clashing with the source node.
            # What happens is that the non-projected source gets evaluated
            # at the projected source coordinates because we have to set
            # self._requested_coordinates for the datasource to avoid floating point
            # lat/lon disagreement issues
            return Node.from_definition(self.source.definition).eval(coords, output=output, _selector=selector)
        else:
            return self.source.eval(coords, output=output, _selector=selector)

    @tl.default('base_ref')
    def _default_base_ref(self):
        return "{}_reprojected".format(self.source.base_ref)
