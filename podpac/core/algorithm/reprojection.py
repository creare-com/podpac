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
from podpac.core.coordinates.coordinates import Coordinates
from podpac.core.interpolation.interpolation import Interpolate
from podpac.core.utils import NodeTrait


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
    coordinates: Coordinates, Node, str, dict
        Coordinates used to evaluate the source. These can be specified as a dictionary, json-formatted string,
        PODPAC Coordinates, or a PODPAC Node, where the node MUST implement the 'coordinates' attribute.
    """

    coordinates = tl.Union(
        [NodeTrait(), tl.Dict(), tl.Unicode(), tl.Instance(Coordinates)],
        help="""Coordinates used to evaluate the source. These can be specified as a dictionary,
                           json-formatted string, PODPAC Coordinates, or a PODPAC Node, where the node MUST implement
                           the 'coordinates' attribute""",
    ).tag(attr=True)

    @tl.validate("coordinates")
    def _validate_coordinates(self, d):
        if isinstance(d["value"], Node) and not hasattr(d["value"], "coordinates"):
            raise ValueError(
                "When specifying the coordinates as a PODPAC Node, this Node must have a 'coordinates' attribute"
            )
        return d["value"]

    @property
    def _coordinates(self):
        if isinstance(self.coordinates, Coordinates):
            return self.coordinates
        elif isinstance(self.coordinates, Node):
            return self.coordinates.coordinates
        elif isinstance(self.coordinates, dict):
            return Coordinates.from_definition(self.coordinates)
        elif isinstance(self.coordinates, string_types):
            return Coordinates.from_json(self.coordinates)
        else:
            raise TypeError("The coordinates attribute is of the wrong type.")

    def _source_eval(self, coordinates, selector, output=None):
        return self.source.eval(self._coordinates, output=output, _selector=selector)

    @property
    def base_ref(self):
        return "{}_reprojected".format(self.source.base_ref)
