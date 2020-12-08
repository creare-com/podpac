from __future__ import division, unicode_literals, print_function, absolute_import

import logging
import copy

from six import string_types
import traitlets as tl

from podpac.core.utils import common_doc, NodeTrait, cached_property
from podpac.core.coordinates import Coordinates
from podpac.core.node import Node
from podpac.core.interpolation.interpolation import InterpolationTrait
from podpac.core.data.datasource import COMMON_DATA_DOC, DataSource

_logger = logging.getLogger(__name__)

# TODO: Move this to algorithm Nodes based on the Interpolation Refactor -- should be much more streamlined now.
class ReprojectedSource(DataSource):
    """Create a DataSource with a different resolution from another Node. This can be used to bilinearly interpolated a
    dataset after averaging over a larger area.

    Attributes
    ----------
    source : Node
        The source node
    reprojected_coordinates : :class:`podpac.Coordinates`
        Coordinates where the source node should be evaluated.
    """

    source = NodeTrait().tag(attr=True)
    reprojected_coordinates = tl.Instance(Coordinates).tag(attr=True)

    # list of attribute names, used by __repr__ and __str__ to display minimal info about the node
    _repr_keys = ["source"]

    def _first_init(self, **kwargs):
        if "reprojected_coordinates" in kwargs:
            if isinstance(kwargs["reprojected_coordinates"], dict):
                kwargs["reprojected_coordinates"] = Coordinates.from_definition(kwargs["reprojected_coordinates"])
            elif isinstance(kwargs["reprojected_coordinates"], string_types):
                kwargs["reprojected_coordinates"] = Coordinates.from_json(kwargs["reprojected_coordinates"])

        return super(ReprojectedSource, self)._first_init(**kwargs)

    @common_doc(COMMON_DATA_DOC)
    def get_coordinates(self):
        """{get_coordinates}"""

        # cannot guarantee that coordinates exist
        if not isinstance(self.source, DataSource):
            return self.reprojected_coordinates

        sc = self.source.coordinates
        rc = self.reprojected_coordinates
        return Coordinates(
            [rc[dim] if dim in rc.dims else self.source.coordinates[dim] for dim in self.source.coordinates.dims],
            validate_crs=False,
        )

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}"""

        data = self.source.eval(coordinates)

        # The following is needed in case the source is an algorithm
        # or compositor node that doesn't have all the dimensions of
        # the reprojected coordinates
        # TODO: What if data has coordinates that reprojected_coordinates doesn't have
        keep_dims = list(data.coords.keys())
        drop_dims = [d for d in coordinates.dims if d not in keep_dims]
        coordinates.drop(drop_dims)
        return data

    @property
    def base_ref(self):
        return "{}_reprojected".format(self.source.base_ref)
