from __future__ import division, unicode_literals, print_function, absolute_import

from six import string_types

import traitlets as tl

from podpac.core.utils import common_doc, NodeTrait
from podpac.core.coordinates import Coordinates
from podpac.core.node import Node
from podpac.core.data.datasource import COMMON_DATA_DOC, DataSource
from podpac.core.data.interpolation import interpolation_trait


class ReprojectedSource(DataSource):
    """Create a DataSource with a different resolution from another Node. This can be used to bilinearly interpolated a
    dataset after averaging over a larger area.
    
    Attributes
    ----------
    source : Node
        The source node
    source_interpolation : str
        Type of interpolation method to use for the source node
    reprojected_coordinates : Coordinates
        Coordinates where the source node should be evaluated. 
    """

    source = NodeTrait().tag(readonly=True)

    # node attrs
    source_interpolation = interpolation_trait().tag(attr=True)
    reprojected_coordinates = tl.Instance(Coordinates).tag(attr=True)

    def _first_init(self, **kwargs):
        if "reprojected_coordinates" in kwargs:
            if isinstance(kwargs["reprojected_coordinates"], dict):
                kwargs["reprojected_coordinates"] = Coordinates.from_definition(kwargs["reprojected_coordinates"])
            elif isinstance(kwargs["reprojected_coordinates"], string_types):
                kwargs["reprojected_coordinates"] = Coordinates.from_json(kwargs["reprojected_coordinates"])

        return super(ReprojectedSource, self)._first_init(**kwargs)

    @common_doc(COMMON_DATA_DOC)
    def get_native_coordinates(self):
        """{get_native_coordinates}
        """
        if isinstance(self.source, DataSource):
            sc = self.source.native_coordinates
        else:  # Otherwise we cannot guarantee that native_coordinates exist
            sc = self.reprojected_coordinates
        rc = self.reprojected_coordinates
        coords = [rc[dim] if dim in rc.dims else sc[dim] for dim in sc.dims]
        return Coordinates(coords)

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        if hasattr(self.source, "interpolation") and self.source_interpolation is not None:
            si = self.source.interpolation
            self.source.interpolation = self.source_interpolation
        elif self.source_interpolation is not None:
            _logger.warning(
                "ReprojectedSource cannot set the 'source_interpolation'"
                " since self.source does not have an 'interpolation' "
                " attribute. \n type(self.source): %s\nself.source: " % (str(type(self.source)), str(self.source))
            )
        data = self.source.eval(coordinates)
        if hasattr(self.source, "interpolation") and self.source_interpolation is not None:
            self.source.interpolation = si
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
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        return "{}_reprojected".format(self.source.base_ref)
