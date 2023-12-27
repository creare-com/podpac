"""
Interpolator implementations
"""

from __future__ import division, unicode_literals, print_function, absolute_import
from six import string_types

import numpy as np
import xarray as xr
import traitlets as tl
from scipy.spatial import cKDTree

# Optional dependencies


# podac imports
from podpac.core.interpolation.interpolator import COMMON_INTERPOLATOR_DOCS, Interpolator, InterpolatorException
from podpac.core.coordinates import Coordinates, UniformCoordinates1d, StackedCoordinates
from podpac.core.coordinates.utils import make_coord_delta, make_coord_value, VALID_DIMENSION_NAMES
from podpac.core.utils import common_doc
from podpac.core.coordinates.utils import get_timedelta
from podpac.core.interpolation.selector import Selector, _higher_precision_time_coords1d, _higher_precision_time_stack


@common_doc(COMMON_INTERPOLATOR_DOCS)
class NoneInterpolator(Interpolator):
    """None Interpolation"""

    dims_supported = VALID_DIMENSION_NAMES
    methods_supported = ["none"]
    method = tl.Unicode(default_value="none")

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def can_interpolate(self, udims, source_coordinates, eval_coordinates):
        """
        {interpolator_interpolate}
        """
        udims_subset = self._filter_udims_supported(udims)

        return udims_subset

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def interpolate(self, udims, source_coordinates, source_data, eval_coordinates, output_data):
        """
        {interpolator_interpolate}
        """
        # Note, some of the following code duplicates code in the Selector class.
        # This duplication is for the sake of optimization

        return source_data

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def can_select(self, udims, source_coordinates, eval_coordinates):
        """
        {interpolator_can_select}
        """
        if not (self.method == "none"):
            return tuple()

        udims_subset = self._filter_udims_supported(udims)
        return udims_subset

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def select_coordinates(self, udims, source_coordinates, eval_coordinates, index_type="numpy"):
        """
        {interpolator_select}
        """
        return source_coordinates.intersect(eval_coordinates, outer=False, return_index=True)
