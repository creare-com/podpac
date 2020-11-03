"""
Interpolator implementations
"""

from __future__ import division, unicode_literals, print_function, absolute_import
from six import string_types

import traitlets as tl
import numpy as np
import xarray as xr

# Optional dependencies


# podac imports
from podpac.core.interpolation.interpolator import COMMON_INTERPOLATOR_DOCS, Interpolator, InterpolatorException
from podpac.core.coordinates import Coordinates, UniformCoordinates1d, StackedCoordinates
from podpac.core.utils import common_doc
from podpac.core.coordinates.utils import get_timedelta


@common_doc(COMMON_INTERPOLATOR_DOCS)
class XarrayInterpolator(Interpolator):
    """Xarray interpolation Interpolation

    {nearest_neighbor_attributes}
    """

    dims_supported = ["lat", "lon", "alt", "time"]
    methods_supported = [
        "nearest",
        "linear",
        "bilinear",
        "quadratic",
        "cubic",
        "zero",
        "slinear",
        "next",
        "previous",
        "splinef2d",
    ]

    # defined at instantiation
    method = tl.Unicode(default_value="nearest")
    fill_value = tl.Union([tl.Unicode(), tl.Float()], default_value=None, allow_none=True)
    fill_nan = tl.Bool(False)

    kwargs = tl.Dict({"bounds_error": False})

    def __repr__(self):
        rep = super(XarrayInterpolator, self).__repr__()
        # rep += '\n\tspatial_tolerance: {}\n\ttime_tolerance: {}'.format(self.spatial_tolerance, self.time_tolerance)
        return rep

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def can_interpolate(self, udims, source_coordinates, eval_coordinates):
        """
        {interpolator_interpolate}
        """
        udims_subset = self._filter_udims_supported(udims)

        # confirm that udims are in both source and eval coordinates
        if self._dim_in(udims_subset, source_coordinates):
            for d in source_coordinates.dims:  # Cannot handle stacked dimensions
                if source_coordinates.is_stacked(d):
                    return tuple()
            return udims_subset
        else:
            return tuple()

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def interpolate(self, udims, source_coordinates, source_data, eval_coordinates, output_data):
        """
        {interpolator_interpolate}
        """
        indexers = []

        coords = {}
        used_dims = set()

        for d in source_coordinates.udims:
            if not source_coordinates.is_stacked(d) and eval_coordinates.is_stacked(d):
                new_dim = [dd for dd in eval_coordinates.dims if d in dd][0]
                coords[d] = xr.DataArray(
                    eval_coordinates[d].coordinates, dims=[new_dim], coords=[eval_coordinates[new_dim].coordinates]
                )

            elif source_coordinates.is_stacked(d) and not eval_coordinates.is_stacked(d):
                raise InterpolatorException("Xarray interpolator cannot handle multi-index (source is points).")
            else:
                # TODO: Check dependent coordinates
                coords[d] = eval_coordinates.coords[d]

        kwargs = self.kwargs.copy()
        kwargs.update({"fill_value": self.fill_value})

        coords["kwargs"] = kwargs

        if self.fill_nan:
            for d in source_coordinates.dims:
                if not np.any(np.isnan(source_data)):
                    break
                source_data = source_data.interpolate_na(method=self.method, dim=d)

        if self.method == "bilinear":
            self.method = "linear"
        output_data = source_data.interp(method=self.method, **coords)

        return output_data
