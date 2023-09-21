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
from podpac.core.coordinates.utils import VALID_DIMENSION_NAMES
from podpac.core.utils import common_doc
from podpac.core.coordinates.utils import get_timedelta


@common_doc(COMMON_INTERPOLATOR_DOCS)
class XarrayInterpolator(Interpolator):
    """Xarray interpolation Interpolation

    Attributes
    ----------
    {interpolator_attributes}

    fill_nan: bool
        Default is False. If True, nan values will be filled before interpolation.
    fill_value: float,str
        Default is None. The value that will be used to fill nan values. This can be a number, or "extrapolate", see `scipy.interpn`/`scipy/interp1d`
    kwargs: dict
        Default is {{"bounds_error": False}}. Additional values to pass to xarray's `interp` method.

    """

    dims_supported = VALID_DIMENSION_NAMES
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
        if self._dim_in(udims_subset, source_coordinates, unstacked=True):
            for d in source_coordinates.udims:  # Cannot handle stacked dimensions
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
        coords = {}
        nn_coords = {}

        for d in udims:
            # Note: This interpolator cannot handle stacked source -- and this is handled in the can_interpolate function
            if source_coordinates[d].size == 1:
                # If the source only has a single coordinate, xarray will automatically throw an error asking for at least 2 coordinates
                # So, we prevent this. Main problem is that this won't respect any tolerances.
                new_dim = [dd for dd in eval_coordinates.dims if d in dd][0]
                nn_coords[d] = xr.DataArray(
                    eval_coordinates[d].coordinates,
                    dims=[new_dim],
                    coords=[eval_coordinates.xcoords[new_dim]],
                )
                continue
            if (
                not source_coordinates.is_stacked(d)
                and eval_coordinates.is_stacked(d)
                and len(eval_coordinates[d].shape) == 1
            ):
                # Handle case for stacked coordinates (i.e. along a curve)
                new_dim = [dd for dd in eval_coordinates.dims if d in dd][0]
                coords[d] = xr.DataArray(
                    eval_coordinates[d].coordinates, dims=[new_dim], coords=[eval_coordinates.xcoords[new_dim]]
                )
            elif (
                not source_coordinates.is_stacked(d)
                and eval_coordinates.is_stacked(d)
                and len(eval_coordinates[d].shape) > 1
            ):
                # Dependent coordinates (i.e. a warped coordinate system)
                keep_coords = {k: v for k, v in eval_coordinates.xcoords.items() if k in eval_coordinates.xcoords[d][0]}
                coords[d] = xr.DataArray(
                    eval_coordinates[d].coordinates, dims=eval_coordinates.xcoords[d][0], coords=keep_coords
                )
            else:
                # TODO: Check dependent coordinates
                coords[d] = eval_coordinates[d].coordinates

        kwargs = self.kwargs.copy()
        kwargs.update({"fill_value": self.fill_value})

        coords["kwargs"] = kwargs

        if self.method == "bilinear":
            self.method = "linear"

        if self.fill_nan:
            for d in source_coordinates.dims:
                if not np.any(np.isnan(source_data)):
                    break
                # use_coordinate=False allows for interpolation when dimension is not monotonically increasing
                source_data = source_data.interpolate_na(method=self.method, dim=d, use_coordinate=False)

        if nn_coords:
            source_data = source_data.sel(method="nearest", **nn_coords)

        output_data = source_data.interp(method=self.method, **coords)

        return output_data.transpose(*eval_coordinates.xdims)
