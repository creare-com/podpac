"""
Interpolator implementations
"""

from __future__ import division, unicode_literals, print_function, absolute_import
from six import string_types

import numpy as np
import traitlets as tl

# Optional dependencies
try:
    import rasterio
    from rasterio import transform
    from rasterio.warp import reproject, Resampling
except:
    rasterio = None

# podac imports
from podpac.core.interpolation.interpolator import COMMON_INTERPOLATOR_DOCS, Interpolator, InterpolatorException
from podpac.core.units import UnitsDataArray
from podpac.core.coordinates import Coordinates, UniformCoordinates1d, StackedCoordinates
from podpac.core.utils import common_doc
from podpac.core.coordinates.utils import get_timedelta


@common_doc(COMMON_INTERPOLATOR_DOCS)
class RasterioInterpolator(Interpolator):
    """Rasterio Interpolation

    Attributes
    ----------
    {interpolator_attributes}
    rasterio_interpolators : list of str
        Interpolator methods available via rasterio
    """

    methods_supported = [
        "nearest",
        "bilinear",
        "cubic",
        "cubic_spline",
        "lanczos",
        "average",
        "mode",
        "gauss",
        "max",
        "min",
        "med",
        "q1",
        "q3",
    ]
    method = tl.Unicode(default_value="nearest")

    dims_supported = ["lat", "lon"]

    # TODO: implement these parameters for the method 'nearest'
    spatial_tolerance = tl.Float(default_value=np.inf)
    time_tolerance = tl.Union([tl.Unicode(), tl.Instance(np.timedelta64, allow_none=True)])

    # TODO: support 'gauss' method?

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def can_interpolate(self, udims, source_coordinates, eval_coordinates):
        """{interpolator_can_interpolate}"""

        # TODO: make this so we don't need to specify lat and lon together
        # or at least throw a warning
        if (
            "lat" in udims
            and "lon" in udims
            and self._dim_in(["lat", "lon"], source_coordinates, eval_coordinates)
            and source_coordinates["lat"].is_uniform
            and source_coordinates["lon"].is_uniform
            and eval_coordinates["lat"].is_uniform
            and eval_coordinates["lon"].is_uniform
        ):

            return udims

        # otherwise return no supported dims
        return tuple()

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def interpolate(self, udims, source_coordinates, source_data, eval_coordinates, output_data):
        """
        {interpolator_interpolate}
        """

        # TODO: handle when udims does not contain both lat and lon
        # if the source data has more dims than just lat/lon is asked, loop over those dims and run the interpolation
        # on those grids
        if len(source_data.dims) > 2:
            keep_dims = ["lat", "lon"]
            return self._loop_helper(
                self.interpolate, keep_dims, udims, source_coordinates, source_data, eval_coordinates, output_data
            )

        with rasterio.Env():
            src_transform = transform.Affine.from_gdal(*source_coordinates.geotransform)
            src_crs = rasterio.crs.CRS.from_proj4(source_coordinates.crs)
            # Need to make sure array is c-contiguous
            source = np.ascontiguousarray(source_data.data)

            dst_transform = transform.Affine.from_gdal(*eval_coordinates.geotransform)
            dst_crs = rasterio.crs.CRS.from_proj4(eval_coordinates.crs)
            # Need to make sure array is c-contiguous
            if not output_data.data.flags["C_CONTIGUOUS"]:
                destination = np.ascontiguousarray(output_data.data)
            else:
                destination = output_data.data

            reproject(
                source,
                np.atleast_2d(destination.squeeze()),  # Needed for legacy compatibility
                src_transform=src_transform,
                src_crs=src_crs,
                src_nodata=np.nan,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=getattr(Resampling, self.method),
            )
            output_data.data[:] = destination

        return output_data
