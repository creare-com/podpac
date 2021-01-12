"""
Interpolator implementations
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np
import traitlets as tl

# Optional dependencies
try:
    import scipy
    from scipy.interpolate import griddata, RectBivariateSpline, RegularGridInterpolator
    from scipy.spatial import KDTree
except:
    scipy = None

# podac imports
from podpac.core.interpolation.interpolator import COMMON_INTERPOLATOR_DOCS, Interpolator, InterpolatorException
from podpac.core.units import UnitsDataArray
from podpac.core.coordinates import Coordinates, UniformCoordinates1d, StackedCoordinates
from podpac.core.utils import common_doc
from podpac.core.coordinates.utils import get_timedelta


@common_doc(COMMON_INTERPOLATOR_DOCS)
class ScipyPoint(Interpolator):
    """Scipy Point Interpolation

    Attributes
    ----------
    {interpolator_attributes}
    """

    methods_supported = ["nearest"]
    method = tl.Unicode(default_value="nearest")
    dims_supported = ["lat", "lon"]

    # TODO: implement these parameters for the method 'nearest'
    spatial_tolerance = tl.Float(default_value=np.inf)
    time_tolerance = tl.Union([tl.Unicode(), tl.Instance(np.timedelta64, allow_none=True)])

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def can_interpolate(self, udims, source_coordinates, eval_coordinates):
        """
        {interpolator_can_interpolate}
        """

        # TODO: make this so we don't need to specify lat and lon together
        # or at least throw a warning
        if (
            "lat" in udims
            and "lon" in udims
            and not self._dim_in(["lat", "lon"], source_coordinates)
            and self._dim_in(["lat", "lon"], source_coordinates, unstacked=True)
            and self._dim_in(["lat", "lon"], eval_coordinates, unstacked=True)
        ):

            return tuple(["lat", "lon"])

        # otherwise return no supported dims
        return tuple()

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def interpolate(self, udims, source_coordinates, source_data, eval_coordinates, output_data):
        """
        {interpolator_interpolate}
        """

        order = "lat_lon" if "lat_lon" in source_coordinates.dims else "lon_lat"

        # calculate tolerance
        if isinstance(eval_coordinates["lat"], UniformCoordinates1d):
            dlat = eval_coordinates["lat"].step
        else:
            dlat = (eval_coordinates["lat"].bounds[1] - eval_coordinates["lat"].bounds[0]) / (
                eval_coordinates["lat"].size - 1
            )

        if isinstance(eval_coordinates["lon"], UniformCoordinates1d):
            dlon = eval_coordinates["lon"].step
        else:
            dlon = (eval_coordinates["lon"].bounds[1] - eval_coordinates["lon"].bounds[0]) / (
                eval_coordinates["lon"].size - 1
            )

        tol = np.linalg.norm([dlat, dlon]) * 8

        if self._dim_in(["lat", "lon"], eval_coordinates):
            pts = np.stack([source_coordinates[dim].coordinates for dim in source_coordinates[order].dims], axis=1)
            if order == "lat_lon":
                pts = pts[:, ::-1]
            pts = KDTree(pts)
            lon, lat = np.meshgrid(eval_coordinates["lon"].coordinates, eval_coordinates["lat"].coordinates)
            dist, ind = pts.query(np.stack((lon.ravel(), lat.ravel()), axis=1), distance_upper_bound=tol)
            mask = ind == source_data[order].size
            ind[mask] = 0  # This is a hack to make the select on the next line work
            # (the masked values are set to NaN on the following line)
            vals = source_data[{order: ind}]
            vals[mask] = np.nan
            # make sure 'lat_lon' or 'lon_lat' is the first dimension
            dims = [dim for dim in source_data.dims if dim != order]
            vals = vals.transpose(order, *dims).data
            shape = vals.shape
            coords = [eval_coordinates["lat"].coordinates, eval_coordinates["lon"].coordinates]
            coords += [source_coordinates[d].coordinates for d in dims]
            vals = vals.reshape(eval_coordinates["lat"].size, eval_coordinates["lon"].size, *shape[1:])
            vals = UnitsDataArray(vals, coords=coords, dims=["lat", "lon"] + dims)
            # and transpose back to the destination order
            output_data.data[:] = vals.transpose(*output_data.dims).data[:]

            return output_data

        elif self._dim_in(["lat", "lon"], eval_coordinates, unstacked=True):
            dst_order = "lat_lon" if "lat_lon" in eval_coordinates.dims else "lon_lat"
            src_stacked = np.stack(
                [source_coordinates[dim].coordinates for dim in source_coordinates[order].dims], axis=1
            )
            new_stacked = np.stack(
                [eval_coordinates[dim].coordinates for dim in source_coordinates[order].dims], axis=1
            )
            pts = KDTree(src_stacked)
            dist, ind = pts.query(new_stacked, distance_upper_bound=tol)
            mask = ind == source_data[order].size
            ind[mask] = 0
            vals = source_data[{order: ind}]
            vals[{order: mask}] = np.nan
            dims = list(output_data.dims)
            dims[dims.index(dst_order)] = order
            output_data.data[:] = vals.transpose(*dims).data[:]

            return output_data


@common_doc(COMMON_INTERPOLATOR_DOCS)
class ScipyGrid(ScipyPoint):
    """Scipy Interpolation

    Attributes
    ----------
    {interpolator_attributes}
    """

    methods_supported = ["nearest", "bilinear", "cubic_spline", "spline_2", "spline_3", "spline_4"]
    method = tl.Unicode(default_value="nearest")

    # TODO: implement these parameters for the method 'nearest'
    spatial_tolerance = tl.Float(default_value=np.inf)
    time_tolerance = tl.Union([tl.Unicode(), tl.Instance(np.timedelta64, allow_none=True)], default_value=None)

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def can_interpolate(self, udims, source_coordinates, eval_coordinates):
        """
        {interpolator_can_interpolate}
        """

        # TODO: make this so we don't need to specify lat and lon together
        # or at least throw a warning
        if (
            "lat" in udims
            and "lon" in udims
            and self._dim_in(["lat", "lon"], source_coordinates)
            and self._dim_in(["lat", "lon"], eval_coordinates, unstacked=True)
        ):

            return ["lat", "lon"]

        # otherwise return no supported dims
        return tuple()

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def interpolate(self, udims, source_coordinates, source_data, eval_coordinates, output_data):
        """
        {interpolator_interpolate}
        """

        if self._dim_in(["lat", "lon"], eval_coordinates):
            return self._interpolate_irregular_grid(
                udims, source_coordinates, source_data, eval_coordinates, output_data, grid=True
            )

        elif self._dim_in(["lat", "lon"], eval_coordinates, unstacked=True):
            eval_coordinates_us = eval_coordinates.unstack()
            return self._interpolate_irregular_grid(
                udims, source_coordinates, source_data, eval_coordinates_us, output_data, grid=False
            )

    def _interpolate_irregular_grid(
        self, udims, source_coordinates, source_data, eval_coordinates, output_data, grid=True
    ):

        if len(source_data.dims) > 2:
            keep_dims = ["lat", "lon"]
            return self._loop_helper(
                self._interpolate_irregular_grid,
                keep_dims,
                udims,
                source_coordinates,
                source_data,
                eval_coordinates,
                output_data,
                grid=grid,
            )

        s = []
        if source_coordinates["lat"].is_descending:
            lat = source_coordinates["lat"].coordinates[::-1]
            s.append(slice(None, None, -1))
        else:
            lat = source_coordinates["lat"].coordinates
            s.append(slice(None, None))
        if source_coordinates["lon"].is_descending:
            lon = source_coordinates["lon"].coordinates[::-1]
            s.append(slice(None, None, -1))
        else:
            lon = source_coordinates["lon"].coordinates
            s.append(slice(None, None))

        data = source_data.data[tuple(s)]

        # remove nan's
        I, J = np.isfinite(lat), np.isfinite(lon)
        coords_i = lat[I], lon[J]
        coords_i_dst = [eval_coordinates["lon"].coordinates, eval_coordinates["lat"].coordinates]

        # Swap order in case datasource uses lon,lat ordering instead of lat,lon
        if source_coordinates.dims.index("lat") > source_coordinates.dims.index("lon"):
            I, J = J, I
            coords_i = coords_i[::-1]
            coords_i_dst = coords_i_dst[::-1]
        data = data[I, :][:, J]

        if self.method in ["bilinear", "nearest"]:
            f = RegularGridInterpolator(
                coords_i, data, method=self.method.replace("bi", ""), bounds_error=False, fill_value=np.nan
            )
            if grid:
                x, y = np.meshgrid(*coords_i_dst)
            else:
                x, y = coords_i_dst
            output_data.data[:] = f((y.ravel(), x.ravel())).reshape(output_data.shape)

        # TODO: what methods is 'spline' associated with?
        elif "spline" in self.method:
            if self.method == "cubic_spline":
                order = 3
            else:
                # TODO: make this a parameter
                order = int(self.method.split("_")[-1])

            f = RectBivariateSpline(coords_i[0], coords_i[1], data, kx=max(1, order), ky=max(1, order))
            output_data.data[:] = f(coords_i_dst[1], coords_i_dst[0], grid=grid).reshape(output_data.shape)

        return output_data
