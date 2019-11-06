"""
Interpolator implementations
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np
import traitlets as tl

# Optional dependencies
try:
    import rasterio
    from rasterio import transform
    from rasterio.warp import reproject, Resampling
except:
    rasterio = None
try:
    import scipy
    from scipy.interpolate import griddata, RectBivariateSpline, RegularGridInterpolator
    from scipy.spatial import KDTree
except:
    scipy = None

# podac imports
from podpac.core.data.interpolator import COMMON_INTERPOLATOR_DOCS, Interpolator, InterpolatorException
from podpac.core.units import UnitsDataArray
from podpac.core.coordinates import Coordinates, UniformCoordinates1d, StackedCoordinates
from podpac.core.utils import common_doc


@common_doc(COMMON_INTERPOLATOR_DOCS)
class NearestNeighbor(Interpolator):
    """Nearest Neighbor Interpolation
    
    {nearest_neighbor_attributes}
    """

    dims_supported = ["lat", "lon", "alt", "time"]
    methods_supported = ["nearest"]

    # defined at instantiation
    method = tl.Unicode(default_value="nearest")
    spatial_tolerance = tl.Float(default_value=np.inf, allow_none=True)
    time_tolerance = tl.Instance(np.timedelta64, allow_none=True)

    def __repr__(self):
        rep = super(NearestNeighbor, self).__repr__()
        # rep += '\n\tspatial_tolerance: {}\n\ttime_tolerance: {}'.format(self.spatial_tolerance, self.time_tolerance)
        return rep

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def can_interpolate(self, udims, source_coordinates, eval_coordinates):
        """
        {interpolator_interpolate}
        """
        udims_subset = self._filter_udims_supported(udims)

        # confirm that udims are in both source and eval coordinates
        # TODO: handle stacked coordinates
        if self._dim_in(udims_subset, source_coordinates, eval_coordinates):
            return udims_subset
        else:
            return tuple()

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def interpolate(self, udims, source_coordinates, source_data, eval_coordinates, output_data):
        """
        {interpolator_interpolate}
        """
        indexers = []

        # select dimensions common to eval_coordinates and udims
        # TODO: this is sort of convoluted implementation
        for dim in eval_coordinates.dims:

            # TODO: handle stacked coordinates
            if isinstance(eval_coordinates[dim], StackedCoordinates):

                # udims within stacked dims that are in the input udims
                udims_in_stack = list(set(udims) & set(eval_coordinates[dim].dims))

                # TODO: how do we choose a dimension to use from the stacked coordinates?
                # For now, choose the first coordinate found in the udims definition
                if udims_in_stack:
                    raise InterpolatorException("Nearest interpolation does not yet support stacked dimensions")
                    # dim = udims_in_stack[0]
                else:
                    continue

            # TODO: handle if the source coordinates contain `dim` within a stacked coordinate
            elif dim not in source_coordinates.dims:
                raise InterpolatorException("Nearest interpolation does not yet support stacked dimensions")

            elif dim not in udims:
                continue

            # set tolerance value based on dim type
            tolerance = None
            if dim == "time" and self.time_tolerance:
                tolerance = self.time_tolerance
            elif dim != "time":
                tolerance = self.spatial_tolerance

            # reindex using xarray
            indexer = {dim: eval_coordinates[dim].coordinates.copy()}
            indexers += [dim]
            source_data = source_data.reindex(method=str("nearest"), tolerance=tolerance, **indexer)

        # at this point, output_data and eval_coordinates have the same dim order
        # this transpose makes sure the source_data has the same dim order as the eval coordinates
        output_data.data = source_data.transpose(*eval_coordinates.dims)

        return output_data


@common_doc(COMMON_INTERPOLATOR_DOCS)
class NearestPreview(NearestNeighbor):
    """Nearest Neighbor (Preview) Interpolation
    
    {nearest_neighbor_attributes}
    """

    methods_supported = ["nearest_preview"]
    method = tl.Unicode(default_value="nearest_preview")
    spatial_tolerance = tl.Float(read_only=True, allow_none=True, default_value=None)

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def can_select(self, udims, source_coordinates, eval_coordinates):
        """
        {interpolator_can_select}
        """
        udims_subset = self._filter_udims_supported(udims)

        # confirm that udims are in both source and eval coordinates
        # TODO: handle stacked coordinates
        if self._dim_in(udims_subset, source_coordinates, eval_coordinates):
            return udims_subset
        else:
            return tuple()

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def select_coordinates(self, udims, source_coordinates, source_coordinates_index, eval_coordinates):
        """
        {interpolator_select}
        """
        new_coords = []
        new_coords_idx = []

        # iterate over the source coordinate dims in case they are stacked
        for src_dim, idx in zip(source_coordinates, source_coordinates_index):

            # TODO: handle stacked coordinates
            if isinstance(source_coordinates[src_dim], StackedCoordinates):
                raise InterpolatorException("NearestPreview select does not yet support stacked dimensions")

            if src_dim in eval_coordinates.dims:
                src_coords = source_coordinates[src_dim]
                dst_coords = eval_coordinates[src_dim]

                if isinstance(dst_coords, UniformCoordinates1d):
                    dst_start = dst_coords.start
                    dst_stop = dst_coords.stop
                    dst_delta = dst_coords.step
                else:
                    dst_start = dst_coords.coordinates[0]
                    dst_stop = dst_coords.coordinates[-1]
                    with np.errstate(invalid="ignore"):
                        dst_delta = (dst_stop - dst_start) / (dst_coords.size - 1)

                if isinstance(src_coords, UniformCoordinates1d):
                    src_start = src_coords.start
                    src_stop = src_coords.stop
                    src_delta = src_coords.step
                else:
                    src_start = src_coords.coordinates[0]
                    src_stop = src_coords.coordinates[-1]
                    with np.errstate(invalid="ignore"):
                        src_delta = (src_stop - src_start) / (src_coords.size - 1)

                ndelta = max(1, np.round(dst_delta / src_delta))
                if src_coords.size == 1:
                    c = src_coords.copy()
                else:
                    c = UniformCoordinates1d(src_start, src_stop, ndelta * src_delta, **src_coords.properties)

                if isinstance(idx, slice):
                    idx = slice(idx.start, idx.stop, int(ndelta))
                else:
                    idx = slice(idx[0], idx[-1], int(ndelta))
            else:
                c = source_coordinates[src_dim]

            new_coords.append(c)
            new_coords_idx.append(idx)

        return Coordinates(new_coords), tuple(new_coords_idx)


@common_doc(COMMON_INTERPOLATOR_DOCS)
class Rasterio(Interpolator):
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

    # TODO: implement these parameters for the method 'nearest'
    spatial_tolerance = tl.Float(default_value=np.inf)
    time_tolerance = tl.Instance(np.timedelta64, allow_none=True)

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

        def get_rasterio_transform(c):
            """Summary
            
            Parameters
            ----------
            c : TYPE
                Description
            
            Returns
            -------
            TYPE
                Description
            """
            west, east = c["lon"].area_bounds
            south, north = c["lat"].area_bounds
            cols, rows = (c["lon"].size, c["lat"].size)
            # print (east, west, south, north)
            return transform.from_bounds(west, south, east, north, cols, rows)

        with rasterio.Env():
            src_transform = get_rasterio_transform(source_coordinates)
            src_crs = {"init": source_coordinates.crs}
            # Need to make sure array is c-contiguous
            if source_coordinates["lat"].is_descending:
                source = np.ascontiguousarray(source_data.data)
            else:
                source = np.ascontiguousarray(source_data.data[::-1, :])

            dst_transform = get_rasterio_transform(eval_coordinates)
            dst_crs = {"init": eval_coordinates.crs}
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
            if eval_coordinates["lat"].is_descending:
                output_data.data[:] = destination
            else:
                output_data.data[:] = destination[::-1, :]

        return output_data


@common_doc(COMMON_INTERPOLATOR_DOCS)
class ScipyPoint(Interpolator):
    """Scipy Point Interpolation
    
    Attributes
    ----------
    {interpolator_attributes}
    """

    methods_supported = ["nearest"]
    method = tl.Unicode(default_value="nearest")

    # TODO: implement these parameters for the method 'nearest'
    spatial_tolerance = tl.Float(default_value=np.inf)
    time_tolerance = tl.Instance(np.timedelta64, allow_none=True)

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
            lon, lat = np.meshgrid(eval_coordinates.coords["lon"], eval_coordinates.coords["lat"])
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
    time_tolerance = tl.Instance(np.timedelta64, allow_none=True)

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

            return udims

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
