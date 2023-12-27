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
class NearestNeighbor(Interpolator):
    """Nearest Neighbor Interpolation

    {nearest_neighbor_attributes}
    """

    dims_supported = VALID_DIMENSION_NAMES
    methods_supported = ["nearest"]

    # defined at instantiation
    method = tl.Unicode(default_value="nearest")
    ambiguous_rounding = tl.Enum(["-infinity", "+infinity", "unbiased"], default_value="-infinity")
    spatial_tolerance = tl.Float(default_value=np.inf, allow_none=True)
    time_tolerance = tl.Union([tl.Unicode(), tl.Instance(np.timedelta64, allow_none=True)])
    alt_tolerance = tl.Float(default_value=np.inf, allow_none=True)
    other_dim_tolerance = tl.Float(default_value=np.inf, allow_none=True)

    # spatial_scale only applies when the source is stacked with time or alt. The supplied value will be assigned a distance of "1'"
    spatial_scale = tl.Float(default_value=1, allow_none=True)
    # time_scale only applies when the source is stacked with lat, lon, or alt. The supplied value will be assigned a distance of "1'"
    time_scale = tl.Union([tl.Unicode(), tl.Instance(np.timedelta64, allow_none=True)])
    # alt_scale only applies when the source is stacked with lat, lon, or time. The supplied value will be assigned a distance of "1'"
    alt_scale = tl.Float(default_value=1, allow_none=True)
    # other dim scale
    other_dim_scale = tl.Float(default_value=1, allow_none=True)

    respect_bounds = tl.Bool(True)
    remove_nan = tl.Bool(False)
    use_selector = tl.Bool(True)

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

        return udims_subset

    def can_select(self, udims, source_coordinates, eval_coordinates):
        selector = super().can_select(udims, source_coordinates, eval_coordinates)
        if self.use_selector:
            return selector
        return ()

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def interpolate(self, udims, source_coordinates, source_data, eval_coordinates, output_data):
        """
        {interpolator_interpolate}
        """
        # Note, some of the following code duplicates code in the Selector class.
        # This duplication is for the sake of optimization

        def is_stacked(d):
            return "_" in d

        if hasattr(source_data, "attrs") and "bounds" in source_data.attrs:
            bounds = source_data.attrs["bounds"]
            if "time" in bounds and bounds["time"]:
                if "time" in eval_coordinates.udims:
                    bounds["time"] = [
                        self._atime_to_float(b, source_coordinates["time"], eval_coordinates["time"])
                        for b in bounds["time"]
                    ]
                else:
                    bounds["time"] = [
                        self._atime_to_float(b, source_coordinates["time"], source_coordinates["time"])
                        for b in bounds["time"]
                    ]

        else:
            bounds = None

        if self.remove_nan:
            # Eliminate nans from the source data. Note, this could turn a uniform griddted dataset into a stacked one
            source_data, source_coordinates = self._remove_nans(source_data, source_coordinates)

        data_index = []
        for d in source_coordinates.dims:
            # Make sure we're supposed to do nearest neighbor interpolation for this UDIM, otherwise skip this dimension
            if len([dd for dd in d.split("_") if dd in udims]) == 0:
                index = self._resize_unstacked_index(np.arange(source_coordinates[d].size), d, eval_coordinates)
                data_index.append(index)
                continue
            source = source_coordinates[d]
            if is_stacked(d):
                if bounds is not None:
                    bound = np.stack([bounds[dd] for dd in d.split("_")], axis=1)
                else:
                    bound = None
                index = self._get_stacked_index(d, source, eval_coordinates, bound)

                if len(source.shape) == 2:  # Handle case of 2D-stacked coordinates
                    ncols = source.shape[1]
                    index1 = index // ncols
                    index1 = self._resize_stacked_index(index1, d, eval_coordinates)
                    # With nD stacked coordinates, there are 'n' indices in the tuple
                    # All of these need to get into the data_index, and in the right order
                    data_index.append(index1)  # This is a hack
                    index = index % ncols  # The second half can go through the usual machinery
                elif len(source.shape) > 2:  # Handle case of nD-stacked coordinates
                    raise NotImplementedError
                index = self._resize_stacked_index(index, d, eval_coordinates)
            elif source_coordinates[d].is_uniform:
                request = eval_coordinates[d]
                if bounds is not None:
                    bound = bounds[d]
                else:
                    bound = None
                index = self._get_uniform_index(d, source, request, bound)
                index = self._resize_unstacked_index(index, d, eval_coordinates)
            else:  # non-uniform coordinates... probably an optimization here
                request = eval_coordinates[d]
                if bounds is not None:
                    bound = bounds[d]
                else:
                    bound = None
                index = self._get_nonuniform_index(d, source, request, bound)
                index = self._resize_unstacked_index(index, d, eval_coordinates)

            data_index.append(index)

        index = tuple(data_index)

        output_data.data[:] = np.array(source_data)[index]

        bool_inds = sum([i == -1 for i in index]).astype(bool)
        output_data.data[bool_inds] = np.nan

        return output_data

    def _remove_nans(self, source_data, source_coordinates):
        index = np.array(np.isnan(source_data), bool)
        if not np.any(index):
            return source_data, source_coordinates

        data = source_data.data[~index]
        coords = np.meshgrid(
            *[source_coordinates[d.split("_")[0]].coordinates for d in source_coordinates.dims], indexing="ij"
        )
        repeat_shape = coords[0].shape
        coords = [c[~index] for c in coords]

        final_dims = [d.split("_")[0] for d in source_coordinates.dims]
        # Add back in any stacked coordinates
        for i, d in enumerate(source_coordinates.dims):
            dims = d.split("_")
            if len(dims) == 1:
                continue
            reshape = np.ones(len(coords), int)
            reshape[i] = -1
            repeats = list(repeat_shape)
            repeats[i] = 1
            for dd in dims[1:]:
                crds = source_coordinates[dd].coordinates.reshape(*reshape)
                for j, r in enumerate(repeats):
                    crds = crds.repeat(r, axis=j)
                coords.append(crds[~index])
                final_dims.append(dd)

        return data, Coordinates([coords], dims=[final_dims])

    def _get_tol(self, dim, source, request):
        if dim in ["lat", "lon"]:
            return self.spatial_tolerance
        if dim == "alt":
            return self.alt_tolerance
        if dim == "time":
            if self.time_tolerance == "":
                return np.inf
            return self._time_to_float(self.time_tolerance, source, request)
        return self.other_dim_tolerance

    def _get_scale(self, dim, source_1d, request_1d):
        if dim in ["lat", "lon"]:
            return 1 / self.spatial_scale
        if dim == "alt":
            return 1 / self.alt_scale
        if dim == "time":
            if self.time_scale == "":
                return 1.0
            return 1 / self._time_to_float(self.time_scale, source_1d, request_1d)
        return self.other_dim_scale

    def _time_to_float(self, time, time_source, time_request):
        dtype0 = time_source.coordinates[0].dtype
        dtype1 = time_request.coordinates[0].dtype
        dtype = dtype0 if dtype0 > dtype1 else dtype1
        time = make_coord_delta(time)
        if isinstance(time, np.timedelta64):
            time1 = (time + np.datetime64("2000")).astype(dtype).astype(float) - (
                np.datetime64("2000").astype(dtype).astype(float)
            )
        return time1

    def _atime_to_float(self, time, time_source, time_request):
        dtype0 = time_source.coordinates[0].dtype
        dtype1 = time_request.coordinates[0].dtype
        dtype = dtype0 if dtype0 > dtype1 else dtype1
        time = make_coord_value(time)
        if isinstance(time, np.datetime64):
            time = time.astype(dtype).astype(float)
        return time

    def _get_stacked_index(self, dim, source, request, bounds=None):
        # The udims are in the order of the request so that the meshgrid calls will be in the right order
        udims = [ud for ud in request.udims if ud in source.udims]

        time_source = time_request = None
        if "time" in udims:
            time_source = source["time"]
            time_request = request["time"]

        tols = np.array([self._get_tol(d, time_source, time_request) for d in udims])[None, :]
        scales = np.array([self._get_scale(d, time_source, time_request) for d in udims])[None, :]
        tol = np.linalg.norm((tols * scales).squeeze())
        src_coords, req_coords_diag = _higher_precision_time_stack(source, request, udims)
        # We need to unwravel the nD stacked coordinates
        ckdtree_source = cKDTree(src_coords.reshape(src_coords.shape[0], -1).T * scales)

        # if the udims are all stacked in the same stack as part of the request coordinates, then we're done.
        # Otherwise we have to evaluate each unstacked set of dimensions independently
        # Note, part of this code is duplicated in the selector
        indep_evals = [ud for ud in udims if not request.is_stacked(ud)]
        # two udims could be stacked, but in different dim groups, e.g. source (lat, lon), request (lat, time), (lon, alt)
        stacked = {d for d in request.dims for ud in udims if ud in d and request.is_stacked(ud)}

        if (len(indep_evals) + len(stacked)) <= 1:  # output is stacked in the same way
            # The ckdtree call below needs the lat/lon pairs in the last axis position
            req_coords = np.moveaxis(req_coords_diag, 0, -1)
        elif (len(stacked) == 0) | (len(indep_evals) == 0 and len(stacked) == len(udims)):
            req_coords = np.stack([i.ravel() for i in np.meshgrid(*req_coords_diag, indexing="ij")], axis=1)
        else:
            # Rare cases? E.g. lat_lon_time_alt source to lon, time_alt, lat destination
            sizes = [request[d].size for d in request.dims]
            reshape = np.ones(len(request.dims), int)
            coords = [None] * len(udims)
            for i in range(len(udims)):
                ii = [ii for ii in range(len(request.dims)) if udims[i] in request.dims[ii]][0]
                reshape[:] = 1
                reshape[ii] = -1
                coords[i] = req_coords_diag[i].reshape(*reshape)
                for j, d in enumerate(request.dims):
                    if udims[i] in d:  # Then we don't need to repeat
                        continue
                    coords[i] = coords[i].repeat(sizes[j], axis=j)
            req_coords = np.stack([i.ravel() for i in coords], axis=1)

        dist, index = ckdtree_source.query(req_coords * scales, k=1)

        if self.respect_bounds:
            if bounds is None:
                bounds = np.stack(
                    [
                        src_coords.reshape(src_coords.shape[0], -1).T.min(0),
                        src_coords.reshape(src_coords.shape[0], -1).T.max(0),
                    ],
                    axis=1,
                )
            # Fix order of bounds
            bounds = bounds[:, [source.udims.index(dim) for dim in udims]]
            index[np.any((req_coords > bounds[1]), axis=-1) | np.any((req_coords < bounds[0]), axis=-1)] = -1

        if tol and tol != np.inf:
            index[dist > tol] = -1

        return index

    def _get_uniform_index(self, dim, source, request, bounds=None):
        tol = self._get_tol(dim, source, request)

        index = (request.coordinates - source.start) / source.step
        rindex = np.around(index).astype(int)
        if self.ambiguous_rounding == "-infinity":
            # Find all the 0.5 and 1.5's that were rounded to even numbers, and make sure they all round down
            I = (index % 0.5) == 0
            rindex[I] = np.floor(index[I])
        elif self.ambiguous_rounding == "+infinity":
            # Find all the 0.5 and 1.5's that were rounded to even numbers, and make sure they all round down
            I = (index % 0.5) == 0
            rindex[I] = np.ceil(index[I])
        else:  # "unbiased", that's the default np.around behavior, so do nothing
            pass

        stop_ind = int(source.size)
        if self.respect_bounds:
            rindex[(rindex < 0) | (rindex >= stop_ind)] = -1
        else:
            rindex = np.clip(rindex, 0, stop_ind - 1)
        if tol and tol != np.inf:
            if dim == "time":
                step = self._time_to_float(source.step, source, request)
            else:
                step = source.step
            rindex[np.abs(index - rindex) * np.abs(step) > tol] = -1

        return rindex

    def _get_nonuniform_index(self, dim, source, request, bounds=None):
        tol = self._get_tol(dim, source, request)

        src, req = _higher_precision_time_coords1d(source, request)
        ckdtree_source = cKDTree(src.reshape(-1, 1))
        dist, index = ckdtree_source.query(req[:].reshape(-1, 1), k=1)
        index[index == source.coordinates.size] = -1

        if self.respect_bounds:
            if bounds is None:
                bounds = [src.min(), src.max()]
            index[(req.ravel() > bounds[1]) | (req.ravel() < bounds[0])] = -1
        if tol and tol != np.inf:
            index[dist > tol] = -1

        return index

    def _resize_unstacked_index(self, index, source_dim, request):
        # When the request is stacked, and the stacked dimensions are n-dimensions where n > 1,
        # Then len(request.shape) != len(request.dims), so it take s a little bit of footwork
        # to get the correct shape for the index
        reshape = np.array(request.shape)
        i = 0
        for dim in request.dims:
            addnext = len(request[dim].shape)
            if source_dim not in dim:
                reshape[i : i + addnext] = 1
            i += addnext
        return index.reshape(*reshape)

    def _resize_stacked_index(self, index, source_dim, request):
        reshape = np.array(request.shape)
        i = 0
        for dim in request.dims:
            addnext = len(request[dim].shape)
            d = dim.split("_")
            if not any([dd in source_dim for dd in d]):
                reshape[i : i + addnext] = 1
            i += addnext
        return index.reshape(*reshape)


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

        # confirm that udims are in source and eval coordinates
        # TODO: handle stacked coordinates
        if self._dim_in(udims_subset, source_coordinates):
            return udims_subset
        else:
            return tuple()

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def select_coordinates(self, udims, source_coordinates, eval_coordinates, index_type="numpy"):
        """
        {interpolator_select}
        """
        new_coords = []
        new_coords_idx = []

        source_coords, source_coords_index = source_coordinates.intersect(
            eval_coordinates, outer=True, return_index=True
        )

        if source_coords.size == 0:
            return source_coords, source_coords_index

        # iterate over the source coordinate dims in case they are stacked
        for src_dim, idx in zip(source_coords, source_coords_index):

            # TODO: handle stacked coordinates
            if isinstance(source_coords[src_dim], StackedCoordinates):
                raise InterpolatorException("NearestPreview select does not yet support stacked dimensions")

            if src_dim in eval_coordinates.dims:
                src_coords = source_coords[src_dim]
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

                ndelta = max(1, np.round(np.abs(dst_delta / src_delta)))
                idx_offset = 0
                if src_coords.size == 1:
                    c = src_coords.copy()
                else:
                    c_test = UniformCoordinates1d(src_start, src_stop, ndelta * src_delta, **src_coords.properties)
                    bounds = source_coordinates[src_dim].bounds
                    # The delta/2 ensures the endpoint is included when there is a floating point rounding error
                    # the delta/2 is more than needed, but does guarantee.
                    src_stop = np.clip(src_stop + ndelta * src_delta / 2, bounds[0], bounds[1])
                    c = UniformCoordinates1d(src_start, src_stop, ndelta * src_delta, **src_coords.properties)
                    if c.size > c_test.size:  # need to adjust the index as well
                        idx_offset = int(ndelta)

                idx_start = idx.start if isinstance(idx, slice) else idx[0]
                idx_stop = idx.stop if isinstance(idx, slice) else idx[-1]
                if idx_stop is not None:
                    idx_stop += idx_offset
                idx = slice(idx_start, idx_stop, int(ndelta))
            else:
                c = source_coords[src_dim]

            new_coords.append(c)
            new_coords_idx.append(idx)

        return Coordinates(new_coords, validate_crs=False), tuple(new_coords_idx)
