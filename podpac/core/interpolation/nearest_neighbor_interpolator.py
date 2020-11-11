"""
Interpolator implementations
"""

from __future__ import division, unicode_literals, print_function, absolute_import
from six import string_types

import numpy as np
import traitlets as tl
from scipy.spatial import cKDTree

# Optional dependencies


# podac imports
from podpac.core.interpolation.interpolator import COMMON_INTERPOLATOR_DOCS, Interpolator, InterpolatorException
from podpac.core.coordinates import Coordinates, UniformCoordinates1d, StackedCoordinates
from podpac.core.utils import common_doc
from podpac.core.coordinates.utils import get_timedelta
from podpac.core.interpolation.selector import Selector, _higher_precision_time_coords1d, _higher_precision_time_stack


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
    time_tolerance = tl.Union([tl.Unicode(), tl.Instance(np.timedelta64, allow_none=True)])
    alt_tolerance = tl.Float(default_value=np.inf, allow_none=True)

    # spatial_scale only applies when the source is stacked with time or alt
    spatial_scale = tl.Float(default_value=1, allow_none=True)
    # time_scale only applies when the source is stacked with lat, lon, or alt
    time_scale = tl.Union([tl.Unicode(), tl.Instance(np.timedelta64, allow_none=True)])
    # alt_scale only applies when the source is stacked with lat, lon, or time
    alt_scale = tl.Float(default_value=1, allow_none=True)

    respect_bounds = tl.Bool(True)
    remove_nan = tl.Bool(True)

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

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def interpolate(self, udims, source_coordinates, source_data, eval_coordinates, output_data):
        """
        {interpolator_interpolate}
        """
        # Note, some of the following code duplicates code in the Selector class.
        # This duplication is for the sake of optimization
        if self.remove_nan:
            # Eliminate nans from the source data. Note, this could turn a uniform griddted dataset into a stacked one
            source_data, source_coordinates = self._remove_nans(source_data, source_coordinates)

        def is_stacked(d):
            return "_" in d

        data_index = []
        for d in source_coordinates.dims:
            source = source_coordinates[d]
            if is_stacked(d):
                index = self._get_stacked_index(d, source, eval_coordinates)
            elif source_coordinates[d].is_uniform:
                request = eval_coordinates[d]
                index = self._get_uniform_index(d, source, request)
            else:  # non-uniform coordinates... probably an optimization here
                request = eval_coordinates[d]
                index = self._get_nonuniform_index(d, source, request)
            data_index.append(index)

        index = tuple(data_index)

        output_data.data[:] = np.array(source_data)[index]

        return output_data

    def _remove_nans(self, source_data, source_coordinates):
        index = np.isnan(source_data)
        if not np.any(index):
            return source_data, source_coordinates

        data = source_data.data[~index]
        coords = np.meshgrid(*[c.coordinates for c in source_coordinates.values()], indexing="ij")
        coords = [c[~index] for c in coords]

        return data, Coordinates([coords], dims=[source_coordinates.udims])

    def _get_tol(self, dim):
        if dim in ["lat", "lon"]:
            return self.spatial_tolerance
        if dim == "alt":
            return self.alt_tolerance
        if dim == "time":
            return self.time_tolerance
        raise NotImplementedError()

    def _get_scale(self, dim):
        if dim in ["lat", "lon"]:
            return self.spatial_scale
        if dim == "alt":
            return self.alt_scale
        if dim == "time":
            return self.time_scale
        raise NotImplementedError()

    def _get_stacked_index(self, dim, source, request):
        # The udims are in the order of the request so that the meshgrid calls will be in the right order
        udims = [ud for ud in request.udims if ud in source.udims]
        tols = np.array([self._get_tol(d) for d in udims])[None, :]
        scales = np.array([self._get_scale(d) for d in udims])[None, :]
        tol = np.linalg.norm((tols * scales).squeeze())
        src_coords, req_coords_diag = _higher_precision_time_stack(source, request, udims)
        ckdtree_source = cKDTree(src_coords * scales)

        # if the udims are all stacked in the same stack as part of the request coordinates, then we're done.
        # Otherwise we have to evaluate each unstacked set of dimensions independently
        indep_evals = [ud for ud in udims if not request.is_stacked(ud)]
        # two udims could be stacked, but in different dim groups, e.g. source (lat, lon), request (lat, time), (lon, alt)
        stacked = {d for d in request.dims for ud in udims if ud in d and request.is_stacked(ud)}

        if (len(indep_evals) + len(stacked)) <= 1:  # output is stacked in the same way
            req_coords = req_coords_diag
        elif (len(stacked) == 0) | (len(indep_evals) == 0 and len(stacked) == len(udims)):
            req_coords = np.stack([i.ravel() for i in np.meshgrid(*req_coords_diag.T, indexing="ij")], axis=1)
        else:
            # Rare cases? E.g. lat_lon_time_alt source to lon, time_alt, lat destination
            c_evals = indep_evals + list(stacked)
            sizes = [request[d].size for d in c_evals]
            reshape = np.ones(len(c_evals), int)
            coords = [None] * len(udims)
            for i in range(len(udims)):
                reshape[:] = 1
                reshape[i] = -1
                coords[i] = req_coords_diag[i].reshape(*reshape)
                for j, d in c_evals:
                    if udims[i] in d:
                        continue
                    coords[i] = coords[i].repeat(sizes[j], axis=j)
            req_coords = np.stack([i.ravel() for i in np.meshgrid(*coords, indexing="ij")], axis=1)

        dist, index = ckdtree_source.query(req_coords * np.array(scales)[None, :], k=1)

        if tol and tol != np.inf:
            index[dist > tol] = -1

        index = self._resize_stacked_index(index, dim, request)
        return index

    def _get_uniform_index(self, dim, source, request):
        tol = self._get_tol(dim)

        index = ((request.coordinates - source.start) / source.step).astype(int)
        rindex = np.around(index).astype(int)
        stop_ind = int(source.size)
        if self.respect_bounds:
            rindex[(rindex < 0) | (rindex >= stop_ind)] = -1
        else:
            rindex = np.clip(rindex, 0, stop_ind)
        if tol and tol != np.inf:
            rindex[np.abs(index - rindex) * source.step > tol] = -1

        index = self._resize_unstacked_index(rindex, dim, request)
        return index

    def _get_nonuniform_index(self, dim, source, request):
        tol = self._get_tol(dim)

        src, req = _higher_precision_time_coords1d(source, request)
        ckdtree_source = cKDTree(src[:, None])
        dist, index = ckdtree_source.query(req[:, None], k=1)
        index[index == source.coordinates.size] = -1

        if self.respect_bounds:
            index[(req > src.max()) | (req < src.min())] = -1
        if tol and tol != np.inf:
            index[dist > tol] = -1

        index = self._resize_unstacked_index(index, dim, request)
        return index

    def _resize_unstacked_index(self, index, source_dim, request):
        reshape = np.ones(len(request.dims), int)
        i = [i for i in range(len(request.dims)) if source_dim in request.dims]
        reshape[i] = -1
        return index.reshape(*reshape)

    def _resize_stacked_index(self, index, source_dim, request):
        reshape = np.ones(len(request.dims), int)
        sizes = [request[d].size for d in request.dims]

        dims = source_dim.split("_")
        for i, dim in enumerate(dims):
            reshape[:] = 1

            if "_" in request.dims[i]:
                if all([d in request.dims[i] for d in dims]):  # Stacked to Stacked
                    return index

            # Examples: lat_lon_time_alt source --> lon, time_alt, lat destination
            #           lat_lon source --> lat, time, lon destination
            reshape[i] = sizes[i]
            for j, rdim in enumerate(request.dims):
                if any([d in request.dims[i] for d in dims]):
                    reshape[j] = sizes[j]

        index = index.reshape(*reshape)
        return index


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
                c = source_coords[src_dim]

            new_coords.append(c)
            new_coords_idx.append(idx)

        return Coordinates(new_coords, validate_crs=False), tuple(new_coords_idx)
