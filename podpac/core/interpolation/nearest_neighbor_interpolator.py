"""
Interpolator implementations
"""

from __future__ import division, unicode_literals, print_function, absolute_import
from six import string_types

import numpy as np
import traitlets as tl

# Optional dependencies


# podac imports
from podpac.core.interpolation.interpolator import COMMON_INTERPOLATOR_DOCS, Interpolator, InterpolatorException
from podpac.core.coordinates import Coordinates, UniformCoordinates1d, StackedCoordinates
from podpac.core.utils import common_doc
from podpac.core.coordinates.utils import get_timedelta


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
                if isinstance(self.time_tolerance, string_types):
                    self.time_tolerance = get_timedelta(self.time_tolerance)
                tolerance = self.time_tolerance
            elif dim != "time":
                tolerance = self.spatial_tolerance

            # reindex using xarray
            indexer = {dim: eval_coordinates[dim].coordinates.copy()}
            indexers += [dim]
            source_data = source_data.reindex(method=str("nearest"), tolerance=tolerance, **indexer)

        # at this point, output_data and eval_coordinates have the same dim order
        # this transpose makes sure the source_data has the same dim order as the eval coordinates
        eval_dims = eval_coordinates.dims
        output_data.data = source_data.part_transpose(eval_dims)

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

        # confirm that udims are in source and eval coordinates
        # TODO: handle stacked coordinates
        if self._dim_in(udims_subset, source_coordinates):
            return udims_subset
        else:
            return tuple()

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def select_coordinates(self, udims, source_coordinates, eval_coordinates):
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
