import numpy as np
from scipy.spatial import cKDTree
import traitlets as tl
import logging

_logger = logging.getLogger(__name__)

from podpac.core.coordinates.coordinates import merge_dims
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates

METHOD = {"nearest": [0], "bilinear": [-1, 1], "linear": [-1, 1], "cubic": [-2, -1, 1, 2]}


class Selector(tl.HasTraits):

    method = tl.Tuple()

    def __init__(self, method=None):
        """
        Params
        -------
        method: str, list
            Either a list of offsets or a type of selection
        """
        if isinstance(method, str):
            self.method = METHOD.get(method)
        else:
            self.method = method

    def select(self, source_coords, request_coords):
        coords = []
        coords_ids = []
        for coord1d in source_coords._coords.values():
            c, ci = self.select1d(coord1d, request_coords)
            coords.append(c)
            coords_inds.append(ci)
        coords = merge_dims(coords)
        coords_inds = self.merge_indices(coords_inds, source_coords.dims, request_coords.dims)

    def select1d(self, source, request):
        if isinstance(source, StackedCoordinates):
            return self.select_stacked(source, request)
        elif source.is_uniform:
            return self.select_uniform(source, request)
        elif source.is_monotonic:
            return self.select_monotonic(source, request)
        else:
            _logger.info("Coordinates are not subselected for source {} with request {}".format(source, request))
            return source, slice(0, None)

    def merge_indices(self, indices, source_dims, request_dims):
        return tuple(indices)

    def select_uniform(self, source, request):
        crds = request[source.name]
        if crds.is_uniform and crds.step < source.step:
            return np.arange(source.size)

        index = (crds.coordinates - source.start) / source.step
        stop_ind = int(np.round((source.stop - source.start) / source.step))
        if len(self.method) > 1:
            flr_ceil = {-1: np.floor(index), 1: np.ceil(index)}
        else:
            flr_ceil = {0: np.round(index)}
        inds = []
        for m in self.method:
            sign = np.sign(m)
            base = flr_ceil[sign]
            inds.append(base + sign * (sign * m - 1))

        inds = np.stack(inds, axis=1).ravel().astype(int)
        return inds[(inds >= 0) & (inds <= stop_ind)]

    def select_monotonic(self, source, request):
        crds = request[source.name]
        ckdtree_source = cKDTree(source.coordinates[:, None])
        _, inds = ckdtree_source.query(crds.coordinates[:, None], k=1)
        return np.sort(np.unique(inds))
