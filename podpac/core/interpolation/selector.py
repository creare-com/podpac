import numpy as np
from scipy.spatial import cKDTree
import traitlets as tl
import logging

_logger = logging.getLogger(__name__)

from podpac.core.coordinates.coordinates import Coordinates
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates

METHOD = {"nearest": [0], "bilinear": [-1, 1], "linear": [-1, 1], "cubic": [-2, -1, 1, 2]}


def _higher_precision_time_stack(coords0, coords1, dims):
    crds0 = []
    crds1 = []
    for d in dims:
        dtype0 = coords0[d].coordinates[0].dtype
        dtype1 = coords1[d].coordinates[0].dtype
        if not np.issubdtype(dtype0, np.datetime64) or not np.issubdtype(dtype1, np.datetime64):
            crds0.append(coords0[d].coordinates)
            crds1.append(coords1[d].coordinates)
            continue
        if dtype0 > dtype1:  # greater means higher precision (smaller unit)
            dtype = dtype0
        else:
            dtype = dtype1
        crds0.append(coords0[d].coordinates.astype(dtype).astype(float))
        crds1.append(coords1[d].coordinates.astype(dtype).astype(float))

    return np.stack(crds0, axis=1), np.stack(crds1, axis=1)


class Selector(tl.HasTraits):
    supported_methods = ["nearest", "linear", "bilinear", "cubic"]

    method = tl.Tuple()
    respect_bounds = tl.Bool(False)

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
        coords_inds = []
        for coord1d in source_coords._coords.values():
            c, ci = self.select1d(coord1d, request_coords)
            coords.append(c)
            coords_inds.append(ci)
        coords = Coordinates(coords)
        coords_inds = self.merge_indices(coords_inds, source_coords.dims, request_coords.dims)
        return coords, coords_inds

    def select1d(self, source, request):
        if isinstance(source, StackedCoordinates):
            ci = self.select_stacked(source, request)
        elif source.is_uniform:
            ci = self.select_uniform(source, request)
        else:
            ci = self.select_nonuniform(source, request)
        # else:
        # _logger.info("Coordinates are not subselected for source {} with request {}".format(source, request))
        # return source, slice(0, None)
        ci = np.sort(np.unique(ci))
        return source[ci], ci

    def merge_indices(self, indices, source_dims, request_dims):
        # For numpy to broadcast correctly, we have to reshape each of the indices
        reshape = np.ones(len(indices), int)
        for i in range(len(indices)):
            reshape[:] = 1
            reshape[i] = -1
            indices[i] = indices[i].reshape(*reshape)
        return tuple(indices)

    def select_uniform(self, source, request):
        crds = request[source.name]
        if crds.is_uniform and crds.step < source.step and not request.is_stacked(source.name):
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
        inds = inds[(inds >= 0) & (inds <= stop_ind)]
        return inds

    def select_nonuniform(self, source, request):
        crds = request[source.name]
        ckdtree_source = cKDTree(source.coordinates[:, None])
        _, inds = ckdtree_source.query(crds.coordinates[:, None], k=len(self.method))
        return inds.ravel()

    def select_stacked(self, source, request):
        udims = [ud for ud in source.udims if ud in request.udims]
        src_coords, req_coords_diag = _higher_precision_time_stack(source, request, udims)
        ckdtree_source = cKDTree(src_coords)
        _, inds = ckdtree_source.query(req_coords_diag, k=len(self.method))
        inds = inds.ravel()

        if np.unique(inds).size == source.size:
            return inds

        if len(udims) == 1:
            return inds

        # if the udims are all stacked in the same stack as part of the request coordinates, then we're done.
        # Otherwise we have to evaluate each unstacked set of dimensions independently
        indep_evals = [ud for ud in udims if not request.is_stacked(ud)]
        # two udims could be stacked, but in different dim groups, e.g. source (lat, lon), request (lat, time), (lon, alt)
        stacked = {d for d in request.dims for ud in udims if ud in d and request.is_stacked(ud)}

        if (len(indep_evals) + len(stacked)) <= 1:
            return inds

        stacked_ud = [d for s in stacked for d in s.split("_") if d in udims]

        c_evals = indep_evals + stacked_ud
        # Since the request are for independent dimensions (we know that already) the order doesn't matter
        inds = [set(self.select1d(source[ce], request)[1]) for ce in c_evals]

        if self.respect_bounds:
            inds = np.array(list(set.intersection(*inds)), int)
        else:
            inds = np.sort(np.array(list(set.union(*inds)), int))
        return inds
