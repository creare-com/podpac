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
    lens = []
    for d in dims:
        c0, c1 = _higher_precision_time_coords1d(coords0[d], coords1[d])
        crds0.append(c0)
        crds1.append(c1)
        lens.append(len(c1))
    if np.all(np.array(lens) == lens[0]):
        crds1 = np.stack(crds1, axis=0)

    return np.stack(crds0, axis=0), crds1


def _higher_precision_time_coords1d(coords0, coords1):
    dtype0 = coords0.coordinates[0].dtype
    dtype1 = coords1.coordinates[0].dtype
    if not np.issubdtype(dtype0, np.datetime64) or not np.issubdtype(dtype1, np.datetime64):
        return coords0.coordinates, coords1.coordinates
    if dtype0 > dtype1:  # greater means higher precision (smaller unit)
        dtype = dtype0
    else:
        dtype = dtype1
    return coords0.coordinates.astype(dtype).astype(float), coords1.coordinates.astype(dtype).astype(float)


def _index2slice(index):
    if index.size == 0:
        return slice(0, 0)
    elif index.size == 1:
        return slice(index[0], index[0] + 1)
    else:
        df = np.diff(index)
        mn = np.min(index)
        mx = np.max(index)
        if np.all(df == df[0]):
            return slice(mn, mx + 1, df[0])
        else:
            return slice(mn, mx + 1)


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

    def select(self, source_coords, request_coords, index_type="numpy"):
        """Sub-selects the source_coords based on the request_coords

        Parameters
        ------------
        source_coords: :class:`podpac.Coordinates`
            The coordinates of the source data
        request_coords: :class:`podpac.Coordinates`
            The coordinates of the request (user eval)
        index_type: str, optional
            Default is 'numpy'. Either "numpy", "xarray", or "slice". The returned index will be compatible with,
            either "numpy" (default) or "xarray" objects, or any
            object that works with tuples of slices ("slice")

        Returns
        --------
        :class:`podpac.Coordinates`:
            The sub-selected source coordinates
        tuple(indices):
            The indices that can be used to sub-select the source coordinates to produce the sub-selected coordinates.
            This is useful for directly indexing into the data type.
        """
        if source_coords.crs.lower() != request_coords.crs.lower():
            request_coords = request_coords.transform(source_coords.crs)
        coords = []
        coords_inds = []
        for coord1d in source_coords._coords.values():
            ci = self._select1d(coord1d, request_coords, index_type)
            ci = np.sort(np.unique(ci))
            if len(coord1d.shape) == 2:  # Handle case of 2D-stacked coordinates
                ncols = coord1d.shape[1]
                ci = (ci // ncols, ci % ncols)
                if index_type == "slice":
                    ci = tuple([_index2slice(cii) for cii in ci])
            elif index_type == "slice":
                ci = _index2slice(ci)
            if len(coord1d.shape) == 3:  # Handle case of 3D-stacked coordinates
                raise NotImplementedError
            c = coord1d[ci]
            coords.append(c)
            coords_inds.append(ci)
        coords = Coordinates(coords)
        if index_type == "numpy":
            coords_inds = self._merge_indices(coords_inds, source_coords.dims, request_coords.dims)
        elif index_type == "xarray":
            # unlike numpy, xarray assumes indexes are orthogonal by default, so the 1d coordinates are already correct
            # unless there are tuple coordinates (nD stacked coords) but those are handled in interpolation_manager
            pass
        return coords, tuple(coords_inds)

    def _select1d(self, source, request, index_type):
        if isinstance(source, StackedCoordinates):
            ci = self._select_stacked(source, request, index_type)
        elif source.is_uniform:
            ci = self._select_uniform(source, request, index_type)
        else:
            ci = self._select_nonuniform(source, request, index_type)
        # else:
        # _logger.info("Coordinates are not subselected for source {} with request {}".format(source, request))
        # return source, slice(0, None)
        return ci

    def _merge_indices(self, indices, source_dims, request_dims):
        # For numpy to broadcast correctly, we have to reshape each of the indices
        reshape = np.ones(len(indices), int)
        new_indices = []
        for i in range(len(indices)):
            reshape[:] = 1
            reshape[i] = -1
            if isinstance(indices[i], tuple):
                # nD stacked coordinates
                # This means the source has shape (N, M, ...)
                # But the coordinates are stacked (i.e. lat_lon with shape N, M for the lon and lat parts)
                new_indices.append(tuple([ind.reshape(*reshape) for ind in indices[i]]))
            else:
                new_indices.append(indices[i].reshape(*reshape))
        return tuple(new_indices)

    def _select_uniform(self, source, request, index_type):
        crds = request[source.name]
        if crds.is_uniform and abs(crds.step) < abs(source.step) and not request.is_stacked(source.name):
            start_ind = np.clip((crds.start - source.start) / source.step, 0, source.size)
            end_ind = np.clip((crds.stop - source.start) / source.step, 0, source.size)
            start = int(np.floor(max(0, min(start_ind, end_ind) + min(self.method) - 1e-6)))
            stop = int(np.ceil(min(source.size, max(start_ind, end_ind) + max(self.method) + 1 + 1e-6)))
            return np.arange(start, stop)

        index = (crds.coordinates - source.start) / source.step
        stop_ind = source.size - 1
        if len(self.method) > 1:
            flr_ceil = {-1: np.floor(index), 1: np.ceil(index)}
        else:
            # In this case, floating point error really matters, so we have to do a test
            up = np.round(index - 1e-6)
            down = np.round(index + 1e-6)
            # When up and down do not agree, make sure both the indices that will be kept.
            index_mid = down  # arbitrarily default to down when both satisfy criteria
            Iup = up != down
            Iup[Iup] = (up[Iup] >= 0) & (up[Iup] <= stop_ind) & ((up[Iup] > down.max()) | (up[Iup] < down.min()))
            index_mid[Iup] = up[Iup]
            flr_ceil = {0: index_mid}

        inds = []
        for m in self.method:
            sign = np.sign(m)
            base = flr_ceil[sign]
            inds.append(base + sign * (sign * m - 1))

        inds = np.stack(inds, axis=1).ravel().astype(int)
        inds = inds[(inds >= 0) & (inds <= stop_ind)]
        return inds

    def _select_nonuniform(self, source, request, index_type):
        src, req = _higher_precision_time_coords1d(source, request[source.name])
        ckdtree_source = cKDTree(src[:, None])
        _, inds = ckdtree_source.query(req[:, None], k=len(self.method))
        inds = inds[inds < source.coordinates.size]
        return inds.ravel()

    def _select_stacked(self, source, request, index_type):
        udims = [ud for ud in source.udims if ud in request.udims]

        # if the udims are all stacked in the same stack as part of the request coordinates, then we can take a shortcut.
        # Otherwise we have to evaluate each unstacked set of dimensions independently
        indep_evals = [ud for ud in udims if not request.is_stacked(ud)]
        # two udims could be stacked, but in different dim groups, e.g. source (lat, lon), request (lat, time), (lon, alt)
        stacked = {d for d in request.dims for ud in udims if ud in d and request.is_stacked(ud)}

        inds = np.array([])
        # Parts of the below code is duplicated in NearestNeighborInterpolotor
        src_coords, req_coords_diag = _higher_precision_time_stack(source, request, udims)
        # For nD stacked coordinates we need to unravel the stacked dimension
        ckdtree_source = cKDTree(src_coords.reshape(src_coords.shape[0], -1).T)
        if (len(indep_evals) + len(stacked)) <= 1:
            req_coords = req_coords_diag.T
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

        _, inds = ckdtree_source.query(req_coords, k=len(self.method))
        inds = inds[inds < source.coordinates.size]
        inds = inds.ravel()
        return inds
