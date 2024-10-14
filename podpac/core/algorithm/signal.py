"""
Signal Summary
"""

import podpac
from collections import OrderedDict

import traitlets as tl
import numpy as np
import xarray as xr
import scipy.signal

from podpac.core.settings import settings
from podpac.core.coordinates import Coordinates, UniformCoordinates1d, ArrayCoordinates1d
from podpac.core.coordinates import add_coord
from podpac.core.node import Node
from podpac.core.algorithm.algorithm import UnaryAlgorithm
from podpac.core.utils import common_doc, ArrayTrait, NodeTrait
from podpac.core.node import COMMON_NODE_DOC


COMMON_DOC = COMMON_NODE_DOC.copy()
COMMON_DOC[
    "full_kernel"
] = """Kernel that contains all the dimensions of the input source, in the correct order.

        Returns
        -------
        np.ndarray
            The dimensionally full convolution kernel"""
COMMON_DOC[
    "validate_kernel"
] = """Checks to make sure the kernel is valid.

        Parameters
        ----------
        proposal : np.ndarray
            The proposed kernel

        Returns
        -------
        np.ndarray
            The valid kernel

        Raises
        ------
        ValueError
            If the kernel is not valid (i.e. incorrect dimensionality). """


class Convolution(UnaryAlgorithm):
    """Compute a general convolution over a source node.

    This node automatically resizes the requested coordinates to avoid edge effects.

    Attributes
    ----------
    source : podpac.Node
        Source node on which convolution will be performed.
    kernel : np.ndarray, optional
        The convolution kernel. This kernel must include the dimensions of source node outputs. The dimensions for this
        array are labelled by `kernel_dims`. Any dimensions not in the source nodes outputs will be summed over.
    kernel_dims : list, optional
        A list of the dimensions for the kernel axes. If the dimensions in this list do not match the
        coordinates in the source, then any extra dimensions in the kernel are removed by adding all the values over that axis
        dimensions in the source are not convolved with any kernel.

    kernel_type : str, optional
        If kernel is not defined, kernel_type will create a kernel based on the inputs, and it will have the
        same number of axes as kernel_dims.
        The format for the created  kernels is '<kernel_type>, <kernel_size>, <kernel_params>'.
        Any kernel defined in `scipy.signal` as well as `mean` can be used. For example:
        kernel_type = 'mean, 8' or kernel_type = 'gaussian,16,8' are both valid.
        Note: These kernels are automatically normalized such that kernel.sum() == 1
    """

    kernel = ArrayTrait(dtype=float).tag(attr=True)
    kernel_dims = tl.List().tag(attr=True)
    # Takes one or the other which is hard to implement in a GUI
    kernel_type = tl.List().tag(attr=True)

    def _first_init(self, kernel=None, kernel_dims=None, kernel_type=None, kernel_ndim=None, **kwargs):
        if kernel_dims is None:
            raise TypeError("Convolution expected 'kernel_dims' to be specified when giving a 'kernel' array")

        if kernel is not None and kernel_type is not None:
            raise TypeError("Convolution expected 'kernel' or 'kernel_type', not both")

        if kernel is None:
            if kernel_type is None:
                raise TypeError("Convolution requires 'kernel' array or 'kernel_type' string")
            kernel = self._make_kernel(kernel_type, len(kernel_dims))

        if len(kernel_dims) != len(np.array(kernel).shape):
            raise TypeError(
                "The kernel_dims should contain the same number of dimensions as the number of axes in 'kernel', but len(kernel_dims) {} != len(kernel.shape) {}".format(
                    len(kernel_dims), len(np.array(kernel).shape)
                )
            )

        kwargs["kernel"] = kernel
        kwargs["kernel_dims"] = kernel_dims
        return super(Convolution, self)._first_init(**kwargs)

    @common_doc(COMMON_DOC)
    def _eval(self, coordinates, output=None, _selector=None):
        """Evaluates this nodes using the supplied coordinates.

        Parameters
        ----------
        coordinates : podpac.Coordinates
            {requested_coordinates}
        output : podpac.UnitsDataArray, optional
            {eval_output}
        _selector: callable(coordinates, request_coordinates)
            {eval_selector}

        Returns
        -------
        {eval_return}
        """
        # The size of this kernel is used to figure out the expanded size
        full_kernel = self.kernel

        # expand the coordinates
        # The next line effectively drops extra coordinates, so we have to add those later in case the
        # source is some sort of reduction Node.
        kernel_dims = [kd for kd in coordinates.dims if kd in self.kernel_dims]
        missing_dims = [kd for kd in coordinates.dims if kd not in self.kernel_dims]

        exp_coords = []
        exp_slice = []
        for dim in kernel_dims:
            coord = coordinates[dim]
            s = full_kernel.shape[self.kernel_dims.index(dim)]
            if s == 1 or not isinstance(coord, (UniformCoordinates1d, ArrayCoordinates1d)):
                exp_coords.append(coord)
                exp_slice.append(slice(None))
                continue

            if isinstance(coord, UniformCoordinates1d):
                s_start = -s // 2
                s_end = max(s // 2 - ((s + 1) % 2), 1)
                # The 1e-14 is for floating point error because if endpoint is slightly
                # in front of step * N then the endpoint is excluded
                # ALSO: MUST use size instead of step otherwise floating point error
                # makes the xarray arrays not align. The following HAS to be true:
                #     np.diff(coord.coordinates).mean() == coord.step
                exp_coords.append(
                    UniformCoordinates1d(
                        add_coord(coord.start, s_start * coord.step),
                        add_coord(coord.stop, s_end * coord.step + 1e-14 * coord.step),
                        size=coord.size - s_start + s_end,  # HAVE to use size, see note above
                        **coord.properties
                    )
                )
                exp_slice.append(slice(-s_start, -s_end))
            elif isinstance(coord, ArrayCoordinates1d):
                if not coord.is_monotonic or coord.size < 2:
                    exp_coords.append(coord)
                    exp_slice.append(slice(None))
                    continue

                arr_coords = coord.coordinates
                delta_start = arr_coords[1] - arr_coords[0]
                extra_start = np.arange(arr_coords[0] - delta_start * (s // 2), arr_coords[0], delta_start)
                delta_end = arr_coords[-1] - arr_coords[-2]
                # The 1e-14 is for floating point error to make sure endpoint is included
                extra_end = np.arange(
                    arr_coords[-1] + delta_end, arr_coords[-1] + delta_end * (s // 2) + delta_end * 1e-14, delta_end
                )
                arr_coords = np.concatenate([extra_start, arr_coords, extra_end])
                exp_coords.append(ArrayCoordinates1d(arr_coords, **coord.properties))
                exp_slice.append(slice(extra_start.size, -extra_end.size))

        # Add missing dims back in -- this is needed in case the source is a reduce node.
        exp_coords += [coordinates[d] for d in missing_dims]

        # Create expanded coordinates
        exp_slice = tuple(exp_slice)
        expanded_coordinates = Coordinates(exp_coords, crs=coordinates.crs, validate_crs=False)

        if settings["DEBUG"]:
            self._expanded_coordinates = expanded_coordinates

        # evaluate source using expanded coordinates, convolve, and then slice out original coordinates
        source = self.source.eval(expanded_coordinates, _selector=_selector)

        kernel_dims_u = kernel_dims
        kernel_dims = self.kernel_dims
        sum_dims = [d for d in kernel_dims if d not in source.dims]
        # Sum out the extra dims
        full_kernel = full_kernel.sum(axis=tuple([kernel_dims.index(d) for d in sum_dims]))
        exp_slice = [exp_slice[i] for i in range(len(kernel_dims_u)) if kernel_dims_u[i] not in sum_dims]
        kernel_dims = [d for d in kernel_dims if d in source.dims]

        # Put the kernel axes in the correct order
        # The (if d in kernel_dims) takes care of "output", which can be optionally present
        full_kernel = full_kernel.transpose([kernel_dims.index(d) for d in source.dims if (d in kernel_dims)])

        # Check for extra dimensions in the source and reshape the kernel appropriately
        if any([d not in kernel_dims for d in source.dims if d != "output"]):
            new_axis = []
            new_exp_slice = []
            for d in source.dims:
                if d in kernel_dims:
                    new_axis.append(slice(None))
                    new_exp_slice.append(exp_slice[kernel_dims.index(d)])
                else:
                    new_axis.append(None)
                    new_exp_slice.append(slice(None))
            full_kernel = full_kernel[tuple(new_axis)]
            exp_slice = new_exp_slice

        if np.any(np.isnan(source)):
            method = "direct"
        else:
            method = "auto"

        if ("output" not in source.dims) or ("output" in source.dims and "output" in kernel_dims):
            result = scipy.signal.convolve(source, full_kernel, mode="same", method=method)
        else:
            # source with multiple outputs
            result = np.stack(
                [
                    scipy.signal.convolve(source.sel(output=output), full_kernel, mode="same", method=method)
                    for output in source.coords["output"]
                ],
                axis=source.dims.index("output"),
            )
        result = result[tuple(exp_slice)]

        if output is None:
            missing_dims = [d for d in coordinates.dims if d not in source.dims]
            output = self.create_output_array(coordinates.drop(missing_dims), data=result)
        else:
            output[:] = result

        return output

    @staticmethod
    def _make_kernel(kernel_type, ndim):
        ktype = kernel_type.split(",")[0]
        size = int(kernel_type.split(",")[1])
        if ktype == "mean":
            k = np.ones([size] * ndim)
        else:
            args = [float(a) for a in kernel_type.split(",")[2:]]
            if hasattr(scipy.signal,ktype): # scipy moved where "gaussian" is placed, this handles similar circumstances
                f = getattr(scipy.signal, ktype)
            else:
                f = getattr(scipy.signal.windows, ktype)
            k1d = f(size, *args)
            k = k1d.copy()
            for i in range(ndim - 1):
                k = np.tensordot(k, k1d, 0)

        return k / k.sum()
