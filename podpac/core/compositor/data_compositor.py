from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np
import xarray as xr
import traitlets as tl

from podpac.core.utils import common_doc
from podpac.core.compositor.compositor import COMMON_COMPOSITOR_DOC, BaseCompositor
from podpac.core.units import UnitsDataArray
from podpac.core.interpolation.interpolation import InterpolationMixin


@common_doc(COMMON_COMPOSITOR_DOC)
class DataCompositor(BaseCompositor):
    """Compositor that combines tiled sources.

    The requested data does not need to be interpolated by the sources before being composited

    Attributes
    ----------
    sources : list
        Source nodes, in order of preference. Later sources are only used where earlier sources do not provide data.
    source_coordinates : :class:`podpac.Coordinates`
        Coordinates that make each source unique. Must the same size as ``sources`` and single-dimensional. Optional.
    """

    @common_doc(COMMON_COMPOSITOR_DOC)
    def composite(self, coordinates, data_arrays, result=None):
        """Composites data_arrays in order that they appear. Once a request contains no nans, the result is returned.

        Parameters
        ----------
        coordinates : :class:`podpac.Coordinates`
            {requested_coordinates}
        data_arrays : generator
            Evaluated source data, in the same order as the sources.
        result : podpac.UnitsDataArray, optional
            {eval_output}

        Returns
        -------
        {eval_return} This composites the sources together until there are no nans or no more sources.
        """

        # TODO: Fix boundary information on the combined data arrays
        res = next(data_arrays)
        for arr in data_arrays:
            res = res.combine_first(arr)
        res = UnitsDataArray(res)

        if result is not None:
            result.data[:] = res.transponse(*result.dims).data
            return result
        return res


class InterpDataCompositor(InterpolationMixin, DataCompositor):
    pass
