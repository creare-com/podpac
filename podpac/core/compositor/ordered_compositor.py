from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np
import traitlets as tl

from podpac.core.node import NodeException
from podpac.core.units import UnitsDataArray
from podpac.core.utils import common_doc
from podpac.core.compositor.compositor import COMMON_COMPOSITOR_DOC, BaseCompositor


@common_doc(COMMON_COMPOSITOR_DOC)
class OrderedCompositor(BaseCompositor):
    """Compositor that combines sources based on their order in self.sources.

    The sources should generally be interpolated before being composited (i.e. not raw datasources).

    Attributes
    ----------
    sources : list
        Source nodes, in order of preference. Later sources are only used where earlier sources do not provide data.
    source_coordinates : :class:`podpac.Coordinates`
        Coordinates that make each source unique. Must the same size as ``sources`` and single-dimensional. Optional.
    multithreading : bool, optional
        Default is True. If True, will always evaluate the compositor in serial, ignoring any MULTITHREADING settings

    """

    multithreading = tl.Bool(False)

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

        if result is None:
            result = self.create_output_array(coordinates)
        else:
            result[:] = np.nan

        mask = UnitsDataArray.create(coordinates, outputs=self.outputs, data=0, dtype=bool)
        for data in data_arrays:
            if self.outputs is None:
                try:
                    data = data.transpose(*result.dims)
                except ValueError:
                    raise NodeException(
                        "Cannot evaluate compositor with requested dims %s. "
                        "The compositor source dims are %s. "
                        "Specify the compositor 'dims' attribute to ignore extra requested dims."
                        % (coordinates.dims, data.dims)
                    )
                self._composite(result, data, mask)
            else:
                for name in data["output"]:
                    self._composite(result.sel(output=name), data.sel(output=name), mask.sel(output=name))

            # stop if the results are full
            if np.all(mask):
                break

        return result

    @staticmethod
    def _composite(result, data, mask):
        source_mask = np.isfinite(data.data)
        b = ~mask & source_mask
        result.data[b.data] = data.data[b.data]
        mask |= source_mask
