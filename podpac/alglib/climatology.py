"""
PODPAC node to compute beta fit of seasonal variables
"""

import logging
import numpy as np
import xarray as xr
import traitlets as tl
from lazy_import import lazy_module
from scipy.stats import beta
from scipy.stats._continuous_distns import FitSolverError

# optional imports
h5py = lazy_module("h5py")

# Internal dependencies
import podpac
from podpac.core.algorithm.stats import DayOfYearWindow

# Set up logging
_log = logging.getLogger(__name__)


class BetaFitDayOfYear(DayOfYearWindow):
    """
    This fits a beta distribution to day of the year in the requested coordinates over a window. It returns the beta
    distribution parameters 'a', and 'b' as part of the output. It may also return a number of percentiles.

    Attributes
    -----------
    percentiles: list, optional
        Default is []. After computing the beta distribution, optionally compute the value of the function for the given
        percentiles in the list. The results will be available as an output named ['d0', 'd1',...] for each entry in
        the list.
    """

    percentiles = tl.List().tag(attr=True)
    rescale = tl.Bool(True).tag(attr=True)

    @property
    def outputs(self):
        return ["a", "b"] + ["d{}".format(i) for i in range(len(self.percentiles))]

    def function(self, data, output):
        # define the fit function
        try:
            data[data == 1] -= 1e-6
            data[data == 0] += 1e-6
            a, b, loc, scale = beta.fit(data, floc=0, fscale=1)
        except FitSolverError as e:
            print(e)
            return output

        # populate outputs for this point
        output.loc[{"output": "a"}] = a
        output.loc[{"output": "b"}] = b
        for ii, d in enumerate(self.percentiles):
            output.loc[{"output": "d" + str(ii)}] = beta.ppf(d, a, b)

        return output

    def rescale_outputs(self, output, scale_max, scale_min):
        output[..., 2:] = (output[..., 2:] * (scale_max - scale_min)) + scale_min
        return output
