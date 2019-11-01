"""
Algorithm Public Module
"""

# REMINDER: update api docs (doc/source/user/api.rst) to reflect changes to this file

from podpac.core.algorithm.algorithm import Algorithm
from podpac.core.algorithm.general import Arithmetic, Generic, CombineOutputs
from podpac.core.algorithm.utility import SinCoords, Arange, CoordData
from podpac.core.algorithm.stats import (
    Min,
    Max,
    Sum,
    Count,
    Mean,
    Median,
    Variance,
    StandardDeviation,
    Skew,
    Kurtosis,
    DayOfYear,
    GroupReduce,
)
from podpac.core.algorithm.coord_select import ExpandCoordinates, SelectCoordinates, YearSubstituteCoordinates
from podpac.core.algorithm.signal import Convolution, SpatialConvolution, TimeConvolution
