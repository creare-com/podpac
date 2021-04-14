import numpy as np
import pytest

from .coordinates_for_tests import COORDINATES
import podpac.datalib
from podpac import Coordinates, clinspace


@pytest.mark.integration
class TestSoilGrids(object):
    def test_common_coordinates(self):
        soil_organic_carbon = podpac.datalib.soilgrids.SoilGridsSOC(layer="soc_0-5cm_Q0.95")
        for ck, c in COORDINATES.items():
            print("Evaluating: ", ck)
            o = soil_organic_carbon.eval(c)
            assert np.any(np.isfinite(o.data))
