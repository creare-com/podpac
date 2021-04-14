import numpy as np
import pytest

from .coordinates_for_tests import COORDINATES
import podpac.datalib
from podpac import Coordinates, clinspace


@pytest.mark.integration
class TestMODIS(object):
    def test_common_coordinates(self):
        modis = podpac.datalib.modis_pds.MODIS(product="MCD43A4.006", data_key="B01")  #  Band 01, 620 - 670nm
        for ck, c in COORDINATES.items():
            print("Evaluating: ", ck)
            o = modis.eval(c)
            assert np.any(np.isfinite(o.data))
