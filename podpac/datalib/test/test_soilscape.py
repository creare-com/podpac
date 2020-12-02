import numpy as np
import pytest

from .coordinates_for_tests import COORDINATES
import podpac.datalib
from podpac import Coordinates, clinspace


@pytest.mark.integration
class TestSoilscape(object):
    def test_common_coordinates(self):
        point_interpolation = {
            "method": "nearest",
            "params": {
                "use_selector": False,
                "remove_nan": True,
                "time_scale": "1,M",
                "respect_bounds": False,
            },
        }
        soilscape = podpac.datalib.soilscape.SoilSCAPE20min(
            site="Canton_OK", data_key="soil_moisture", interpolation=point_interpolation
        )
        for ck, c in COORDINATES.items():
            if "cosmos" in ck:
                continue
            print("Evaluating: ", ck)
            o = soilscape.eval(c)
            assert np.any(np.isfinite(o.data))
