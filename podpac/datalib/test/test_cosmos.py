import numpy as np
import pytest

from .coordinates_for_tests import COORDINATES
import podpac.datalib.cosmos_stations
from podpac import Coordinates, clinspace


@pytest.mark.integration
class TestCOSMOS(object):
    def test_common_coordinates(self):
        point_interpolation = {
            "method": "nearest",
            "params": {"use_selector": False, "remove_nan": True, "time_scale": "1,M", "respect_bounds": False},
        }
        cosmos = podpac.datalib.cosmos_stations.COSMOSStations()
        cosmos_raw = podpac.datalib.cosmos_stations.COSMOSStationsRaw()
        cosmos_filled = podpac.datalib.cosmos_stations.COSMOSStations(interpolation=point_interpolation)
        for ck, c in COORDINATES.items():
            if ck != "cosmos_region":
                continue
            print("Evaluating: ", ck)
            o_f = cosmos_filled.eval(c)
            assert np.any(np.isfinite(o_f.data))
            o = cosmos.eval(c)
            o_r = cosmos.eval(c)
            if "soilscape" in ck:
                assert np.any(np.isnan(o.data))
                assert np.any(np.isnan(o_r.data))
                continue
            assert np.any(np.isfinite(o.data))
            assert np.any(np.isfinite(o_r.data))
