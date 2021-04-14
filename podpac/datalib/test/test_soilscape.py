import numpy as np
import pytest

from .coordinates_for_tests import COORDINATES
import podpac.datalib


@pytest.mark.integration
class TestSoilscape(object):
    def test_common_coordinates(self):
        point_interpolation = {
            "method": "nearest",
            "params": {"use_selector": False, "remove_nan": True, "time_scale": "1,M", "respect_bounds": False},
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

    def test_site_raw(self):
        sm = podpac.datalib.soilscape.SoilSCAPE20minRaw(site="Canton_OK", data_key="soil_moisture")
        coords_source = sm.make_coordinates(time=sm.sources[0].coordinates["time"][:5])
        coords_interp_time = sm.make_coordinates(time="2016-01-01")
        coords_interp_alt = sm.make_coordinates(time=sm.sources[0].coordinates["time"][:5], depth=5)
        o1 = sm.eval(coords_source)
        o2 = sm.eval(coords_interp_time)
        o3 = sm.eval(coords_interp_alt)

    def test_site_interpolated(self):
        sm = podpac.datalib.soilscape.SoilSCAPE20min(site="Canton_OK", data_key="soil_moisture")
        coords_source = sm.make_coordinates(time=sm.sources[0].coordinates["time"][:5])
        coords_interp_time = sm.make_coordinates(time="2016-01-01")
        coords_interp_alt = sm.make_coordinates(time=sm.sources[0].coordinates["time"][:5], depth=5)
        o1 = sm.eval(coords_source)
        o2 = sm.eval(coords_interp_time)
        o3 = sm.eval(coords_interp_alt)
