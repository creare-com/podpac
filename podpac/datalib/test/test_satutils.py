import datetime

import numpy as np
import pytest

import podpac

STAC_API_URL = "https://earth-search.aws.element84.com/v0"


@pytest.mark.integration
class TestLandsat8(object):
    def test_get_times(self):
        lat = [39.5, 40.0]
        lon = [65, 66]
        coords = podpac.Coordinates([lat, lon], dims=["lat", "lon"])

        node = podpac.datalib.satutils.Landsat8(asset="B01")
        times = node.get_times(coords)

        assert isinstance(times, podpac.Coordinates)
        assert times.dims == ("time",)

    def test_eval_with_source_times(self):
        lat = [39.5, 40.0]
        lon = [65, 66]
        spatial_coords = podpac.Coordinates([lat, lon], dims=["lat", "lon"])

        node = podpac.datalib.satutils.Landsat8(stac_api_url=STAC_API_URL, asset="B01")
        time_coords = node.get_times(spatial_coords)
        eval_coords = podpac.coordinates.merge_dims([spatial_coords, time_coords[:3]])
        output = node.eval(eval_coords)
        import pdb

        pdb.set_trace()  # breakpoint f4d8247e //

    def test_time_tol(self):
        pass


@pytest.mark.skip(reason="requester pays")
@pytest.mark.integration
class TestSentinel2(object):
    def test_sentinel2(self):
        lat = [39.5, 40.5]
        lon = [-110, -105]
        time = ["2020-12-09", "2020-12-10"]
        c = podpac.Coordinates([lat, lon, time], dims=["lat", "lon", "time"])

        with podpac.settings:
            podpac.settings["AWS_REQUESTER_PAYS"] = True
            node = podpac.datalib.satutils.Sentinel2(stac_api_url=STAC_API_URL, asset="B01")
            output = node.eval(c)
