import datetime

import numpy as np
import pytest

import podpac

STAC_API_URL = "https://earth-search.aws.element84.com/v0"


@pytest.mark.integration
class TestLandsat8(object):
    def test_landsat8(self):
        lat = [39.5, 40.5]
        lon = [-110, -105]
        time = ["2020-12-09", "2020-12-10"]
        c = podpac.Coordinates([lat, lon, time], dims=["lat", "lon", "time"])

        node = podpac.datalib.satutils.Landsat8(
            stac_api_url=STAC_API_URL,
            asset="B01",
        )
        output = node.eval(c)
        assert np.isfinite(output).sum() > 0


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
            node = podpac.datalib.satutils.Sentinel2(
                stac_api_url=STAC_API_URL, asset="B01", aws_region_name="eu-central-1"
            )
            output = node.eval(c)
            assert np.isfinite(output).sum() > 0
