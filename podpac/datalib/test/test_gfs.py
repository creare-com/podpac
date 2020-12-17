import datetime

import numpy as np
import pytest

import podpac


@pytest.mark.integration
class TestGFS(object):
    parameter = "SOIM"
    level = "0-10 m DPTH"

    def test_source(self):
        now = datetime.datetime.now()
        yesterday = now - datetime.timedelta(1)

        # specify source date/time and forecast
        gfs_soim = podpac.datalib.gfs.GFSSourceRaw(
            parameter=self.parameter,
            level=self.level,
            date=yesterday.strftime("%Y%m%d"),
            hour="1200",
            forecast="003",
            anon=True,
        )

        o = gfs_soim.eval(gfs_soim.coordinates)

    def test_composited(self):
        now = datetime.datetime.now()
        yesterday = now - datetime.timedelta(1)
        tomorrow = now + datetime.timedelta(1)

        # specify source date/time, select forecast at evaluation
        gfs_soim = podpac.datalib.gfs.GFS(
            parameter=self.parameter, level=self.level, date=yesterday.strftime("%Y%m%d"), hour="1200", anon=True
        )

        # whole world forecast at this time tomorrow
        coords = gfs_soim.sources[0].coordinates
        c = podpac.Coordinates([coords["lat"], coords["lon"], tomorrow], dims=["lat", "lon", "time"])
        o = gfs_soim.eval(c)

        # time series: get the forecast at lat=42, lon=275 every hour for the next 6 hours
        start = now
        stop = now + datetime.timedelta(hours=6)
        c = podpac.Coordinates([42, 282, podpac.crange(start, stop, "1,h")], dims=["lat", "lon", "time"])
        o = gfs_soim.eval(c)

    def test_latest(self):
        now = datetime.datetime.now()
        tomorrow = now + datetime.timedelta(1)

        # get latest source, select forecast at evaluation
        gfs_soim = podpac.datalib.gfs.GFSLatest(parameter=self.parameter, level=self.level, anon=True)

        # latest whole world forecast at this time tomorrow
        coords = gfs_soim.sources[0].coordinates
        c = podpac.Coordinates([coords["lat"], coords["lon"], tomorrow], dims=["lat", "lon", "time"])
        o = gfs_soim.eval(c)
