import datetime

import pytest
import s3fs

import podpac
from podpac.datalib import gfs


@pytest.mark.skip("Broken, GFS data source structure changed. ")
@pytest.mark.integration
class TestGFS(object):
    parameter = "SOIM"
    level = "0-10 m DPTH"

    @classmethod
    def setup_class(cls):
        # find an existing date
        s3 = s3fs.S3FileSystem(anon=True)
        prefix = "%s/%s/%s/" % (gfs.BUCKET, cls.parameter, cls.level)
        dates = [path.replace(prefix, "") for path in s3.ls(prefix)]
        cls.date = dates[0]

    def test_source(self):
        # specify source datetime and forecast
        gfs_soim = gfs.GFSSourceRaw(
            parameter=self.parameter,
            level=self.level,
            date=self.date,
            hour="1200",
            forecast="003",
            anon=True,
        )

        o = gfs_soim.eval(gfs_soim.coordinates)

    def test_composited(self):
        # specify source datetime, select forecast at evaluation from time coordinates
        gfs_soim = gfs.GFS(parameter=self.parameter, level=self.level, date=self.date, hour="1200", anon=True)

        # whole world forecast at 15:30
        forecast_time = datetime.datetime.strptime(self.date + " 15:30", "%Y%m%d %H:%M")
        coords = gfs_soim.sources[0].coordinates
        c = podpac.Coordinates([coords["lat"], coords["lon"], forecast_time], dims=["lat", "lon", "time"])
        o = gfs_soim.eval(c)

        # time series: get the forecast at lat=42, lon=275 every hour for 6 hours
        start = forecast_time
        stop = forecast_time + datetime.timedelta(hours=6)
        c = podpac.Coordinates([42, 282, podpac.crange(start, stop, "1,h")], dims=["lat", "lon", "time"])
        o = gfs_soim.eval(c)

    def test_latest(self):
        # get latest source, select forecast at evaluation
        gfs_soim = gfs.GFSLatest(parameter=self.parameter, level=self.level, anon=True)

        # latest whole world forecast
        forecast_time = datetime.datetime.strptime(gfs_soim.date + " " + gfs_soim.hour, "%Y%m%d %H%M")
        coords = gfs_soim.sources[0].coordinates
        c = podpac.Coordinates([coords["lat"], coords["lon"], forecast_time], dims=["lat", "lon", "time"])
        o = gfs_soim.eval(c)
