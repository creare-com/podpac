from __future__ import division, unicode_literals, print_function, absolute_import

import datetime

import traitlets as tl
import numpy as np

from lazy_import import lazy_module

s3fs = lazy_module("s3fs")

# Internal imports
from podpac.data import DataSource, Rasterio
from podpac.coordinates import Coordinates, merge_dims
from podpac.utils import cached_property, DiskCacheMixin
from podpac.core.authentication import S3Mixin

BUCKET = "noaa-gfs-pds"


class GFSSource(DiskCacheMixin, Rasterio):
    parameter = tl.Unicode().tag(attr=True)
    level = tl.Unicode().tag(attr=True)
    date = tl.Unicode().tag(attr=True)
    hour = tl.Unicode().tag(attr=True)
    forecast = tl.Unicode().tag(attr=True)

    @property
    def source(self):
        return "s3://%s/%s/%s/%s/%s/%s" % (BUCKET, self.parameter, self.level, self.date, self.hour, self.forecast)


# TODO time interpolation
class GFS(S3Mixin, DiskCacheMixin, DataSource):
    parameter = tl.Unicode().tag(attr=True)
    level = tl.Unicode().tag(attr=True)
    date = tl.Unicode().tag(attr=True)
    hour = tl.Unicode().tag(attr=True)

    cache_native_coordinates = tl.Bool(True)

    @property
    def prefix(self):
        return "%s/%s/%s/%s/%s/" % (BUCKET, self.parameter, self.level, self.date, self.hour)

    @cached_property(use_cache_ctrl=True)
    def forecasts(self):
        return [path.replace(self.prefix, "") for path in self.s3.find(self.prefix)]

    @cached_property
    def sources(self):
        params = {
            "parameter": self.parameter,
            "level": self.level,
            "date": self.date,
            "hour": self.hour,
            "cache_ctrl": self.cache_ctrl,
            "s3": self.s3,
        }
        return np.array([GFSSource(forecast=forecast, **params) for forecast in self.forecasts])

    def get_native_coordinates(self):
        nc = self.sources[0].native_coordinates
        base_time = datetime.datetime.strptime("%s %s" % (self.date, self.hour), "%Y%m%d %H%M")
        forecast_times = [base_time + datetime.timedelta(hours=int(h)) for h in self.forecasts]
        tc = Coordinates(
            [[dt.strftime("%Y-%m-%d %H:%M") for dt in forecast_times]], dims=["time"], crs=nc.crs, validate_crs=False
        )
        return merge_dims([nc, tc])

    def get_data(self, coordinates, coordinates_index):
        data = self.create_output_array(coordinates)
        for i, source in enumerate(self.sources[coordinates_index[2]]):
            data[:, :, i] = source.eval(coordinates.drop("time"))
        return data


def GFSLatest(parameter=None, level=None, **kwargs):
    # date
    date = datetime.datetime.now().strftime("%Y%m%d")

    # hour
    prefix = "%s/%s/%s/%s/" % (BUCKET, parameter, level, date)
    s3 = s3fs.S3FileSystem(anon=True)
    hours = set([path.replace(prefix, "")[:4] for path in s3.find(prefix)])
    if not hours:
        raise RuntimeError("No data found at '%s'" % prefix)
    hour = max(hours)

    # node
    return GFS(parameter=parameter, level=level, date=date, hour=hour, **kwargs)


if __name__ == "__main__":
    import datetime
    import podpac

    # switch to 'disk' cache to cache s3 data
    cache_ctrl = ["ram"]
    # cache_ctrl = ['ram', 'disk']

    parameter = "SOIM"
    level = "0-10 m DPTH"

    now = datetime.datetime.now()
    yesterday = now - datetime.timedelta(1)
    tomorrow = now + datetime.timedelta(1)

    # GFSSource (specify source date/time and forecast)
    print("GFSSource node (parameter, level, date, hour)")
    gfs_soim = GFSSource(
        parameter=parameter,
        level=level,
        date=yesterday.strftime("%Y%m%d"),
        hour="1200",
        forecast="003",
        cache_ctrl=cache_ctrl,
        anon=True,
    )

    o = gfs_soim.eval(gfs_soim.native_coordinates)
    print(o)

    # GFS (specify source date/time, select forecast at evaluation)
    print("GFS node (parameter, level, date, hour)")
    gfs_soim = GFS(
        parameter=parameter,
        level=level,
        date=yesterday.strftime("%Y%m%d"),
        hour="1200",
        cache_ctrl=cache_ctrl,
        anon=True,
    )

    # whole world forecast at this time tomorrow
    c = Coordinates(
        [gfs_soim.native_coordinates["lat"], gfs_soim.native_coordinates["lon"], tomorrow], dims=["lat", "lon", "time"]
    )
    o = gfs_soim.eval(c)
    print(o)

    # time series: get the forecast at lat=42, lon=275 every hour for the next 6 hours
    start = now
    stop = now + datetime.timedelta(hours=6)
    c = Coordinates([42, 282, podpac.crange(start, stop, "1,h")], dims=["lat", "lon", "time"])
    o = gfs_soim.eval(c)
    print(o)

    # latest (get latest source, select forecast at evaluation)
    print("GFSLatest node (parameter, level)")
    gfs_soim = GFSLatest(parameter=parameter, level=level, cache_ctrl=cache_ctrl, anon=True)
    c = Coordinates(
        [gfs_soim.native_coordinates["lat"], gfs_soim.native_coordinates["lon"], tomorrow], dims=["lat", "lon", "time"]
    )
    o = gfs_soim.eval(c)
    print(o)
