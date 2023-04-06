from __future__ import division, unicode_literals, print_function, absolute_import

import datetime

import traitlets as tl
import numpy as np

from lazy_import import lazy_module

s3fs = lazy_module("s3fs")

# Internal imports
from podpac.core.data.rasterio_source import Rasterio
from podpac.core.authentication import S3Mixin
from podpac.coordinates import Coordinates
from podpac.utils import cached_property, DiskCacheMixin
from podpac.compositor import TileCompositor

BUCKET = "noaa-gfs-pds"


class GFSSourceRaw(DiskCacheMixin, Rasterio):
    """Raw GFS data from S3

    Attributes
    ----------
    parameter : str
        parameter, e.g. 'SOIM'.
    level : str
        depth, e.g. "0-10 m DPTH"
    date : str
        source date in '%Y%m%d' format, e.g. '20200130'
    hour : str
        source hour, e.g. '1200'
    forecast : str
        forecast time in hours from the source date and hour, e.g. '003'
    """

    parameter = tl.Unicode().tag(attr=True)
    level = tl.Unicode().tag(attr=True)
    date = tl.Unicode().tag(attr=True)
    hour = tl.Unicode().tag(attr=True)
    forecast = tl.Unicode().tag(attr=True)

    @property
    def source(self):
        return "s3://%s/%s/%s/%s/%s/%s" % (BUCKET, self.parameter, self.level, self.date, self.hour, self.forecast)


class GFS(S3Mixin, DiskCacheMixin, TileCompositor):
    """Composited and interpolated GFS data from S3

    Attributes
    ----------
    parameter : str
        parameter, e.g. 'SOIM'.
    level : str
        source depth, e.g. "0-10 m DPTH"
    date : str
        source date in '%Y%m%d' format, e.g. '20200130'
    hour : str
        source hour, e.g. '1200'
    """

    parameter = tl.Unicode().tag(attr=True, required=True)
    level = tl.Unicode().tag(attr=True, required=True)
    date = tl.Unicode().tag(attr=True, required=True)
    hour = tl.Unicode().tag(attr=True, required=True)

    @property
    def _repr_keys(self):
        return ["parameter", "level", "date", "hour"] + super()._repr_keys

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
        }
        return np.array([GFSSourceRaw(forecast=forecast, **params) for forecast in self.forecasts])

    @cached_property
    def source_coordinates(self):
        base_time = datetime.datetime.strptime("%s %s" % (self.date, self.hour), "%Y%m%d %H%M")
        forecast_times = [base_time + datetime.timedelta(hours=int(h)) for h in self.forecasts]
        return Coordinates(
            [[dt.strftime("%Y-%m-%d %H:%M") for dt in forecast_times]], dims=["time"], validate_crs=False
        )


def GFSLatest(parameter=None, level=None, **kwargs):
    """
    The latest composited and interpolated GFS data from S3

    Arguments
    ---------
    parameter : str
        parameter, e.g. 'SOIM'.
    level : str
        source depth, e.g. "0-10 m DPTH"

    Returns
    -------
    node : GFS
        GFS node with the latest forecast data available for the given parameter and level.
    """

    s3 = s3fs.S3FileSystem(anon=True)

    # get latest date
    prefix = "%s/%s/%s/" % (BUCKET, parameter, level)
    dates = [path.replace(prefix, "") for path in s3.ls(prefix)]
    if not dates:
        raise RuntimeError("No data found at '%s'" % prefix)
    date = max(dates)

    # get latest hour
    prefix = "%s/%s/%s/%s/" % (BUCKET, parameter, level, date)
    hours = [path.replace(prefix, "") for path in s3.ls(prefix)]
    if not hours:
        raise RuntimeError("No data found at '%s'" % prefix)
    hour = max(hours)

    # node
    return GFS(parameter=parameter, level=level, date=date, hour=hour, **kwargs)
