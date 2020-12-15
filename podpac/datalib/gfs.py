from __future__ import division, unicode_literals, print_function, absolute_import

import datetime

import traitlets as tl
import numpy as np

from lazy_import import lazy_module

s3fs = lazy_module("s3fs")

# Internal imports
from podpac.core.data.rasterio_source import RasterioBase
from podpac.core.compositor.data_compositor import InterpDataCompositor
from podpac.data import DataSource
from podpac.coordinates import Coordinates
from podpac.utils import cached_property, DiskCacheMixin
from podpac.core.authentication import S3Mixin

BUCKET = "noaa-gfs-pds"


class GFSSourceRaw(DiskCacheMixin, RasterioBase):
    """ Raw GFS data from S3

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


# TODO time interpolation
class GFS(S3Mixin, DiskCacheMixin, InterpDataCompositor):
    """ Composited and interpolated GFS data from S3

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

    parameter = tl.Unicode().tag(attr=True)
    level = tl.Unicode().tag(attr=True)
    date = tl.Unicode().tag(attr=True)
    hour = tl.Unicode().tag(attr=True)

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
