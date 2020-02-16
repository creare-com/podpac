from __future__ import division, unicode_literals, print_function, absolute_import

import logging
import datetime

import traitlets as tl
import numpy as np

# Helper utility for optional imports
from lazy_import import lazy_module

# Optional Imports
rasterio = lazy_module("rasterio")
boto3 = lazy_module("boto3")
botocore = lazy_module("botocore")

# Internal imports
from podpac.data import DataSource, Rasterio
from podpac.coordinates import Coordinates, merge_dims

BUCKET = "noaa-gfs-pds"

s3 = boto3.resource("s3")
s3.meta.client.meta.events.register("choose-signer.s3.*", botocore.handlers.disable_signing)
bucket = s3.Bucket(BUCKET)

# TODO add time to native_coordinates
class GFSSource(Rasterio):
    parameter = tl.Unicode().tag(attr=True)
    level = tl.Unicode().tag(attr=True)
    date = tl.Unicode().tag(attr=True)
    hour = tl.Unicode().tag(attr=True)
    forecast = tl.Unicode().tag(attr=True)

    @property
    def source(self):
        return "%s/%s/%s/%s/%s" % (self.parameter, self.level, self.date, self.hour, self.forecast)


class GFS(DataSource):
    parameter = tl.Unicode().tag(attr=True)
    level = tl.Unicode().tag(attr=True)
    date = tl.Unicode().tag(attr=True)
    hour = tl.Unicode().tag(attr=True)

    @property
    def sources(self):
        params = {
            "parameter": self.parameter,
            "level": self.level,
            "date": self.date,
            "hour": self.hour,
            "cache_ctrl": self.cache_ctrl,
        }
        self._sources = np.array([GFSSource(forecast=h, **params) for h in self.forecasts])  # can we load this lazily?

    # def init(self):
    #     # TODO check prefix and the options at the next level

    #     prefix = "%s/%s/%s/%s/" % (self.parameter, self.level, self.date, self.hour)
    #     forecasts = [obj.key.replace(prefix, "") for obj in bucket.objects.filter(Prefix=prefix)]
    #     if not forecasts:
    #         raise ValueError("Not found: '%s/*'" % prefix)

    @property
    def native_coordinates(self):
        nc = self._sources[0].native_coordinates
        base_time = datetime.datetime.strptime("%s %s" % (self.date, self.hour), "%Y%m%d %H%M")
        forecast_times = [base_time + datetime.timedelta(hours=int(h)) for h in self.forecasts]
        tc = Coordinates([[dt.strftime("%Y-%m-%d %H:%M") for dt in forecast_times]], dims=["time"])
        return merge_dims([nc, tc])

    def get_data(self, coordinates, coordinates_index):
        data = self.create_output_array(coordinates)
        for i, source in enumerate(self._sources[coordinates_index[2]]):
            data[:, :, i] = source.eval(coordinates.drop("time"))
        return data


def GFSLatest(parameter=None, level=None, **kwargs):
    # date
    date = datetime.datetime.now().strftime("%Y%m%d")

    # hour
    prefix = "%s/%s/%s/" % (self.parameter, self.level, self.date)
    objs = bucket.objects.filter(Prefix=prefix)
    hours = set(obj.key.split("/")[3] for obj in objs)
    if not hours:
        raise RuntimeError("TODO")
    hour = max(hours)

    # node
    return GFS(parameter=parameter, level=level, data=date, hour=hour, **kwargs)
