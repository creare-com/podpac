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

    def init(self):
        self._logger = logging.getLogger(__name__)

        # check if the key exists
        try:
            s3.Object(BUCKET, self._key).load()
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise ValueError("Not found: '%s'" % self._key)  # TODO list options
            else:
                raise

    @property
    def _key(self):
        return "%s/%s/%s/%s/%s" % (self.parameter, self.level, self.date, self.hour, self.forecast)

    @tl.default("nan_vals")
    def _get_nan_vals(self):
        return [self.dataset.nodata]
        # return list(self.dataset.nodatavals) # which?

    @property
    def source(self):
        return self._key

    @tl.default("dataset")
    def open_dataset(self):
        """Opens the data source"""

        cache_key = "fileobj"
        with rasterio.MemoryFile() as f:
            if self.cache_ctrl and self.has_cache(key=cache_key):
                data = self.get_cache(key=cache_key)
                f.write(data)
            else:
                self._logger.info("Downloading S3 fileobj (Bucket: %s, Key: %s)" % (BUCKET, self._key))
                s3.Object(BUCKET, self._key).download_fileobj(f)
                f.seek(0)
                self.cache_ctrl and self.put_cache(f.read(), key=cache_key)
                f.seek(0)

            dataset = f.open()

        return dataset


class GFS(DataSource):
    parameter = tl.Unicode().tag(attr=True)
    level = tl.Unicode().tag(attr=True)
    date = tl.Unicode().tag(attr=True)
    hour = tl.Unicode().tag(attr=True)

    @property
    def source(self):
        return "%s/%s/%s/%s" % (self.parameter, self.level, self.date, self.hour)

    def init(self):
        # TODO check prefix and the options at the next level

        self._prefix = "%s/%s/%s/%s/" % (self.parameter, self.level, self.date, self.hour)
        self.forecasts = [obj.key.replace(self._prefix, "") for obj in bucket.objects.filter(Prefix=self._prefix)]

        if not self.forecasts:
            raise ValueError("Not found: '%s/*'" % self._prefix)

        params = {
            "parameter": self.parameter,
            "level": self.level,
            "date": self.date,
            "hour": self.hour,
            "cache_ctrl": self.cache_ctrl,
        }
        self._sources = np.array([GFSSource(forecast=h, **params) for h in self.forecasts])  # can we load this lazily?

        nc = self._sources[0].native_coordinates
        base_time = datetime.datetime.strptime("%s %s" % (self.date, self.hour), "%Y%m%d %H%M")
        forecast_times = [base_time + datetime.timedelta(hours=int(h)) for h in self.forecasts]
        tc = Coordinates([[dt.strftime("%Y-%m-%d %H:%M") for dt in forecast_times]], dims=["time"], crs=nc.crs)
        self.set_trait("native_coordinates", merge_dims([nc, tc]))

    def get_data(self, coordinates, coordinates_index):
        data = self.create_output_array(coordinates)
        for i, source in enumerate(self._sources[coordinates_index[2]]):
            data[:, :, i] = source.eval(coordinates.drop("time"))
        return data


class GFSLatest(GFS):
    # TODO raise exception if date or hour is in init args

    def init(self):
        now = datetime.datetime.now()

        # date
        self.set_trait("date", now.strftime("%Y%m%d"))

        # hour
        prefix = "%s/%s/%s/" % (self.parameter, self.level, self.date)
        objs = bucket.objects.filter(Prefix=prefix)
        hours = set(obj.key.split("/")[3] for obj in objs)
        if hours:
            self.set_trait("hour", max(hours))

        super(GFSLatest, self).init()
