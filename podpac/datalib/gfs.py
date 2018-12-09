
from __future__ import division, unicode_literals, print_function, absolute_import

import logging
import datetime

import traitlets as tl
import numpy as np
import boto3
import botocore
import rasterio

from podpac.data import DataSource, Rasterio
from podpac.coordinates import Coordinates, merge_dims

BUCKET = 'noaa-gfs-pds'

s3 = boto3.resource('s3')
s3.meta.client.meta.events.register('choose-signer.s3.*', botocore.handlers.disable_signing)
bucket = s3.Bucket(BUCKET)

# TODO add time to native_coordinates
class GFSSource(Rasterio):
    parameter = tl.Unicode(readonly=True).tag(attr=True)
    level = tl.Unicode(readonly=True).tag(attr=True)
    date = tl.Unicode(readonly=True).tag(attr=True)
    hour = tl.Unicode(readonly=True).tag(attr=True)
    forecast = tl.Unicode(readonly=True).tag(attr=True)
    dataset = tl.Any(readonly=True)

    def init(self):
        self._logger = logging.getLogger(__name__)
        
        # check if the key exists
        try:
            s3.Object(BUCKET, self._key).load()
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                raise ValueError("Not found: '%s'" % self._key) # TODO list options
            else:
                raise
    
    @property
    def _key(self):
        return '%s/%s/%s/%s/%s' % (self.parameter, self.level, self.date, self.hour, self.forecast)

    @property
    def source(self):
        return self._key

    @tl.default('dataset')
    def open_dataset(self):
        """Opens the data source"""

        cache_key = 'fileobj'
        with rasterio.Env(), rasterio.MemoryFile() as f:
            if self.cache_ctrl and self.has_cache(key=cache_key):
                data = self.get_cache(key=cache_key)
                f.write(data)
            else:
                self._logger.info('Downloading S3 fileobj (Bucket: %s, Key: %s)' % (BUCKET, self._key))
                s3.Object(BUCKET, self._key).download_fileobj(f)
                f.seek(0)
                self.cache_ctrl and self.put_cache(f.read(), key=cache_key)
            
            dataset = f.open()

        return dataset

    def get_native_coordinates(self):
        # TODO the lat is still -0.125 to 359.875 instead of -180 to 180
        c = super(GFSSource, self).get_native_coordinates()
        return Coordinates([c['lat'][::-1], c['lon']])

class GFS(DataSource):
    parameter = tl.Unicode(readonly=True).tag(attr=True)
    level = tl.Unicode(readonly=True).tag(attr=True)
    date = tl.Unicode(readonly=True).tag(attr=True)
    hour = tl.Unicode(readonly=True).tag(attr=True)

    @property
    def source(self):
        return '%s/%s/%s/%s' % (self.parameter, self.level, self.date, self.hour)

    def init(self):
        # TODO check prefix and the options at the next level
        
        self._prefix = '%s/%s/%s/%s/' % (self.parameter, self.level, self.date, self.hour)
        self.forecasts = [obj.key.replace(self._prefix, '') for obj in bucket.objects.filter(Prefix=self._prefix)]
        
        if not self.forecasts:
            raise ValueError("Not found: '%s/*'" % self._prefix)

        params = {'parameter': self.parameter, 'level': self.level, 'date': self.date, 'hour': self.hour, 'cache_ctrl': self.cache_ctrl}
        self._sources = np.array([GFSSource(forecast=h, **params) for h in self.forecasts]) # can we load this lazily?

        nc = self._sources[0].native_coordinates
        base_time = datetime.datetime.strptime('%s %s' % (self.date, self.hour), '%Y%m%d %H%M')
        forecast_times = [base_time + datetime.timedelta(hours=int(h)) for h in self.forecasts]
        tc = Coordinates([[dt.strftime('%Y-%m-%d %H:%M') for dt in forecast_times]], dims=['time'])
        self.native_coordinates = merge_dims([nc, tc])

    def get_data(self, coordinates, coordinates_index):
        data = self.create_output_array(coordinates)
        for i, source in enumerate(self._sources[coordinates_index[2]]):
            source.eval(coordinates.drop('time'), output=data[:, :, i])
        return data

class GFSLatest(GFS):
    # TODO raise exception if date or hour is in init args

    def init(self):
        now = datetime.datetime.now()
        
        # date
        self.date = now.strftime('%Y%m%d')

        # hour
        prefix = '%s/%s/%s/' % (self.parameter, self.level, self.date)
        objs = bucket.objects.filter(Prefix=prefix)
        hours = set(obj.key.split('/')[3] for obj in objs)
        if hours:
            self.hour = max(hours)

        super(GFSLatest, self).init()
        