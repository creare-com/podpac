
from __future__ import division, unicode_literals, print_function, absolute_import

import logging

import traitlets as tl
import numpy as np
import boto3
import botocore
import rasterio

from podpac.data import Rasterio
from podpac.compositor import Compositor

BUCKET = 'noaa-gfs-pds'

s3 = boto3.resource('s3')
s3.meta.client.meta.events.register('choose-signer.s3.*', botocore.handlers.disable_signing)

# TODO add time to native_coordinates
class GFSSource(Rasterio):
    parameter = tl.Unicode().tag(attr=True)
    level = tl.Unicode().tag(attr=True)
    date = tl.Unicode().tag(attr=True)
    hour = tl.Unicode().tag(attr=True)
    forecast = tl.Unicode().tag(attr=True)
    dataset = tl.Any()

    def init(self):
        self._logger = logging.getLogger(__name__)
        
        # check if the key exists
        try:
            s3.Object(BUCKET, self._key).load()
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                raise ValueError(self._key) # TODO list options
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
        with rasterio.MemoryFile() as f:
            if self.has_cache(key=cache_key):
                data = self.get_cache(key=cache_key)
                f.write(data)
            else:
                self._logger.info('Downloading S3 fileobj (Bucket: %s, Key: %s)' % (BUCKET, self._key))
                s3.Object(BUCKET, self._key).download_fileobj(f)
                f.seek(0)
                self.put_cache(f.read(), key=cache_key)
            
            dataset = f.open()

        return dataset

class GFS(Compositor):
    parameter = tl.Unicode().tag(attr=True)
    level = tl.Unicode().tag(attr=True)
    date = tl.Unicode().tag(attr=True)
    hour = tl.Unicode().tag(attr=True)

    @property
    def sources(self):
        params = {
            'parameter': self.parameter, 
            'level': self.level, 
            'date': self.date, 
            'hour': self.hour,
            'cache_ctrl': self.cache_ctrl
        }

        bucket = s3.Bucket(BUCKET)
        prefix = '%s/%s/%s/%s/' % (self.parameter, self.level, self.date, self.hour)
        objs = bucket.objects.filter(Prefix='%s/%s/%s/%s/003' % (self.parameter, self.level, self.date, self.hour))
        sources = [GFSSource(forecast=obj.key.replace(prefix, ''), **params) for obj in objs]
        return np.array(sources)