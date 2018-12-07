
from __future__ import division, unicode_literals, print_function, absolute_import

import traitlets as tl

import boto3, botocore
from botocore.handlers import disable_signing
import rasterio

from podpac.data import Rasterio
from podpac.coordinates import Coordinates, clinspace

BUCKET = 'noaa-gfs-pds'

class GFSSource(Rasterio):
    parameter = tl.Unicode().tag(attr=True)
    level = tl.Unicode().tag(attr=True)
    date = tl.Unicode().tag(attr=True)
    hour = tl.Unicode().tag(attr=True)
    forecast = tl.Unicode().tag(attr=True)
    dataset = tl.Any()

    def init(self):
        self._s3 = boto3.resource('s3')
        self._s3.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
        # self._bucket = s3.Bucket(BUCKET)
        
        # check if the key exists
        try:
            self._s3.Object(BUCKET, self._key).load()
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                raise ValueError(self.key) # TODO list options
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

        with rasterio.MemoryFile() as f:
            self._s3.Object(BUCKET, self._key).download_fileobj(f)
            dataset = f.open()
            return dataset

if __name__ == '__main__':
    from matplotlib import pyplot
    c = Coordinates([clinspace(43, 42, 1000), clinspace(-73, -72, 1000)], dims=['lat', 'lon'])
    node = GFSSource(parameter='SOIM', level='10-40 m DPTH', date='20181206', hour='1200', forecast='006')

    output = node.eval(node.native_coordinates)

    output.plot()
    pyplot.show(False)