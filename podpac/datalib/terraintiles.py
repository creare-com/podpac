"""
Terrain Tiles

Hosted on AWS S3
https://registry.opendata.aws/terrain-tiles/

Description
    Gridded elevation tiles
Resource type
    S3 Bucket
Amazon Resource Name (ARN)
    arn:aws:s3:::elevation-tiles-prod
AWS Region
    us-east-1


Documentation: https://mapzen.com/documentation/terrain-tiles/
"""


import os
import re
from datetime import datetime
import logging
import io

import traitlets as tl
import boto3
from botocore.handlers import disable_signing
import numpy as np

from podpac import settings
from podpac.data import Rasterio
from podpac.compositor import OrderedCompositor
from podpac.coordinates import crange, Coordinates


####
# module attributes
####
BUCKET = 'elevation-tiles-prod'

####
# private module attributes
####

# create log for module
_log = logging.getLogger(__name__)

# regex for finding files
_radar_re = re.compile(r'^\d{4}/\d{2}/\d{2}/(....)/')
_scan_re = re.compile(r'^\d{4}/\d{2}/\d{2}/..../(?:(?=(.*.gz))|(?=(.*V0*.gz))|(?=(.*V0*)))')

# s3 handling
_s3 = boto3.resource('s3')
_s3.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)  # allows no password
_bucket_name = BUCKET
_bucket = _s3.Bucket(_bucket_name)


class TerrainTilesSource(Rasterio):
    """DataSource to handle individual TerrainTiles raster files
    
    Parameters
    ----------
    source : str
        Filename of the sourcefile
    
    Attributes
    ----------
    prefix : str
        prefix to the filename (:attr:`source`) within the S3 bucket
    """

    source = tl.Unicode()
    process_in = tl.Enum(['cache'], default_value='cache')

    def init(self):
        pass

    @tl.default('dataset')
    def open_dataset(self):
        self._download_file()   # download the file to ram or cache the first time it is accessed
        super(TerrainTilesSource, self).open_dataset()

    # def get_data(self, coordinates, coordinates_index):
        # super(TerrainTilesSource, self).get_data(coordinates, coordinates_index)

    def _download_file(self):
        """Download/load file from s3
        """

        # download into memory
        # NOT IMPLEMENTED YET
        # https://boto3.readthedocs.io/en/latest/reference/services/s3.html#S3.Client.download_fileobj
        if self.process_in == 'ram':
            _log.debug('Downloading terrain tile {} to ram'.format(self.source))
            
            data = io.BytesIO()
            _bucket.download_fileobj(self.source, data)
            return data 

        # download file to cache directory
        else:
            filename = os.path.split(self.source)[1]  # get filename off of source
            filename_safe = filename.replace('\\', '').replace(':', '').replace('/', '')  # sanitize filename

            cache_path = os.path.join(settings['CACHE_DIR'], 'terraintiles')
            cache_filepath = os.path.join(cache_path, filename_safe)  # path to file in cache

            # make the cach directory if it hasn't been made already
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)

            # don't re-download the same file
            if not os.path.exists(cache_filepath):
                _log.debug('Downloading terrain tile {} to cache'.format(cache_filepath))
                _bucket.download_file(self.source, cache_filepath)
            
            with open(filename_safe, 'rb') as data:
                return data


class Nexrad(OrderedCompositor):
    """Nexrad data interface (https://docs.opendata.aws/noaa-nexrad/readme.html)

    The Next Generation Weather Radar (NEXRAD) is a network of 160 high-resolution
    Doppler radar sites that detects precipitation and atmospheric movement and
    disseminates data in approximately 5 minute intervals from each site. NEXRAD enables
    severe storm prediction and is used by researchers and commercial enterprises
    to study and address the impact of weather across multiple sectors.
    
    https://www.ncdc.noaa.gov/data-access/radar-data/radar-display-tools

    Attributes
    ----------
    radar_sites : list
        list of radar names to use. if empty, get all
    """
    
    # inputs
    field = tl.Unicode(default_value='relectivity')       # must be one of fields available in the pyart Radar class
                                # http://arm-doe.github.io/pyart-docs-travis/user_reference/generated/pyart.core.Radar.html#pyart-core-radar
    stations = tl.List(default_value=['FOP1'])           # list avail radar with :meth:`get_radars()`
    process_in = tl.Unicode(default_value='cache')          # 'ram' or 'cache'


    @tl.default('sources')
    def _default_sources(self):
        """SMAPDateFolder objects pointing to specific SMAP folders

        Returns
        -------
        np.ndarray of :class:`NexradSource`
            Array of :class:`NexradSource` instances
        """
        # dates = self.get_available_times_dates()[1]
        # src_objs = np.array([
        #     SMAPDateFolder(product=self.product, version=self.version, folder_date=date,
        #                    shared_coordinates=self.shared_coordinates,
        #                    auth_session=self.auth_session,
        #                    layerkey=self.layerkey)
        #     for date in dates])
        # return src_objs
        np.ndarray([])


    def select_sources(self, coordinates):
        pass

    def get_shared_coordinates(self):
        pass

    def get_source_coordinates(self):
        pass

    def find_coordinates(self):
        pass



############
# Utilities
############


def get_radars(coordinates):
    """Query for radars within podpac coordinate bounds

    This method allows you to get the available radars in a given time and spatial area.
    Note that this is fairly inefficient querying in time.
    It queries for radars across time by listing the contents of the S3 bucket on each day within
    the time coordinates.
    Large time periods will take a long time to query.
    
    Parameters
    ----------
    coordinates : :class:`podpac.coordinates.Coordinates`
        Find available radars within coordinates
    
    Returns
    -------
    list
        list of radars available within the coordinate bounds
    """
    _log.debug('Getting radars for coordinates')

    radars = [key for key in NEXRAD_LOCATIONS]

    if 'lat' in coordinates:
        bounds = coordinates['lat'].bounds
        radars = [r for r in radars if \
                 (NEXRAD_LOCATIONS[r]['lat'] >= bounds[0] and NEXRAD_LOCATIONS[r]['lat'] <= bounds[1])]

    if 'lon' in coordinates:
        bounds = coordinates['lon'].bounds
        radars = [r for r in radars if \
                 (NEXRAD_LOCATIONS[r]['lon'] >= bounds[0] and NEXRAD_LOCATIONS[r]['lon'] <= bounds[1])]

                 
    if 'time' in coordinates:
        bounds = coordinates['time'].bounds
        _log.debug('Getting radars from s3 bucket between {} and {}'.format(bounds[0], bounds[1]))

        # get each day in the list
        radars_in_time = []
        # TODO: handle when bounds[0] == bounds[1] (same day)
        for dt in np.arange(bounds[0], bounds[1], dtype='datetime64[D]'):
            prefix = _build_prefix(dt)

            # ask for all the radars on this day
            resp = _bucket.meta.client.list_objects(Bucket=BUCKET,
                                                         Prefix=prefix,
                                                         Delimiter='/')

            for entry in resp['CommonPrefixes']:
                match = _radar_re.match(entry['Prefix'])
                if match is not None:
                    radar = match.group(1)

                    # if the radar is not in the pyart list, then immediately append it to master list
                    # this means that it is a new radar and we don't support geoqueries yet
                    if radar not in NEXRAD_LOCATIONS and radar not in radars:
                        radars.append(radar)
                    else:
                        radars_in_time.append(radar)

        radars = list(set(radars) & set(radars_in_time))

    return radars


def _build_prefix(dt, station=None):
    """Build URL prefix
    
    Parameters
    ----------
    dt : :class:`np.datetime64`, datetime
        Datetime to use to build prefix
    station : None, optional
        station id to use for prefix
    
    Returns
    -------
    str
        Bucket URL
    """
    if isinstance(dt, np.datetime64):
        dt = dt.astype(datetime)

    prefix = '{:04}/{:02}/{:02}/'.format(dt.year, dt.month, dt.day)

    if station is not None:
        prefix += '{}/'.format(station.upper())

    return prefix
