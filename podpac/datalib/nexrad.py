"""
NEXRAD Support

This module relies on the `pyart` module published by the DOE:
https://github.com/ARM-DOE/pyart

```bash
$ pip install arm_pyart
```

Some code adopted from from https://github.com/aarande/nexradaws
Attribution: Aaron Anderson
"""


import os
import re
from datetime import datetime
import logging

import traitlets as tl
import boto3
from botocore.handlers import disable_signing
import numpy as np

from podpac import settings
from podpac.data import DataSource
from podpac.compositor import OrderedCompositor
from podpac.coordinates import crange, Coordinates

# module requires pyart
try:
    import pyart
    from pyart.io.nexrad_common import NEXRAD_LOCATIONS
except ImportError as e:
    raise ImportError('The `nexrad` datalib relies on the `pyart` module (https://github.com/ARM-DOE/pyart).' + \
                      'Install pyart using pip: \n\n$ pip install arm_pyart')

####
# module attributes
####
BUCKET = 'noaa-nexrad-level2'

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
_bucket_name = 'noaa-nexrad-level2'
_bucket = _s3.Bucket(_bucket_name)


class NexradSource(DataSource):
    """DataSource to handle single Nexrad file
    
    Hosted on AWS S3 at https://noaa-nexrad-level2.s3.amazonaws.com/
    
    The NEXRAD Level II archive data is hosted in the noaa-nexrad-level2
    Amazon S3 bucket in the us-east-1 AWS region.
    The address for the public bucket is: https://noaa-nexrad-level2.s3.amazonaws.com.
    
    Each volume scan file of archival data is available as an object in Amazon S3.
    The basic data format is:
    
    ``/<Year>/<Month>/<Day>/<NEXRAD Station/>/<filename>``
    
    Where:
    
        ``<Year>`` is the year the data was collected
        ``<Month>`` is the month of the year the data was collected
        ``<Day>`` is the day of the month the data was collected
        ``<NEXRAD Station>`` is the NEXRAD ground station (map of ground stations)
        ``<filename>`` is the name of the file containing the data. These are compressed files (compressed with gzip).
        The file name has more precise timestamp information.
    
    All files in the archive use the same compressed format (.gz).
    The data file names are, for example, KAKQ20010101_080138.gz.
    The file naming convention is:
    
    ``GGGGYYYYMMDD_TTTTTT``
    
    Where:
    
        ``GGGG`` = Ground station ID (map of ground stations)
        ``YYYY`` = year
        ``MM`` = month
        ``DD`` = day
        ``TTTTTT`` = time when data started to be collected (GMT)
    
    Note that the 2015 files have an additional field on the file name.
    It adds “_V06” to the end of the file name. An example is KABX20150303_001050_V06.gz.
    
    The full historical archive from NOAA from June 1991 to present is available.
    
    See https://docs.opendata.aws/noaa-nexrad/readme.html for more information.
    
    Parameters
    ----------
    field : str
        radar field to parse. Defaults to 'reflectivity'.
        See ``fields`` attribute within pyart ``Radar`` class:
        http://arm-doe.github.io/pyart-docs-travis/user_reference/generated/pyart.core.Radar.html#pyart-core-radar
    process_in : ['ram', 'cache']
        Where to process the file from S3 bucket. Defaults to 'cache'.
        Note: 'ram' option is not currently working
    source : str
        Filename of the sourcefile
    
    Attributes
    ----------
    datetime : datetime
        datetime of the radar reading
    prefix : str
        prefix to the filename (:attr:`source`) within the S3 bucket
    radar : :class:`pyart.core.Radar`
        Radar class of the loaded archive file.
    station : str
        Radar station of the data source

    """

    source = tl.Unicode()
    field = tl.Unicode(default_value='reflectivity')
    process_in = tl.Enum(['ram', 'cache'], default_value='cache')
    
    station = tl.Unicode()      # set by the filename
    datetime = tl.Instance(datetime)
    prefix = tl.Unicode()       # prefix for the path to the filename
    radar = tl.Instance(pyart.core.Radar, allow_none=True)

    def init(self):
        self.station = self.source[0:4]
        self.datetime = datetime.strptime(self.source[4:19], '%Y%m%d_%H%M%S')
        self.prefix = _build_prefix(self.datetime, station=self.station)

    def get_data(self, coordinates, coordinates_index):

        # get file from s3 and load with pyart
        if self.radar is None:
            self.radar = self._load_archive()

        # get field from radar object
        data = self.radar.fields[self.field]['data']

        # set nan_vals with radar fill value
        self.nan_vals = [data.fill_value]

        # fill and stack data
        data = np.hstack(data.filled())

        # return filled data at coordinates_index
        # return data[coordinates_index]
        return data

    def get_native_coordinates(self):

        # TODO: convert this to regex
        dt = np.datetime64(datetime.strptime(self.source[4:19], '%Y%m%d_%H%M%S'))

        # get file from s3 and load with pyart
        if self.radar is None:
            self.radar = self._load_archive()

        # get lat/lon/alt coordinates
        lat = self.radar.gate_latitude['data']
        lon = self.radar.gate_longitude['data']
        # alt = self.radar.gate_altitude['data']  # TODO: add altitutde
        lat_lon = (np.hstack(lat), np.hstack(lon))

        return Coordinates([lat_lon, [dt]], dims=['lat_lon', 'time'])

    def _load_archive(self):
        """Download nexrad archive file to cache or ram and load into pyart.core.Radar class
        
        Returns
        -------
        pyart.core.Radar
            Radar class of the nexrad archive file
        """
        
        # filepath within the S3 bucket
        filepath = self.prefix + self.source

        # sanitize filename
        filename_safe = self.source.replace('\\', '').replace(':', '').replace('/', '')

        print(filepath)

        # download into memory
        # https://boto3.readthedocs.io/en/latest/reference/services/s3.html#S3.Client.download_fileobj
        if self.process_in == 'ram':
            _log.debug('Downloading nexrad radar file {} to ram'.format(filepath))
            
            with open(filename_safe, 'wb') as data:
                _bucket.download_fileobj(filepath, data)
                # method supports file-like objects (https://github.com/ARM-DOE/pyart/blob/master/pyart/io/common.py#L24)
                radar = pyart.io.read_nexrad_archive(data)
            return radar

        # download file to cache directory
        else:
            _log.debug('Downloading nexrad radar file {} to cache'.format(filepath))

            tmppath = os.path.join(settings['CACHE_DIR'], 'nexrad')
            if not os.path.exists(tmppath):
                os.makedirs(tmppath)

            tmpfilepath = os.path.join(tmppath, filename_safe)
            if not os.path.exists(tmpfilepath):
                _bucket.download_file(filepath, tmpfilepath)
            
            radar = pyart.io.read_nexrad_archive(tmpfilepath)
            return radar


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
