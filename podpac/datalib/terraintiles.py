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

Attribution
-----------
- Some source adapted from https://github.com/tilezen/joerd

Attributes
----------
BUCKET : str
    AWS S3 bucket
TILE_FORMATS : list
    list of support tile formats

Notes
-----

See https://github.com/racemap/elevation-service/blob/master/tileset.js
for example skadi implementation
"""


import os
import re
from itertools import product
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
TILE_FORMATS = ['terrarium', 'normal', 'geotiff']  # TODO: Support skadi format

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
    source : str, :class:`io.BytesIO`
        Filename of the sourcefile, or bytes of the files
    process_in : ['ram', 'cache']
        Where to process the file from S3 bucket. Defaults to 'cache'.
        Note: 'ram' option is not yet supported
    
    Attributes
    ----------
    dataset : TYPE
        Description
    
    Deleted Attributes
    ------------------
    prefix : str
        prefix to the filename (:attr:`source`) within the S3 bucket
    """

    # parameters
    source = tl.Union([tl.Unicode(), tl.Instance(io.BytesIO)])
    process_in = tl.Enum(['ram', 'cache'], default_value='cache')

    # attributes
    dataset = tl.Any()

    @tl.default('dataset')
    def _open_dataset(self):
        self.source = self._download_file()   # download the file to cache the first time it is accessed
        return super(TerrainTilesSource, self)._open_dataset()

    # def get_data(self, coordinates, coordinates_index):
        # super(TerrainTilesSource, self).get_data(coordinates, coordinates_index)

    def _download_file(self):
        """Download/load file from s3
        
        Returns
        -------
        str or :class:`io.BytesIO`
            full path to the downloaded tile in the cache, or bytes from BytesIO
        """

        # download into memory
        # NOT IMPLEMENTED YET
        # https://boto3.readthedocs.io/en/latest/reference/services/s3.html#S3.Client.download_fileobj
        if self.process_in == 'ram':
            _log.debug('Downloading terrain tile {} to ram'.format(self.source))
            
            dataset = io.BytesIO()
            _bucket.download_fileobj(self.source, dataset)
            return dataset

        # download file to cache directory
        else:
            filename = os.path.split(self.source)[1]  # get filename off of source
            filename_safe = filename.replace('\\', '').replace(':', '').replace('/', '')  # sanitize filename

            cache_path = os.path.join(settings['CACHE_DIR'], 'terraintiles', os.path.split(self.source)[0])
            cache_filepath = os.path.join(cache_path, filename_safe)  # path to file in cache

            # make the cach directory if it hasn't been made already
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)

            # don't re-download the same file
            if not os.path.exists(cache_filepath):
                _log.debug('Downloading terrain tile {} to cache'.format(cache_filepath))
                _bucket.download_file(self.source, cache_filepath)

            return cache_filepath



class TerrainTiles(OrderedCompositor):
    """TerrainTiles Compositor
    """
    
    # inputs
    process_in = tl.Enum(['ram', 'cache'], default_value='cache')

    @tl.default('sources')
    def _default_sources(self):
        """
        """
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


def get_zoom_levels(tile_format='geotiff'):
    """Get available zoom levels
    
    Parameters
    ----------
    tile_format : str, optional
        Tile format to query. Defaults to 'geotiff'
        Available formats: :attr:`TILE_FORMATS`
    
    Raises
    ------
    TypeError
    
    Returns
    -------
    list of int
        list of zoom levels
    """

    # check format (`skadi` format not supported)
    if tile_format not in TILE_FORMATS:
        raise TypeError("format must be one of {}".format(TILE_FORMATS))

    zoom_re = re.compile(r'^.*\/(\d*)\/')
    prefix = '{}/'.format(tile_format)

    # get list of objects in bucket
    resp = _bucket.meta.client.list_objects(Bucket=BUCKET, Prefix=prefix, Delimiter='/')

    zoom_levels = []
    for entry in resp['CommonPrefixes']:
        match = zoom_re.match(entry['Prefix'])
        if match is not None:
            zoom_levels.append(int(match.group(1)))

    # sort from low to high
    zoom_levels.sort()

    return zoom_levels

def get_tiles(coordinates, zoom):
    """Query for tiles within podpac coordinate bounds
    
    This method allows you to get the available tiles in a given spatial area.
    
    Parameters
    ----------
    coordinates : :class:`podpac.coordinates.Coordinates`
        Find available tiles within coordinates
    zoom : int, optional
        zoom level
    
    Raises
    ------
    TypeError
        Description
    
    Returns
    -------
    list of tuple
        list of tile urls for coordinates and zoom level
    """
    _log.debug('Getting tiles for coordinates {}'.format(coordinates))

    if 'lat' not in coordinates or 'lon' not in coordinates:
        raise TypeError('input coordinates must have lat and lon dimensions to get tiles')

    # point coordinates
    if 'lat_lon' in coordinates.dims or 'lon_lat' in coordinates.dims:
        lat_lon = zip(coordinates['lat'].coordinates, coordinates['lon'].coordinates)

        tiles = []
        for (lat, lon) in lat_lon:
            tile = _get_tiles_point(lat, lon, zoom)
            if tile not in tiles:
                tiles.append(tile)

    # gridded coordinates
    else:
        lat_bounds = coordinates['lat'].bounds
        lon_bounds = coordinates['lon'].bounds

        tiles = _get_tiles_grid(lat_bounds, lon_bounds, zoom)

    return tiles


############
# Private Utilites
############

def _get_tiles_grid(lat_bounds, lon_bounds, zoom):
    """
    Convert geographic bounds into a list of tile coordinates at given zoom.
    Adapted from https://github.com/tilezen/joerd
    
    Parameters
    ----------
    lat_bounds : :class:`np.array` of float
        [min, max] bounds from lat coordinates
    lon_bounds : :class:`np.array` of float
        [min, max] bounds from lon coordinates
    zoom : int
        zoom level
    
    Returns
    -------
    list of tuple
        list of tuples (x, y, zoom) describing the tiles to cover coordinates
    """

    # convert to mercator
    xm_min, ym_min = _mercator(lat_bounds[1], lon_bounds[0])
    xm_max, ym_max = _mercator(lat_bounds[0], lon_bounds[1])

    # convert to tile-space bounding box
    xmin, ymin = _mercator_to_tilespace(xm_min, ym_min, zoom)
    xmax, ymax = _mercator_to_tilespace(xm_max, ym_max, zoom)

    # generate a list of tiles
    xs = range(xmin, xmax+1)
    ys = range(ymin, ymax+1)

    tiles = [(x, y, zoom) for (y, x) in product(ys, xs)]

    return tiles

def _get_tiles_point(lat, lon, zoom):
    """Get tiles at a single point and zoom level
    
    Parameters
    ----------
    lat : float
        latitude
    lon : float
        longitude
    zoom : int
        zoom level
    
    Returns
    -------
    tuple
        (x, y, zoom) tile url
    """
    xm, ym = _mercator(lat, lon)
    x, y = _mercator_to_tilespace(xm, ym, zoom)

    return x, y, zoom

def _mercator(lat, lon):
    """Convert latitude, longitude to x, y mercator coordinate at given zoom
    Adapted from https://github.com/tilezen/joerd

    Parameters
    ----------
    lat : float
        latitude
    lon : float
        longitude
    
    Returns
    -------
    tuple
        (x, y) float mercator coordinates
    """
    # convert to radians
    x1, y1 = lon * np.pi/180, lat * np.pi/180

    # project to mercator
    x, y = x1, np.log(np.tan(0.25 * np.pi + 0.5 * y1))

    return x, y

def _mercator_to_tilespace(xm, ym, zoom):
    """Convert mercator to tilespace coordinates
    
    Parameters
    ----------
    x : float
        mercator x coordinate
    y : float
        mercator y coordinate
    zoom : int
        zoom level
    
    Returns
    -------
    tuple
        (x, y) int tile coordinates
    """

    tiles = 2 ** zoom
    diameter = 2 * np.pi
    x = int(tiles * (xm + np.pi) / diameter)
    y = int(tiles * (np.pi - ym) / diameter)

    return x, y

def _build_prefix(tile_format, zoom=None, x=None):
    """Build S3 URL prefix
    
    The S3 bucket is organized {tile_format}/{z}/{x}/{y}.tif
    
    Parameters
    ----------
    tile_format : str
        One of 'terrarium', 'normal', 'geotiff'
    zoom : int
        zoom level
    x : int
        x tilespace coordinate
    
    Returns
    -------
    str
        Bucket prefix
    
    Raises
    ------
    TypeError
    """

    if tile_format not in TILE_FORMATS:
        raise TypeError("format must be one of {}".format(TILE_FORMATS))

    prefix = '{}/'.format(tile_format)
    if zoom is not None:
        prefix += '{}/'.format(zoom)

    if x is not None:
        prefix += '{}/'.format(x)

    return prefix
