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
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


from podpac import settings
from podpac.data import Rasterio
from podpac.compositor import OrderedCompositor
from podpac.interpolators import Rasterio as RasterioInterpolator


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
    process_in = tl.Enum(['ram', 'cache'], default_value='cache')  # Note: 'ram' is not yet supported

    # attributes
    dataset = tl.Any()

    @tl.default('dataset')
    def _open_dataset(self):
        self.source = self._download_file()   # download the file to cache the first time it is accessed
        
        # TODO: this is a temporary solution to reproject coordinates to WGS84
        # reproject dataset to 'EPSG:4326'
        with rasterio.open(self.source) as src:
            self.source = self._reproject(src, {'init': 'epsg:4326'})
        
        # opens source file
        return super(TerrainTilesSource, self)._open_dataset()

    def get_data(self, coordinates, coordinates_index):
        data = super(TerrainTilesSource, self).get_data(coordinates, coordinates_index)
        data.data[data.data < 0] = np.nan
        return data


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

    def _reproject(self, src, dst_crs):
        # https://rasterio.readthedocs.io/en/latest/topics/reproject.html#reprojecting-a-geotiff-dataset
         
        # calculate default transform
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        # write out new file with new projection
        dst_filename = '{}.wgs8484.tif'.format(self.source.replace('.tif', ''))
        with rasterio.open(dst_filename, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs)

        # return filename
        return dst_filename


class TerrainTiles(OrderedCompositor):
    """Terrain Tiles gridded elevation tiles data library

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
    
    Parameters
    ----------
    process_in : ['ram', 'cache']
        Where to process the file from S3 bucket. Defaults to 'cache'.
        Note: 'ram' option is not yet supported
    zoom : int
        Zoom level of tiles. Defaults to 7.
    tile_format : str
        one of :attr:`TILE_FORMATS`

    Attributes
    ----------
    source : str
        compositor source identifier for cache

    """
    
    # parameters
    process_in = tl.Enum(['ram', 'cache'], default_value='cache')  # Note: 'ram' is not yet supported
    zoom = tl.Int(default_value=4)
    tile_format = tl.Enum(TILE_FORMATS, default_value='geotiff')

    # attributes
    source = BUCKET

    # TODO: This is how I believe this should be implemented, but it feels very inefficient
    # @tl.default('sources')
    # def _default_sources(self):
    #     """Default sources are all the tiles for the requested tile_format and zoom level

    #     Returns
    #     -------
    #     :class:`np.ndarray` of :class:`TerrainTileSource`
    #         TerrainTilesSource for each tile at the :attr:`zoom` level
    #     """

    #     # get all the tile definitions for the requested zoom level
    #     tiles = _get_tile_tuples(self.zoom)

    #     # get source urls
    #     sources = [self._tile_url(self.tile_format, self.zoom, x, y) for (x, y, zoom) in tiles]

    #     # create TerrainTilesSource classes for each url source
    #     return np.array([TerrainTilesSource(source=source, process_in=self.process_in) \
    #                        for source in sources])


    def select_sources(self, coordinates):
        # get all the tile sources for the requested zoom level and coordinates
        sources = get_tile_urls(self.tile_format, self.zoom, coordinates)
        
        # create TerrainTilesSource classes for each url source
        self.sources = np.array([self._create_source(source) for source in sources])

        return self.sources


    def get_source_coordinates(self):
        """ this would require us to download all raster files """
        pass

    def _create_source(self, source):
        return TerrainTilesSource(source=source, process_in=self.process_in)

############
# Utilities
############


def get_zoom_levels(tile_format='geotiff'):
    """Get available zoom levels for certain tile formats
    
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

def get_tile_urls(tile_format, zoom, coordinates=None):
    """Get tile urls for a specific zoom level and geospatial coordinates
    
    Parameters
    ----------
    tile_format : str
        format of the tile to get
    zoom : int
        zoom level
    coordinates : :class:`podpac.Coordinates`, optional
        only return tiles within coordinates
    
    Returns
    -------
    list of str
        list of tile urls
    """

    # get all the tile definitions for the requested zoom level
    tiles = _get_tile_tuples(zoom, coordinates)
    
    # get source urls
    return [_tile_url(tile_format, x, y, z) for (x, y, z) in tiles]


############
# Private Utilites
############

def _get_tile_tuples(zoom, coordinates=None):
    """Query for tiles within podpac coordinates
    
    This method allows you to get the available tiles in a given spatial area.
    This will work for all :attr:`TILE_FORMAT` types
    
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
        list of tile tuples (x, y, zoom) for zoom level and coordinates
    """

    # if no coordinates are supplied, get all tiles for zoom level
    if coordinates is None:
        # clip lat to +/- 85.051129 because that's all that spherical mercator
        tiles = _get_tiles_grid([-85.051129, 85.051129], [-180, 180], zoom)

    # down select tiles based on coordinates
    else:

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

def _tile_url(tile_format, x, y, zoom):
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
    y : int
        x tilespace coordinate
    
    Returns
    -------
    str
        Bucket prefix
    
    Raises
    ------
    TypeError
    """

    tile_url = '{tile_format}/{zoom}/{x}/{y}.{ext}'
    ext = {
        'geotiff': 'tif',
        'normal': 'png',
        'terrarium': 'png'
    }

    return tile_url.format(tile_format=tile_format, zoom=zoom, x=x, y=y, ext=ext[tile_format])

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
