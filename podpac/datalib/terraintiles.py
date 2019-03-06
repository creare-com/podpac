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

import traitlets as tl
import numpy as np

from podpac.data import Rasterio
from podpac.compositor import OrderedCompositor
from podpac.interpolators import Rasterio as RasterioInterpolator, ScipyGrid, ScipyPoint
from podpac.data import interpolation_trait
from lazy_import import lazy_module


# optional imports
boto3 = lazy_module('boto3')
botocore = lazy_module('botocore')
rasterio = lazy_module('rasterio')

####
# module attributes
####
BUCKET = 'elevation-tiles-prod'
TILE_FORMATS = ['terrarium', 'normal', 'geotiff']  # TODO: Support skadi format

####
# private module attributes
####

# create log for module
_logger = logging.getLogger(__name__)

# s3 handling
_s3 = boto3.resource('s3')
_s3.meta.client.meta.events.register('choose-signer.s3.*', botocore.handlers.disable_signing)  # allows no password
_bucket_name = BUCKET
_bucket = _s3.Bucket(_bucket_name)


class TerrainTilesSource(Rasterio):
    """DataSource to handle individual TerrainTiles raster files
    
    Parameters
    ----------
    source : str
        Path to the sourcefile on S3
    
    Attributes
    ----------
    dataset : :class:`rasterio.io.DatasetReader`
        rasterio dataset
    """

    # parameters
    source = tl.Unicode()

    # attributes
    dataset = tl.Any()
    interpolation = interpolation_trait(default_value={
        'method': 'nearest',
        'interpolators': [RasterioInterpolator, ScipyGrid, ScipyPoint]
    })

    @tl.default('dataset')
    def open_dataset(self):
        """Opens the data source"""

        cache_key = 'fileobj'
        with rasterio.MemoryFile() as f:
            if self.cache_ctrl and self.has_cache(key=cache_key):
                data = self.get_cache(key=cache_key)
                f.write(data)
            else:
                _logger.info('Downloading S3 fileobj (Bucket: %s, Key: %s)' % (BUCKET, self.source))
                _bucket.download_fileobj(self.source, f)
                f.seek(0)
                self.cache_ctrl and self.put_cache(f.read(), key=cache_key)
            f.seek(0)
            
            dataset = f.open()
            reprojected_dataset = self._reproject(dataset, {'init': 'epsg:4326'})  # reproject dataset into WGS84

        return reprojected_dataset

    def get_data(self, coordinates, coordinates_index):
        data = super(TerrainTilesSource, self).get_data(coordinates, coordinates_index)
        data.data[data.data < 0] = np.nan
        # data.data[data.data < 0] = np.nan  # TODO: handle really large values
        return data
    
    def download(self, path='terraintiles'):
        """
        Download the TerrainTile file from S3 to a local file.

        Parameters
        ----------
        path : str
            Subdirectory to put files. Defaults to 'terraintiles'.
            Within this directory, the tile files will retain the same directory structure as on S3.
        """

        filename = os.path.split(self.source)[1]  # get filename off of source
        joined_path = os.path.join(path, os.path.split(self.source)[0])  # path to file
        filepath = os.path.abspath(os.path.join(joined_path, filename))

        # make the directory if it hasn't been made already
        if not os.path.exists(joined_path):
            os.makedirs(joined_path)

        # download the file
        _logger.debug('Downloading terrain tile {} to filepath: {}'.format(self.source, filepath))
        _bucket.download_file(self.source, filepath)


    def _reproject(self, src_dataset, dst_crs):
        # https://rasterio.readthedocs.io/en/latest/topics/reproject.html#reprojecting-a-geotiff-dataset
         
        # calculate default transform
        transform, width, height = rasterio.warp.calculate_default_transform(src_dataset.crs,
                                                                             dst_crs,
                                                                             src_dataset.width,
                                                                             src_dataset.height,
                                                                             *src_dataset.bounds)

        kwargs = src_dataset.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        # write out new file with new projection
        with rasterio.MemoryFile() as f:
            with f.open(**kwargs) as dataset:
                for i in range(1, src_dataset.count + 1):
                    rasterio.warp.reproject(
                        source=rasterio.band(src_dataset, i),
                        destination=rasterio.band(dataset, i),
                        src_transform=src_dataset.transform,
                        src_crs=src_dataset.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs)
            return f.open()

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
    zoom : int
        Zoom level of tiles. Defaults to 6.
    tile_format : str
        One of :attr:`TILE_FORMATS`. Defaults to 'geotiff'

    """
    
    # parameters
    zoom = tl.Int(default_value=6).tag(attr=True)
    tile_format = tl.Enum(TILE_FORMATS, default_value='geotiff').tag(attr=True)

    @property
    def source(self):
        """
        S3 Bucket source of TerrainTiles

        Returns
        -------
        str
        """
        return BUCKET

    def select_sources(self, coordinates):
        # get all the tile sources for the requested zoom level and coordinates
        sources = get_tile_urls(self.tile_format, self.zoom, coordinates)
        
        # create TerrainTilesSource classes for each url source
        self.sources = np.array([self._create_source(source) for source in sources])

        return self.sources

    def download(self, path='terraintiles'):
        """
        Download active terrain tile source files to local directory

        Parameters
        ----------
        path : str
            Subdirectory to put files. Defaults to 'terraintiles'.
            Within this directory, the tile files will retain the same directory structure as on S3.
        """

        try:
            for source in self.sources:
                source.download(path)
        except tl.TraitError as e:
            raise ValueError('No terrain tile sources selected. Evaluate node at coordinates to select sources.')

    def _create_source(self, source):
        return TerrainTilesSource(source=source)



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

        _logger.debug('Getting tiles for coordinates {}'.format(coordinates))

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
