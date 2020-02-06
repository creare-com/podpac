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
- See required attribution when using terrain tiles:
  https://github.com/tilezen/joerd/blob/master/docs/attribution.md

Attributes
----------
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
from io import BytesIO

import traitlets as tl
import numpy as np

from podpac.data import Rasterio
from podpac.compositor import OrderedCompositor
from podpac.interpolators import Rasterio as RasterioInterpolator, ScipyGrid, ScipyPoint
from podpac.data import interpolation_trait

from lazy_import import lazy_module

# optional imports
s3fs = lazy_module("s3fs")
rasterio = lazy_module("rasterio")

####
# private module attributes
####

# create log for module
_logger = logging.getLogger(__name__)
_s3 = s3fs.S3FileSystem(anon=True)


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
    source = tl.Unicode().tag(readonly=True)

    # attributes
    interpolation = interpolation_trait(
        default_value={"method": "nearest", "interpolators": [RasterioInterpolator, ScipyGrid, ScipyPoint]}
    ).tag(readonly=True)

    @tl.default("crs")
    def _default_crs(self):
        if "geotiff" in self.source:
            return "EPSG:3857"
        if "terrarium" in self.source:
            return "EPSG:3857"
        if "normal" in self.source:
            return "EPSG:3857"

    @tl.default("dataset")
    def open_dataset(self):
        """Opens the data source"""

        cache_key = "fileobj"
        with rasterio.MemoryFile() as f:

            # load data from cache
            if self.cache_ctrl and self.has_cache(key=cache_key):
                _logger.debug("Retrieving terrain tile {} from cache'".format(self.source))
                data = self.get_cache(key=cache_key)
                f.write(data)

            else:

                # try finding local file first
                try:
                    with open(self.source, "rb") as localfile:
                        data = localfile.read()

                # download and put in cache
                except FileNotFoundError:
                    _logger.info("Downloading S3 fileobj: {}".format(self.source))
                    with _s3.open(self.source, "rb") as s3file:
                        data = s3file.read()

                # write to memory file
                f.write(data)

                # put data in the cache
                _logger.debug("Caching terrain tile {} in key 'fileobj'".format(self.source))
                self.cache_ctrl  # confirm this is initialized
                self.put_cache(data, key=cache_key)

            f.seek(0)
            dataset = f.open()

        return dataset

    def get_data(self, coordinates, coordinates_index):
        data = super(TerrainTilesSource, self).get_data(coordinates, coordinates_index)
        data.data[data.data < 0] = np.nan
        # data.data[data.data < 0] = np.nan  # TODO: handle really large values
        return data

    def download(self, path="terraintiles"):
        """
        Download the TerrainTile file from S3 to a local file.
        This is a convience method for users and not used by PODPAC machinery.

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
        _logger.debug("Downloading terrain tile {} to filepath: {}".format(self.source, filepath))
        _s3.get(self.source, filepath)


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
        One of ['geotiff', 'terrarium', 'normal']. Defaults to 'geotiff'
        PODPAC node can only evaluate 'geotiff' formats.
        Other tile_formats can be specified for :meth:`download`
        No support for 'skadi' formats at this time.
    bucket : str
        Bucket of the terrain tiles.
        Defaults to 'elevation-tiles-prod'
    """

    # parameters
    zoom = tl.Int(default_value=6).tag(attr=True)
    tile_format = tl.Enum(["geotiff", "terrarium", "normal"], default_value="geotiff").tag(attr=True)
    bucket = tl.Unicode(default_value="elevation-tiles-prod").tag(attr=True)

    @tl.default("sources")
    def _default_sources(self):
        return np.array([])

    @property
    def source(self):
        """
        S3 Bucket source of TerrainTiles

        Returns
        -------
        str
        """
        return self.bucket

    def select_sources(self, coordinates):
        # get all the tile sources for the requested zoom level and coordinates
        sources = get_tile_urls(self.tile_format, self.zoom, coordinates)

        # create TerrainTilesSource classes for each url source
        self.sources = np.array([self._create_source(source) for source in sources])

        return self.sources

    def download(self, path="terraintiles"):
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
            raise ValueError("No terrain tile sources selected. Evaluate node at coordinates to select sources.")

    def _create_source(self, source):
        return TerrainTilesSource(source="{}/{}".format(self.bucket, source))


############
# Utilities
############
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
        # get whole world
        tiles = _get_tiles_grid([-90, 90], [-180, 180], zoom)

    # down select tiles based on coordinates
    else:
        _logger.debug("Getting tiles for coordinates {}".format(coordinates))

        if "lat" not in coordinates or "lon" not in coordinates:
            raise TypeError("input coordinates must have lat and lon dimensions to get tiles")

        # transform to WGS84 (epsg:4326) to use the mapzen example for transforming coordinates to tilespace
        # it doesn't seem to conform to standard google tile indexing
        c = coordinates.transform("epsg:4326")

        # point coordinates
        if "lat_lon" in c.dims or "lon_lat" in c.dims:
            lat_lon = zip(c["lat"].coordinates, c["lon"].coordinates)

            tiles = []
            for (lat, lon) in lat_lon:
                tile = _get_tiles_point(lat, lon, zoom)
                if tile not in tiles:
                    tiles.append(tile)

        # gridded coordinates
        else:
            lat_bounds = c["lat"].bounds
            lon_bounds = c["lon"].bounds

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

    tile_url = "{tile_format}/{zoom}/{x}/{y}.{ext}"
    ext = {"geotiff": "tif", "normal": "png", "terrarium": "png"}

    return tile_url.format(tile_format=tile_format, zoom=zoom, x=x, y=y, ext=ext[tile_format])


def _get_tiles_grid(lat_bounds, lon_bounds, zoom):
    """
    Convert geographic bounds into a list of tile coordinates at given zoom.
    Adapted from https://github.com/tilezen/joerd
    
    Parameters
    ----------
    lat_bounds : :class:`np.array` of float
        [min, max] bounds from lat (y) coordinates
    lon_bounds : :class:`np.array` of float
        [min, max] bounds from lon (x) coordinates
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
    xs = range(xmin, xmax + 1)
    ys = range(ymin, ymax + 1)

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
    x1, y1 = lon * np.pi / 180, lat * np.pi / 180

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
