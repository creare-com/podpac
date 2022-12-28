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

import podpac
from podpac.core.data.rasterio_source import RasterioRaw
from podpac.compositor import TileCompositorRaw
from podpac.interpolators import InterpolationMixin
from podpac.interpolators import RasterioInterpolator, ScipyGrid, ScipyPoint
from podpac.utils import cached_property
from podpac.authentication import S3Mixin

####
# private module attributes
####

# create log for module
_logger = logging.getLogger(__name__)
ZOOM_SIZES = [
    8271.5169531233,
    39135.75848200978,
    19567.87924100587,
    9783.939620502935,
    4891.969810250487,
    2445.9849051252454,
    1222.9924525636013,
    611.4962262818025,
    305.7481131408976,
    152.8740565714275,
    76.43702828571375,
    38.218514142856876,
    19.109257072407146,
    9.554628536203573,
    4.777314268103609,
]


class TerrainTilesSourceRaw(RasterioRaw):
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

    anon = tl.Bool(True)

    @tl.default("crs")
    def _default_crs(self):
        if "geotiff" in self.source:
            return "EPSG:3857"
        if "terrarium" in self.source:
            return "EPSG:3857"
        if "normal" in self.source:
            return "EPSG:3857"

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
        joined_path = os.path.join(path, os.path.split(self.source)[0].replace("s3://", ""))  # path to file
        filepath = os.path.abspath(os.path.join(joined_path, filename))

        # make the directory if it hasn't been made already
        if not os.path.exists(joined_path):
            os.makedirs(joined_path)

        # download the file
        _logger.debug("Downloading terrain tile {} to filepath: {}".format(self.source, filepath))
        self.s3.get(self.source, filepath)

    # this is a little crazy, but I get floating point issues with indexing if i don't round to 7 decimal digits
    def get_coordinates(self):
        coordinates = super(TerrainTilesSourceRaw, self).get_coordinates()

        for dim in coordinates:
            coordinates[dim] = np.round(coordinates[dim].coordinates, 6)

        return coordinates


class TerrainTilesComposite(TileCompositorRaw):
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
        Zoom level of tiles, in [0, ..., 14]. Defaults to 7. A value of "-1" will automatically determine the zoom level.
        WARNING: When automatic zoom is used, evaluating points (stacked lat,lon) uses the maximum zoom level (level 14)
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
    zoom = tl.Int(default_value=-1).tag(attr=True)
    tile_format = tl.Enum(["geotiff", "terrarium", "normal"], default_value="geotiff").tag(attr=True)
    bucket = tl.Unicode(default_value="elevation-tiles-prod").tag(attr=True)
    sources = []  # these are loaded as needed
    urls = tl.List(trait=tl.Unicode()).tag(attr=True)  # Maps directly to sources
    dims = ["lat", "lon"]
    anon = tl.Bool(True)

    def _zoom(self, coordinates):
        if self.zoom >= 0:
            return self.zoom
        crds = coordinates.transform("EPSG:3857")
        if coordinates.is_stacked("lat") or coordinates.is_stacked("lon"):
            return len(ZOOM_SIZES) - 1
        steps = []
        for crd in crds.values():
            if crd.name not in ["lat", "lon"]:
                continue
            if crd.size == 1:
                continue
            if isinstance(crd, podpac.coordinates.UniformCoordinates1d):
                steps.append(np.abs(crd.step))
            elif isinstance(crd, podpac.coordinates.ArrayCoordinates1d):
                steps.append(np.abs(np.diff(crd.coordinates)).min())
            else:
                continue
        if not steps:
            return len(ZOOM_SIZES) - 1

        step = min(steps) / 2
        zoom = 0
        for z, zs in enumerate(ZOOM_SIZES):
            zoom = z
            if zs < step:
                break
        return zoom

    def select_sources(self, coordinates, _selector=None):
        # get all the tile sources for the requested zoom level and coordinates
        sources = get_tile_urls(self.tile_format, self._zoom(coordinates), coordinates)
        urls = ["s3://{}/{}".format(self.bucket, s) for s in sources]

        # create TerrainTilesSourceRaw classes for each url source
        self.sources = self._create_composite(urls)
        if self.trait_is_defined("interpolation") and self.interpolation is not None:
            for s in self.sources:
                if s.has_trait("interpolation"):
                    s.set_trait("interpolation", self.interpolation)
        return self.sources

    def find_coordinates(self):
        return [podpac.coordinates.union([source.coordinates for source in self.sources])]

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
            for source in self.sources[0].sources:
                source.download(path)
        except tl.TraitError as e:
            raise ValueError("No terrain tile sources selected. Evaluate node at coordinates to select sources.") from e

    def _create_composite(self, urls):
        # Share the s3 connection
        sample_source = TerrainTilesSourceRaw(
            source=urls[0],
            cache_ctrl=self.cache_ctrl,
            force_eval=self.force_eval,
            cache_output=self.cache_output,
            cache_dataset=True,
        )
        return [
            TerrainTilesSourceRaw(
                source=url,
                s3=sample_source.s3,
                cache_ctrl=self.cache_ctrl,
                force_eval=self.force_eval,
                cache_output=self.cache_output,
                cache_dataset=True,
            )
            for url in urls
        ]


class TerrainTiles(InterpolationMixin, TerrainTilesComposite):
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

    pass


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

        if "lat" not in coordinates.udims or "lon" not in coordinates.udims:
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
    x, y = x1, np.log(np.tan(0.25 * np.pi + 0.5 * y1) + 1e-32)

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

    tiles = 2**zoom
    diameter = 2 * np.pi
    x = int(tiles * (xm + np.pi) / diameter)
    y = int(tiles * (np.pi - ym) / diameter)

    return x, y


if __name__ == "__main__":
    from podpac import Coordinates, clinspace

    c = Coordinates([clinspace(40, 43, 1000), clinspace(-76, -72, 1000)], dims=["lat", "lon"])
    c2 = Coordinates(
        [clinspace(40, 43, 1000), clinspace(-76, -72, 1000), ["2018-01-01", "2018-01-02"]], dims=["lat", "lon", "time"]
    )

    print("TerrainTiles")
    node = TerrainTiles(tile_format="geotiff", zoom=8)
    output = node.eval(c)
    print(output)

    output = node.eval(c2)
    print(output)

    print("TerrainTiles cached")
    node = TerrainTiles(tile_format="geotiff", zoom=8, cache_ctrl=["ram", "disk"])
    output = node.eval(c)
    print(output)

    # tile urls
    print("get tile urls")
    print(np.array(get_tile_urls("geotiff", 1)))
    print(np.array(get_tile_urls("geotiff", 9, coordinates=c)))

    print("done")
