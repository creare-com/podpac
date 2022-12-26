"""
Satellite data access using sat-utils (https://github.com/sat-utils) developed by Development Seed

Supports access to:

- Landsat 8 on AWS OpenData: https://registry.opendata.aws/landsat-8/
- Sentinel 2
"""

import logging
import datetime
import os

import numpy as np
import traitlets as tl
from lazy_import import lazy_module

satsearch = lazy_module("satsearch")

# Internal dependencies
import podpac
from podpac.compositor import TileCompositor
from podpac.core.data.rasterio_source import RasterioRaw
from podpac.core.units import UnitsDataArray
from podpac.authentication import S3Mixin
from podpac import settings

_logger = logging.getLogger(__name__)


def _get_asset_info(item, name):
    """for forwards/backwards compatibility, convert B0x to/from Bx as needed"""

    if name in item.assets:
        return item.assets[name]
    elif name.replace("B", "B0") in item.assets:
        # Bx -> B0x
        return item.assets[name.replace("B", "B0")]
    elif name.replace("B0", "B") in item.assets:
        # B0x -> Bx
        return item.assets[name.replace("B0", "B")]
    else:
        available = [key for key in item.assets.keys() if key not in ["thumbnail", "overview", "info", "metadata"]]
        raise KeyError("asset '%s' not found. Available assets: %s" % (name, avaialable))


def _get_s3_url(item, asset_name):
    """convert to s3:// urls
    href: https://landsat-pds.s3.us-west-2.amazonaws.com/c1/L8/034/033/LC08_L1TP_034033_20201209_20201218_01_T1/LC08_L1TP_034033_20201209_20201218_01_T1_B2.TIF
    url:  s3://landsat-pds/c1/L8/034/033/LC08_L1TP_034033_20201209_20201218_01_T1/LC08_L1TP_034033_20201209_20201218_01_T1_B2.TIF
    """

    info = _get_asset_info(item, asset_name)

    if info["href"].startswith("s3://"):
        return info["href"]

    elif info["href"].startswith("https://"):
        root, key = info["href"][8:].split("/", 1)
        bucket = root.split(".")[0]
        return "s3://%s/%s" % (bucket, key)

    else:
        raise ValueError("Could not parse satutils asset href '%s'" % info["href"])


class SatUtilsSource(RasterioRaw):
    date = tl.Unicode(help="item.properties.datetime from sat-utils item").tag(attr=True)

    def get_coordinates(self):
        # get spatial coordinates from rasterio over s3
        spatial_coordinates = super(SatUtilsSource, self).get_coordinates()
        time = podpac.Coordinates([self.date], dims=["time"], crs=spatial_coordinates.crs)
        return podpac.coordinates.merge_dims([spatial_coordinates, time])


class SatUtils(S3Mixin, TileCompositor):
    """
    PODPAC DataSource node to access the data using sat-utils developed by Development Seed
    See https://github.com/sat-utils

    See :class:`podpac.compositor.OrderedCompositor` for more information.

    Parameters
    ----------
    collection : str, optional
        Specifies the collection for satellite data.
        Options include "landsat-8-l1", "sentinel-2-l1c".
        Defaults to all collections.
    query : dict, optional
        Dictionary of properties to query on, supports eq, lt, gt, lte, gte
        Passed through to the sat-search module.
        See https://github.com/sat-utils/sat-search/blob/master/tutorial-1.ipynb
        Defaults to None
    asset : str, optional
        Asset to download from the satellite image.
        The asset must be a band name or a common extension name, see https://github.com/radiantearth/stac-spec/tree/master/extensions/eo
        See also the Assets section of this tutorial: https://github.com/sat-utils/sat-stac/blob/master/tutorial-2.ipynb
        Defaults to "B3" (green)
    min_bounds_span : dict, optional
        Default is {}. When specified, gives the minimum bounds that will be used for a coordinate in the query, so
        it works properly. If a user specified a lat, lon point, the query may fail since the min/max values for
        lat/lon are the same. When specified, these bounds will be padded by the following for latitude (as an example):
        [lat - min_bounds_span['lat'] / 2, lat + min_bounds_span['lat'] / 2]
    """

    stac_api_url = tl.Unicode().tag(attr=True)
    collection = tl.Unicode(default_value=None, allow_none=True).tag(attr=True)
    asset = tl.Unicode().tag(attr=True)
    query = tl.Dict(default_value=None, allow_none=True).tag(attr=True)
    anon = tl.Bool(default_value=False).tag(attr=True)
    min_bounds_span = tl.Dict(allow_none=True).tag(attr=True)

    @tl.default("interpolation")
    def _default_interpolation(self):
        # this default interpolation enables NN interpolation without having to expand past the bounds of the query
        # we're relying on satutils to give us the nearest neighboring tile here.
        return {"method": "nearest", "params": {"respect_bounds": False}}

    @tl.default("stac_api_url")
    def _get_stac_api_url_from_env(self):
        if "STAC_API_URL" not in os.environ:
            raise TypeError(
                "STAC endpoint required. Please define the SatUtils 'stac_api_url' or 'STAC_API_URL' environmental variable"
            )

        return os.environ

    def select_sources(self, coordinates, _selector=None):
        result = self.search(coordinates)

        if result.found() == 0:
            _logger.warning(
                "Sat Utils did not find any items for collection {}. Ensure that sat-stac is installed, or try with a different set of coordinates (self.search(coordinates)).".format(
                    self.collection
                )
            )
            return []

        return [
            SatUtilsSource(source=_get_s3_url(item, self.asset), date=item.properties["datetime"], anon=self.anon)
            for item in result.items()
        ]

    def search(self, coordinates):
        """
        Query data from sat-utils interface within PODPAC coordinates

        Parameters
        ----------
        coordinates : :class:`podpac.Coordinates`
            PODPAC coordinates specifying spatial and temporal bounds

        Raises
        ------
        ValueError
            Error raised when no spatial or temporal bounds are provided

        Returns
        -------
        search : :class:`satsearch.search.Search`
            Results form sat-search
        """

        # Ensure Coordinates are in decimal lat-lon
        coordinates = coordinates.transform("epsg:4326")

        time_bounds = None
        if "time" in coordinates.udims:
            time_bounds = [
                str(np.datetime64(bound, "s"))
                for bound in coordinates["time"].bounds
                if isinstance(bound, np.datetime64)
            ]
            if len(time_bounds) < 2:
                raise ValueError("Time coordinates must be of type np.datetime64")

            if self.min_bounds_span != None and "time" in self.min_bounds_span:
                time_span, time_unit = self.min_bounds_span["time"].split(",")
                time_delta = np.timedelta64(int(time_span), time_unit)
                time_bounds_dt = [np.datetime64(tb) for tb in time_bounds]
                timediff = np.diff(time_bounds_dt)
                if timediff < time_delta:
                    pad = (time_delta - timediff) / 2
                    time_bounds = [str((time_bounds_dt[0] - pad)[0]), str((time_bounds_dt[1] + pad)[0])]

        bbox = None
        if "lat" in coordinates.udims or "lon" in coordinates.udims:
            lat = coordinates["lat"].bounds
            lon = coordinates["lon"].bounds
            if (self.min_bounds_span != None) and ("lat" in self.min_bounds_span) and ("lon" in self.min_bounds_span):
                latdiff = np.diff(lat)
                londiff = np.diff(lon)
                if latdiff < self.min_bounds_span["lat"]:
                    pad = ((self.min_bounds_span["lat"] - latdiff) / 2)[0]
                    lat = [lat[0] - pad, lat[1] + pad]

                if londiff < self.min_bounds_span["lon"]:
                    pad = ((self.min_bounds_span["lon"] - londiff) / 2)[0]
                    lon = [lon[0] - pad, lon[1] + pad]

            bbox = [lon[0], lat[0], lon[1], lat[1]]

        # TODO: do we actually want to limit an open query?
        if time_bounds is None and bbox is None:
            raise ValueError("No time or spatial coordinates requested")

        # search dict
        search_kwargs = {}

        search_kwargs["url"] = self.stac_api_url

        if time_bounds is not None:
            search_kwargs["datetime"] = "{start_time}/{end_time}".format(
                start_time=time_bounds[0], end_time=time_bounds[1]
            )

        if bbox is not None:
            search_kwargs["bbox"] = bbox

        if self.query is not None:
            search_kwargs["query"] = self.query
        else:
            search_kwargs["query"] = {}

        if self.collection is not None:
            search_kwargs["collections"] = [self.collection]

        # search with sat-search
        _logger.debug("sat-search searching with {}".format(search_kwargs))
        search = satsearch.Search(**search_kwargs)
        _logger.debug("sat-search found {} items".format(search.found()))

        return search


class Landsat8(SatUtils):
    """
    Landsat 8 on AWS OpenData
    https://registry.opendata.aws/landsat-8/

    Leverages sat-utils (https://github.com/sat-utils) developed by Development Seed

    Parameters
    ----------
    asset : str, optional
        Asset to download from the satellite image.
        For Landsat8, this includes: 'B01','B02','B03','B04','B05','B06','B07','B08','B09','B10','B11','B12'
        The asset must be a band name or a common extension name, see https://github.com/radiantearth/stac-spec/tree/master/extensions/eo
        See also the Assets section of this tutorial: https://github.com/sat-utils/sat-stac/blob/master/tutorial-2.ipynb
    query : dict, optional
        Dictionary of properties to query on, supports eq, lt, gt, lte, gte
        Passed through to the sat-search module.
        See https://github.com/sat-utils/sat-search/blob/master/tutorial-1.ipynb
        Defaults to None
    min_bounds_span : dict, optional
        Default is {}. When specified, gives the minimum bounds that will be used for a coordinate in the query, so
        it works properly. If a user specified a lat, lon point, the query may fail since the min/max values for
        lat/lon are the same. When specified, these bounds will be padded by the following for latitude (as an example):
        [lat - min_bounds_span['lat'] / 2, lat + min_bounds_span['lat'] / 2]
    """

    collection = "landsat-8-l1-c1"
    anon = True


class Sentinel2(SatUtils):
    """
    Sentinel 2 on AWS OpenData
    https://registry.opendata.aws/sentinel-2/

    Leverages sat-utils (https://github.com/sat-utils) developed by Development Seed.

    Note this data source requires the requester to pay, so you must set podpac settings["AWS_REQUESTER_PAYS"] = True

    Parameters
    ----------
    asset : str, optional
        Asset to download from the satellite image.
        For Sentinel2, this includes: 'tki','B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12
        The asset must be a band name or a common extension name, see https://github.com/radiantearth/stac-spec/tree/master/extensions/eo
        See also the Assets section of this tutorial: https://github.com/sat-utils/sat-stac/blob/master/tutorial-2.ipynb
    query : dict, optional
        Dictionary of properties to query on, supports eq, lt, gt, lte, gte
        Passed through to the sat-search module.
        See https://github.com/sat-utils/sat-search/blob/master/tutorial-1.ipynb
        Defaults to None
    min_bounds_span : dict, optional
        Default is {}. When specified, gives the minimum bounds that will be used for a coordinate in the query, so
        it works properly. If a user specified a lat, lon point, the query may fail since the min/max values for
        lat/lon are the same. When specified, these bounds will be padded by the following for latitude (as an example):
        [lat - min_bounds_span['lat'] / 2, lat + min_bounds_span['lat'] / 2]
    """

    collection = "sentinel-s2-l1c"
