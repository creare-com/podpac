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

import podpac
from podpac.compositor import TileCompositor
from podpac.data import Rasterio
from podpac.authentication import S3Mixin
from podpac.core.coordinates import ArrayCoordinates1d

_logger = logging.getLogger(__name__)


def _get_asset_info(item, name):
    """ for forwards/backwards compatibility, convert B0x to/from Bx as needed """

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
    """ convert to s3:// urls
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


class SatUtilsSource(Rasterio):
    date = tl.Any()

    def get_coordinates(self):
        # get spatial coordinates from rasterio over s3
        spatial_coordinates = super(SatUtilsSource, self).get_coordinates()
        time = podpac.Coordinates([self.date], dims=["time"], crs=spatial_coordinates.crs)
        return podpac.coordinates.merge_dims([spatial_coordinates, time])


from podpac.core.compositor.tile_compositor import TileCompositorRaw


class SatUtils(S3Mixin, TileCompositorRaw):
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

    stac_api_url = tl.Unicode("https://earth-search.aws.element84.com/v0").tag(attr=True)
    collection = tl.Unicode(default_value=None, allow_none=True).tag(attr=True)
    asset = tl.Unicode().tag(attr=True)
    query = tl.Dict(default_value=None, allow_none=True).tag(attr=True)

    time_tol = tl.Instance(np.timedelta64, default_value=np.timedelta64(2, "D"), allow_none=True)

    @tl.validate("time_tol")
    def validate_time_tol(self, d):
        if d["value"] is None:
            return None

        try:
            value = get_timedelta(d["value"])
        except:
            raise TypeError("Invalid time_tol '%s' of type %s" % (d[value], type(d["value"])))

        if value < 0:
            raise ValueError("Invalid time_tol '%s', time_tol cannot be negative" % value)

        return value

    def select_sources(self, coordinates, _selector=None):
        items = self._search(coordinates)
        sources = [SatUtilsSource(source=_get_s3_url(item, self.asset), date=item.datetime) for item in items]
        import pdb

        pdb.set_trace()  # breakpoint 2b5e462b //

        return sources

    # def _eval(self, coordinates, output=None, _selector=None):
    #     tiles = self._search(coordinates)

    #     native_time_coordinates = coordinates.copy()
    #     native_time_coordinates['time'] = ArrayCoordinates1d(sorted([tile.datetime for tile in items]))
    #     import pdb; pdb.set_trace()  # breakpoint c31cf643 //

    #     return super(self)._eval(native_time_coordinates, output=output, _selector=_selector)

    def _search(self, coordinates):
        search_kwargs = {}

        search_kwargs["url"] = self.stac_api_url
        search_kwargs["collections"] = [self.collection]

        if self.query is not None:
            search_kwargs["query"] = self.query

        # bbox
        # TODO spatial tolerance
        coordinates = coordinates.transform("EPSG:4326")
        lat = coordinates["lat"].bounds
        lon = coordinates["lon"].bounds
        search_kwargs["bbox"] = [lon[0], lat[0], lon[1], lat[1]]

        # time bounds
        # TODO time tolerance
        lo, hi = coordinates["time"].bounds
        search_kwargs["datetime"] = "%s/%s" % (lo, hi)

        # search
        items = satsearch.Search(**search_kwargs).items()

        # # outer time intersection
        # dates = np.unique([item.datetime for item in items]) # also sorted
        # lo = max(dates < coordinates['time'].bounds[0])
        # hi = min(dates > coordinates['time'].bounds[1])
        # return [item for item in items if lo <= item.datetime <= hi]

        return items

    def get_times(self, coordinates=None):
        if "lat" not in coordinates.udims or "lon" not in coordinates.udims:
            raise ValueError("coordinates must contain spatial dimensions (lat and lon)")
        coordinates = coordinates.transform("EPSG:4326")
        lat = coordinates["lat"].bounds
        lon = coordinates["lon"].bounds
        bbox = [lon[0], lat[0], lon[1], lat[1]]

        search_kwargs = {}
        search_kwargs["url"] = self.stac_api_url
        search_kwargs["collections"] = [self.collection]
        if self.query is not None:
            search_kwargs["query"] = self.query
        search_kwargs["bbox"] = bbox

        result = satsearch.Search(**search_kwargs)
        return podpac.Coordinates([[item.datetime for item in result.items()]], dims=[["time"]])


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
