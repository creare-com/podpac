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
from satsearch import Search

# Internal dependencies
import podpac
from podpac.compositor import OrderedCompositor
from podpac.data import Rasterio
from podpac.core.units import UnitsDataArray
from podpac.core.node import node_eval
from podpac.authentication import S3Mixin
from podpac import settings

_logger = logging.getLogger(__name__)


class SatUtilsSource(Rasterio):
    date = tl.Unicode(help="item.properties.datetime from sat-utils item").tag(attr=True)

    def get_coordinates(self):
        # get original coordinates
        spatial_coordinates = super(SatUtilsSource, self).get_coordinates()

        # lookup available dates and use pre-fetched lat and lon bounds. Make sure CRS is set to spatial coords
        time = podpac.Coordinates([self.date], dims=["time"], crs=spatial_coordinates.crs)

        # merge dims
        return podpac.coordinates.merge_dims([time, spatial_coordinates])

    def get_data(self, coordinates, coordinates_index):

        # create array for time, lat, lon
        data = self.create_output_array(coordinates)

        # eval in space for single time
        data[0, :, :] = super(SatUtilsSource, self).get_data(coordinates.drop("time"), coordinates_index[1:])

        return data


class SatUtils(S3Mixin, OrderedCompositor):
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

    # required
    collection = tl.Unicode(default_value=None, allow_none=True).tag(attr=True)
    query = tl.Dict(default_value=None, allow_none=True).tag(attr=True)
    asset = tl.Unicode(default_value=None, allow_none=True).tag(attr=True)

    min_bounds_span = tl.Dict(allow_none=True).tag(attr=True)

    # attributes
    _search = None

    @property
    def sources(self):
        if not self._search:
            raise AttributeError("Run `node.eval` or `node.search` with coordinates to define `node.sources` property")

        items = self._search.items()

        if self.asset is None:
            raise ValueError("Asset type must be defined. Use `list_assets` method")

        if self.asset not in items[0].assets:
            raise ValueError(
                'Requested asset "{}" is not available in item assets: {}'.format(
                    self.asset, list(items[0].assets.keys())
                )
            )

        # generate s3:// urls instead of https:// so that file-loader can handle
        s3_urls = [
            item.assets[self.asset]["href"].replace("https://", "s3://").replace(".s3.amazonaws.com", "")
            for item in items
        ]

        return [
            SatUtilsSource(source=s3_urls[item_idx], date=item.properties["datetime"])
            for item_idx, item in enumerate(items)
        ]

    @node_eval
    def eval(self, coordinates, output=None):
        # update sources with search
        _ = self.search(coordinates)

        # run normal eval once self.data is prepared
        return super(SatUtils, self).eval(coordinates, output)

    ##########
    # Data I/O
    ##########
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
        bbox = None

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
        search = {}
        if time_bounds is not None:
            search["time"] = "{start_time}/{end_time}".format(start_time=time_bounds[0], end_time=time_bounds[1])

        if bbox is not None:
            search["bbox"] = bbox

        if self.query is not None:
            search["query"] = self.query
        else:
            search["query"] = {}

        # note, this will override the collection in "query"
        if self.collection is not None:
            search["query"]["collection"] = {"eq": self.collection}

        # search with sat-search
        self._search = Search(**search)
        _logger.debug("sat-search found {} items".format(self._search.found()))

        return self._search

    def list_assets(self):
        """List available assets (or bands) within data source.
        You must run `search` with coordinates before you can list the assets available for those coordinates
        
        Returns
        -------
        list
            list of string asset names
        """
        if not self._search:
            raise AttributeError("Run `node.eval` or `node.search` with coordinates to be able to list assets")
        else:
            items = self._search.items()
            return list(items[0].assets.keys())


class Landsat8(SatUtils):
    """
    Landsat 8 on AWS OpenData
    https://registry.opendata.aws/landsat-8/

    Leverages sat-utils (https://github.com/sat-utils) developed by Development Seed
    See :class:`podpac.datalib.satutils.SatUtils` for attributes
    """

    collection = "landsat-8-l1"


class Sentinel2(SatUtils):
    """
    Sentinel 2 on AWS OpenData
    https://registry.opendata.aws/sentinel-2/
    Note this data source requires the requester to pay, so you must set podpac settings["AWS_REQUESTER_PAYS"] = True

    Leverages sat-utils (https://github.com/sat-utils) developed by Development Seed.
    See :class:`podpac.datalib.satutils.SatUtils` for attributes
    """

    collection = "sentinel-2-l1c"
