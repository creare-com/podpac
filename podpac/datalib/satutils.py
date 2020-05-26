"""
Satellite data access using sat-utils (https://github.com/sat-utils) developed by Development Seed

Supports access to:

- Landsat 8 on AWS OpenData: https://registry.opendata.aws/landsat-8/
- Sentinel 2
"""

import logging
import datetime

import numpy as np
import traitlets as tl
from satsearch import Search

# Internal dependencies
from podpac import Coordinates, Node
from podpac.compositor import OrderedCompositor
from podpac.data import DataSource
from podpac import authentication
from podpac import settings
from podpac import cached_property
from podpac.core.units import UnitsDataArray
from podpac.core.node import node_eval

_logger = logging.getLogger(__name__)


class SatUtils(DataSource):
    """
    PODPAC DataSource node to access the data using sat-utils developed by Development Seed
    See https://github.com/sat-utils
    
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
        Defaults to "MTL"
    min_bounds_span: dict, optional
        Default is {}. When specified, gives the minimum bounds that will be used for a coordinate in the query, so
        it works properly. If a user specified a lat, lon point, the query may fail since the min/max values for 
        lat/lon are the same. When specified, these bounds will be padded by the following for latitude (as an example): 
        [lat - min_bounds_span['lat'] / 2, lat + min_bounds_span['lat'] / 2]

    Attributes
    ----------
    data : :class:`podpac.UnitsDataArray`
        The data array compiled from downloaded satellite imagery
    """

    # required
    collection = tl.Unicode(default_value=None, allow_none=True).tag(attr=True)
    query = tl.Dict(default_value=None, allow_none=True).tag(attr=True)
    asset = tl.Unicode(default_value="MTL").tag(attr=True)

    min_bounds_span = tl.Dict(allow_none=True).tag(attr=True)

    # attributes
    data = tl.Any(allow_none=True)

    @property
    def coordinates(self):
        if self.data is None:
            _log.warning("No coordinates found in SatUtils source")
            return Coordinates([], dims=[])

        return Coordinates.from_xarray(self.data.coords, crs=self.data.attrs["crs"])

    def get_data(self, coordinates, coordinates_index):
        if self.data is not None:
            da = self.data[coordinates_index]
            return da
        else:
            _log.warning("No data found in SatUtils source")
            return np.array([])

    @node_eval
    def eval(self, coordinates, output=None):

        search = self.search(coordinates)

        # download data for coordinate bounds, then handle that data as an H5PY node
        # zip_files = self._download(coordinates)
        # self.data = self._read_zips(zip_files)  # reads each file in zip archive and creates single dataarray

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
        s = Search(**search)
        _logger.debug("sat-search found {} items".format(s.found()))

        return s

    def download(self, coordinates):
        """
        Download data from sat-utils Interface within PODPAC coordinates
        
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
        zipfile.ZipFile
            Returns zip file byte-str to downloaded data
        """

        pass
