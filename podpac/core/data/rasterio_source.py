from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict

from six import string_types
import traitlets as tl
import numpy as np
import pyproj
import logging

from podpac.core.lazy_loading import lazy_module

rasterio = lazy_module("rasterio")
boto3 = lazy_module("boto3")

from podpac.core.utils import common_doc, cached_property
from podpac.core.coordinates import UniformCoordinates1d, Coordinates, merge_dims
from podpac.core.data.datasource import COMMON_DATA_DOC
from podpac.core.data.file_source import BaseFileSource
from podpac.core.authentication import S3Mixin

_logger = logging.getLogger(__name__)


@common_doc(COMMON_DATA_DOC)
class Rasterio(S3Mixin, BaseFileSource):
    """Create a DataSource using rasterio.

    Attributes
    ----------
    source : str, :class:`io.BytesIO`
        Path to the data source
    dataset : :class:`rasterio._io.RasterReader`
        A reference to the datasource opened by rasterio
    coordinates : :class:`podpac.Coordinates`
        {coordinates}
    band : int
        The 'band' or index for the variable being accessed in files such as GeoTIFFs. Use None for all bounds.
    crs : str, optional
        The coordinate reference system. Normally this will come directly from the file, but this allows users to
        specify the crs in case this information is missing from the file.
    aws_https: bool
        Default is True. If False, will not use https when reading from AWS. This is useful for debugging when SSL certificates are invalid.
    prefer_overviews: bool, optional
        Default is False. If True, will pull data from an overview with the closest resolution (step size) matching the
        median resolution in the request.
    prefer_overviews_closest: bool, optional
        Default is False. If True, will find the closest overview instead of the closest

    See Also
    --------
    Rasterio : Interpolated rasterio datasource for general use.
    """

    band = tl.CInt(allow_none=True).tag(attr=True)
    crs = tl.Unicode(allow_none=True, default_value=None).tag(attr=True)

    driver = tl.Unicode(allow_none=True, default_value=None)
    aws_https = tl.Bool(True).tag(attr=True)
    prefer_overviews = tl.Bool(False).tag(attr=True)
    prefer_overviews_closest = tl.Bool(False).tag(attr=True)
    coordinate_index_type = tl.Unicode("slice")
    _overview_grids: "tl.Dict[int, Coordinates]" = tl.Dict()  # overview grids read so far, keyed by level

    @cached_property
    def dataset(self):
        return self.open_dataset(self.source)

    def open_dataset(self, source, overview_level=None):
        envargs = {"AWS_HTTPS": self.aws_https}
        kwargs = {}
        if overview_level is not None:
            kwargs = {"overview_level": overview_level}
        if source.startswith("s3://"):
            if self.aws_get_auth_from_env:
                # let GDAL find and refresh credentials itself
                envargs["session"] = rasterio.session.DummySession()
            else:
                envargs["session"] = rasterio.session.AWSSession(
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    region_name=self.aws_region_name,
                    requester_pays=self.aws_requester_pays,
                    aws_unsigned=self.anon,
                )

            with rasterio.env.Env(**envargs) as env:
                _logger.debug("Rasterio environment options: {}".format(env.options))
                # /vsis3/ stops rasterio from injecting a boto3 session with frozen credentials
                return rasterio.open("/vsis3/" + source[len("s3://") :], **kwargs)
        else:
            return rasterio.open(source, **kwargs)

    @tl.default("band")
    def _band_default(self):
        if self.outputs is not None and self.output is not None:
            return self.outputs.index(self.output)
        elif self.outputs is None:
            return 1
        else:
            return None  # All bands

    # -------------------------------------------------------------------------
    # public api methods
    # -------------------------------------------------------------------------

    @cached_property
    def nan_vals(self):
        return np.unique(np.array(self.dataset.nodatavals).astype(self.dtype)).tolist()

    def close_dataset(self):
        """Closes the file for the datasource"""
        self.dataset.close()

    @common_doc(COMMON_DATA_DOC)
    def get_coordinates(self):
        """{get_coordinates}

        The default implementation tries to find the lat/lon coordinates based on dataset.affine.
        It cannot determine the alt or time dimensions, so child classes may
        have to overload this method.
        """

        # check to see if the coordinates are rotated used affine
        affine = self.dataset.transform
        validate_crs = True
        if self.crs is not None:
            crs = self.crs
        elif (
            isinstance(self.dataset.crs, rasterio.crs.CRS) or isinstance(self.dataset.crs, dict)
        ) and "init" in self.dataset.crs:
            crs = self.dataset.crs["init"].upper()
            validate_crs = False
        else:
            try:
                crs = pyproj.CRS(self.dataset.crs).to_wkt()
            except pyproj.exceptions.CRSError:
                raise RuntimeError("Unexpected rasterio crs '%s'" % self.dataset.crs)

        return Coordinates.from_geotransform(affine.to_gdal(), self.dataset.shape, crs, validate_crs)

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}"""
        data = self.create_output_array(coordinates)
        slc = coordinates_index

        # read data within coordinates_index window
        window = ((slc[0].start, slc[0].stop), (slc[1].start, slc[1].stop))

        if self.outputs is not None:  # read all the bands
            raster_data = self.dataset.read(out_shape=(len(self.outputs),) + tuple(coordinates.shape), window=window)
            raster_data = np.moveaxis(raster_data, 0, 2)
        else:  # read the requested band
            raster_data = self.dataset.read(self.band, out_shape=tuple(coordinates.shape)[:2], window=window)

        # set raster data to output array
        data.data.ravel()[:] = raster_data.ravel()
        return data

    def _get_window_coords(self, coordinates, new_coords):
        new_coords, slc = new_coords.intersect(coordinates, return_index=True, outer=True)
        window = ((slc[0].start, slc[0].stop), (slc[1].start, slc[1].stop))
        return window, new_coords

    def _eval(self, coordinates, output=None, _selector=None):
        if self.prefer_overviews:
            # select from the overview grid instead of the full-resolution grid
            grid = self._overview_grid(self._overview_level(coordinates))
            scale = [n / m for n, m in zip(self.coordinates.shape, grid.shape)]
            selector = _selector

            def overview_selector(source_coordinates, request_coordinates, index_type=None):  # NOSONAR(S1172)
                if selector is None:
                    # no interpolator: read the overview pixels within the request bounds
                    selected, index = grid.intersect(request_coordinates, outer=True, return_index=True)
                else:
                    selected, index = selector(grid, request_coordinates, index_type="slice")
                if selected.size == 0:
                    return selected, index
                # drop any step so the coordinates cover whole window which get_data reads
                index = tuple(slice(*i.indices(n)[:2]) for i, n in zip(index, grid.shape))
                # full-resolution window which get_data will read at the overview level
                window = tuple(slice(i.start * s, i.stop * s) for i, s in zip(index, scale))
                return grid[index], window

            return super()._eval(coordinates, output=output, _selector=overview_selector)

        return super()._eval(coordinates, output=output, _selector=_selector)

    def _overview_grid(self, level: int | None) -> Coordinates:
        """Get the grid of an overview level.

        The overview is opened once to read its exact shape and transform, then the grid is kept in `_overview_grids`.

        Parameters
        ----------
        level : int | None
            Index into `overviews`, or None for full resolution.

        Returns
        -------
        Coordinates
            Overview grid, with any non-spatial dims of the source. The source coordinates if level is None.
        """
        if level is None:
            return self.coordinates
        if level not in self._overview_grids:
            with self.open_dataset(self.source, level) as dataset:
                grid = Coordinates.from_geotransform(dataset.transform.to_gdal(), dataset.shape, self.coordinates.crs)
            # keep any non-spatial dims of the source
            grid = merge_dims([grid, self.coordinates.drop(["lat", "lon"])])
            self._overview_grids[level] = grid.transpose(*self.coordinates.dims)
        return self._overview_grids[level]

    def _overview_level(self, coordinates: Coordinates) -> int | None:
        """Find the overview matching the median request resolution.

        Parameters
        ----------
        coordinates : Coordinates
            Requested coordinates, already transformed to the source crs.

        Returns
        -------
        int | None
            Index into `overviews` of the overview to read, or None to read full resolution.
        """
        if not isinstance(self.coordinates["lat"], UniformCoordinates1d):
            return None

        # Figure out how much coarser the request is than the actual data
        reduction_factor = np.inf
        for c in ["lat", "lon"]:
            values = np.unique(coordinates[c].coordinates)
            if values.size == 1:
                reduction_factor = 0
                break
            # median ignores outliers due to fine steps near the poles in web mercator or gaps between clustered points
            min_delta = np.median(np.diff(values))
            reduction_factor = min(reduction_factor, np.abs(min_delta / self.coordinates[c].step))

        # Find the overview that's closest to this reduction factor
        eps = 1e-6  # allowable epsilon for an overview to be chosen
        if (reduction_factor < 2 - eps) or (len(self.overviews) == 0):
            return None
        diffs = reduction_factor - np.array(self.overviews)
        if self.prefer_overviews_closest:
            diffs = np.abs(diffs)
        else:
            diffs[diffs < -eps] = np.inf
        return int(np.argmin(diffs))

    # -------------------------------------------------------------------------
    # additional methods and properties
    # -------------------------------------------------------------------------

    @property
    def overviews(self):
        # all bands share overview factors, so use the first band when reading all of them
        return self.dataset.overviews(self.band if self.band is not None else 1)

    @property
    def tags(self):
        return self.dataset.tags()

    @property
    def subdatasets(self):
        return self.dataset.subdatasets

    @property
    def band_count(self):
        """The number of bands"""

        return self.dataset.count

    @cached_property
    def band_descriptions(self):
        """A description of each band contained in dataset.tags

        Returns
        -------
        OrderedDict
            Dictionary of band_number: band_description pairs. The band_description values are a dictionary, each
            containing a number of keys -- depending on the metadata
        """

        return OrderedDict((i, self.dataset.tags(i + 1)) for i in range(self.band_count))

    @cached_property
    def band_keys(self):
        """An alternative view of band_descriptions based on the keys present in the metadata

        Returns
        -------
        dict
            Dictionary of metadata keys, where the values are the value of the key for each band.
            For example, band_keys['TIME'] = ['2015', '2016', '2017'] for a dataset with three bands.
        """

        keys = {k for i in range(self.band_count) for k in self.band_descriptions[i]}  # set
        return {k: [self.band_descriptions[i].get(k) for i in range(self.band_count)] for k in keys}

    def get_band_numbers(self, key, value):
        """Return the bands that have a key equal to a specified value.

        Parameters
        ----------
        key : str / list
            Key present in the metadata of the band. Can be a single key, or a list of keys.
        value : str / list
            Value of the key that should be returned. Can be a single value, or a list of values

        Returns
        -------
        np.ndarray
            An array of band numbers that match the criteria
        """
        if not hasattr(key, "__iter__") or isinstance(key, string_types):
            key = [key]

        if not hasattr(value, "__iter__") or isinstance(value, string_types):
            value = [value]

        match = np.ones(self.band_count, bool)
        for k, v in zip(key, value):
            match = match & (np.array(self.band_keys[k]) == v)
        matches = np.nonzero(match)[0] + 1

        return matches
