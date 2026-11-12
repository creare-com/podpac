from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d

from six import string_types
import traitlets as tl
import numpy as np
import pyproj
import logging
import contextlib

from lazy_import import lazy_module

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
        Default is False. If True, will pull data from an overview with the closest resolution (step size) matching the smallest resolution
        in the request.
    prefer_overviews_closest: bool, optional
        Default is False. If True, will find the closest overview instead of the closest

    See Also
    --------
    Rasterio : Interpolated rasterio datasource for general use.
    """

    band = tl.CInt(allow_none=True).tag(attr=True)
    crs = tl.Unicode(allow_none=True, default_value=None).tag(attr=True)

    driver = tl.Unicode(allow_none=True, default_value=None)
    coordinate_index_type = tl.Unicode()
    aws_https = tl.Bool(True).tag(attr=True)
    prefer_overviews = tl.Bool(False).tag(attr=True)
    prefer_overviews_closest = tl.Bool(False).tag(attr=True)
    _dataset = tl.Instance("rasterio.io.DatasetReader", default_value=None, allow_none=True)

    @tl.default("coordinate_index_type")
    def _default_coordinate_index_type(self):
        if self.prefer_overviews:
            return "numpy"
        else:
            return "slice"

    @property
    def dataset(self):
        if self._dataset is None or self._dataset.closed:
            self._dataset = self.open_dataset(self.source)
        return self._dataset

    def _env(self, source):
        """Rasterio environment for accessing the source.

        Returns
        -------
        rasterio.env.Env | contextlib.nullcontext
            Context manager for environment.
        """
        if not source.startswith("s3://"):
            return contextlib.nullcontext()

        envargs = {"AWS_HTTPS": self.aws_https}
        if self.aws_get_auth_from_env:
            envargs["session"] = rasterio.session.AWSSession()
        else:
            envargs["session"] = rasterio.session.AWSSession(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region_name,
                requester_pays=self.aws_requester_pays,
                aws_unsigned=self.anon,
            )
        return rasterio.env.Env(**envargs)

    def open_dataset(self, source, overview_level=None):
        kwargs = {}
        if overview_level is not None:
            kwargs = {"overview_level": overview_level}
        with self._env(source) as env:
            if env is not None:
                _logger.debug("Rasterio environment options: {}".format(env.options))
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
        """Closes the file for the datasource."""
        if self._dataset is not None:
            self._dataset.close()

    def _read(self, dataset, *args, **kwargs):
        """Read data from the source.

        If the read fails, the handle is closed and GDAL's HTTP cache is cleared (rasterio>=1.5) before
        the error is raised. A dropped or truncated HTTP response can be cached by GDAL in both the open handle
        and its process-wide HTTP cache, so without dropping them every later read of the affected tiles
        could keep failing.

        Parameters
        ----------
        dataset : rasterio.io.DatasetReader
            The open dataset to read from.
        *args, **kwargs
            Passed to rasterio data reader.

        Returns
        -------
        np.ndarray
            The data read.
        """
        try:
            return dataset.read(*args, **kwargs)
        except rasterio.errors.RasterioIOError:
            try:
                from rasterio.cache import invalidate
            except ImportError:  # rasterio < 1.5
                invalidate = None

            if invalidate is not None:
                with self._env(self.source):
                    invalidate(self.source)
            dataset.close()
            raise

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
        if self.prefer_overviews:
            return self.get_data_overviews(coordinates)

        data = self.create_output_array(coordinates)
        slc = coordinates_index

        # read data within coordinates_index window
        window = ((slc[0].start, slc[0].stop), (slc[1].start, slc[1].stop))

        if self.outputs is not None:  # read all the bands
            raster_data = self._read(
                self.dataset, out_shape=(len(self.outputs),) + tuple(coordinates.shape), window=window
            )
            raster_data = np.moveaxis(raster_data, 0, 2)
        else:  # read the requested band
            raster_data = self._read(self.dataset, self.band, out_shape=tuple(coordinates.shape)[:2], window=window)

        # set raster data to output array
        data.data.ravel()[:] = raster_data.ravel()
        return data

    def _get_window_coords(self, coordinates, new_coords):
        new_coords, slc = new_coords.intersect(coordinates, return_index=True, outer=True)
        window = ((slc[0].start, slc[0].stop), (slc[1].start, slc[1].stop))
        return window, new_coords

    def get_data_overviews(self, coordinates):
        # Figure out how much coarser the request is than the actual data
        reduction_factor = np.inf
        for c in ["lat", "lon"]:
            crd = coordinates[c]
            if crd.size == 1:
                reduction_factor = 0
                break
            if isinstance(crd, UniformCoordinates1d):
                min_delta = crd.step
            elif isinstance(crd, ArrayCoordinates1d) and crd.is_monotonic:
                min_delta = crd.deltas.min()
            else:
                raise NotImplementedError(
                    "The Rasterio node with prefer_overviews=True currently does not support request coordinates type {}".format(
                        coordinates
                    )
                )
            reduction_factor = min(
                reduction_factor, np.abs(min_delta / self.coordinates[c].step)  # self.coordinates is always uniform
            )
        # Find the overview that's closest to this reduction factor
        if (reduction_factor < 2) or (len(self.overviews) == 0):  # Then we shouldn't use an overview
            overview_level = None
        else:
            diffs = reduction_factor - np.array(self.overviews)
            if self.prefer_overviews_closest:
                diffs = np.abs(diffs)
            else:
                diffs[diffs < 0] = np.inf
            overview_level = np.argmin(diffs)

        # Now read the data
        if overview_level is None:
            dataset = self.dataset
        else:
            dataset = self.open_dataset(self.source, overview_level)
        try:
            # read data within coordinates_index window at the resolution of the overview
            # Rasterio will then automatically pull from the overview

            new_coords = Coordinates.from_geotransform(
                dataset.transform.to_gdal(), dataset.shape, crs=self.coordinates.crs
            )
            window, new_coords = self._get_window_coords(coordinates, new_coords)
            missing_coords = self.coordinates.drop(["lat", "lon"])

            new_coords = merge_dims([new_coords, missing_coords])
            new_coords = new_coords.transpose(*self.coordinates.dims)
            coordinates_shape = new_coords.shape[:2]

            # The following lines are *nearly* copied/pasted from get_data
            if self.outputs is not None:  # read all the bands
                raster_data = self._read(dataset, out_shape=(len(self.outputs),) + coordinates_shape, window=window)
                raster_data = np.moveaxis(raster_data, 0, 2)
            else:  # read the requested band
                raster_data = self._read(dataset, self.band, out_shape=coordinates_shape, window=window)

            # set raster data to output array
            data = self.create_output_array(new_coords)
            data.data.ravel()[:] = raster_data.ravel()
        except (rasterio.errors.RasterioError, OSError, ValueError, IndexError, MemoryError) as e:
            _logger.exception("Error occurred when reading overview with Rasterio: {}".format(e))
            raise
        finally:
            if overview_level is not None:
                dataset.close()

        return data

    # -------------------------------------------------------------------------
    # additional methods and properties
    # -------------------------------------------------------------------------

    @property
    def overviews(self):
        return self.dataset.overviews(self.band)

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
