from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
import io
import re

from six import string_types
import traitlets as tl
import numpy as np
import pyproj
import logging

from lazy_import import lazy_module

rasterio = lazy_module("rasterio")

from podpac.core.utils import common_doc, cached_property
from podpac.core.coordinates import UniformCoordinates1d, Coordinates
from podpac.core.data.datasource import COMMON_DATA_DOC, DATA_DOC
from podpac.core.data.file_source import BaseFileSource, LoadFileMixin
from podpac.core.interpolation.interpolation import InterpolationMixin

_logger = logging.getLogger(__name__)


@common_doc(COMMON_DATA_DOC)
class RasterioRaw(LoadFileMixin, BaseFileSource):
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
    read_as_filename : bool, optional
        Default is False. If True, the file will be read using rasterio.open(self.source) instead of being automatically
        parsed to handle ftp, s3, in-memory files, etc.

    See Also
    --------
    Rasterio : Interpolated rasterio datasource for general use.
    """

    # dataset = tl.Instance(rasterio.DatasetReader).tag(readonly=True)
    band = tl.CInt(allow_none=True).tag(attr=True)
    crs = tl.Unicode(allow_none=True, default_value=None).tag(attr=True)
    driver = tl.Unicode(allow_none=True, default_value=None)
    read_from_source = tl.Bool(False).tag(attr=True)
    coordinate_index_type = "slice"
    aws_https = tl.Bool(True)

    @cached_property
    def dataset(self):
        if self.source.startswith("s3://"):
            _logger.info("Loading AWS resource: %s" % self.source)
            with rasterio.env.Env(aws_unsigned=self.anon, AWS_HTTPS=self.aws_https) as env:
                _logger.debug("Rasterio sees these AWS credentials:", env.options)
                dataset = rasterio.open(self.source)  # This should pull AWS credentials automatically
                return dataset
        elif re.match(".*:.*:.*", self.source):
            # i.e. user supplied a non-file-looking string like 'HDF4_EOS:EOS_GRID:"MOD13Q1.A2013033.h08v05.006.2015256072248.hdf":MODIS_Grid_16DAY_250m_500m_VI:"250m 16 days NDVI"'
            # This also includes many subdatsets as part of GDAL data drivers; https://gdal.org/drivers/raster/index.html
            self.set_trait("read_from_source", True)
            return rasterio.open(self.source)
        else:
            return super(RasterioRaw, self).dataset

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

    def open_dataset(self, fp, **kwargs):
        if self.read_from_source:
            return rasterio.open(self.source)

        with rasterio.MemoryFile() as mf:
            mf.write(fp.read())
            return mf.open(driver=self.driver)

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

        if self.crs is not None:
            crs = self.crs
        elif isinstance(self.dataset.crs, rasterio.crs.CRS) and "init" in self.dataset.crs:
            crs = self.dataset.crs["init"].upper()
        elif isinstance(self.dataset.crs, dict) and "init" in self.dataset.crs:
            crs = self.dataset.crs["init"].upper()
        else:
            try:
                crs = pyproj.CRS(self.dataset.crs).to_wkt()
            except pyproj.exceptions.CRSError:
                raise RuntimeError("Unexpected rasterio crs '%s'" % self.dataset.crs)

        return Coordinates.from_geotransform(affine.to_gdal(), self.dataset.shape, crs)

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

    # -------------------------------------------------------------------------
    # additional methods and properties
    # -------------------------------------------------------------------------

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
        matches = np.where(match)[0] + 1

        return matches


class Rasterio(InterpolationMixin, RasterioRaw):
    """ Rasterio datasource with interpolation. """

    pass
