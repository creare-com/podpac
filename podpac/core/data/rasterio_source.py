from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict

import traitlets as tl
import numpy as np
from six import string_types

from lazy_import import lazy_module, lazy_class

rasterio = lazy_module("rasterio")

from podpac.core.utils import common_doc
from podpac.core.coordinates import UniformCoordinates1d, Coordinates
from podpac.core.data.datasource import COMMON_DATA_DOC, DATA_DOC
from podpac.core.data.file_source import BaseFileSource, LoadFileMixin


@common_doc(COMMON_DATA_DOC)
class Rasterio(LoadFileMixin, BaseFileSource):
    """Create a DataSource using rasterio.
 
    Attributes
    ----------
    source : str, :class:`io.BytesIO`
        Path to the data source
    dataset : :class:`rasterio._io.RasterReader`
        A reference to the datasource opened by rasterio
    native_coordinates : :class:`podpac.Coordinates`
        {native_coordinates}
    band : int
        The 'band' or index for the variable being accessed in files such as GeoTIFFs
    """

    # dataset = tl.Instance(rasterio.DatasetReader).tag(readonly=True)
    band = tl.CInt(default_value=1).tag(attr=True)

    # -------------------------------------------------------------------------
    # public api methods
    # -------------------------------------------------------------------------

    @property
    def nan_vals(self):
        return list(self.dataset.nodatavals)

    def open_dataset(self, fp):
        with rasterio.MemoryFile(fp.read()) as mf:
            return mf.open()

    def close_dataset(self):
        """Closes the file for the datasource
        """
        self.dataset.close()

    @common_doc(COMMON_DATA_DOC)
    @property
    def native_coordinates(self):
        """{get_native_coordinates}
        
        The default implementation tries to find the lat/lon coordinates based on dataset.affine.
        It cannot determine the alt or time dimensions, so child classes may
        have to overload this method.
        """

        # check to see if the coordinates are rotated used affine
        affine = self.dataset.transform
        if affine[1] != 0.0 or affine[3] != 0.0:
            raise NotImplementedError("Rotated coordinates are not yet supported")

        if isinstance(self.dataset.crs, rasterio.crs.CRS):
            crs = self.dataset.crs.wkt
        elif isinstance(self.dataset.crs, dict) and "init" in self.dataset.crs:
            crs = self.dataset.crs["init"].upper()
        else:
            try:
                crs = pyproj.CRS(self.dataset.crs).to_wkt()
            except:
                raise RuntimeError("Unexpected rasterio crs '%s'" % self.dataset.crs)

        # get ul and lr pixel centers
        left, top = self.dataset.xy(0, 0)
        right, bottom = self.dataset.xy(self.dataset.width - 1, self.dataset.height - 1)
        lat = UniformCoordinates1d(bottom, top, size=self.dataset.height, name="lat")
        lon = UniformCoordinates1d(left, right, size=self.dataset.width, name="lon")
        return Coordinates([lat, lon], dims=["lat", "lon"], crs=crs)

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        data = self.create_output_array(coordinates)
        slc = coordinates_index

        # read data within coordinates_index window
        window = ((slc[0].start, slc[0].stop), (slc[1].start, slc[1].stop))
        raster_data = self.dataset.read(self.band, out_shape=coordinates.shape, window=window)

        # set raster data to output array
        data.data.ravel()[:] = raster_data.ravel()
        return data

    # -------------------------------------------------------------------------
    # additional methods and properties
    # -------------------------------------------------------------------------

    @property
    def band_count(self):
        """The number of bands"""

        return self.dataset.count

    @property
    def band_descriptions(self):
        """ A description of each band contained in dataset.tags
        
        Returns
        -------
        OrderedDict
            Dictionary of band_number: band_description pairs. The band_description values are a dictionary, each 
            containing a number of keys -- depending on the metadata
        """

        if not hasattr(self, "_band_descriptions"):
            self._band_descriptions = OrderedDict((i, self.dataset.tags(i + 1)) for i in range(self.band_count))

        return self._band_descriptions

    @property
    def band_keys(self):
        """An alternative view of band_descriptions based on the keys present in the metadata
        
        Returns
        -------
        dict
            Dictionary of metadata keys, where the values are the value of the key for each band. 
            For example, band_keys['TIME'] = ['2015', '2016', '2017'] for a dataset with three bands.
        """

        if not hasattr(self, "_band_keys"):
            keys = {k for i in range(self.band_count) for k in self.band_descriptions[i]}  # set
            self._band_keys = {k: [self.band_descriptions[i].get(k) for i in range(self.band_count)] for k in keys}

        return self._band_keys

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
