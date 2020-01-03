"""
Datasources from files
"""

from __future__ import division, unicode_literals, print_function, absolute_import

from io import BytesIO
from collections import OrderedDict
from six import string_types

import numpy as np
import traitlets as tl
import pandas as pd
import xarray as xr

from podpac.core.settings import settings
from podpac.core.utils import common_doc, trait_is_defined
from podpac.core.data.datasource import COMMON_DATA_DOC, DataSource
from podpac.core.coordinates import Coordinates, UniformCoordinates1d, ArrayCoordinates1d, StackedCoordinates
from podpac.core.coordinates.utils import Dimension, VALID_DIMENSION_NAMES

# Optional dependencies
from lazy_import import lazy_module, lazy_class

rasterio = lazy_module("rasterio")
h5py = lazy_module("h5py")
boto3 = lazy_module("boto3")
requests = lazy_module("requests")
zarr = lazy_module("zarr")
zarrGroup = lazy_class("zarr.Group")
s3fs = lazy_module("s3fs")


@common_doc(COMMON_DATA_DOC)
class DatasetSource(DataSource):
    """
    Base class for dataset/file datasources.

    This class facilitates setting the native_coordinates from the coordinates defined in the file, including
    decoding datetimes when necessary. The coordinates are automatically read from the dataset when possible, and
    methods are provided for customization when necessary.

    Attributes
    ----------
    source : str
        Path to the data source file.
    dataset
        Dataset object.
    native_coordinates : Coordinates
        {native_coordinates}
    lat_key : str
        latitude key, default 'lat'
    lon_key : str
        longitude key, default 'lon'
    time_key : str
        time key, default 'time'
    alt_key : str
        altitude key, default 'alt'
    data_key : str
        data key
    output_keys : list
        list of data keys, for multiple-output nodes
    crs : str
        Coordinate reference system of the coordinates.
    cf_time : bool
        decode CF datetimes
    cf_units : str
        units, when decoding CF datetimes
    cf_calendar : str
        calendar, when decoding CF datetimes
    """

    source = tl.Unicode(default_value=None, allow_none=True).tag(readonly=True)
    data_key = tl.Unicode(allow_none=True).tag(attr=True)
    output_keys = tl.List(allow_none=True).tag(attr=True)
    lat_key = tl.Unicode(allow_none=True, default_value="lat").tag(attr=True)
    lon_key = tl.Unicode(allow_none=True, default_value="lon").tag(attr=True)
    time_key = tl.Unicode(allow_none=True, default_value="time").tag(attr=True)
    alt_key = tl.Unicode(allow_none=True, default_value="alt").tag(attr=True)
    crs = tl.Unicode(allow_none=True, default_value=None).tag(attr=True)
    cf_time = tl.Bool(False).tag(attr=True)
    cf_units = tl.Unicode(allow_none=True).tag(attr=True)
    cf_calendar = tl.Unicode(allow_none=True).tag(attr=True)

    dataset = tl.Any().tag(readonly=True)

    @tl.default("data_key")
    def _default_data_key(self):
        return None

    @tl.default("output_keys")
    def _default_output_keys(self):
        return None

    @tl.validate("output")
    def _validate_output(self, d):
        return d["value"]

    def init(self):
        super(DatasetSource, self).init()

        # check the dataset and dims
        self.dataset
        self.dims

        # validation and defaults for data_key, output_keys, outputs, and output
        if self.data_key is not None and self.output_keys is not None:
            raise TypeError("%s cannot have both 'data_key' or 'output_keys' defined" % self.__class__.__name__)

        if self.data_key is None and self.output_keys is None:
            available_keys = self.available_keys
            if len(available_keys) == 1:
                self.set_trait("data_key", available_keys[0])
            else:
                self.set_trait("output_keys", available_keys)

        if self.outputs is not None:
            if self.data_key is not None:
                raise TypeError("outputs must be None for single-output nodes")
            if len(self.outputs) != len(self.output_keys):
                raise ValueError(
                    "outputs and output_keys size mismatch (%d != %d)" % (len(self.outputs), len(self.output_keys))
                )
        else:
            self.set_trait("outputs", self.output_keys)

        if self.output is not None:
            if self.outputs is None:
                raise TypeError("Invalid output '%s' (output must be None for single-output nodes)." % self.output)
            if self.output not in self.outputs:
                raise ValueError("Invalid output '%s' (available outputs are %s)" % (self.output, self.outputs))

    @common_doc(COMMON_DATA_DOC)
    def get_native_coordinates(self):
        """{native_coordinates}
        """
        cs = []
        dims = self.dims
        for dim in dims:
            if dim == "lat":
                cs.append(self.get_lat())
            elif dim == "lon":
                cs.append(self.get_lon())
            elif dim == "time":
                cs.append(self.get_time())
            elif dim == "alt":
                cs.append(self.get_alt())

        return Coordinates(cs, dims=dims, crs=self.crs)

    def get_lat(self):
        """Get the native latitude coordinates from the dataset."""
        return self.dataset[self.lat_key]

    def get_lon(self):
        """Get the native longitude coordinates from the dataset."""
        return self.dataset[self.lon_key]

    def get_time(self):
        """Get the native time coordinates from the dataset, decoding datetimes if requested."""
        values = self.dataset[self.time_key]
        if self.cf_time:
            values = xr.coding.times.decode_cf_datetime(values, self.cf_units, self.cf_calendar)
        return values

    def get_alt(self):
        """Get the native altitude coordinates from the dataset."""
        return self.dataset[self.alt_key]

    def close_dataset(self):
        """ Close the dataset. Subclasses should implement as needed. """
        pass

    @property
    def dims(self):
        raise NotImplementedError

    @property
    def available_keys(self):
        raise NotImplementedError

    @property
    @common_doc(COMMON_DATA_DOC)
    def base_definition(self):
        """Base node definition for DatasetSource nodes.
        
        Returns
        -------
        {definition_return}
        """

        d = super(DatasetSource, self).base_definition

        # remove unnecessary attrs
        attrs = d.get("attrs", {})
        if self.data_key is None and "data_key" in attrs:
            del attrs["data_key"]
        if self.output_keys is None and "output_keys" in attrs:
            del attrs["output_keys"]
        if self.crs is None and "crs" in attrs:
            del attrs["crs"]
        if self.outputs == self.output_keys and "outputs" in attrs:
            del attrs["outputs"]
        if "lat" not in self.dims and "lat_key" in attrs:
            del attrs["lat_key"]
        if "lon" not in self.dims and "lon_key" in attrs:
            del attrs["lon_key"]
        if "alt" not in self.dims and "alt_key" in attrs:
            del attrs["alt_key"]
        if "time" not in self.dims and "time_key" in attrs:
            del attrs["time_key"]
        if self.cf_time is False:
            if "cf_time" in attrs:
                del attrs["cf_time"]
            if "cf_units" in attrs:
                del attrs["cf_units"]
            if "cf_calendar" in attrs:
                del attrs["cf_calendar"]

        return d


@common_doc(COMMON_DATA_DOC)
class Dataset(DatasetSource):
    """Create a DataSource node using xarray.open_dataset.
    
    Attributes
    ----------
    source : str
        Path to the dataset file.
    dataset : xarray.Dataset
        Dataset object.
    native_coordinates : Coordinates
        {native_coordinates}
    data_key : str
        data key, default 'data'
    lat_key : str
        latitude key, default 'lat'
    lon_key : str
        longitude key, default 'lon'
    time_key : str
        time key, default 'time'
    alt_key : str
        altitude key, default 'alt'
    crs : str
        Coordinate reference system of the coordinates
    extra_dim : dict
        In cases where the data contain dimensions other than ['lat', 'lon', 'time', 'alt'], these dimensions need to be selected. 
        For example, if the data contains ['lat', 'lon', 'channel'], the second channel can be selected using `extra_dim=dict(channel=1)`
    """

    dataset = tl.Instance(xr.Dataset).tag(readonly=True)

    # node attrs
    extra_dim = tl.Dict(allow_none=True, default_value=None).tag(attr=True)

    @tl.default("dataset")
    def _open_dataset(self):
        return xr.open_dataset(self.source)

    def close_dataset(self):
        self.dataset.close()

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        if self.data_key is not None:
            data = self.dataset[self.data_key]
            data = data.transpose(*self.dataset.dims)
        else:
            data = self.dataset[self.output_keys].to_array(dim="output")
            tdims = tuple(self.dataset.dims) + ("output",)
            data = data.transpose(*tdims)
        return self.create_output_array(coordinates, data.data[coordinates_index])

    @property
    def dims(self):
        """dataset coordinate dims"""
        lookup = {self.lat_key: "lat", self.lon_key: "lon", self.alt_key: "alt", self.time_key: "time"}
        for dim in self.dataset.dims:
            if dim not in lookup:
                raise ValueError(
                    "Unexpected dimension '%s' in xarray dataset (source '%s'). "
                    "Use 'lat_key', 'lon_key', 'time_key' and 'alt_key' to select dataset dimensions"
                    % (dim, self.source)
                )

        return [lookup[dim] for dim in self.dataset.dims]

    @property
    def available_keys(self):
        """available data keys"""
        return list(self.dataset.keys())

    @property
    @common_doc(COMMON_DATA_DOC)
    def base_definition(self):
        """Base node definition for DatasetSource nodes.
        
        Returns
        -------
        {definition_return}
        """

        d = super(Dataset, self).base_definition

        # remove unnecessary attrs
        attrs = d.get("attrs", {})
        if self.extra_dim is None and "extra_dim" in attrs:
            del attrs["extra_dim"]

        return d


@common_doc(COMMON_DATA_DOC)
class CSV(DatasetSource):
    """Create a DataSource from a .csv file.

    This class assumes that the data has a storage format such as:
    header 1,   header 2,   header 3, ...
    row1_data1, row1_data2, row1_data3, ...
    row2_data1, row2_data2, row2_data3, ...
    
    Attributes
    ----------
    source : str
        Path to the csv file
    header : int, None
        Row number containing the column names, default 0. Use None for no header.
    dataset : pd.DataFrame
        Raw Pandas DataFrame used to read the data
    native_coordinates : Coordinates
        {native_coordinates}
    data_key : str, int
        data column number or column title, default 'data'
    lat_key : str, int
        latitude column number or column title, default 'lat'
    lon_key : str, int
        longitude column number or column title, default 'lon'
    time_key : str, int
        time column number or column title, default 'time'
    alt_key : str, int
        altitude column number or column title, default 'alt'
    crs : str
        Coordinate reference system of the coordinates
    """

    header = tl.Any(default_value=0).tag(attr=True)
    lat_key = tl.Union([tl.Unicode(), tl.Int()], default_value="lat").tag(attr=True)
    lon_key = tl.Union([tl.Unicode(), tl.Int()], default_value="lon").tag(attr=True)
    time_key = tl.Union([tl.Unicode(), tl.Int()], default_value="time").tag(attr=True)
    alt_key = tl.Union([tl.Unicode(), tl.Int()], default_value="alt").tag(attr=True)
    data_key = tl.Union([tl.Unicode(), tl.Int()], allow_none=True, default_value=None).tag(attr=True)
    output_keys = tl.Union([tl.List(tl.Unicode()), tl.List(tl.Int())], allow_none=True, default_value=None).tag(
        attr=True
    )

    dataset = tl.Instance(pd.DataFrame).tag(readonly=True)

    @tl.default("dataset")
    def _open_dataset(self):
        return pd.read_csv(self.source, parse_dates=True, infer_datetime_format=True, header=self.header)

    def _get_key(self, key):
        return self.dataset.columns[key] if isinstance(key, int) else key

    def _get_col(self, key):
        return key if isinstance(key, int) else self.dataset.columns.get_loc(key)

    def get_lat(self):
        """Get latitude coordinates from the csv file."""
        return self.dataset[self._get_key(self.lat_key)].values

    def get_lon(self):
        """Get longitude coordinates from the csv file."""
        return self.dataset[self._get_key(self.lon_key)].values

    def get_time(self):
        """Get time coordinates from the csv file."""
        return self.dataset[self._get_key(self.time_key)].values

    def get_alt(self):
        """Get altitude coordinates from the csv file."""
        return self.dataset[self._get_key(self.alt_key)].values

    @common_doc(COMMON_DATA_DOC)
    def get_native_coordinates(self):
        """{get_native_coordinates}
        
        Note: CSV files have StackedCoordinates.
        """

        coords = super(CSV, self).get_native_coordinates()
        stacked = StackedCoordinates(list(coords.values()))
        return Coordinates([stacked], **coords.properties)

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        if self.data_key is not None:
            I = self._get_col(self.data_key)
        else:
            I = [self._get_col(key) for key in self.output_keys]
        data = self.dataset.iloc[coordinates_index[0], I]
        return self.create_output_array(coordinates, data=data)

    @property
    def dims(self):
        """dataset coordinate dims"""
        lookup = {
            self._get_key(self.lat_key): "lat",
            self._get_key(self.lon_key): "lon",
            self._get_key(self.alt_key): "alt",
            self._get_key(self.time_key): "time",
        }
        return [lookup[key] for key in self.dataset.columns if key in lookup]

    @property
    def available_keys(self):
        """available data keys"""
        dims_keys = [self.lat_key, self.lon_key, self.alt_key, self.time_key]
        return [key for key in self.dataset.columns if key not in dims_keys]

    @property
    @common_doc(COMMON_DATA_DOC)
    def base_definition(self):
        """Base node definition for DatasetSource nodes.
        
        Returns
        -------
        {definition_return}
        """

        d = super(CSV, self).base_definition

        # remove unnecessary attrs
        attrs = d.get("attrs", {})
        if self.header == 0 and "header" in attrs:
            del attrs["header"]

        return d


@common_doc(COMMON_DATA_DOC)
class H5PY(DatasetSource):
    """Create a DataSource node using h5py.
    
    Attributes
    ----------
    source : str
        Path to the h5py file
    dataset : h5py.File
        The h5py file object used to read the file
    native_coordinates : Coordinates
        {native_coordinates}
    file_mode : str, optional
        Default is 'r'. The mode used to open the HDF5 file. Options are r, r+, w, w- or x, a (see h5py.File).
    data_key : str, int
        data key, default 'data'
    lat_key : str, int
        latitude coordinates key, default 'lat'
    lon_key : str, int
        longitude coordinates key, default 'lon'
    time_key : str, int
        time coordinates key, default 'time'
    alt_key : str, int
        altitude coordinates key, default 'alt'
    crs : str
        Coordinate reference system of the coordinates
    cf_time : bool
        decode CF datetimes
    cf_units : str
        units, when decoding CF datetimes
    cf_calendar : str
        calendar, when decoding CF datetimes
    """

    file_mode = tl.Unicode(default_value="r").tag(readonly=True)

    @tl.default("dataset")
    def _open_dataset(self):
        # TODO: dataset should not open by default
        # prefer with as: syntax
        return h5py.File(self.source, self.file_mode)

    def close_dataset(self):
        """Closes the file. """
        self.dataset.close()

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        data = self.create_output_array(coordinates)
        if self.data_key is not None:
            data[:] = self.dataset[self.data_key][coordinates_index]
        else:
            for key, name in zip(self.output_keys, self.outputs):
                data.sel(output=name)[:] = self.dataset[key][coordinates_index]
        return data

    def attrs(self, key="/"):
        """Dataset or group key for which attributes will be summarized.
        """
        return dict(self.dataset[key].attrs)

    @property
    def dims(self):
        lookup = {self.lat_key: "lat", self.lon_key: "lon", self.alt_key: "alt", self.time_key: "time"}
        return [lookup[key] for key in H5PY._find_h5py_keys(self.dataset) if key in lookup]

    @property
    def available_keys(self):
        dims_keys = [self.lat_key, self.lon_key, self.alt_key, self.time_key]
        return [key for key in H5PY._find_h5py_keys(self.dataset) if key not in dims_keys]

    @staticmethod
    def _find_h5py_keys(obj, keys=[]):
        if isinstance(obj, (h5py.Group, h5py.File)):
            for k in obj.keys():
                keys = H5PY._find_h5py_keys(obj[k], keys)
        else:
            keys.append(obj.name)
            return keys
        keys = list(set(keys))
        keys.sort()
        return keys


class Zarr(DatasetSource):
    """Create a DataSource node using zarr.
    
    Attributes
    ----------
    source : str
        Path to the Zarr archive
    file_mode : str, optional
        Default is 'r'. The mode used to open the Zarr archive. Options are r, r+, w, w- or x, a.
    dataset : zarr.Group
        The h5py file object used to read the file
    native_coordinates : Coordinates
        {native_coordinates}
    data_key : str, int
        data key, default 'data'
    lat_key : str, int
        latitude coordinates key, default 'lat'
    lon_key : str, int
        longitude coordinates key, default 'lon'
    time_key : str, int
        time coordinates key, default 'time'
    alt_key : str, int
        altitude coordinates key, default 'alt'
    crs : str
        Coordinate reference system of the coordinates
    cf_time : bool
        decode CF datetimes
    cf_units : str
        units, when decoding CF datetimes
    cf_calendar : str
        calendar, when decoding CF datetimes
    """

    file_mode = tl.Unicode(default_value="r").tag(readonly=True)

    # optional inputs
    access_key_id = tl.Unicode()
    secret_access_key = tl.Unicode()
    region_name = tl.Unicode()

    @tl.default("access_key_id")
    def _get_access_key_id(self):
        return settings["AWS_ACCESS_KEY_ID"]

    @tl.default("secret_access_key")
    def _get_secret_access_key(self):
        return settings["AWS_SECRET_ACCESS_KEY"]

    @tl.default("region_name")
    def _get_region_name(self):
        return settings["AWS_REGION_NAME"]

    @tl.default("dataset")
    def _open_dataset(self):
        if self.source is None:
            raise TypeError("Zarr node requires 'source' or 'dataset'")

        if self.source.startswith("s3://"):
            root = self.source.strip("s3://")
            kwargs = {"region_name": self.region_name}
            s3 = s3fs.S3FileSystem(key=self.access_key_id, secret=self.secret_access_key, client_kwargs=kwargs)
            s3map = s3fs.S3Map(root=root, s3=s3, check=False)
            store = s3map
        else:
            store = str(self.source)  # has to be a string in Python2.7 for local files

        try:
            return zarr.open(store, mode=self.file_mode)
        except ValueError:
            raise ValueError("No Zarr store found at path '%s'" % self.source)

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        data = self.create_output_array(coordinates)
        if self.data_key is not None:
            data[:] = self.dataset[self.data_key][coordinates_index]
        else:
            for key, name in zip(self.output_keys, self.outputs):
                data.sel(output=name)[:] = self.dataset[key][coordinates_index]
        return data

    @property
    def dims(self):
        """dataset coordinate dims"""
        lookup = {self.lat_key: "lat", self.lon_key: "lon", self.alt_key: "alt", self.time_key: "time"}
        return [lookup[key] for key in self.dataset if key in lookup]

    @property
    def available_keys(self):
        """available data keys"""
        dim_keys = [self.lat_key, self.lon_key, self.alt_key, self.time_key]
        return [key for key in self.dataset if key not in dim_keys]


# TODO
@common_doc(COMMON_DATA_DOC)
class Rasterio(DataSource):
    r"""Create a DataSource using Rasterio.
 
    Parameters
    ----------
    source : str, :class:`io.BytesIO`
        Path to the data source
    band : int
        The 'band' or index for the variable being accessed in files such as GeoTIFFs

    Attributes
    ----------
    dataset : :class:`rasterio._io.RasterReader`
        A reference to the datasource opened by rasterio
    native_coordinates : :class:`podpac.Coordinates`
        {native_coordinates}


    Notes
    ------
    The source could be a path to an s3 bucket file, e.g.: s3://landsat-pds/L8/139/045/LC81390452014295LGN00/LC81390452014295LGN00_B1.TIF  
    In that case, make sure to set the environmental variable: 
    * Windows: set CURL_CA_BUNDLE=<path_to_conda_env>\Library\ssl\cacert.pem
    * Linux: export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
    """

    source = tl.Union([tl.Unicode(), tl.Instance(BytesIO)]).tag(readonly=True)
    dataset = tl.Any().tag(readonly=True)

    # node attrs
    band = tl.CInt(1).tag(attr=True)

    @tl.default("dataset")
    def _open_dataset(self):
        """Opens the data source
        
        Returns
        -------
        :class:`rasterio.io.DatasetReader`
            Rasterio dataset
        """

        # TODO: dataset should not open by default
        # prefer with as: syntax

        if isinstance(self.source, BytesIO):
            # https://rasterio.readthedocs.io/en/latest/topics/memory-files.html
            # TODO: this is still not working quite right - likely need to work
            # out the BytesIO format or how we are going to read/write in memory
            with rasterio.MemoryFile(self.source) as memfile:
                return memfile.open(driver="GTiff")

        # local file
        else:
            return rasterio.open(self.source)

    def close_dataset(self):
        """Closes the file for the datasource
        """
        self.dataset.close()

    @common_doc(COMMON_DATA_DOC)
    def get_native_coordinates(self):
        """{get_native_coordinates}
        
        The default implementation tries to find the lat/lon coordinates based on dataset.affine.
        It cannot determine the alt or time dimensions, so child classes may
        have to overload this method.
        """

        # check to see if the coordinates are rotated used affine
        affine = self.dataset.transform
        if affine[1] != 0.0 or affine[3] != 0.0:
            raise NotImplementedError("Rotated coordinates are not yet supported")

        try:
            crs = self.dataset.crs["init"].upper()
        except:
            crs = None

        # get bounds
        left, bottom, right, top = self.dataset.bounds

        # rasterio reads data upside-down from coordinate conventions, so lat goes from top to bottom
        lat = UniformCoordinates1d(top, bottom, size=self.dataset.height, name="lat")
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
        raster_data = self.dataset.read(self.band, out_shape=tuple(coordinates.shape), window=window)

        # set raster data to output array
        data.data.ravel()[:] = raster_data.ravel()
        return data

    @property
    def band_count(self):
        """The number of bands

        Returns
        -------
        int
            The number of bands in the dataset
        """

        if not hasattr(self, "_band_count"):
            self._band_count = self.dataset.count

        return self._band_count

    @property
    def band_descriptions(self):
        """A description of each band contained in dataset.tags
        
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
        if (not hasattr(key, "__iter__") or isinstance(key, string_types)) and (
            not hasattr(value, "__iter__") or isinstance(value, string_types)
        ):
            key = [key]
            value = [value]

        match = np.ones(self.band_count, bool)
        for k, v in zip(key, value):
            match = match & (np.array(self.band_keys[k]) == v)
        matches = np.where(match)[0] + 1

        return matches
