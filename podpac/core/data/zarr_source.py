import traitlets as tl
import numpy as np

from lazy_import import lazy_module, lazy_class

zarr = lazy_module("zarr")
zarrGroup = lazy_class("zarr.Group")
s3fs = lazy_module("s3fs")

from podpac.core.authentication import S3Mixin
from podpac.core.utils import common_doc, cached_property
from podpac.core.data.datasource import COMMON_DATA_DOC, DATA_DOC
from podpac.core.data.file_source import BaseFileSource, FileKeysMixin


class Zarr(S3Mixin, FileKeysMixin, BaseFileSource):
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
    coordinate_index_type = "slice"

    def _get_store(self):
        if self.source.startswith("s3://"):
            root = self.source.strip("s3://")
            kwargs = {"region_name": self.aws_region_name}
            s3 = s3fs.S3FileSystem(key=self.aws_access_key_id, secret=self.aws_secret_access_key, client_kwargs=kwargs)
            s3map = s3fs.S3Map(root=root, s3=s3, check=False)
            store = s3map
        else:
            store = str(self.source)  # has to be a string in Python2.7 for local files
        return store

    @cached_property
    def dataset(self):
        store = self._get_store()

        try:
            return zarr.open(store, mode=self.file_mode)
        except ValueError:
            raise ValueError("No Zarr store found at path '%s'" % self.source)

    # -------------------------------------------------------------------------
    # public api methods
    # -------------------------------------------------------------------------

    @cached_property
    def dims(self):
        if not isinstance(self.data_key, list):
            key = self.data_key
        else:
            key = self.data_key[0]
        try:
            return self.dataset[key].attrs["_ARRAY_DIMENSIONS"]
        except:
            lookup = {self.lat_key: "lat", self.lon_key: "lon", self.alt_key: "alt", self.time_key: "time"}
            return [lookup[key] for key in self.dataset if key in lookup]

    def _add_keys(self, base_keys):
        keys = base_keys.copy()
        for bk in base_keys:
            try:
                new_keys = [bk + "/" + k for k in self.dataset[bk].keys()]
                keys.extend(new_keys)

                # Remove the group key
                keys.pop(keys.index(bk))
            except AttributeError:
                pass
        return keys

    @cached_property
    def keys(self):
        keys = list(self.dataset.keys())
        full_keys = self._add_keys(keys)
        while keys != full_keys:
            keys = full_keys.copy()
            full_keys = self._add_keys(keys)

        return full_keys

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        data = self.create_output_array(coordinates)
        if not isinstance(self.data_key, list):
            data[:] = self.dataset[self.data_key][coordinates_index]
        else:
            for key, name in zip(self.data_key, self.outputs):
                data.sel(output=name)[:] = self.dataset[key][coordinates_index]
        return data
