import os
import traitlets as tl
import numpy as np

from lazy_import import lazy_module, lazy_class, lazy_function

zarr = lazy_module("zarr")
zarr_open = lazy_function("zarr.convenience.open")
zarr_open_consolidated = lazy_function("zarr.convenience.open_consolidated")
zarrGroup = lazy_class("zarr.Group")

from podpac.core.authentication import S3Mixin
from podpac.core.utils import common_doc, cached_property
from podpac.core.data.datasource import COMMON_DATA_DOC, DATA_DOC
from podpac.core.data.file_source import BaseFileSource, FileKeysMixin
from podpac.core.interpolation.interpolation import InterpolationMixin


class ZarrRaw(S3Mixin, FileKeysMixin, BaseFileSource):
    """Create a DataSource node using zarr.

    Attributes
    ----------
    source : str
        Path to the Zarr archive
    file_mode : str, optional
        Default is 'r'. The mode used to open the Zarr archive. Options are r, r+, w, w- or x, a.
    dataset : zarr.Group
        The h5py file object used to read the file
    coordinates : :class:`podpac.Coordinates`
        {coordinates}
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

    See Also
    --------
    Zarr : Interpolated Zarr Datasource for general use.
    """

    # Doesnt support other dims
    file_mode = tl.Unicode(default_value="r").tag(readonly=True)
    coordinate_index_type = "slice"
    _consolidated = False

    def _get_store(self):
        if self.source.startswith("s3://"):
            s3fs = lazy_module("s3fs")
            root = self.source.strip("s3://")
            s3map = s3fs.S3Map(root=root, s3=self.s3, check=False)
            store = s3map
        else:
            store = str(self.source)  # has to be a string in Python2.7 for local files
        return store

    def chunk_exists(self, index=None, chunk_str=None, data_key=None, chunks=None, list_dir=[]):
        """
        Test to see if a chunk exists for a particular slice.
        Note: Only the start of the index is used.

        Parameters
        -----------
        index: tuple(slice), optional
            Default is None. A tuple of slices indicating the data that the users wants to access
        chunk_str: str, optional
            Default is None. A string equivalent to the filename of the chunk (.e.g. "1.0.5")
        data_key: str, optional
            Default is None. The data_key for the zarr array that will be queried.
        chunks: list, optional
            Defaut is None. The chunk structure of the zarr array. If not provided will use self.dataset[data_key].chunks
        list_dir: list, optional
            A list of existing paths -- used in lieu of 'exist' calls
        """

        if not data_key:
            data_key = ""

        if not chunks:
            if data_key:
                chunks = self.dataset[data_key].chunks
            else:
                chunks = self.dataset.chunks

        if index:
            chunk_str = ".".join([str(int(s.start // chunks[i])) for i, s in enumerate(index)])

        if not index and not chunk_str:
            raise ValueError("Either the index or chunk_str needs to be specified")

        path = os.path.join(self.source, data_key, chunk_str)
        if self.source.startswith("s3:"):
            path = path.replace("\\", "/")
        else:
            path = path.replace("/", os.sep)
        if list_dir:
            return path in list_dir

        if self.source.startswith("s3:"):
            fs = self.s3
        else:
            fs = os.path

        return fs.exists(path)

    def list_dir(self, data_key=None):
        za = self.dataset
        if data_key:
            za = za[data_key]
        else:
            data_key = ""

        path = os.path.join(self.source, data_key)
        if self.source.startswith("s3:"):
            path = path.replace("\\", "/")
            ld = ["s3://" + p for p in self.s3.ls(path)]
        else:
            path = path.replace("/", os.sep)
            ld = [os.path.join(path, p) for p in os.listdir(path)]

        return ld

    @cached_property
    def dataset(self):
        store = self._get_store()
        try:
            # import zarr.open
            # import zarr.open_consolidated
            if self.file_mode == "r":
                try:
                    self._consolidated = True
                    return zarr_open_consolidated(store)
                except KeyError:
                    pass  # No consolidated metadata available
            self._consolidated = False
            return zarr_open(store, mode=self.file_mode)
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
        """{get_data}"""
        data = self.create_output_array(coordinates)
        if not isinstance(self.data_key, list):
            data[:] = self.dataset[self.data_key][coordinates_index]
        else:
            for key, name in zip(self.data_key, self.outputs):
                data.sel(output=name)[:] = self.dataset[key][coordinates_index]
        return data


class Zarr(InterpolationMixin, ZarrRaw):
    """Zarr Datasource with Interpolation."""

    pass
