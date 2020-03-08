import traitlets as tl

from lazy_import import lazy_module, lazy_class

h5py = lazy_module("h5py")

from podpac.core.utils import common_doc, cached_property
from podpac.core.data.datasource import COMMON_DATA_DOC, DATA_DOC
from podpac.core.data.file_source import BaseFileSource, FileKeysMixin


@common_doc(COMMON_DATA_DOC)
class H5PY(FileKeysMixin, BaseFileSource):
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

    @cached_property
    def dataset(self):
        return h5py.File(self.source, self.file_mode)

    def close_dataset(self):
        """Closes the file. """
        self.dataset.close()

    # -------------------------------------------------------------------------
    # public api methods
    # -------------------------------------------------------------------------

    @cached_property
    def dims(self):
        """ dataset coordinate dims """
        try:
            key = self.data_key or self.output_keys[0]
            return self.dataset[key].attrs["_ARRAY_DIMENSIONS"]
        except:
            lookup = {self.lat_key: "lat", self.lon_key: "lon", self.alt_key: "alt", self.time_key: "time"}
            return [lookup[key] for key in self.keys if key in lookup]

    @cached_property
    def keys(self):
        return H5PY._find_h5py_keys(self.dataset)

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

    # -------------------------------------------------------------------------
    # additional methods and properties
    # -------------------------------------------------------------------------

    def attrs(self, key="/"):
        """Dataset or group key for which attributes will be summarized.
        """
        return dict(self.dataset[key].attrs)

    @staticmethod
    def _find_h5py_keys(obj, keys=[]):
        # recursively find keys

        if isinstance(obj, (h5py.Group, h5py.File)):
            for k in obj.keys():
                keys = H5PY._find_h5py_keys(obj[k], keys)
        else:
            keys.append(obj.name)
            return keys
        keys = sorted(list(set(keys)))
        return keys
