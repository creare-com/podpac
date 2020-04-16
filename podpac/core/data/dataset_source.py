import xarray as xr
import traitlets as tl

from podpac.core.utils import common_doc, cached_property
from podpac.core.data.datasource import COMMON_DATA_DOC, DATA_DOC
from podpac.core.data.file_source import BaseFileSource, FileKeysMixin, LoadFileMixin


@common_doc(COMMON_DATA_DOC)
class Dataset(FileKeysMixin, LoadFileMixin, BaseFileSource):
    """Create a DataSource node using xarray.open_dataset.
    
    Attributes
    ----------
    source : str
        Path to the dataset file.
        In addition to local paths, file://, http://, ftp://, and s3:// transport protocols are supported.
    dataset : xarray.Dataset
        Dataset object.
    coordinates : Coordinates
        {coordinates}
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

    # dataset = tl.Instance(xr.Dataset).tag(readonly=True)
    extra_dim = tl.Dict(allow_none=True).tag(attr=True)

    @tl.default("extra_dim")
    def _default_extra_dim(self):
        return None

    # -------------------------------------------------------------------------
    # public api properties and methods
    # -------------------------------------------------------------------------

    def open_dataset(self, fp):
        return xr.open_dataset(fp)

    def close_dataset(self):
        self.dataset.close()

    @cached_property
    def dims(self):
        """dataset coordinate dims"""
        lookup = {self.lat_key: "lat", self.lon_key: "lon", self.alt_key: "alt", self.time_key: "time"}
        return [lookup[dim] for dim in self.dataset.dims]

    @cached_property
    def keys(self):
        return list(self.dataset.keys())

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """

        if not isinstance(self.data_key, list):
            data = self.dataset[self.data_key]
            data = data.transpose(*self.dataset.dims)
        else:
            data = self.dataset[self.data_key].to_array(dim="output")
            tdims = tuple(self.dataset.dims) + ("output",)
            data = data.transpose(*tdims)

        return self.create_output_array(coordinates, data.data[coordinates_index])
