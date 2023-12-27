import xarray as xr
import traitlets as tl

from podpac.core.utils import common_doc, cached_property
from podpac.core.data.datasource import COMMON_DATA_DOC, DATA_DOC
from podpac.core.data.file_source import BaseFileSource, FileKeysMixin, LoadFileMixin
from podpac.core.interpolation.interpolation import InterpolationMixin
from podpac.core.coordinates.coordinates import Coordinates


@common_doc(COMMON_DATA_DOC)
class DatasetRaw(FileKeysMixin, LoadFileMixin, BaseFileSource):
    """Create a DataSource node using xarray.open_dataset.

    Attributes
    ----------
    source : str
        Path to the dataset file.
        In addition to local paths, file://, http://, ftp://, and s3:// transport protocols are supported.
    dataset : xarray.Dataset
        Dataset object.
    coordinates : :class:`podpac.Coordinates`
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
    selection : dict
        Extra dimension(s) selection. Select one coordinate by index for each extra dimension.
        This is necessary when the data contains dimensions other than 'lat', 'lon', 'time', and 'alt'.
        For example, with dims `('lat', 'lon', 'channel')`, use `{{'channel': 1}}`.
    infer_podpac_coords: bool
        If True, load the coordinates from the dataset coords directly. Default is False.
        This is particularly useful if the file was saved using PODPAC.

    See Also
    --------
    Dataset : Interpolated xarray dataset source for general use.
    """

    # selection lets you use other dims
    # dataset = tl.Instance(xr.Dataset).tag(readonly=True)
    selection = tl.Dict(allow_none=True, default_value=None).tag(attr=True)
    infer_podpac_coords = tl.Bool(False).tag(attr=True)
    decode_cf = tl.Bool(True)
    coordinate_index_type = "xarray"

    # -------------------------------------------------------------------------
    # public api properties and methods
    # -------------------------------------------------------------------------

    def open_dataset(self, fp):
        return xr.open_dataset(fp, decode_cf=self.decode_cf)

    def close_dataset(self):
        super(DatasetRaw, self).close_dataset()
        self.dataset.close()

    @cached_property
    def dims(self):
        """dataset coordinate dims"""
        lookup = {self.lat_key: "lat", self.lon_key: "lon", self.alt_key: "alt", self.time_key: "time"}
        return [lookup[dim] for dim in self.dataset.dims if dim in lookup]

    @cached_property
    def keys(self):
        return list(self.dataset.keys())

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}"""

        if not isinstance(self.data_key, list):
            data = self.dataset[self.data_key][self.selection or {}]
            data = data.transpose(*[self._lookup_key(dim) for dim in self.dims])
        else:
            data = self.dataset[self.data_key].to_array(dim="output")[self.selection or {}]
            tdims = tuple(self.dataset.dims) + ("output",)
            data = data.transpose(*tdims)

        return self.create_output_array(coordinates, data[coordinates_index])

    @common_doc(COMMON_DATA_DOC)
    def get_coordinates(self):
        """{get_coordinates}"""
        if self.infer_podpac_coords:
            return Coordinates.from_xarray(self.dataset, crs=self.crs)
        return super().get_coordinates()


class Dataset(InterpolationMixin, DatasetRaw):
    """xarray dataset source with interpolation."""

    pass
