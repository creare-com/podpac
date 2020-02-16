import pandas as pd
import traitlets as tl

from podpac.core.utils import common_doc
from podpac.core.coordinates import Coordinates, StackedCoordinates
from podpac.core.data.datasource import COMMON_DATA_DOC, DATA_DOC
from podpac.core.data.file_source import BaseFileSource, FileKeysMixin, LoadFileMixin


@common_doc(COMMON_DATA_DOC)
class CSV(FileKeysMixin, LoadFileMixin, BaseFileSource):
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
    output_keys = tl.Union([tl.List(tl.Unicode()), tl.List(tl.Int())], allow_none=True).tag(attr=True)

    @tl.default("data_key")
    def _default_data_key(self):
        return super(CSV, self)._default_data_key()

    @tl.default("output_keys")
    def _default_output_keys(self):
        return super(CSV, self)._default_output_keys()

    # -------------------------------------------------------------------------
    # public api methods
    # -------------------------------------------------------------------------

    def open_dataset(self, f):
        return pd.read_csv(f, parse_dates=True, infer_datetime_format=True, header=self.header)

    @property
    def dims(self):
        """ list of dataset coordinate dimensions """
        if not hasattr(self, "_dims"):
            lookup = {
                self._get_key(self.lat_key): "lat",
                self._get_key(self.lon_key): "lon",
                self._get_key(self.alt_key): "alt",
                self._get_key(self.time_key): "time",
            }
            self._dims = [lookup[key] for key in self.dataset.columns if key in lookup]

        return self._dims

    @property
    def keys(self):
        """available data keys"""
        return self.dataset.columns.tolist()

    @common_doc(COMMON_DATA_DOC)
    @property
    def native_coordinates(self):
        """{get_native_coordinates}
        
        Note: CSV files have StackedCoordinates.
        """

        coords = super(CSV, self).native_coordinates
        if len(coords) == 1:
            return coords
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

    # -------------------------------------------------------------------------
    # helper methods
    # -------------------------------------------------------------------------

    def _lookup_key(self, dim):
        lookup = {"lat": self.lat_key, "lon": self.lon_key, "alt": self.alt_key, "time": self.time_key}
        return self._get_key(lookup[dim])

    def _get_key(self, key):
        return self.dataset.columns[key] if isinstance(key, int) else key

    def _get_col(self, key):
        return key if isinstance(key, int) else self.dataset.columns.get_loc(key)
