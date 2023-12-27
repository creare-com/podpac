import pandas as pd
import traitlets as tl

from podpac.core.utils import common_doc, cached_property
from podpac.core.coordinates import Coordinates, StackedCoordinates
from podpac.core.data.datasource import COMMON_DATA_DOC, DATA_DOC
from podpac.core.data.file_source import BaseFileSource, FileKeysMixin, LoadFileMixin
from podpac.core.interpolation.interpolation import InterpolationMixin


@common_doc(COMMON_DATA_DOC)
class CSVRaw(FileKeysMixin, LoadFileMixin, BaseFileSource):
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
    coordinates : :class:`podpac.Coordinates`
        {coordinates}
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

    See Also
    --------
    CSV : Interpolated CSV file datasource for general use.
    """

    # No support here for custom Dimension names? selection in dataset_source.py
    header = tl.Any(default_value=0).tag(attr=True)
    lat_key = tl.Union([tl.Unicode(), tl.Int()], default_value="lat").tag(attr=True)
    lon_key = tl.Union([tl.Unicode(), tl.Int()], default_value="lon").tag(attr=True)
    time_key = tl.Union([tl.Unicode(), tl.Int()], default_value="time").tag(attr=True)
    alt_key = tl.Union([tl.Unicode(), tl.Int()], default_value="alt").tag(attr=True)
    data_key = tl.Union([tl.Unicode(), tl.Int(), tl.List(trait=tl.Unicode()), tl.List(trait=tl.Int())]).tag(attr=True)

    @tl.default("data_key")
    def _default_data_key(self):
        return super(CSVRaw, self)._default_data_key()

    @tl.validate("data_key")
    def _validate_data_key(self, d):
        keys = d["value"]
        if not isinstance(keys, list):
            keys = [d["value"]]

        if isinstance(keys[0], int):
            for col in keys:
                if col not in self.available_data_cols:
                    raise ValueError("Invalid data_key %d, available columns are %s" % (col, self.available_data_cols))
        else:
            for key in keys:
                if key not in self.available_data_keys:
                    raise ValueError("Invalid data_key '%s', available keys are %s" % (key, self.available_data_keys))

        return d["value"]

    @tl.default("outputs")
    def _default_outputs(self):
        if not isinstance(self.data_key, list):
            return None
        else:
            return [self._get_key(elem) for elem in self.data_key]

    # -------------------------------------------------------------------------
    # public api methods
    # -------------------------------------------------------------------------

    def open_dataset(self, f):
        return pd.read_csv(f, parse_dates=True, infer_datetime_format=True, header=self.header)

    @cached_property
    def dims(self):
        """list of dataset coordinate dimensions"""
        lookup = {
            self._get_key(self.lat_key): "lat",
            self._get_key(self.lon_key): "lon",
            self._get_key(self.alt_key): "alt",
            self._get_key(self.time_key): "time",
        }
        return [lookup[key] for key in self.dataset.columns if key in lookup]

    @cached_property
    def keys(self):
        """available data keys"""
        return self.dataset.columns.tolist()

    @cached_property
    def available_data_keys(self):
        """available data keys"""

        dim_keys = [self._get_key(key) for key in [self.lat_key, self.lon_key, self.alt_key, self.time_key]]
        keys = [key for key in self.keys if key not in dim_keys]
        if len(keys) == 0:
            raise ValueError("No data keys found in '%s'" % self.source)
        return keys

    @cached_property
    def available_data_cols(self):
        return [self._get_col(key) for key in self.available_data_keys]

    @common_doc(COMMON_DATA_DOC)
    def get_coordinates(self):
        """{get_coordinates}

        Note: CSV files have StackedCoordinates.
        """

        coords = super(CSVRaw, self).get_coordinates()
        if len(coords) == 1:
            return coords
        stacked = StackedCoordinates(list(coords.values()))
        return Coordinates([stacked], validate_crs=False, **coords.properties)

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}"""

        if not isinstance(self.data_key, list):
            I = self._get_col(self.data_key)
        else:
            I = [self._get_col(key) for key in self.data_key]
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


class CSV(InterpolationMixin, CSVRaw):
    """CSV datasource with interpolation."""

    pass
