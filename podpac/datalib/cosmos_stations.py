from __future__ import division, unicode_literals, print_function, absolute_import

import re
import json
import logging
from six import string_types
from dateutil import parser
from io import StringIO

try:
    import cPickle  # Python 2.7
except:
    import _pickle as cPickle

import numpy as np
import traitlets as tl

# Optional dependencies
from lazy_import import lazy_module

bs4 = lazy_module("bs4")

import podpac
from podpac.core.utils import _get_from_url, cached_property
from podpac.data import DataSource
from podpac.compositor import TileCompositor


_logger = logging.getLogger(__name__)


def _convert_str_to_vals(properties):
    IGNORE_KEYS = ["sitenumber"]
    for k, v in properties.items():
        if not isinstance(v, string_types) or k in IGNORE_KEYS:
            continue
        try:
            if "," in v:
                properties[k] = tuple([float(vv) for vv in v.split(",")])
            else:
                properties[k] = float(v)
        except ValueError:
            try:
                properties[k] = np.datetime64(v)
            except ValueError:
                pass
    return properties


class COSMOSStation(DataSource):
    _repr_keys = ["label", "network", "location"]

    url = tl.Unicode("http://cosmos.hwr.arizona.edu/Probes/StationDat/")
    station_data = tl.Dict().tag(attr=True)

    @cached_property
    def raw_data(self):
        _logger.info("Downloading station data from {}".format(self.station_data_url))

        r = _get_from_url(self.station_data_url)
        if r is None:
            raise ConnectionError(
                "COSMOS data cannot be retrieved. Is the site {} down?".format(self.station_calibration_url)
            )
        return r.text

    @cached_property
    def data_columns(self):
        return self.raw_data.split("\n", 1)[0].split(" ")

    @property
    def site_number(self):
        return str(self.station_data["sitenumber"])

    @property
    def station_data_url(self):
        return self.url + self.site_number + "/smcounts.txt"

    @property
    def station_calibration_url(self):
        return self.url + self.site_number + "/calibrationInfo.php"

    @property
    def station_properties_url(self):
        return self.url + self.site_number + "/index.php"

    def get_data(self, coordinates, coordinates_index):
        data = np.loadtxt(StringIO(self.raw_data), skiprows=1, usecols=self.data_columns.index("SOILM"))[
            coordinates_index[0]
        ]
        data[data > 100] = np.nan
        data[data < 0] = np.nan
        data /= 100.0  # Make it fractional
        return self.create_output_array(coordinates, data=data.reshape(coordinates.shape))

    def get_coordinates(self):
        lat_lon = self.station_data["location"]
        time = np.atleast_2d(
            np.loadtxt(
                StringIO(self.raw_data),
                skiprows=1,
                usecols=[self.data_columns.index("YYYY-MM-DD"), self.data_columns.index("HH:MM")],
                dtype=str,
            )
        )
        if time.size == 0:
            time = np.datetime64("NaT")
        else:
            time = np.array([t[0] + "T" + t[1] for t in time], np.datetime64)
        c = podpac.Coordinates([time, [lat_lon[0], lat_lon[1]]], ["time", ["lat", "lon"]])
        return c

    @property
    def label(self):
        return self.station_data["label"]

    @property
    def network(self):
        return self.station_data["network"]

    @property
    def location(self):
        return self.station_data["location"]

    @cached_property(use_cache_ctrl=True)
    def calibration_data(self):
        cd = _get_from_url(self.station_calibration_url)
        if cd is None:
            raise ConnectionError(
                "COSMOS data cannot be retrieved. Is the site {} down?".format(self.station_calibration_url)
            )
        cd = cd.json()
        cd["items"] = [_convert_str_to_vals(i) for i in cd["items"]]
        return cd

    @cached_property(use_cache_ctrl=True)
    def site_properties(self):
        r = _get_from_url(self.station_properties_url)
        if r is None:
            raise ConnectionError(
                "COSMOS data cannot be retrieved. Is the site {} down?".format(self.station_properties_url)
            )
        soup = bs4.BeautifulSoup(r.text, "lxml")
        regex = re.compile("Soil Organic Carbon")
        loc = soup.body.findAll(text=regex)[0].parent.parent
        label, value = loc.findAll("div")
        labels = [l.strip() for l in label.children if "br" not in str(l)]
        values = [l.strip() for l in value.children if "br" not in str(l) and l.strip() != ""]

        properties = {k: v for k, v in zip(labels, values)}

        return _convert_str_to_vals(properties)


class COSMOSStations(TileCompositor):
    url = tl.Unicode("http://cosmos.hwr.arizona.edu/Probes/")
    stations_url = tl.Unicode("sitesNoLegend.js")
    dims = ["lat", "lon", "time"]

    from podpac.style import Style

    style = Style(colormap="jet")

    ## PROPERTIES
    @cached_property(use_cache_ctrl=True)
    def _stations_data_raw(self):
        url = self.url + self.stations_url
        r = _get_from_url(url)
        if r is None:
            raise ConnectionError("COSMOS data cannot be retrieved. Is the site {} down?".format(url))

        t = r.text

        # Fix the JSON
        t_f = re.sub(':\s?",', ': "",', t)  # Missing closing parenthesis
        if t_f[-5:] == ",\n]}\n":  # errant comma
            t_f = t_f[:-5] + "\n]}\n"

        return t_f

    @cached_property
    def stations_data(self):
        stations = json.loads(self._stations_data_raw)
        stations["items"] = [_convert_str_to_vals(i) for i in stations["items"]]
        return stations

    @cached_property(use_cache_ctrl=True)
    def source_coordinates(self):
        lat_lon = np.array(self.stations_value("location"))[self.has_data]
        c = podpac.Coordinates([[lat_lon[:, 0], lat_lon[:, 1]]], ["lat_lon"])
        return c

    @cached_property
    def has_data(self):
        return ~(np.array(self.stations_value("lastdat")) == "YYYY-MM-DD")

    @cached_property
    def sources(self):
        return np.array([COSMOSStation(station_data=item) for item in self.stations_data["items"]])[self.has_data]

    @property
    def available_data_keys(self):
        return list(self.stations_data["items"][0].keys())

    ## UTILITY FUNCTIONS
    def stations_value(self, key, stations_data=None):
        """Returns a list of values for all the station for a particular key

        Parameters
        -----------
        key: str
           Key describing the station data. See self.available_data_keys for available keys.

        Returns
        --------
        list
            A list of the values for the keys for each station
        """
        if key not in self.available_data_keys:
            raise ValueError("Input key {} is not in available keys {}".format(key, self.available_data_keys))

        return self._stations_value(key, stations_data)

    def _stations_value(self, key, stations_data=None):
        """helper function for stations_value"""
        if stations_data is None:
            stations_data = self.stations_data

        return [i[key] for i in stations_data["items"]]

    @property
    def stations_label(self):
        return self.stations_value("label")

    def label_from_latlon(self, lat_lon):
        """Returns the COSMOS station's label given it's lat/lon coordinates

        Parameters
        -----------
        lat_lon : podpac.Coordinates
            The lat/lon locations whose station name will be returned. Note, the lat/lon coordinates have to match
            exactly the coordinates given in station_data[N]['location'], where N is the station.
            This should be Coordinates object with 'lat_lon' stacked coordinates as one of the dimensions.

        Returns
        --------
        list
            List of COSMOS station names corresponding to the given coordinates. If a coordinate has no match, then
            "None" is returned.
        """
        if "lon_lat" in lat_lon.dims:
            lat_lon = lat_lon.transpose("lon_lat")
        elif "lat_lon" not in lat_lon.dims:
            raise ValueError("The coordinates object must have a stacked 'lat_lon' dimension.")

        labels_map = {s["location"]: s["label"] for s in self.stations_data["items"]}
        labels = [labels_map.get(ll, None) for ll in lat_lon.xcoords["lat_lon"]]
        return labels

    def latlon_from_label(self, label):
        """Returns the lat/lon coordinates of COSMOS stations that match the given labels

        Parameters
        ------------
        label: str, list
            Strings that partially describe a COSMOS station label.

        Returns
        --------
        podpac.Coordinates
            The coordinates of the COSMOS stations matching the input data
        """
        if not isinstance(label, list):
            label = [label]

        ind = self._get_label_inds(label)
        if ind.size == 0:
            return podpac.Coordinates([])  # Empty

        lat_lon = np.array(self.stations_value("location"))[ind].squeeze()
        c = podpac.Coordinates([[lat_lon[0], lat_lon[1]]], ["lat_lon"])

        return c

    def _get_label_inds(self, label):
        """Helper function to get source indices for partially matched labels"""
        ind = []
        for lab in label:
            ind.extend([i for i, l in enumerate(self.stations_label) if lab.lower() in l.lower()])

        ind = np.unique(ind)
        return ind

    def get_calibration_data(self, label=None, lat_lon=None):
        """Returns the calibration information for a station. Users must supply a label or lat_lon coordinates.

        Parameters
        ------------
        label: str, List (optional)
            Labels describing the station.

        lat_lon: podpac.Coordinates (optional)
            Coordinates of the COSMOS station. Note, this object has to have a 'lat_lon' dimension which matches exactly
            with the COSMOS stations.

        Returns
        --------
        list
            A list of dictionaries containing the calibration data for the requested stations.
        """

        if label is None and lat_lon is None:
            raise ValueError("Must supply either 'label' or 'lat_lon'")

        if lat_lon is not None:
            label = self.label_from_latlon(lat_lon)

        if isinstance(label, string_types):
            label = [label]

        inds = self._get_label_inds(label)

        return [self.sources[i].calibration_data for i in inds]

    def get_site_properties(self, label=None, lat_lon=None):
        """Returns the site properties for a station. Users must supply a label or lat_lon coordinates.

        Parameters
        ------------
        label: str, List (optional)
            Labels describing the station.

        lat_lon: podpac.Coordinates (optional)
            Coordinates of the COSMOS station. Note, this object has to have a 'lat_lon' dimension which matches exactly
            with the COSMOS stations.

        Returns
        --------
        list
            A list of dictionaries containing the properties for the requested stations.
        """

        if label is None and lat_lon is None:
            raise ValueError("Must supply either 'label' or 'lat_lon'")

        if lat_lon is not None:
            label = self.label_from_latlon(lat_lon)

        if isinstance(label, string_types):
            label = [label]

        inds = self._get_label_inds(label)

        return [self.sources[i].site_properties for i in inds]

    def get_station_data(self, label=None, lat_lon=None):
        """Returns the station data. Users must supply a label or lat_lon coordinates.

        Parameters
        ------------
        label: str, List (optional)
            Labels describing the station.

        lat_lon: podpac.Coordinates (optional)
            Coordinates of the COSMOS station. Note, this object has to have a 'lat_lon' dimension which matches exactly
            with the COSMOS stations.

        Returns
        --------
        list
            A list of dictionaries containing the data for the requested stations.
        """

        if label is None and lat_lon is None:
            raise ValueError("Must supply either 'label' or 'lat_lon'")

        if lat_lon is not None:
            label = self.label_from_latlon(lat_lon)

        if isinstance(label, string_types):
            label = [label]

        inds = self._get_label_inds(label)

        return [self.stations_data["items"][i] for i in inds]


if __name__ == "__main__":
    bounds = {"lat": [40, 46], "lon": [-78, -68]}
    cs = COSMOSStations(
        cache_ctrl=["ram", "disk"],
        interpolation={"method": "nearest", "params": {"use_selector": False, "remove_nan": True, "time_scale": "1,M"}},
    )
    csr = COSMOSStations(
        cache_ctrl=["ram", "disk"],
        interpolation={"method": "nearest", "params": {"use_selector": False, "remove_nan": True, "time_scale": "1,M"}},
    )

    sd = cs.stations_data
    ci = cs.source_coordinates.select(bounds)
    ce = podpac.coordinates.merge_dims(
        [podpac.Coordinates([podpac.crange("2018-05-01", "2018-06-01", "1,D", "time")]), ci]
    )
    cg = podpac.Coordinates(
        [
            podpac.clinspace(ci["lat"].bounds[1], ci["lat"].bounds[0], 12, "lat"),
            podpac.clinspace(ci["lon"].bounds[1], ci["lon"].bounds[0], 16, "lon"),
            ce["time"],
        ]
    )
    o = cs.eval(ce)
    o_r = csr.eval(ce)
    og = cs.eval(cg)

    # Test helper functions
    labels = cs.stations_label
    lat_lon = cs.latlon_from_label("Manitou")
    labels = cs.label_from_latlon(lat_lon)
    lat_lon2 = cs.latlon_from_label("No Match Here")
    cal = cs.get_calibration_data("Manitou")
    props = cs.get_site_properties("Manitou")

    from matplotlib import rcParams

    rcParams["axes.labelsize"] = 12
    rcParams["xtick.labelsize"] = 10
    rcParams["ytick.labelsize"] = 10
    rcParams["legend.fontsize"] = 8
    rcParams["lines.linewidth"] = 2
    rcParams["font.size"] = 12

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    fig = plt.figure(figsize=(6.5, 3), dpi=300)
    plt.plot(o.time, o.data, "o-")
    ax = plt.gca()
    plt.ylim(0, 1)
    plt.legend(cs.label_from_latlon(ce))
    # plt.plot(o_r.time, o_r.data, ".-")
    plt.ylabel("Soil Moisture ($m^3/m^3$)")
    plt.xlabel("Date")
    # plt.xticks(rotation=90)
    fig.autofmt_xdate()
    ax.fmt_xdata = mdates.DateFormatter("%m-%d")
    plt.title("COSMOS Data for 2018 over lat (40, 46) by lon (-78,-68)")
    plt.tight_layout()
    plt.show()

    print("Done")
