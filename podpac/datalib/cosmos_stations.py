from __future__ import division, unicode_literals, print_function, absolute_import

import traitlets as tl
import re
import numpy as np
from dateutil import parser
import requests
import json

try:
    import cPickle  # Python 2.7
except:
    import _pickle as cPickle
from io import StringIO

from podpac.core.utils import optional_import

# Optional dependencies
from lazy_import import lazy_module, lazy_class

bs4 = lazy_module("bs4")

import podpac
from podpac.core.utils import cached_property


class COSMOSStation(podpac.data.DataSource):
    url = tl.Unicode("http://cosmos.hwr.arizona.edu/Probes/StationDat/")
    station_data = tl.Dict().tag(attr=True)

    raw_data = tl.Unicode()

    @tl.default("interpolation")
    def _interpolation_default(self):
        return {"method": "nearest", "params": {"spatial_tolerance": 1.1}}

    @tl.default("raw_data")
    def get_raw_data(self):
        r = requests.get(self.station_data_url)
        return r.text

    data_columns = tl.List()

    @tl.default("data_columns")
    def _default_data_colums(self):
        return self.raw_data.split("\n", 1)[0].split(" ")

    @property
    def station_data_url(self):
        sitenumber = self.station_data["sitenumber"]
        return self.url + str(sitenumber) + "/smcounts.txt"

    def get_data(self, coordinates, coordinates_index):
        data = np.loadtxt(StringIO(self.raw_data), skiprows=1, usecols=self.data_columns.index("SOILM"))[
            coordinates_index[0]
        ]
        data[data > 100] = np.nan
        data[data < 0] = np.nan
        data /= 100.0  # Make it fractional
        return self.create_output_array(coordinates, data=data[:, None])

    def get_native_coordinates(self):
        lat_lon = [float(v) for v in self.station_data["location"].split(",")]
        time = np.loadtxt(
            StringIO(self.raw_data),
            skiprows=1,
            usecols=[self.data_columns.index("YYYY-MM-DD"), self.data_columns.index("HH:MM")],
            dtype=str,
        )
        time = np.array([t[0] + "T" + t[1] for t in time], np.datetime64)
        c = podpac.Coordinates([time, lat_lon], ["time", "lat_lon"], ctype="point")
        c["lat"].ctype = "point"
        c["lon"].ctype = "point"
        return c

    def __repr__(self):
        return "%s, %s (%s)" % (self.station_data["label"], self.station_data["network"], self.station_data["location"])


class COSMOSStations(podpac.compositor.OrderedCompositor):
    url = tl.Unicode("http://cosmos.hwr.arizona.edu/Probes/")
    stations_url = tl.Unicode("sitesNoLegend.js")

    @cached_property
    def stations_data(self):
        return self._stations_data()

    @cache_func("stations_data")
    def _stations_data(self):
        url = self.url + self.stations_url
        r = requests.get(url)
        t = r.text
        # Fix the JSON
        t_f = re.sub(':\s?",', ': "",', t)  # Missing closing parenthesis
        if t_f[-5:] == ",\n]}\n":  # errant comma
            t_f = t_f[:-5] + "\n]}\n"
        stations = json.loads(t_f)
        return stations

        if t[-55:-52] == ' ",':  # missing close quote
            t = t[:-55] + t[-55:].replace(' ",', '"",')
        stations = json.loads(t)
        return stations

    @tl.default("sources")
    def _sources_default(self):
        return np.array([COSMOSStation(station_data=item) for item in self.stations_data["items"]])

    @tl.default("source_coordinates")
    def _source_coordinates_default(self):
        lat_lon = np.array([s["location"].split(",") for s in self.stations_data["items"]], dtype=float)
        c = podpac.Coordinates([lat_lon.T], ["lat_lon"], ctype="point")
        c["lat"].ctype = "point"
        c["lon"].ctype = "point"
        return c

    def label_from_latlon(self, lat_lon):
        labels_map = {s["location"]: s["label"] for s in self.stations_data["items"]}
        labels = [labels_map[str(ll.item())[1:-1].replace(" ", "")] for ll in lat_lon]
        return labels


if __name__ == "__main__":
    coords = podpac.Coordinates([podpac.clinspace(40, 46, 3, "lat"), podpac.clinspace(-78, -68, 3, "lon")])
    cs = COSMOSStations()
    ci = cs.source_coordinates.intersect(coords)
    ce = podpac.coordinates.merge_dims(
        [podpac.Coordinates([podpac.crange("2018-05-01", "2018-06-01", "1,D", "time")]), ci]
    )
    o = cs.eval(ce)

    from matplotlib.pyplot import plot, show, legend, ylim, ylabel, xlabel

    plot(o.time, o.data, "-")
    ylim(0, 1)
    legend(cs.label_from_latlon(o.lat_lon))
    ylabel("Soil Moisture ($m^3/m^3$)")
    xlabel("Date")
    show()

    print ("Done")
