"""
Airmoss summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import re
from collections import OrderedDict

import requests
from bs4 import BeautifulSoup
import numpy as np
import traitlets as tl

import podpac
from podpac.compositor import OrderedCompositor
from podpac.data import PyDAP


class AirMOSS_Source(PyDAP):
    """Summary

    Attributes
    ----------
    source : str
        URL of the OpenDAP server.
    data_key : str
        PyDAP 'key' for the data to be retrieved from the server. Default 'sm1'.
    """

    data_key = tl.Unicode("sm1").tag(attr=True)
    nan_vals = [-9999.0]

    @podpac.cached_property(use_cache_ctrl=True)
    @tl.default("native_coordinates")
    def _default_native_coordinates(self):
        lon = self.dataset["lon"]
        lat = self.dataset["lat"]
        t = self.dataset["time"]

        lons = podpac.crange(lon[0], lon[-1], lon[1] - lon[0])
        lats = podpac.crange(lat[0], lat[-1], lat[1] - lat[0])

        date_url_re = re.compile("[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}")
        base_date = date_url_re.search(t.attributes["units"]).group()
        times = t.astype("timedelta64[h]") + np.array(base_date, "datetime64")

        return podpac.Coordinates([times, lats, lons], dims=["time", "lat", "lon"])

    def get_data(self, coordinates, coordinates_index):
        data = self.dataset[self.datakey].array[tuple(coordinates_index)]
        d = self.create_output_array(coordinates, data=data.reshape(coordinates.shape))
        return d


class AirMOSS_Site(OrderedCompositor):
    """Summary

    Attributes
    ----------
    product : TYPE
        Description
    site : TYPE
        Description
    """

    product = tl.Enum(["L4RZSM"], default_value="L4RZSM").tag(attr=True)
    site = tl.Unicode("").tag(attr=True)
    base_dir_url = "https://thredds.daac.ornl.gov/thredds/catalog/ornldaac/1421/catalog.html"

    @podpac.cached_property
    @tl.default("native_coordinates")
    def _default_native_coordinates(self):
        lon = self.dataset["lon"]
        lat = self.dataset["lat"]
        times = self.available_dates
        lons = podpac.crange(lon[0], lon[-1], lon[1] - lon[0])
        lats = podpac.crange(lat[0], lat[-1], lat[1] - lat[0])
        return podpac.Coordinates([times, lats, lons], dims=["time", "lat", "lon"])

    @podpac.cached_property
    def available_dates(self):
        date_url_re = re.compile("[0-9]{8}")
        soup = BeautifulSoup(requests.get(self.base_dir_url).text, "lxml")
        times = []
        for aa in soup.find_all("a"):
            text = aa.get_text()
            if self.site in text:
                m = date_url_re.search(text)
                if m:
                    t = m.group()
                    times.append(np.datetime64("-".join([t[:4], t[4:6], t[6:]])))
        return np.array(times.sort())


class AirMOSS(OrderedCompositor):
    """Summary

    Attributes
    ----------
    product : TYPE
        Description
    """

    product = tl.Enum(["L4RZSM"], default_value="L4RZSM").tag(attr=True)
    base_dir_url = "https://thredds.daac.ornl.gov/thredds/catalog/ornldaac/1421/catalog.html"

    @podpac.cached_property
    def available_sites(self):
        site_url_re = re.compile(self.product + "_.*_" + "[0-9]{8}.*\.nc4")
        soup = BeautifulSoup(requests.get(self.base_dir_url).text, "lxml")
        sites = OrderedDict()
        for aa in soup.find_all("a"):
            text = aa.get_text()
            m = site_url_re.match(text)
            if m:
                site = text.split("_")[1]
                sites[site] = 1 + sites.get(site, 0)
        return sites


if __name__ == "__main__":
    # source
    source = "https://thredds.daac.ornl.gov/thredds/dodsC/ornldaac/1421/L4RZSM_BermsP_20121025_v5.nc4"
    am_source = AirMOSS_Source(source=source, interpolation="nearest_preview")
    print(am_source.native_coordinates)

    coords = am_source.native_coordinates[::10, ::10]
    o = source_node.eval(coords)
    print(o)

    # product and site
    am_site = AirMOSS_Site(product="L4RZSM", site="BermsP", interpolation="nearest_preview")
    print(am_site.available_dates)

    coords = podpac.Coordinates([coords["lat"], coords["lon"], am.available_dates[::10]], dims=["lat", "lon", "time"])
    o = am_site.eval(coords)
    print(o)

    # product
    am = AirMoss(product="L4RZSM")
    print(am_site.available_sites)
    o = am.eval(coords)
    print(o)

    print("Done")
