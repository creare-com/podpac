"""
OGC-compliant datasources over HTTP
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import os
import re
from copy import deepcopy
from io import BytesIO

import numpy as np
import traitlets as tl

from podpac.core.settings import settings
from podpac.core.utils import common_doc, cached_property
from podpac.core.node import Node, NodeException
from podpac.core.coordinates import Coordinates
from podpac.core.coordinates import StackedCoordinates, Coordinates1d, UniformCoordinates1d, ArrayCoordinates1d


# Optional dependencies
from lazy_import import lazy_module, lazy_class

bs4 = lazy_module("bs4")
lxml = lazy_module("lxml")  # used by bs4 so want to check if it's available
owslib_wcs = lazy_module("owslib.wcs")
rasterio = lazy_module("rasterio")


class WCS2Error(NodeException):
    pass


class WCS(Node):
    source = tl.Unicode().tag(attr=True)
    layer = tl.Unicode().tag(attr=True)
    version = tl.Unicode(default_value="1.0.0.").tag(attr=True)  # TODO 1.0.0 deprecated?
    # interpolation = tl.Unicode().tag(attr=True) # TODO

    # max_size = tl.Long(default_value=None, allow_none=True) # TODO
    format = tl.Unicode(default_value="geotiff")
    crs = tl.Unicode(default_value="EPSG:4326")

    _repr_keys = ["source", "layer"]

    _requested_coordinates = tl.Instance(Coordinates, allow_none=True)
    _evaluated_coordinates = tl.Instance(Coordinates)

    @cached_property
    def client(self):
        return owslib_wcs.WebCoverageService(self.source, version=self.version)

    @cached_property
    def coordinates(self):
        # TODO select correct boundingbox by crs?

        metadata = self.client.contents[self.layer]

        # coordinates
        w, s, e, n = metadata.boundingBoxWGS84
        low = metadata.grid.lowlimits
        high = metadata.grid.highlimits
        xsize = int(high[0]) - int(low[0])
        ysize = int(high[1]) - int(low[1])

        coords = []
        coords.append(UniformCoordinates1d(s, n, size=ysize, name="lat"))
        coords.append(UniformCoordinates1d(w, e, size=xsize, name="lon"))

        if metadata.timepositions or metadata.timelimits:
            import pdb

            pdb.set_trace()  # breakpoint 8546c30e //

        return Coordinates(coords, crs=self.crs)

    def _eval(self, coordinates, output=None, _selector=None):
        """{get_data}

        """

        # store requested coordinates for debugging
        if settings["DEBUG"]:
            self._requested_coordinates = coordinates

        # check for missing dimensions
        for dim in self.coordinates:
            if dim not in coordinates.udims:
                raise ValueError("Cannot evaluate these coordinates, missing dim '%s'" % c.name)

        # remove extra dimensions
        extra = [
            c.name
            for c in coordinates.values()
            if (isinstance(c, Coordinates1d) and c.name not in self.coordinates.udims)
            or (isinstance(c, StackedCoordinates) and all(dim not in self.coordinates.udims for dim in c.dims))
        ]
        coordinates = coordinates.drop(extra)

        # store input coordinates to evaluated coordinates
        self._evaluated_coordinates = deepcopy(coordinates)

        # transform coordinates into native crs if different
        if self.coordinates.crs.lower() != coordinates.crs.lower():
            coordinates = coordinates.transform(self.coordinates.crs)

        # -------------------------------------------------------------------------------------------------------------
        # the following section deviates from the Datasource node
        # (no coordinates intersection or selection)
        # -------------------------------------------------------------------------------------------------------------

        # TODO stacked coordinates
        # TODO time
        coordinates = coordinates.transpose("lat", "lon")

        w = coordinates["lon"].start - coordinates["lon"].step / 2.0
        e = coordinates["lon"].stop + coordinates["lon"].step / 2.0
        s = coordinates["lat"].start - coordinates["lat"].step / 2.0
        n = coordinates["lat"].stop + coordinates["lat"].step / 2.0
        width = coordinates["lon"].size
        height = coordinates["lat"].size

        response = self.client.getCoverage(
            identifier=self.layer,
            bbox=(w, n, e, s),
            width=width,
            height=height,
            crs=self.crs,
            format=self.format,
            version=self.version,
        )
        content = response.read()

        # check for errors
        xml = bs4.BeautifulSoup(content, "lxml")
        error = xml.find("serviceexception")
        if error:
            raise WCS2Error(error.text)

        # get data using rasterio
        with rasterio.MemoryFile() as mf:
            mf.write(content)
            dataset = mf.open(driver="GTiff")

        data = dataset.read(1).astype(float)
        data[np.isin(data, dataset.nodatavals)] = np.nan

        # -------------------------------------------------------------------------------------------------------------
        # the above section deviates from the Datasource node
        # -------------------------------------------------------------------------------------------------------------

        data = self.create_output_array(coordinates, data=data)
        data = data.part_transpose(self._evaluated_coordinates.dims)
        if output is None:
            output = data
        else:
            output.data[:] = data.data

        # save output to private for debugging
        if settings["DEBUG"]:
            self._output = output

        return output

    @staticmethod
    def get_layers(source):
        client = owslib_wcs.WebCoverageService(source)
        return list(client.contents)
