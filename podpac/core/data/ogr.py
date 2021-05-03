import os.path

import numpy as np
import traitlets as tl

from lazy_import import lazy_module

gdal = lazy_module("osgeo.gdal")
ogr = lazy_module("osgeo.ogr")

from podpac import Node, Coordinates, cached_property, settings
from podpac.core.utils import common_doc
from podpac.core.node import COMMON_NODE_DOC


class OGR(Node):
    """ """

    source = tl.Unicode().tag(attr=True)
    layer = tl.Unicode().tag(attr=True)
    attribute = tl.Unicode().tag(attr=True)
    nan_vals = tl.List().tag(attr=True)
    nan_val = tl.Any(np.nan).tag(attr=True)
    driver = tl.Unicode()

    _repr_keys = ["source", "layer", "attribute"]

    # debug traits
    _requested_coordinates = tl.Instance(Coordinates, allow_none=True)
    _evaluated_coordinates = tl.Instance(Coordinates, allow_none=True)

    @tl.validate("driver")
    def _validate_driver(self, d):
        ogr.GetDriverByName(d["value"])
        return d["value"]

    @tl.validate("source")
    def _validate_source(self, d):
        if not os.path.exists(d["value"]):
            raise ValueError("OGR source not found '%s'" % d["value"])
        return d["value"]

    @cached_property
    def datasource(self):
        driver = ogr.GetDriverByName(self.driver)
        return driver.Open(self.source, 0)

    @cached_property
    def extents(self):
        layer = self.datasource.GetLayerByName(self.layer)
        return layer.GetExtent()

    @common_doc(COMMON_NODE_DOC)
    def _eval(self, coordinates, output=None, _selector=None):
        if "lat" not in coordinates.udims or "lon" not in coordinates.udims:
            raise RuntimeError("OGR source requires lat and lon dims")

        requested_coordinates = coordinates
        coordinates = coordinates.udrop(["time", "alt"], ignore_missing=True)

        data = self._get_data(coordinates)

        if output is None:
            output = self.create_output_array(coordinates, data=data)
        else:
            output.data[:] = data

        output.data[np.isin(output.data, self.nan_vals)] = self.nan_val

        if settings["DEBUG"]:
            self._requested_coordinates = requested_coordinates
            self._evaluated_coordinates = coordinates

        return output

    def _get_data(self, coordinates):
        nan_val = 0

        # create target datasource
        driver = gdal.GetDriverByName("MEM")
        target = driver.Create("", coordinates["lon"].size, coordinates["lat"].size, gdal.GDT_Float64)
        target.SetGeoTransform(coordinates.geotransform)
        band = target.GetRasterBand(1)
        band.SetNoDataValue(nan_val)
        band.Fill(nan_val)

        # rasterize
        layer = self.datasource.GetLayerByName(self.layer)
        gdal.RasterizeLayer(target, [1], layer, options=["ATTRIBUTE=%s" % self.attribute])

        data = band.ReadAsArray(buf_type=gdal.GDT_Float64).copy()
        data[data == nan_val] = np.nan
        return nan_val

    def _get_data_at_points(self, points):
        """
        Read vector data at specified points into a numpy array

        :param points: list of Point objects
        """
        vector = np.full((len(points),), np.nan)
        eps = 1e-6

        # Rasterize each point individually
        for (i, p) in enumerate(points):
            geotransform_i = np.array([p.lon - eps / 2.0, eps, 0.0, p.lat - eps / 2.0, 0.0, -1.0 * eps])
            gc_i = GridCoordinates(geotransform=geotransform_i, y_size=1, x_size=1)

            vector[i] = self._get_data_at_gridcoordinates(gc_i)[0, 0]

        return vector
