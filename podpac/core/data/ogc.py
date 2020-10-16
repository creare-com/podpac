"""
OGC-compliant datasources over HTTP
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import os
import re
from io import BytesIO

import numpy as np
import traitlets as tl

from podpac.core.settings import settings
from podpac.core.utils import common_doc, cached_property
from podpac.core.data.datasource import COMMON_DATA_DOC, DataSource
from podpac.core.coordinates import Coordinates, UniformCoordinates1d, ArrayCoordinates1d
from podpac.core.interpolation.interpolation import InterpolationMixin


# Optional dependencies
from lazy_import import lazy_module, lazy_class

bs4 = lazy_module("bs4")
lxml = lazy_module("lxml")  # used by bs4 so want to check if it's available
rasterio = lazy_module("rasterio")
requests = lazy_module("requests")
# esri
RasterToNumPyArray = lazy_module("arcpy.RasterToNumPyArray")
urllib3 = lazy_module("urllib3")
certifi = lazy_module("certifi")


WCS_DEFAULT_VERSION = "1.0.0"
WCS_DEFAULT_CRS = "EPSG:4326"


class WCSBase(DataSource):
    """Create a DataSource from an OGC-compliant WCS service
    
    Attributes
    ----------
    crs : 'str'
        Default is EPSG:4326 (WGS84 Geodic) EPSG number for the coordinate reference system that the data should
        be returned in.
    layer_name : str
        Name of the WCS layer that should be fetched from the server
    source : str
        URL of the WCS server endpoint
    version : str
        Default is 1.0.0. WCS version string.
    wcs_coordinates : :class:`podpac.Coordinates`
        The coordinates of the WCS source
    """

    source = tl.Unicode().tag(attr=True)
    layer_name = tl.Unicode().tag(attr=True)
    version = tl.Unicode(default_value=WCS_DEFAULT_VERSION).tag(attr=True)
    crs = tl.Unicode(default_value=WCS_DEFAULT_CRS).tag(attr=True)

    # list of attribute names, used by __repr__ and __str__ to display minimal info about the node
    _repr_keys = ["source", "interpolation"]

    _get_capabilities_qs = tl.Unicode("SERVICE=WCS&REQUEST=DescribeCoverage&" "VERSION={version}&COVERAGE={layer}")
    _get_data_qs = tl.Unicode(
        "SERVICE=WCS&VERSION={version}&REQUEST=GetCoverage&"
        "FORMAT=GeoTIFF&COVERAGE={layer}&"
        "BBOX={w},{s},{e},{n}&CRS={crs}&RESPONSE_CRS={crs}&"
        "WIDTH={width}&HEIGHT={height}&TIME={time}"
    )

    @property
    def capabilities_url(self):
        """Constructs the url that requests the WCS capabilities
        
        Returns
        -------
        str
            The url that requests the WCS capabilities
        """

        return self.source + "?" + self._get_capabilities_qs.format(version=self.version, layer=self.layer_name)

    @cached_property
    def wcs_coordinates(self):
        """ Coordinates reported by the WCS service.
        
        Returns
        -------
        Coordinates
        
        Notes
        -------
        This assumes a `time`, `lat`, `lon` order for the coordinates, and currently doesn't handle `alt` coordinates
        
        Raises
        ------
        Exception
            Raises this if the required dependencies are not installed.
        """

        if requests is not None:
            capabilities = requests.get(self.capabilities_url)
            if capabilities.status_code != 200:
                raise Exception("Could not get capabilities from WCS server")
            capabilities = capabilities.text

        # TODO: remove support urllib3 - requests is sufficient
        elif urllib3 is not None:
            if certifi is not None:
                http = urllib3.PoolManager(ca_certs=certifi.where())
            else:
                http = urllib3.PoolManager()

            r = http.request("GET", self.capabilities_url)
            capabilities = r.data
            if r.status != 200:
                raise Exception("Could not get capabilities from WCS server:" + self.capabilities_url)
        else:
            raise Exception("Do not have a URL request library to get WCS data.")

        if (
            lxml is not None
        ):  # could skip using lxml and always use html.parser instead, which seems to work but lxml might be faster
            capabilities = bs4.BeautifulSoup(capabilities, "lxml")
        else:
            capabilities = bs4.BeautifulSoup(capabilities, "html.parser")

        domain = capabilities.find("wcs:spatialdomain")
        pos = domain.find("gml:envelope").get_text().split()
        lonlat = np.array(pos, float).reshape(2, 2)
        grid_env = domain.find("gml:gridenvelope")
        low = np.array(grid_env.find("gml:low").text.split(), int)
        high = np.array(grid_env.find("gml:high").text.split(), int)
        size = high - low
        dlondlat = (lonlat[1, :] - lonlat[0, :]) / size
        bottom = lonlat[:, 1].min() + dlondlat[1] / 2
        top = lonlat[:, 1].max() - dlondlat[1] / 2
        left = lonlat[:, 0].min() + dlondlat[0] / 2
        right = lonlat[:, 0].max() - dlondlat[0] / 2

        timedomain = capabilities.find("wcs:temporaldomain")
        if timedomain is None:
            return Coordinates(
                [
                    UniformCoordinates1d(top, bottom, size=size[1], name="lat"),
                    UniformCoordinates1d(left, right, size=size[0], name="lon"),
                ]
            )

        date_re = re.compile("[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}")
        times = str(timedomain).replace("<gml:timeposition>", "").replace("</gml:timeposition>", "").split("\n")
        times = np.array([t for t in times if date_re.match(t)], np.datetime64)

        if len(times) == 0:
            return Coordinates(
                [
                    UniformCoordinates1d(top, bottom, size=size[1], name="lat"),
                    UniformCoordinates1d(left, right, size=size[0], name="lon"),
                ]
            )

        return Coordinates(
            [
                ArrayCoordinates1d(times, name="time"),
                UniformCoordinates1d(top, bottom, size=size[1], name="lat"),
                UniformCoordinates1d(left, right, size=size[0], name="lon"),
            ]
        )

    @common_doc(COMMON_DATA_DOC)
    def get_coordinates(self):
        """{get_coordinates}
            
        Notes
        ------
        This is a little tricky and doesn't fit into the usual PODPAC method, as the service is actually doing the 
        data wrangling for us...
        """

        # TODO update so that we don't rely on _requested_coordinates if possible
        if not self._requested_coordinates:
            return self.wcs_coordinates

        cs = []
        for dim in self.wcs_coordinates.dims:
            if dim in self._requested_coordinates.dims:
                c = self._requested_coordinates[dim]
                if c.size == 1:
                    cs.append(ArrayCoordinates1d(c.coordinates[0], name=dim))
                elif isinstance(c, UniformCoordinates1d):
                    cs.append(UniformCoordinates1d(c.bounds[0], c.bounds[1], abs(c.step), name=dim))
                else:
                    # TODO: generalize/fix this
                    # WCS calls require a regular grid, could (otherwise we have to do multiple WCS calls)
                    cs.append(UniformCoordinates1d(c.bounds[0], c.bounds[1], size=c.size, name=dim))
            else:
                cs.append(self.wcs_coordinates[dim])
        c = Coordinates(cs)
        return c

    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        
        Raises
        ------
        Exception
            Raises this if there is a network error or required dependencies are not installed.
        """
        output = self.create_output_array(coordinates)
        dotime = "time" in self.wcs_coordinates.dims

        wbound = coordinates["lon"].bounds[0] - coordinates["lon"].step / 2.0
        ebound = coordinates["lon"].bounds[1] + coordinates["lon"].step / 2.0
        sbound = coordinates["lat"].bounds[0] - coordinates["lat"].step / 2.0
        nbound = coordinates["lat"].bounds[1] + coordinates["lat"].step / 2.0

        if "time" in coordinates.dims and dotime:
            sd = np.timedelta64(0, "s")
            times = [str(t + sd) for t in coordinates["time"].coordinates]
        else:
            times = [""]

        if len(times) > 1:
            for i, time in enumerate(times):
                url = (
                    self.source
                    + "?"
                    + self._get_data_qs.format(
                        version=self.version,
                        layer=self.layer_name,
                        w=wbound,
                        e=ebound,
                        s=sbound,
                        n=nbound,
                        width=coordinates["lon"].size,
                        height=coordinates["lat"].size,
                        time=time,
                        crs=self.crs,
                    )
                )

                if not dotime:
                    url = url.replace("&TIME=", "")

                if requests is not None:
                    data = requests.get(url)
                    if data.status_code != 200:
                        raise Exception("Could not get data from WCS server:" + url)
                    io = BytesIO(bytearray(data.content))
                    content = data.content

                # TODO: remove support urllib3 - requests is sufficient
                elif urllib3 is not None:
                    if certifi is not None:
                        http = urllib3.PoolManager(ca_certs=certifi.where())
                    else:
                        http = urllib3.PoolManager()
                    r = http.request("GET", url)
                    if r.status != 200:
                        raise Exception("Could not get capabilities from WCS server:" + url)
                    content = r.data
                    io = BytesIO(bytearray(r.data))
                else:
                    raise Exception("Do not have a URL request library to get WCS data.")

                try:
                    try:  # This works with rasterio v1.0a8 or greater, but not on python 2
                        with rasterio.open(io) as dataset:
                            output.data[i, ...] = dataset.read()
                    except Exception as e:  # Probably python 2
                        print(e)
                        tmppath = os.path.join(settings.cache_path, "wcs_temp.tiff")

                        if not os.path.exists(os.path.split(tmppath)[0]):
                            os.makedirs(os.path.split(tmppath)[0])

                        # TODO: close tmppath? os does this on remove?
                        open(tmppath, "wb").write(content)

                        with rasterio.open(tmppath) as dataset:
                            output.data[i, ...] = dataset.read()

                        os.remove(tmppath)  # Clean up

                except ImportError:
                    # Writing the data to a temporary tiff and reading it from there is hacky
                    # However reading directly from r.data or io doesn't work
                    # Should improve in the future
                    open("temp.tiff", "wb").write(r.data)
                    output.data[i, ...] = RasterToNumPyArray("temp.tiff")
        else:
            time = times[0]

            url = (
                self.source
                + "?"
                + self._get_data_qs.format(
                    version=self.version,
                    layer=self.layer_name,
                    w=wbound,
                    e=ebound,
                    s=sbound,
                    n=nbound,
                    width=coordinates["lon"].size,
                    height=coordinates["lat"].size,
                    time=time,
                    crs=self.crs,
                )
            )
            if not dotime:
                url = url.replace("&TIME=", "")
            if requests is not None:
                data = requests.get(url)
                if data.status_code != 200:
                    raise Exception("Could not get data from WCS server:" + url)
                io = BytesIO(bytearray(data.content))
                content = data.content

            # TODO: remove support urllib3 - requests is sufficient
            elif urllib3 is not None:
                if certifi is not None:
                    http = urllib3.PoolManager(ca_certs=certifi.where())
                else:
                    http = urllib3.PoolManager()
                r = http.request("GET", url)
                if r.status != 200:
                    raise Exception("Could not get capabilities from WCS server:" + url)
                content = r.data
                io = BytesIO(bytearray(r.data))
            else:
                raise Exception("Do not have a URL request library to get WCS data.")

            try:
                try:  # This works with rasterio v1.0a8 or greater, but not on python 2
                    with rasterio.open(io) as dataset:
                        if dotime:
                            output.data[0, ...] = dataset.read()
                        else:
                            output.data[:] = dataset.read()
                except Exception as e:  # Probably python 2
                    print(e)
                    tmppath = os.path.join(settings.cache_path, "wcs_temp.tiff")
                    if not os.path.exists(os.path.split(tmppath)[0]):
                        os.makedirs(os.path.split(tmppath)[0])
                    open(tmppath, "wb").write(content)
                    with rasterio.open(tmppath) as dataset:
                        output.data[:] = dataset.read()
                    os.remove(tmppath)  # Clean up
            except ImportError:
                # Writing the data to a temporary tiff and reading it from there is hacky
                # However reading directly from r.data or io doesn't work
                # Should improve in the future
                open("temp.tiff", "wb").write(r.data)
                try:
                    output.data[:] = RasterToNumPyArray("temp.tiff")
                except:
                    raise Exception("Rasterio or Arcpy not available to read WCS feed.")
        if not coordinates["lat"].is_descending:
            if dotime:
                output.data[:] = output.data[:, ::-1, :]
            else:
                output.data[:] = output.data[::-1, :]

        return output

    @property
    def base_ref(self):
        """ definition base_ref """
        if not self.layer_name:
            return super(WCS, self).base_ref

        return self.layer_name.rsplit(".", 1)[1]


class WCS(InterpolationMixin, WCSBase):
    pass
