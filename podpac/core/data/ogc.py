"""
OGC-compliant datasources over HTTP
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import logging
from operator import mul
from functools import reduce

import traitlets as tl

from podpac.core.utils import common_doc, cached_property
from podpac.core.data.datasource import DataSource
from podpac.core.interpolation.interpolation import InterpolationMixin
from podpac.core.node import NodeException
from podpac.core.coordinates import Coordinates
from podpac.core.coordinates import UniformCoordinates1d, ArrayCoordinates1d

# Optional dependencies
from lazy_import import lazy_module, lazy_class

bs4 = lazy_module("bs4")
lxml = lazy_module("lxml")  # used by bs4 so want to check if it's available
owslib_wcs = lazy_module("owslib.wcs")
rasterio = lazy_module("rasterio")


logger = logging.getLogger(__name__)


class WCSError(NodeException):
    pass


class WCSBase(DataSource):
    """
    Access data from a WCS source.

    Attributes
    ----------
    source : str
        WCS server url
    layer : str
        layer name (required)
    version : str
        WCS version, passed through to all requests (default '1.0.0')
    format : str
        Data format, passed through to the GetCoverage requests (default 'geotiff')
    crs : str
        coordinate reference system, passed through to the GetCoverage requests (default 'EPSG:4326')
    interpolation : str
        Interpolation, passed through to the GetCoverage requests. 
    max_size : int
        maximum request size, optional.
        If provided, the coordinates will be tiled into multiple requests.
    """

    source = tl.Unicode().tag(attr=True)
    layer = tl.Unicode().tag(attr=True)
    version = tl.Unicode(default_value="1.0.0").tag(attr=True)
    interpolation = tl.Unicode(default_value=None, allow_none=True).tag(attr=True)

    format = tl.CaselessStrEnum(["geotiff", "geotiff_byte"], default_value="geotiff")
    crs = tl.Unicode(default_value="EPSG:4326")
    max_size = tl.Long(default_value=None, allow_none=True)

    _repr_keys = ["source", "layer"]

    _requested_coordinates = tl.Instance(Coordinates, allow_none=True)
    _evaluated_coordinates = tl.Instance(Coordinates)

    @cached_property
    def client(self):
        return owslib_wcs.WebCoverageService(self.source, version=self.version)

    def get_coordinates(self):
        """
        Get the full WCS grid.
        """

        metadata = self.client.contents[self.layer]

        # TODO select correct boundingbox by crs

        # coordinates
        w, s, e, n = metadata.boundingBoxWGS84
        low = metadata.grid.lowlimits
        high = metadata.grid.highlimits
        xsize = int(high[0]) - int(low[0])
        ysize = int(high[1]) - int(low[1])

        coords = []
        coords.append(UniformCoordinates1d(s, n, size=ysize, name="lat"))
        coords.append(UniformCoordinates1d(w, e, size=xsize, name="lon"))

        if metadata.timepositions:
            coords.append(ArrayCoordinates1d(metadata.timepositions, name="time"))

        if metadata.timelimits:
            raise NotImplementedError("TODO")

        return Coordinates(coords, crs=self.crs)

    def _eval(self, coordinates, output=None, _selector=None):
        """Evaluates this node using the supplied coordinates.

        This method intercepts the DataSource._eval method in order to use the requested coordinates directly when
        they are a uniform grid.

        Parameters
        ----------
        coordinates : :class:`podpac.Coordinates`
            {requested_coordinates}

            An exception is raised if the requested coordinates are missing dimensions in the DataSource.
            Extra dimensions in the requested coordinates are dropped.
        output : :class:`podpac.UnitsDataArray`, optional
            {eval_output}
        _selector: callable(coordinates, request_coordinates)
            {eval_selector}

        Returns
        -------
        {eval_return}

        Raises
        ------
        ValueError
            Cannot evaluate these coordinates
        """

        # the datasource does do this, but we need to do it here to correctly select the correct case
        if self.coordinates.crs.lower() != coordinates.crs.lower():
            coordinates = coordinates.transform(self.coordinates.crs)

        # for a uniform grid, use the requested coordinates (the WCS server will interpolate)
        if (
            ("lat" in coordinates.dims and "lon" in coordinates.dims)
            and (coordinates["lat"].is_uniform or coordinates["lat"].size == 1)
            and (coordinates["lon"].is_uniform or coordinates["lon"].size == 1)
        ):

            def selector(rsc, rsci, coordinates):
                return coordinates, tuple(slice(None) for dim in coordinates)

            return super()._eval(coordinates, output=output, _selector=selector)

        # for uniform stacked, unstack to use the requested coordinates (the WCS server will interpolate)
        if (
            ("lat" in coordinates.udims and coordinates.is_stacked("lat"))
            and ("lon" in coordinates.udims and coordinates.is_stacked("lon"))
            and (coordinates["lat"].is_uniform or coordinates["lat"].size == 1)
            and (coordinates["lon"].is_uniform or coordinates["lon"].size == 1)
        ):

            def selector(rsc, rsci, coordinates):
                unstacked = coordinates.unstack()
                unstacked = unstacked.drop("alt", ignore_missing=True)  # if lat_lon_alt
                return unstacked, tuple(slice(None) for dim in unstacked)

            udata = super()._eval(coordinates, output=None, _selector=selector)
            data = udata.data.diagonal()  # get just the stacked data
            if output is None:
                output = self.create_output_array(coordinates, data=data)
            else:
                output.data[:] = data
            return output

        # otherwise, pass-through (podpac will select and interpolate)
        return super()._eval(coordinates, output=output, _selector=_selector)

    def _get_data(self, coordinates, coordinates_index):
        """{get_data}

        """

        # transpose the coordinates to match the response data
        if "time" in coordinates:
            coordinates = coordinates.transpose("time", "lat", "lon")
        else:
            coordinates = coordinates.transpose("lat", "lon")

        # determine the chunk size (if applicable)
        if self.max_size is not None:
            shape = []
            s = 1
            for n in coordinates.shape:
                r = self.max_size // s
                if r == 0:
                    shape.append(1)
                elif r < n:
                    shape.append(r)
                else:
                    shape.append(n)
                s *= n
            shape = tuple(shape)
        else:
            shape = coordinates.shape

        # request each chunk and composite the data
        output = self.create_output_array(coordinates)
        for chunk, slc in coordinates.iterchunks(shape, return_slices=True):
            output[slc] = self._get_chunk(chunk)

        return output

    def _get_chunk(self, coordinates):
        if coordinates["lon"].size == 1:
            w = coordinates["lon"].coordinates[0]
            e = coordinates["lon"].coordinates[0]
        else:
            w = coordinates["lon"].start - coordinates["lon"].step / 2.0
            e = coordinates["lon"].stop + coordinates["lon"].step / 2.0

        if coordinates["lat"].size == 1:
            s = coordinates["lat"].coordinates[0]
            n = coordinates["lat"].coordinates[0]
        else:
            s = coordinates["lat"].start - coordinates["lat"].step / 2.0
            n = coordinates["lat"].stop + coordinates["lat"].step / 2.0

        width = coordinates["lon"].size
        height = coordinates["lat"].size

        kwargs = {}

        if "time" in coordinates:
            kwargs["time"] = coordinates["time"].coordinates.astype(str).tolist()

        if isinstance(self.interpolation, str):
            kwargs["interpolation"] = self.interpolation

        logger.info(
            "WCS GetCoverage (source=%s, layer=%s, bbox=%s, shape=%s)"
            % (self.source, self.layer, (w, n, e, s), (width, height))
        )

        response = self.client.getCoverage(
            identifier=self.layer,
            bbox=(w, n, e, s),
            width=width,
            height=height,
            crs=self.crs,
            format=self.format,
            version=self.version,
            **kwargs
        )
        content = response.read()

        # check for errors
        xml = bs4.BeautifulSoup(content, "lxml")
        error = xml.find("serviceexception")
        if error:
            raise WCSError(error.text)

        # get data using rasterio
        with rasterio.MemoryFile() as mf:
            mf.write(content)
            dataset = mf.open(driver="GTiff")

        if "time" in coordinates and coordinates["time"].size > 1:
            # this should be easy to do, I'm just not sure how the data comes back.
            # is each time in a different band?
            raise NotImplementedError("TODO")

        data = dataset.read(1).astype(float)
        return data

    @classmethod
    def get_layers(cls, source=None):
        if source is None:
            source = cls.source
        client = owslib_wcs.WebCoverageService(source)
        return list(client.contents)


class WCS(InterpolationMixin, WCSBase):
    pass
