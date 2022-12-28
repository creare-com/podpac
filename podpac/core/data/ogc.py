"""
OGC-compliant datasources over HTTP
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import logging
from operator import mul
from functools import reduce

import traitlets as tl
import pyproj

from podpac.core.utils import common_doc, cached_property, resolve_bbox_order
from podpac.core.data.datasource import DataSource
from podpac.core.interpolation.interpolation import InterpolationMixin, InterpolationTrait
from podpac.core.node import NodeException
from podpac.core.coordinates import Coordinates
from podpac.core.coordinates import UniformCoordinates1d, ArrayCoordinates1d, Coordinates1d, StackedCoordinates

# Optional dependencies
from lazy_import import lazy_module, lazy_class

bs4 = lazy_module("bs4")
lxml = lazy_module("lxml")  # used by bs4 so want to check if it's available
owslib_wcs = lazy_module("owslib.wcs")
owslib_util = lazy_module("owslib.util")
rasterio = lazy_module("rasterio")


logger = logging.getLogger(__name__)


class MockWCSClient(tl.HasTraits):
    source = tl.Unicode()
    version = tl.Enum(["1.0.0"], default_value="1.0.0")
    headers = None
    cookies = None
    auth = tl.Any()

    def getCoverage(
        self,
        identifier=None,
        bbox=None,
        time=None,
        format=None,
        crs=None,
        width=None,
        height=None,
        resx=None,
        resy=None,
        resz=None,
        parameter=None,
        method="Get",
        timeout=30,
        **kwargs
    ):
        """Request and return a coverage from the WCS as a file-like object
        note: additional **kwargs helps with multi-version implementation
        core keyword arguments should be supported cross version
        example:
        cvg=wcs.getCoverage(identifier=['TuMYrRQ4'], timeSequence=['2792-06-01T00:00:00.0'], bbox=(-112,36,-106,41),
                            format='cf-netcdf')
        is equivalent to:
        http://myhost/mywcs?SERVICE=WCS&REQUEST=GetCoverage&IDENTIFIER=TuMYrRQ4&VERSION=1.1.0&BOUNDINGBOX=-180,-90,180,90&TIME=2792-06-01T00:00:00.0&FORMAT=cf-netcdf
        """
        from owslib.util import makeString
        from urllib.parse import urlencode
        from owslib.util import openURL

        if logger.isEnabledFor(logging.DEBUG):
            msg = "WCS 1.0.0 DEBUG: Parameters passed to GetCoverage: identifier={}, bbox={}, time={}, format={}, crs={}, width={}, height={}, resx={}, resy={}, resz={}, parameter={}, method={}, other_arguments={}"  # noqa
            logger.debug(
                msg.format(
                    identifier, bbox, time, format, crs, width, height, resx, resy, resz, parameter, method, str(kwargs)
                )
            )

        base_url = self.source

        logger.debug("WCS 1.0.0 DEBUG: base url of server: %s" % base_url)

        # process kwargs
        request = {"version": self.version, "request": "GetCoverage", "service": "WCS"}
        assert len(identifier) > 0
        request["Coverage"] = identifier
        # request['identifier'] = ','.join(identifier)
        if bbox:
            request["BBox"] = ",".join([makeString(x) for x in bbox])
        else:
            request["BBox"] = None
        if time:
            request["time"] = ",".join(time)
        if crs:
            request["crs"] = crs
        request["format"] = format
        if width:
            request["width"] = width
        if height:
            request["height"] = height
        if resx:
            request["resx"] = resx
        if resy:
            request["resy"] = resy
        if resz:
            request["resz"] = resz

        # anything else e.g. vendor specific parameters must go through kwargs
        if kwargs:
            for kw in kwargs:
                request[kw] = kwargs[kw]

        # encode and request
        data = urlencode(request)
        logger.debug("WCS 1.0.0 DEBUG: Second part of URL: %s" % data)

        u = openURL(base_url, data, method, self.cookies, auth=self.auth, timeout=timeout, headers=self.headers)
        return u


class WCSError(NodeException):
    pass


class WCSRaw(DataSource):
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
    allow_mock_client : bool
        Default is False. If True, a mock client will be used to make WCS requests. This allows returns
        from servers with only partial WCS implementations.
    username : str
        Username for servers that require authentication
    password : str
        Password for servers that require authentication

    See Also
    --------
    WCS : WCS datasource with podpac interpolation.
    """

    source = tl.Unicode().tag(attr=True, required=True)
    layer = tl.Unicode().tag(attr=True, required=True)
    version = tl.Unicode(default_value="1.0.0").tag(attr=True)
    interpolation = InterpolationTrait(default_value=None, allow_none=True).tag(attr=True)
    allow_mock_client = tl.Bool(False).tag(attr=True)
    username = tl.Unicode(allow_none=True)
    password = tl.Unicode(allow_none=True)

    format = tl.CaselessStrEnum(["geotiff", "geotiff_byte"], default_value="geotiff")
    crs = tl.Unicode(default_value="EPSG:4326")
    max_size = tl.Long(default_value=None, allow_none=True)
    wcs_kwargs = tl.Dict(help="Additional query parameters sent to the WCS server")

    _repr_keys = ["source", "layer"]

    _requested_coordinates = tl.Instance(Coordinates, allow_none=True)
    _evaluated_coordinates = tl.Instance(Coordinates)
    coordinate_index_type = "slice"

    @property
    def auth(self):
        if self.username and self.password:
            return owslib_util.Authentication(username=self.username, password=self.password)
        return None

    @cached_property
    def client(self):
        try:
            return owslib_wcs.WebCoverageService(self.source, version=self.version, auth=self.auth)
        except Exception as e:
            if self.allow_mock_client:
                logger.warning(
                    "The OWSLIB Client could not be used. Server endpoint likely does not implement GetCapabilities"
                    "requests. Using Mock client instead. Error was {}".format(e)
                )
                return MockWCSClient(source=self.source, version=self.version, auth=self.auth)
            else:
                raise e

    def get_coordinates(self):
        """
        Get the full WCS grid.
        """

        metadata = self.client.contents[self.layer]

        # coordinates
        bbox = metadata.boundingBoxWGS84
        crs = "EPSG:4326"
        logging.debug("WCS available boundingboxes: {}".format(metadata.boundingboxes))
        for bboxes in metadata.boundingboxes:
            if bboxes["nativeSrs"] == self.crs:
                bbox = bboxes["bbox"]
                crs = self.crs
                break

        low = metadata.grid.lowlimits
        high = metadata.grid.highlimits
        xsize = int(high[0]) - int(low[0])
        ysize = int(high[1]) - int(low[1])

        # Based on https://www.ctps.org/geoserver/web/wicket/bookmarkable/org.geoserver.wcs.web.demo.WCSRequestBuilder;jsessionid=9E2AA99F95410C694D05BA609F25527C?0
        # The above link points to a geoserver implementation, which is the reference implementation.
        # WCS version 1.0.0 always has order lon/lat while version 1.1.1 actually follows the CRS
        if self.version == "1.0.0":
            rbbox = {"lat": [bbox[1], bbox[3], ysize], "lon": [bbox[0], bbox[2], xsize]}
        else:
            rbbox = resolve_bbox_order(bbox, crs, (xsize, ysize))

        coords = []
        coords.append(UniformCoordinates1d(rbbox["lat"][0], rbbox["lat"][1], size=rbbox["lat"][2], name="lat"))
        coords.append(UniformCoordinates1d(rbbox["lon"][0], rbbox["lon"][1], size=rbbox["lon"][2], name="lon"))

        if metadata.timepositions:
            coords.append(ArrayCoordinates1d(metadata.timepositions, name="time"))

        if metadata.timelimits:
            raise NotImplementedError("TODO")

        return Coordinates(coords, crs=crs)

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
        # The mock client cannot figure out the real coordinates, so just duplicate the requested coordinates
        if isinstance(self.client, MockWCSClient):
            if not coordinates["lat"].is_uniform or not coordinates["lon"].is_uniform:
                raise NotImplementedError(
                    "When using the Mock WCS client, the requested coordinates need to be uniform."
                )
            self.set_trait("_coordinates", coordinates)
            self.set_trait("crs", coordinates.crs)

        # remove extra dimensions
        extra = [
            c.name
            for c in coordinates.values()
            if (isinstance(c, Coordinates1d) and c.name not in self.coordinates.udims)
            or (isinstance(c, StackedCoordinates) and all(dim not in self.coordinates.udims for dim in c.dims))
        ]
        coordinates = coordinates.drop(extra)

        # the datasource does do this, but we need to do it here to correctly select the correct case
        if self.coordinates.crs.lower() != coordinates.crs.lower():
            coordinates = coordinates.transform(self.coordinates.crs)

        # for a uniform grid, use the requested coordinates (the WCS server will interpolate)
        if (
            ("lat" in coordinates.dims and "lon" in coordinates.dims)
            and (coordinates["lat"].is_uniform or coordinates["lat"].size == 1)
            and (coordinates["lon"].is_uniform or coordinates["lon"].size == 1)
        ):

            def selector(rsc, coordinates, index_type=None):
                return coordinates, None

            return super()._eval(coordinates, output=output, _selector=selector)

        # for uniform stacked, unstack to use the requested coordinates (the WCS server will interpolate)
        if (
            ("lat" in coordinates.udims and coordinates.is_stacked("lat"))
            and ("lon" in coordinates.udims and coordinates.is_stacked("lon"))
            and (coordinates["lat"].is_uniform or coordinates["lat"].size == 1)
            and (coordinates["lon"].is_uniform or coordinates["lon"].size == 1)
        ):

            def selector(rsc, coordinates, index_type=None):
                unstacked = coordinates.unstack()
                unstacked = unstacked.drop("alt", ignore_missing=True)  # if lat_lon_alt
                return unstacked, None

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
        """{get_data}"""

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
        for i, (chunk, slc) in enumerate(coordinates.iterchunks(shape, return_slices=True)):
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

        kwargs = self.wcs_kwargs.copy()

        if "time" in coordinates:
            kwargs["time"] = coordinates["time"].coordinates.astype(str).tolist()

        if isinstance(self.interpolation, str):
            kwargs["interpolation"] = self.interpolation

        logger.info(
            "WCS GetCoverage (source=%s, layer=%s, bbox=%s, shape=%s, time=%s)"
            % (self.source, self.layer, (w, n, e, s), (width, height), kwargs.get("time"))
        )

        crs = pyproj.CRS(coordinates.crs)
        bbox = (min(w, e), min(s, n), max(e, w), max(n, s))
        # Based on the spec I need the following line, but
        # all my tests on other servers suggests I don't need this...
        # if crs.axis_info[0].direction == "north":
        #     bbox = (min(s, n), min(w, e), max(n, s), max(e, w))

        response = self.client.getCoverage(
            identifier=self.layer,
            bbox=bbox,
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
            try:
                dataset = mf.open(driver="GTiff")
            except rasterio.RasterioIOError:
                raise WCSError("Could not read file with contents:", content)

        if "time" in coordinates and coordinates["time"].size > 1:
            # this should be easy to do, I'm just not sure how the data comes back.
            # is each time in a different band?
            raise NotImplementedError("TODO")

        data = dataset.read().astype(float).squeeze()

        # Need to fix the order of the data in the case of multiple bands
        if len(data.shape) == 3:
            data = data.transpose((1, 2, 0))

        # Need to fix the data order. The request and response order is always the same in WCS, but not in PODPAC
        if n > s:  # By default it returns the data upside down, so this is backwards
            data = data[::-1]
        if e < w:
            data = data[:, ::-1]

        return data

    @classmethod
    def get_layers(cls, source=None):
        if source is None:
            source = cls.source
        client = owslib_wcs.WebCoverageService(source)
        return list(client.contents)


class WCS(InterpolationMixin, WCSRaw):
    """WCS datasource with podpac interpolation."""

    coordinate_index_type = tl.Unicode("slice", read_only=True)
