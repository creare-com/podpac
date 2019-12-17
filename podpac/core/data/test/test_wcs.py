import os
from six import string_types

import pytest
import requests
from traitlets import TraitError

from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.data.wcs import WCS, WCS_DEFAULT_VERSION, WCS_DEFAULT_CRS


class TestWCS(object):
    """test WCS data source
    TODO: this needs to be reworked with real examples
    """

    source = "WCSsource"
    layer_name = "layer"

    with open(os.path.join(os.path.dirname(__file__), "assets/capabilites.xml"), "r") as f:
        capabilities = f.read()

    # TODO load a better geotiff example so get_data works below
    with open(os.path.join(os.path.dirname(__file__), "assets/RGB.byte.tif"), "rb") as f:
        geotiff = f.read()

    def mock_requests(self, cap_status_code=200, data_status_code=200):
        def mock_get(url=None):
            r = requests.Response()

            # get capabilities
            if "REQUEST=DescribeCoverage" in url:
                r.status_code = cap_status_code
                try:
                    r._content = bytes(self.capabilities, "utf-8")
                except:  # Python 2.7
                    r._content = bytes(self.capabilities)
            # get geotiff
            else:
                r.status_code = data_status_code
                r._content = self.geotiff

            return r

        requests.get = mock_get

    def test_wcs_defaults(self):
        """test global WCS defaults"""

        assert WCS_DEFAULT_VERSION
        assert WCS_DEFAULT_CRS

    def test_init(self):
        """test basic init of class"""

        node = WCS(source=self.source)
        assert isinstance(node, WCS)

    def test_traits(self):
        """ check each of the WCS traits """

        WCS(source=self.source)
        with pytest.raises(TraitError):
            WCS(source=5)

        WCS(layer_name=self.layer_name)
        with pytest.raises(TraitError):
            WCS(layer_name=5)

        node = WCS()
        assert node.version == WCS_DEFAULT_VERSION
        with pytest.raises(TraitError):
            WCS(version=5)

        node = WCS()
        assert node.crs == WCS_DEFAULT_CRS
        with pytest.raises(TraitError):
            WCS(crs=5)

    def test_get_capabilities_url(self):
        """test the capabilities url generation"""

        node = WCS(source=self.source)
        url = node.get_capabilities_url
        assert isinstance(url, string_types)
        assert node.source in url

    def test_get_wcs_coordinates(self):
        """get wcs coordinates"""

        import podpac.core.data.wcs
        import urllib3
        import lxml

        # requests
        self.mock_requests()
        node = WCS(source=self.source)
        coordinates = node.wcs_coordinates

        assert isinstance(coordinates, Coordinates)
        assert coordinates["lat"]
        assert coordinates["lon"]
        assert coordinates["time"]

        # bad status code return
        self.mock_requests(cap_status_code=400)
        with pytest.raises(Exception):
            node = WCS(source=self.source)
            coordinates = node.wcs_coordinates

        # no lxml
        podpac.core.data.wcs.lxml = None

        self.mock_requests()
        node = WCS(source=self.source)
        coordinates = node.wcs_coordinates

        assert isinstance(coordinates, Coordinates)
        assert coordinates["lat"]
        assert coordinates["lon"]
        assert coordinates["time"]

        # urllib3
        podpac.core.data.wcs.requests = None

        # no requests, urllib3
        podpac.core.data.wcs.urllib3 = None

        node = WCS(source=self.source)
        with pytest.raises(Exception):
            node.get_wcs_coordinates()

        # put all dependencies back
        podpac.core.data.wcs.requests = requests
        podpac.core.data.wcs.urllib3 = urllib3
        podpac.core.data.wcs.lxml = lxml

    def test_get_native_coordinates(self):
        """get native coordinates"""

        self.mock_requests()
        node = WCS(source=self.source)

        # equal to wcs coordinates when no eval coordinates
        native_coordinates = node.native_coordinates
        wcs_coordinates = node.wcs_coordinates
        assert native_coordinates == wcs_coordinates

        # with eval coordinates
        # TODO: use real eval coordinates
        node._output_coordinates = native_coordinates
        native_coordinates = node.native_coordinates

        assert isinstance(native_coordinates, Coordinates)
        # TODO: one returns monotonic, the other returns uniform
        assert native_coordinates == node._output_coordinates
        assert native_coordinates["lat"]
        assert native_coordinates["lon"]
        assert native_coordinates["time"]

    def test_get_data(self):
        """get data from wcs server"""

        self.mock_requests()
        node = WCS(source=self.source)
        lat = node.native_coordinates["lat"].coordinates
        lon = node.native_coordinates["lon"].coordinates
        time = node.native_coordinates["time"].coordinates

        # no time
        notime_coordinates = Coordinates(
            [clinspace(lat[0], lat[-2], 10), clinspace(lon[0], lon[-2], 10), "2006-06-14T17:00:00"],
            dims=["lat", "lon", "time"],
        )

        with pytest.raises(ValueError):
            output = node.eval(notime_coordinates)
            assert isinstance(output, UnitsDataArray)
            assert output.native_coordinates["lat"][0] == node.native_coordinates["lat"][0]

        # time
        time_coordinates = Coordinates(
            [clinspace(lat[0], lat[-2], 10), clinspace(lon[0], lon[-2], 10), clinspace(time[0], time[-1], len(time))],
            dims=["lat", "lon", "time"],
        )

        with pytest.raises(ValueError):
            output = node.eval(time_coordinates)
            assert isinstance(output, UnitsDataArray)

        # requests exceptions
        self.mock_requests(data_status_code=400)
        with pytest.raises(Exception):
            output = node.eval(time_coordinates)
        with pytest.raises(Exception):
            output = node.eval(time_coordinates)
