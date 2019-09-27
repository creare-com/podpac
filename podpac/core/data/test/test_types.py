"""
Test podpac.core.data.types module
"""

import os
from collections import OrderedDict
from io import BytesIO
import urllib3
import lxml
from six import string_types
import shutil

import pytest

import numpy as np
from traitlets import TraitError
from xarray.core.coordinates import DataArrayCoordinates
import pydap
import pydap.client
import rasterio
import boto3
import botocore
import requests
import zarr

import podpac
from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.units import UnitsDataArray
from podpac.core.node import COMMON_NODE_DOC, Node
from podpac.core.data.datasource import COMMON_DATA_DOC, DataSource
from podpac.core.data.types import WCS_DEFAULT_VERSION, WCS_DEFAULT_CRS
from podpac.core.data.types import Array, PyDAP, Rasterio, WCS, ReprojectedSource, CSV, H5PY, Zarr
from podpac.core.settings import settings

# Trying to fix test
pydap.client.open_url


def test_allow_missing_modules():
    """TODO: Allow user to be missing rasterio and scipy"""
    pass


class TestArray(object):
    """Test Array datasource class (formerly Array)"""

    data = np.random.rand(11, 11)
    coordinates = Coordinates([clinspace(-25, 25, 11), clinspace(-25, 25, 11)], dims=["lat", "lon"])

    def test_source_trait(self):
        """ must be an ndarray """

        node = Array(source=self.data, native_coordinates=self.coordinates)

        # list is coercable to array
        node = Array(source=[0, 1, 1], native_coordinates=self.coordinates)

        # this list is not coercable to array
        # Starting with numpy 0.16, this is now allowed!
        # with pytest.raises(TraitError):
        # node = Array(source=[0, [0, 1]], native_coordinates=self.coordinates)

    def test_get_data(self):
        """ defined get_data function"""

        source = self.data
        node = Array(source=source, native_coordinates=self.coordinates)
        output = node.eval(self.coordinates)

        assert isinstance(output, UnitsDataArray)
        assert output.values[0, 0] == source[0, 0]
        assert output.values[4, 5] == source[4, 5]

    def test_native_coordinates(self):
        """test that native coordinates get defined"""

        node = Array(source=self.data)
        with pytest.raises(NotImplementedError):
            node.get_native_coordinates()

        node = Array(source=self.data, native_coordinates=self.coordinates)
        assert node.native_coordinates

        node = Array(source=self.data, native_coordinates=self.coordinates)
        native_coordinates = node.native_coordinates
        get_native_coordinates = node.get_native_coordinates()
        assert native_coordinates
        assert get_native_coordinates
        assert native_coordinates == get_native_coordinates

    def test_base_definition(self):
        node = Array(source=self.data)
        d = node.base_definition
        source = np.array(d["source"])
        np.testing.assert_array_equal(source, self.data)

    def test_definition(self):
        node = Array(source=self.data)
        node2 = Node.from_definition(node.definition)
        assert isinstance(node2, Array)
        np.testing.assert_array_equal(node2.source, self.data)


class TestPyDAP(object):
    """test pydap datasource"""

    source = "http://demo.opendap.org"
    username = "username"
    password = "password"
    datakey = "key"

    # mock parameters and data
    data = np.random.rand(11, 11)  # mocked from pydap endpoint
    coordinates = Coordinates([clinspace(-25, 25, 11), clinspace(-25, 25, 11)], dims=["lat", "lon"])

    def mock_pydap(self):
        def open_url(url, session=None):
            base = pydap.model.BaseType(name="key", data=self.data)
            dataset = pydap.model.DatasetType(name="dataset")
            dataset["key"] = base
            return dataset

        pydap.client.open_url = open_url

    def test_init(self):
        """test basic init of class"""

        node = PyDAP(source=self.source, datakey=self.datakey, username=self.username, password=self.password)
        assert isinstance(node, PyDAP)

        node = MockPyDAP()
        assert isinstance(node, MockPyDAP)

    def test_traits(self):
        """ check each of the pydap traits """

        with pytest.raises(TraitError):
            PyDAP(source=5, datakey=self.datakey)

        with pytest.raises(TraitError):
            PyDAP(source=self.source, datakey=5)

        nodes = [PyDAP(source=self.source, datakey=self.datakey), MockPyDAP()]

        # TODO: in traitlets, if you already define variable, it won't enforce case on
        # redefinition
        with pytest.raises(TraitError):
            nodes[0].username = 5

        with pytest.raises(TraitError):
            nodes[0].password = 5

        for node in nodes:
            with pytest.raises(TraitError):
                node.auth_class = "auth_class"

            with pytest.raises(TraitError):
                node.auth_session = "auth_class"

            with pytest.raises(TraitError):
                node.dataset = [1, 2, 3]

    def test_auth_session(self):
        """test auth_session attribute and traitlet default """

        # default to none if no username and password
        node = PyDAP(source=self.source, datakey=self.datakey)
        assert node.auth_session is None

        # default to none if no auth_class
        node = PyDAP(source=self.source, datakey=self.datakey, username=self.username, password=self.password)
        assert node.auth_session is None

    def test_dataset(self):
        """test dataset trait """
        self.mock_pydap()

        node = PyDAP(source=self.source, datakey=self.datakey)
        assert isinstance(node.dataset, pydap.model.DatasetType)

    def test_get_data(self):
        """test get_data function of pydap"""
        self.mock_pydap()

        node = PyDAP(source=self.source, datakey=self.datakey, native_coordinates=self.coordinates)
        output = node.eval(self.coordinates)
        assert isinstance(output, UnitsDataArray)
        assert output.values[0, 0] == self.data[0, 0]

        node = MockPyDAP(native_coordinates=self.coordinates)
        output = node.eval(self.coordinates)
        assert isinstance(output, UnitsDataArray)

    def test_native_coordinates(self):
        """test native coordinates of pydap datasource"""
        pass

    def test_keys(self):
        """test return of dataset keys"""
        self.mock_pydap()

        node = MockPyDAP(native_coordinates=self.coordinates)
        keys = node.keys
        assert "key" in keys


class TestCSV(object):
    """ test csv data source
    """

    source = os.path.join(os.path.dirname(__file__), "assets/points.csv")

    def test_init(self):
        try:
            node = CSV(source=self.source)
            raise Exception("No error raised when keys not specified")
        except TypeError:
            pass

        node = CSV(source=self.source, lat_col=0, lon_col=1, time_col=2, alt_col=3, data_col=4)
        assert node._lat_col == 0
        assert node._lon_col == 1
        assert node._time_col == 2
        assert node._alt_col == 3
        assert node._data_col == 4

        node = CSV(source=self.source, lat_col="lat", lon_col="lon", time_col="time", alt_col="alt", data_col="data")
        assert node._lat_col == 0
        assert node._lon_col == 1
        assert node._time_col == 2
        assert node._alt_col == 3
        assert node._data_col == 4

    def test_native_coordinates(self):
        node = CSV(source=self.source, lat_col=0, lon_col=1, time_col=2, alt_col=3, data_col="data")
        nc = node.native_coordinates
        assert nc.size == 5
        np.testing.assert_array_equal(nc["lat"].coordinates, [0, 1, 1, 1, 1])
        np.testing.assert_array_equal(nc["lon"].coordinates, [0, 0, 2, 2, 2])
        np.testing.assert_array_equal(nc["alt"].coordinates, [0, 0, 0, 0, 4])

    def test_data(self):
        node = CSV(source=self.source, lat_col=0, lon_col=1, time_col=2, alt_col=3, data_col="data")
        d = node.eval(node.native_coordinates)
        np.testing.assert_array_equal(d, [0, 1, 2, 3, 4])


class TestRasterio(object):
    """test rasterio data source"""

    source = os.path.join(os.path.dirname(__file__), "assets/RGB.byte.tif")
    band = 1

    def test_init(self):
        """test basic init of class"""

        node = Rasterio(source=self.source, band=self.band)
        assert isinstance(node, Rasterio)

        node = MockRasterio()
        assert isinstance(node, MockRasterio)

    def test_traits(self):
        """ check each of the rasterio traits """

        with pytest.raises(TraitError):
            Rasterio(source=5, band=self.band)

        with pytest.raises(TraitError):
            Rasterio(source=self.source, band="test")

    def test_dataset(self):
        """test dataset attribute and trait default """

        node = Rasterio(source=self.source, band=self.band)
        try:
            RasterReader = rasterio._io.RasterReader  # Rasterio < v1.0
        except:
            RasterReader = rasterio.io.DatasetReader  # Rasterio >= v1.0
        assert isinstance(node.dataset, RasterReader)

        node.close_dataset()

    def test_default_native_coordinates(self):
        """test default native coordinates implementations"""

        node = Rasterio(source=self.source)
        native_coordinates = node.get_native_coordinates()
        assert isinstance(native_coordinates, Coordinates)
        assert len(native_coordinates["lat"]) == 718

    def test_get_data(self):
        """test default get_data method"""

        node = Rasterio(source=self.source)
        native_coordinates = node.get_native_coordinates()
        output = node.eval(native_coordinates)

        assert isinstance(output, UnitsDataArray)

    def test_band_descriptions(self):
        """test band count method"""
        node = Rasterio(source=self.source)
        bands = node.band_descriptions
        assert bands and isinstance(bands, OrderedDict)

    def test_band_count(self):
        """test band descriptions methods"""
        node = Rasterio(source=self.source)
        count = node.band_count
        assert count and isinstance(count, int)

    def test_band_keys(self):
        """test band keys methods"""
        node = Rasterio(source=self.source)
        keys = node.band_keys
        assert keys and isinstance(keys, dict)

    def test_get_band_numbers(self):
        """test band numbers methods"""
        node = Rasterio(source=self.source)
        numbers = node.get_band_numbers("STATISTICS_MINIMUM", "0")
        assert isinstance(numbers, np.ndarray)
        np.testing.assert_array_equal(numbers, np.arange(3) + 1)


class TestH5PY(object):
    source = os.path.join(os.path.dirname(__file__), "assets/h5raster.hdf5")

    def test_init(self):
        node = H5PY(source=self.source, datakey="data/init", latkey="coords/lat", lonkey="coords/lon")
        node.dataset
        node.close_dataset()

    def test_native_coordinates(self):
        node = H5PY(
            source=self.source, datakey="data/init", latkey="coords/lat", lonkey="coords/lon", dims=["lat", "lon"]
        )

        nc = node.native_coordinates
        assert node.native_coordinates.shape == (3, 4)
        np.testing.assert_array_equal(node.native_coordinates["lat"].coordinates, [45.1, 45.2, 45.3])
        np.testing.assert_array_equal(node.native_coordinates["lon"].coordinates, [-100.1, -100.2, -100.3, -100.4])

    def test_data(self):
        node = H5PY(
            source=self.source, datakey="data/init", latkey="coords/lat", lonkey="coords/lon", dims=["lat", "lon"]
        )

        o = node.eval(node.native_coordinates)
        np.testing.assert_array_equal(o.data.ravel(), np.arange(12))

    def test_keys(self):
        node = H5PY(source=self.source, datakey="data/init", latkey="coords/lat", lonkey="coords/lon")
        assert node.keys == ["/coords/lat", "/coords/lon", "/data/init"]

    def test_attrs(self):
        node = H5PY(source=self.source, datakey="data/init", latkey="coords/lat", lonkey="coords/lon")
        assert node.attrs() == {}
        assert node.attrs("data") == {"test": "test"}
        assert node.attrs("coords/lat") == {"unit": "degrees"}
        assert node.attrs("coords/lon") == {"unit": "degrees"}
        assert node.attrs("coords") == {"crs": "EPSG:4326s"}


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

        import podpac.core.data.types

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
        podpac.core.data.types.lxml = None

        self.mock_requests()
        node = WCS(source=self.source)
        coordinates = node.wcs_coordinates

        assert isinstance(coordinates, Coordinates)
        assert coordinates["lat"]
        assert coordinates["lon"]
        assert coordinates["time"]

        # urllib3
        podpac.core.data.types.requests = None

        # no requests, urllib3
        podpac.core.data.types.urllib3 = None

        node = WCS(source=self.source)
        with pytest.raises(Exception):
            node.get_wcs_coordinates()

        # put all dependencies back
        podpac.core.data.types.requests = requests
        podpac.core.data.types.urllib3 = urllib3
        podpac.core.data.types.lxml = lxml

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


class TestReprojectedSource(object):

    """Test Reprojected Source
    TODO: this needs to be reworked with real examples
    """

    source = Node()
    data = np.random.rand(11, 11)
    native_coordinates = Coordinates([clinspace(-25, 25, 11), clinspace(-25, 25, 11)], dims=["lat", "lon"])
    reprojected_coordinates = Coordinates([clinspace(-25, 50, 11), clinspace(-25, 50, 11)], dims=["lat", "lon"])

    def test_init(self):
        """test basic init of class"""

        node = ReprojectedSource(source=self.source)
        assert isinstance(node, ReprojectedSource)

    def test_traits(self):
        """ check each of the s3 traits """

        ReprojectedSource(source=self.source)
        with pytest.raises(TraitError):
            ReprojectedSource(source=5)

        ReprojectedSource(source_interpolation="bilinear")
        with pytest.raises(TraitError):
            ReprojectedSource(source_interpolation=5)

        ReprojectedSource(reprojected_coordinates=self.reprojected_coordinates)
        with pytest.raises(TraitError):
            ReprojectedSource(reprojected_coordinates=5)

    def test_native_coordinates(self):
        """test native coordinates"""

        # error if no source has coordinates
        with pytest.raises(Exception):
            node = ReprojectedSource(source=Node())
            node.native_coordinates

        # source as Node
        node = ReprojectedSource(source=self.source, reprojected_coordinates=self.reprojected_coordinates)
        assert isinstance(node.native_coordinates, Coordinates)
        assert node.native_coordinates["lat"].coordinates[0] == self.reprojected_coordinates["lat"].coordinates[0]

    def test_get_data(self):
        """test get data from reprojected source"""
        datanode = Array(source=self.data, native_coordinates=self.native_coordinates)
        node = ReprojectedSource(source=datanode, reprojected_coordinates=datanode.native_coordinates)
        output = node.eval(node.native_coordinates)
        assert isinstance(output, UnitsDataArray)

    def test_base_ref(self):
        """test base ref"""

        node = ReprojectedSource(source=self.source, reprojected_coordinates=self.reprojected_coordinates)
        ref = node.base_ref
        assert "_reprojected" in ref

    def test_base_definition(self):
        """test definition"""

        node = ReprojectedSource(source=self.source, reprojected_coordinates=self.reprojected_coordinates)
        d = node.base_definition
        assert d["attrs"]["reprojected_coordinates"] == self.reprojected_coordinates

    def test_deserialize_reprojected_coordinates(self):
        node1 = ReprojectedSource(source=self.source, reprojected_coordinates=self.reprojected_coordinates)
        node2 = ReprojectedSource(source=self.source, reprojected_coordinates=self.reprojected_coordinates.definition)
        node3 = ReprojectedSource(source=self.source, reprojected_coordinates=self.reprojected_coordinates.json)

        assert node1.reprojected_coordinates == self.reprojected_coordinates
        assert node2.reprojected_coordinates == self.reprojected_coordinates
        assert node3.reprojected_coordinates == self.reprojected_coordinates


#####
# Implemented Data Source Classes
#####
class MockPyDAP(PyDAP):
    """mock pydap data source """

    source = "http://demo.opendap.org"
    username = "username"
    password = "password"
    datakey = "key"

    def get_native_coordinates(self):
        return self.native_coordinates


class MockRasterio(Rasterio):
    """mock rasterio data source """

    source = os.path.join(os.path.dirname(__file__), "assets/RGB.byte.tif")
    band = 1

    def get_native_coordinates(self):
        return self.native_coordinates


class TestZarr(object):
    path = os.path.join(os.path.dirname(__file__), "assets", "zarr")

    def test_local(self):
        node = Zarr(source=self.path, datakey="a", dims=["lat", "lon"])

    def test_local_invalid_path(self):
        with pytest.raises(ValueError, match="No Zarr store found"):
            Zarr(source="/does/not/exist", datakey="a", dims=["lat", "lon"])

    def test_invalid_datakey(self):
        with pytest.raises(ValueError, match="Zarr data key 'other' not found"):
            Zarr(source=self.path, datakey="other", dims=["lat", "lon"])

    def test_invalid_dims(self):
        with pytest.raises(TraitError):
            Zarr(source=self.path, datakey="a", dims=["y", "x"])

        with pytest.raises(ValueError, match="Zarr time key 'time' not found"):
            Zarr(source=self.path, datakey="a", dims=["lat", "lon", "time"])

    def test_invalid_dimkey(self):
        with pytest.raises(TypeError, match="Zarr node 'latkey' is required"):
            Zarr(source=self.path, datakey="a", dims=["lat", "lon"], latkey=None)

        with pytest.raises(TypeError, match="Zarr node 'lonkey' is required"):
            Zarr(source=self.path, datakey="a", dims=["lat", "lon"], lonkey=None)

        # these can be None because they are not in the store
        Zarr(source=self.path, datakey="a", dims=["lat", "lon"], timekey=None)
        Zarr(source=self.path, datakey="a", dims=["lat", "lon"], altkey=None)

        with pytest.raises(ValueError, match="Zarr lat key 'latitude' not found"):
            Zarr(source=self.path, datakey="a", dims=["lat", "lon"], latkey="latitude")

        with pytest.raises(ValueError, match="Zarr lon key 'longitude' not found"):
            Zarr(source=self.path, datakey="a", dims=["lat", "lon"], lonkey="longitude")

    def test_native_coordinates(self):
        node = Zarr(source=self.path, datakey="a", dims=["lat", "lon"])
        assert node.native_coordinates == podpac.Coordinates([[0, 1, 2], [10, 20, 30, 40]], dims=["lat", "lon"])

    def test_eval(self):
        c = podpac.Coordinates([0, 10], dims=["lat", "lon"])
        a = Zarr(source=self.path, datakey="a", dims=["lat", "lon"])
        b = Zarr(source=self.path, datakey="b", dims=["lat", "lon"])

        assert a.eval(c)[0, 0] == 0.0
        assert b.eval(c)[0, 0] == 1.0

    @pytest.mark.aws
    def test_s3(self):
        path = "s3://podpac-internal-test/drought_parameters.zarr"
        node = Zarr(source=path, datakey="d0", dims=["lat", "lon", "time"])

    def test_group(self):
        dataset = zarr.open(self.path, "r")
        node = Zarr(dataset=dataset, datakey="a", dims=["lat", "lon"])
