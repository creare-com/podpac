import pytest
import requests
import traitlets as tl
import numpy as np

import podpac.datalib
from podpac.datalib import smap

from podpac import settings

# dummy class mixing in custom Earthdata requests Session
class SomeSmapNode(smap.SMAPSessionMixin):
    pass


class TestSMAPSessionMixin(object):
    url = "urs.earthdata.nasa.gov"

    def test_hostname(self):
        node = SomeSmapNode()
        assert node.hostname == self.url

    def test_auth_required(self):
        # make sure auth is deleted from setttings, if it was already there

        # auth required
        with settings:
            if "username@urs.earthdata.nasa.gov" in settings:
                del settings["username@urs.earthdata.nasa.gov"]

            if "password@urs.earthdata.nasa.gov" in settings:
                del settings["password@urs.earthdata.nasa.gov"]

            node = SomeSmapNode()

            # throw auth error
            with pytest.raises(ValueError, match="username"):
                node.session

            node.set_credentials(username="testuser", password="testpass")

            assert node.session
            assert node.session.auth == ("testuser", "testpass")
            assert isinstance(node.session, requests.Session)

    def test_set_credentials(self):
        with settings:
            node = SomeSmapNode()
            node.set_credentials(username="testuser", password="testpass")

            assert settings["username@{}".format(self.url)] == "testuser"
            assert settings["password@{}".format(self.url)] == "testpass"

    def test_session(self):
        with settings:

            node = SomeSmapNode()
            node.set_credentials(username="testuser", password="testpass")

            assert node.session
            assert node.session.auth == ("testuser", "testpass")
            assert isinstance(node.session, requests.Session)

    def test_earth_data_session_rebuild_auth(self):
        class Dum(object):
            pass

        with settings:
            node = SomeSmapNode()
            node.set_credentials(username="testuser", password="testpass")

            prepared_request = Dum()
            prepared_request.headers = {"Authorization": 0}
            prepared_request.url = "https://example.com"

            response = Dum()
            response.request = Dum()
            response.request.url = "https://example2.com"

            node.session.rebuild_auth(prepared_request, response)


@pytest.mark.integration
class TestSMAPSource(object):
    source = "https://n5eil02u.ecs.nsidc.org/opendap/SMAP/SPL4SMGP.004/2015.03.31/SMAP_L4_SM_gph_20150331T013000_Vv4030_001.h5"

    def setup_class(cls):
        # check source with smap session
        smap_session = smap.SMAPSessionMixin()
        response = smap_session.session.head(cls.source)
        assert response.status_code == 200

    def test_product(self):
        node = smap.SMAPSourceRaw(source=self.source)
        assert node.product == "SPL4SMGP"

    def test_version(self):
        node = smap.SMAPSourceRaw(source=self.source)
        assert node.version == 4

    def test_data_key(self):
        node = smap.SMAPSourceRaw(source=self.source)
        assert node.data_key == "Geophysical_Data_sm_surface"

    def test_lat_key(self):
        node = smap.SMAPSourceRaw(source=self.source)
        assert node.lat_key == "cell_lat"

    def test_lon_key(self):
        node = smap.SMAPSourceRaw(source=self.source)
        assert node.lon_key == "cell_lon"

    def test_available_times(self):
        node = smap.SMAPSourceRaw(source=self.source)
        np.testing.assert_array_equal(node.available_times, np.array(["2015-03-31T01:30:00"], dtype=np.datetime64))

    def test_coordinates(self):
        node = smap.SMAPSourceRaw(source=self.source)
        coordinates = node.coordinates
        assert isinstance(coordinates, podpac.Coordinates)
        assert coordinates.dims == ("time", "lat", "lon")
        assert coordinates.shape == (1, 1624, 3856)

    def test_eval_raw(self):
        time = "2015-03-31T01:30:00"
        lat = podpac.clinspace(36, 37, 5)
        lon = podpac.clinspace(36, 37, 5)
        coords = podpac.Coordinates([time, lat, lon], dims=["time", "lat", "lon"])

        node = smap.SMAPSourceRaw(source=self.source, cache_ctrl=["ram"])
        output = node.eval(coords)
        assert output.shape == (1, 13, 12)  # raw source coordinates


@pytest.mark.integration
class TestSMAPProperties(object):
    pass


@pytest.mark.integration
class TestSMAPDateFolder(object):
    product = "SPL4SMGP"
    folder_date = "2015.03.31"

    def test_folder_url(self):
        node = smap.SMAPDateFolder(product=self.product, folder_date=self.folder_date)
        assert node.folder_url == "https://n5eil02u.ecs.nsidc.org/opendap/SMAP/SPL4SMGP.004/2015.03.31"
        response = node.session.head("%s/contents.html" % node.folder_url)
        assert response.status_code == 200

    def test_sources(self):
        node = smap.SMAPDateFolder(product=self.product, folder_date=self.folder_date)
        assert len(node.sources) == 8
        assert all(isinstance(source, smap.SMAPSourceRaw) for source in node.sources)

    def test_is_source_coordinates_complete(self):
        node = smap.SMAPDateFolder(product=self.product, folder_date=self.folder_date)
        assert node.is_source_coordinates_complete

        node = smap.SMAPDateFolder(product="SPL2SMAP_S", folder_date=self.folder_date)
        assert not node.is_source_coordinates_complete

    def test_source_coordinates(self):
        node = smap.SMAPDateFolder(product=self.product, folder_date=self.folder_date)
        source_coordinates = node.source_coordinates
        assert isinstance(source_coordinates, podpac.Coordinates)
        assert source_coordinates.dims == ("time",)

    def test_eval_raw(self):
        time = ["2015-03-31T12:00:00"]
        lat = podpac.clinspace(36, 37, 5)
        lon = podpac.clinspace(36, 37, 5)
        coords = podpac.Coordinates([time, lat, lon], dims=["time", "lat", "lon"])

        node = smap.SMAPDateFolder(product=self.product, folder_date=self.folder_date, cache_ctrl=["ram"])
        output = node.eval(coords)
        assert output.shape == (2, 13, 12)  # raw source coordinates


@pytest.mark.integration
class TestSMAP(object):
    product = "SPL4SMGP"

    def test_base_url(self):
        node = smap.SMAP(product=self.product)
        assert node.base_url == "https://n5eil02u.ecs.nsidc.org/opendap/SMAP"
        response = node.session.head("%s/contents.html" % node.base_url)
        assert response.status_code == 200

    def test_version(self):
        node = smap.SMAP(product=self.product)
        assert node.version == 4

    def test_layer_key(self):
        node = smap.SMAP(product=self.product)
        assert node.layer_key == "{rdk}sm_surface"

    def test_shared_coordinates(self):
        node = smap.SMAP(product=self.product)
        shared_coordinates = node.shared_coordinates
        assert isinstance(shared_coordinates, podpac.Coordinates)
        assert shared_coordinates.dims == ("lat", "lon")
        assert shared_coordinates.shape == (1624, 3856)

    def test_available_dates(self):
        node = smap.SMAP(product=self.product)
        available_dates = node.available_dates
        assert isinstance(available_dates, list)
        assert len(available_dates) > 0
        assert len(available_dates) == len(set(available_dates))

    def test_sources(self):
        node = smap.SMAP(product=self.product)
        sources = node.sources
        assert isinstance(sources, list)
        assert all(isinstance(s, smap.SMAPDateFolder) for s in sources)

    def test_source_coordinates(self):
        node = smap.SMAP(product=self.product)
        source_coordinates = node.source_coordinates
        assert isinstance(source_coordinates, podpac.Coordinates)
        assert source_coordinates.dims == ("time",)
        assert source_coordinates.shape == (len(node.sources),)

    def test_base_ref(self):
        node = smap.SMAP(product=self.product)
        assert node.base_ref == "SMAP_%s" % self.product

    def test_eval(self):
        time = ["2015-03-31T12:00:00", "2015-04-01T12:00:00"]
        lat = podpac.clinspace(36, 37, 5)
        lon = podpac.clinspace(36, 37, 5)
        coords = podpac.Coordinates([time, lat, lon], dims=["time", "lat", "lon"])

        with podpac.settings:
            node = smap.SMAP(product=self.product, cache_ctrl=["ram"])
            output = node.eval(coords)
            assert output.shape == (2, 5, 5)

    def test_eval_raw(self):
        time = ["2015-03-31T12:00:00", "2015-04-01T12:00:00"]
        lat = podpac.clinspace(36, 37, 5)
        lon = podpac.clinspace(36, 37, 5)
        coords = podpac.Coordinates([time, lat, lon], dims=["time", "lat", "lon"])

        node = smap.SMAPRaw(product=self.product, cache_ctrl=["ram"])
        output = node.eval(coords)
        assert output.shape == (8, 13, 12)


@pytest.mark.integration
class TestSMAPBestAvailable(object):
    def test_sources(self):
        node = smap.SMAPBestAvailable()
        assert node.sources[0].product == "SPL2SMAP_S"
        assert node.sources[1].product == "SPL4SMAU"

    def test_eval(self):
        time = ["2015-03-31T12:00:00", "2015-04-01T12:00:00"]
        lat = podpac.clinspace(36, 37, 5)
        lon = podpac.clinspace(36, 37, 5)
        coords = podpac.Coordinates([time, lat, lon], dims=["time", "lat", "lon"])

        node = smap.SMAPBestAvailable(cache_ctrl=["ram"])
        output = node.eval(coords)


@pytest.mark.integration
class TestGetSMAPSources(object):
    pass
