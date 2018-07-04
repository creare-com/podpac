"""
Test podpac.core.data.type module
"""

import pytest

import numpy as np
from traitlets import TraitError
from xarray.core.coordinates import DataArrayCoordinates

import pydap

from podpac.core import data
from podpac.core.units import UnitsDataArray
from podpac.core.data.data import COMMON_DATA_DOC
from podpac.core.node import COMMON_NODE_DOC 
from podpac.core.data.type import COMMON_DOC, NumpyArray, PyDAP
from podpac.core.coordinate import Coordinate


####
# Tests
####
class TestType(object):
    """Test podpac.core.data.type module"""

    def test_allow_missing_modules(self):
        """TODO: Allow user to be missing rasterio and scipy"""
        pass

    def test_common_doc(self):
        """Test that all COMMON_DATA_DOC keys make it into the COMMON_DOC and overwrite COMMON_NODE_DOC keys"""

        for key in COMMON_DATA_DOC:
            assert key in COMMON_DOC and COMMON_DOC[key] == COMMON_DATA_DOC[key]

        for key in COMMON_NODE_DOC:
            if key in COMMON_DATA_DOC:
                assert key in COMMON_DOC and COMMON_DOC[key] != COMMON_NODE_DOC[key]
            else:
                assert key in COMMON_DOC and COMMON_DOC[key] == COMMON_NODE_DOC[key]

    class TestArray(object):
        """Test Array datasource class (formerly NumpyArray)"""

        data = np.random.rand(11, 11)
        coordinates = Coordinate(lat=(-25, 25, 11), lon=(-25, 25, 11), order=['lat', 'lon'])

        def test_source_trait(self):
            """ must be an ndarry """
            
            node = NumpyArray(source=self.data, native_coordinates=self.coordinates)
            assert isinstance(node, NumpyArray)

            with pytest.raises(TraitError):
                node = NumpyArray(source=[0, 1, 1], native_coordinates=self.coordinates)

        def test_get_data(self):
            """ defined get_data function"""
            
            source = self.data
            node = NumpyArray(source=source, native_coordinates=self.coordinates)
            output = node.execute(self.coordinates)

            assert isinstance(output, UnitsDataArray)
            assert output.values[0, 0] == source[0, 0]
            assert output.values[4, 5] == source[4, 5]

        def test_native_coordinates(self):
            """test that native coordinates get defined"""
            
            node = NumpyArray(source=self.data)
            with pytest.raises(NotImplementedError):
                node.get_native_coordinates()

            node = NumpyArray(source=self.data, native_coordinates=self.coordinates)
            assert node.native_coordinates

            # TODO: get rid of this when this returns native_coordinates by default
            with pytest.raises(NotImplementedError):
                node.get_native_coordinates()


    class TestPyDAP(object):
        """test pydap datasource"""

        source = 'http://demo.opendap.org'
        username = 'username'
        password = 'password'
        datakey = 'key'

        # mock parameters and data
        data = np.random.rand(11, 11)   # mocked from pydap endpoint
        coordinates = Coordinate(lat=(-25, 25, 11), lon=(-25, 25, 11), order=['lat', 'lon'])

        def mock_pydap(self):

            def open_url(url, session=None):
                base = pydap.model.BaseType(name='key', data=self.data)
                dataset = pydap.model.DatasetType(name='dataset')
                dataset['key'] = base
                return dataset

            pydap.client.open_url = open_url

        def test_init(self):
            """test basic init of class"""

            node = PyDAP(source=self.source,
                         datakey=self.datakey,
                         username=self.username,
                         password=self.password)
            assert isinstance(node, PyDAP)

            node = MockPyDAP()
            assert isinstance(node, MockPyDAP)

        def test_traits(self):
            """ check each of the pydap traits """

            with pytest.raises(TraitError):
                PyDAP(source=5, datakey=self.datakey)

            with pytest.raises(TraitError):
                PyDAP(source=self.source, datakey=5)

            nodes = [
                PyDAP(source=self.source, datakey=self.datakey),
                MockPyDAP()
            ]

            # TODO: in traitlets, if you already define variable, it won't enforce case on
            # redefinition
            with pytest.raises(TraitError):
                nodes[0].username = 5

            with pytest.raises(TraitError):
                nodes[0].password = 5

            for node in nodes:
                with pytest.raises(TraitError):
                    node.auth_class = 'auth_class'

                with pytest.raises(TraitError):
                    node.auth_session = 'auth_class'

                with pytest.raises(TraitError):
                    node.dataset = [1, 2, 3]

        def test_auth_session(self):
            """test auth_session attribute and traitlet default """

            # default to none if no username and password
            node = PyDAP(source=self.source, datakey=self.datakey)
            assert node.auth_session is None

            # default to none if no auth_class
            node = PyDAP(source=self.source, datakey=self.datakey,
                         username=self.username, password=self.password)
            assert node.auth_session is None

        def test_dataset(self):
            """test dataset attribute and traitlet default """
            self.mock_pydap()

            node = PyDAP(source=self.source, datakey=self.datakey)

            # override/reset source on dataset opening
            node._open_dataset(source='newsource')
            assert node.source == 'newsource'
            assert isinstance(node.dataset, pydap.model.DatasetType)

        def test_source(self):
            """test source attribute and trailet observer """
            self.mock_pydap()

            node = PyDAP(source=self.source,
                         datakey=self.datakey,
                         native_coordinates=self.coordinates)

            # observe source
            node._update_dataset(change={'old': None})
            assert node.source == self.source

            output = node._update_dataset(change={'new': 'newsource', 'old': 'oldsource'})
            assert node.source == 'newsource'
            assert node.native_coordinates == self.coordinates
            assert isinstance(node.dataset, pydap.model.DatasetType)

        def test_get_data(self):
            """test get_data function of pydap"""
            self.mock_pydap()

            node = PyDAP(source=self.source,
                         datakey=self.datakey,
                         native_coordinates=self.coordinates)
            output = node.execute(self.coordinates)
            assert isinstance(output, UnitsDataArray)
            assert output.values[0, 0] == self.data[0, 0]

            node = MockPyDAP(native_coordinates=self.coordinates)
            output = node.execute(self.coordinates)
            assert isinstance(output, UnitsDataArray)


        def test_native_coordinates(self):
            """test native coordinates of pydap datasource"""
            pass

        def test_keys(self):
            """test return of dataset keys"""
            self.mock_pydap()

            node = MockPyDAP(native_coordinates=self.coordinates)
            keys = node.keys
            assert 'key' in keys


    class TestRasterio(object):
        """test rasterio data source"""

        source = 'path/to/rasterio'
        band = 1


class MockPyDAP(PyDAP):
    """mock pydap data source """
    
    source = 'http://demo.opendap.org'
    username = 'username'
    password = 'password'
    datakey = 'key'

    def get_native_coordinates(self):
        return self.native_coordinates
