from __future__ import division, unicode_literals, print_function, absolute_import

import os
from collections import OrderedDict
import json
import six

try:
    import urllib.parse as urllib
except:  # Python 2.7
    import urlparse as urllib

import pytest
import numpy as np
import xarray as xr
from pint.errors import DimensionalityError, UndefinedUnitError
from pint import UnitRegistry

ureg = UnitRegistry()
import traitlets as tl

import podpac
from podpac.core import common_test_utils as ctu
from podpac.core.utils import ArrayTrait
from podpac.core.units import UnitsDataArray
from podpac.core.node import Node, NodeException
from podpac.core.cache import CacheCtrl, RamCacheStore


class TestNode(object):
    def test_eval_not_implemented(self):
        n = Node()
        with pytest.raises(NotImplementedError):
            n.eval(None)

        with pytest.raises(NotImplementedError):
            n.eval(None, output=None)

    def test_find_coordinates_not_implemented(self):
        n = Node()
        with pytest.raises(NotImplementedError):
            n.find_coordinates()

    def test_eval_group(self):
        class MyNode(Node):
            def eval(self, coordinates, output=None):
                return self.create_output_array(coordinates)

        c1 = podpac.Coordinates([[0, 1], [0, 1]], dims=["lat", "lon"])
        c2 = podpac.Coordinates([[10, 11], [10, 11, 12]], dims=["lat", "lon"])
        g = podpac.coordinates.GroupCoordinates([c1, c2])

        n = MyNode()
        outputs = n.eval_group(g)
        assert isinstance(outputs, list)
        assert len(outputs) == 2
        assert isinstance(outputs[0], UnitsDataArray)
        assert isinstance(outputs[1], UnitsDataArray)
        assert outputs[0].shape == (2, 2)
        assert outputs[1].shape == (2, 3)

        # invalid
        with pytest.raises(Exception):
            n.eval_group(c1)

        with pytest.raises(Exception):
            n.eval(g)

    def test_units(self):
        n = Node(units="meters")

        with pytest.raises(UndefinedUnitError):
            Node(units="abc")


class TestCreateOutputArray(object):
    def test_create_output_array_default(self):
        c = podpac.Coordinates([podpac.clinspace((0, 0), (1, 1), 10), [0, 1, 2]], dims=["lat_lon", "time"])
        node = Node()

        output = node.create_output_array(c)
        assert isinstance(output, UnitsDataArray)
        assert output.shape == c.shape
        assert output.dtype == node.dtype
        assert output.crs == c.crs
        assert np.all(np.isnan(output))

    def test_create_output_array_data(self):
        c = podpac.Coordinates([podpac.clinspace((0, 0), (1, 1), 10), [0, 1, 2]], dims=["lat_lon", "time"])
        node = Node()

        output = node.create_output_array(c, data=0)
        assert isinstance(output, UnitsDataArray)
        assert output.shape == c.shape
        assert output.dtype == node.dtype
        assert output.crs == c.crs
        assert np.all(output == 0.0)

    def test_create_output_array_dtype(self):
        c = podpac.Coordinates([podpac.clinspace((0, 0), (1, 1), 10), [0, 1, 2]], dims=["lat_lon", "time"])
        node = Node(dtype=bool)

        output = node.create_output_array(c, data=0)
        assert isinstance(output, UnitsDataArray)
        assert output.shape == c.shape
        assert output.dtype == node.dtype
        assert output.crs == c.crs
        assert np.all(~output)

    def test_create_output_array_units(self):
        c = podpac.Coordinates([podpac.clinspace((0, 0), (1, 1), 10), [0, 1, 2]], dims=["lat_lon", "time"])
        node = Node(units="meters")

        output = node.create_output_array(c)
        assert isinstance(output, UnitsDataArray)

        from podpac.core.units import ureg as _ureg

        assert output.units == _ureg.meters

    def test_create_output_array_crs(self):
        crs = "+proj=merc +lat_ts=56.5 +ellps=GRS80"
        c = podpac.Coordinates([podpac.clinspace((0, 0), (1, 1), 10), [0, 1, 2]], dims=["lat_lon", "time"], crs=crs)
        node = Node()

        output = node.create_output_array(c)
        assert output.crs == crs


class TestCaching(object):
    @classmethod
    def setup_class(cls):
        class MyNode(Node):
            pass

        cls.node = MyNode(cache_ctrl=CacheCtrl([RamCacheStore()]))
        cls.node.rem_cache(key="*", coordinates="*")

        cls.coords = podpac.Coordinates([0, 0], dims=["lat", "lon"])
        cls.coords2 = podpac.Coordinates([1, 1], dims=["lat", "lon"])

    @classmethod
    def teardown_class(cls):
        cls.node.rem_cache(key="*", coordinates="*")

    def setup_method(self, method):
        self.node.rem_cache(key="*", coordinates="*")

    def teardown_method(self, method):
        self.node.rem_cache(key="*", coordinates="*")

    def test_has_cache(self):
        assert not self.node.has_cache("test")

        self.node.put_cache(0, "test")
        assert self.node.has_cache("test")
        assert not self.node.has_cache("test", coordinates=self.coords)

    def test_has_coordinates(self):
        assert not self.node.has_cache("test", coordinates=self.coords)

        self.node.put_cache(0, "test", coordinates=self.coords)

        assert not self.node.has_cache("test")
        assert self.node.has_cache("test", coordinates=self.coords)
        assert not self.node.has_cache("test", coordinates=self.coords2)

    def test_get_put_cache(self):
        with pytest.raises(NodeException):
            self.node.get_cache("test")

        self.node.put_cache(0, "test")
        assert self.node.get_cache("test") == 0

    def test_get_put_coordinates(self):
        with pytest.raises(NodeException):
            self.node.get_cache("test")
        with pytest.raises(NodeException):
            self.node.get_cache("test", coordinates=self.coords)
        with pytest.raises(NodeException):
            self.node.get_cache("test", coordinates=self.coords2)

        self.node.put_cache(0, "test")
        self.node.put_cache(1, "test", coordinates=self.coords)
        self.node.put_cache(2, "test", coordinates=self.coords2)

        assert self.node.get_cache("test") == 0
        assert self.node.get_cache("test", coordinates=self.coords) == 1
        assert self.node.get_cache("test", coordinates=self.coords2) == 2

    def test_put_overwrite(self):
        self.node.put_cache(0, "test")
        assert self.node.get_cache("test") == 0

        with pytest.raises(NodeException):
            self.node.put_cache(1, "test")

        self.node.put_cache(1, "test", overwrite=True)
        assert self.node.get_cache("test") == 1

    def test_rem_all(self):
        self.node.put_cache(0, "a")
        self.node.put_cache(0, "b")
        self.node.put_cache(0, "a", coordinates=self.coords)
        self.node.put_cache(0, "c", coordinates=self.coords)
        self.node.put_cache(0, "c", coordinates=self.coords2)
        self.node.put_cache(0, "d", coordinates=self.coords)

        self.node.rem_cache(key="*", coordinates="*")
        assert not self.node.has_cache("a")
        assert not self.node.has_cache("b")
        assert not self.node.has_cache("a", coordinates=self.coords)
        assert not self.node.has_cache("c", coordinates=self.coords)
        assert not self.node.has_cache("c", coordinates=self.coords2)
        assert not self.node.has_cache("d", coordinates=self.coords)

    def test_rem_key(self):
        self.node.put_cache(0, "a")
        self.node.put_cache(0, "b")
        self.node.put_cache(0, "a", coordinates=self.coords)
        self.node.put_cache(0, "c", coordinates=self.coords)
        self.node.put_cache(0, "c", coordinates=self.coords2)
        self.node.put_cache(0, "d", coordinates=self.coords)

        self.node.rem_cache(key="a", coordinates="*")

        assert not self.node.has_cache("a")
        assert not self.node.has_cache("a", coordinates=self.coords)
        assert self.node.has_cache("b")
        assert self.node.has_cache("c", coordinates=self.coords)
        assert self.node.has_cache("c", coordinates=self.coords2)
        assert self.node.has_cache("d", coordinates=self.coords)

    def test_rem_coordinates(self):
        self.node.put_cache(0, "a")
        self.node.put_cache(0, "b")
        self.node.put_cache(0, "a", coordinates=self.coords)
        self.node.put_cache(0, "c", coordinates=self.coords)
        self.node.put_cache(0, "c", coordinates=self.coords2)
        self.node.put_cache(0, "d", coordinates=self.coords)

        self.node.rem_cache(key="*", coordinates=self.coords)

        assert self.node.has_cache("a")
        assert not self.node.has_cache("a", coordinates=self.coords)
        assert self.node.has_cache("b")
        assert not self.node.has_cache("c", coordinates=self.coords)
        assert self.node.has_cache("c", coordinates=self.coords2)
        assert not self.node.has_cache("d", coordinates=self.coords)

    def test_rem_key_coordinates(self):
        self.node.put_cache(0, "a")
        self.node.put_cache(0, "b")
        self.node.put_cache(0, "a", coordinates=self.coords)
        self.node.put_cache(0, "c", coordinates=self.coords)
        self.node.put_cache(0, "c", coordinates=self.coords2)
        self.node.put_cache(0, "d", coordinates=self.coords)

        self.node.rem_cache(key="a", coordinates=self.coords)

        assert self.node.has_cache("a")
        assert not self.node.has_cache("a", coordinates=self.coords)
        assert self.node.has_cache("b")
        assert self.node.has_cache("c", coordinates=self.coords)
        assert self.node.has_cache("c", coordinates=self.coords2)
        assert self.node.has_cache("d", coordinates=self.coords)


class TestCachePropertyDecorator(object):
    def test_cache_property_decorator(self):
        class Test(podpac.Node):
            a = tl.Int(1).tag(attr=True)
            b = tl.Int(1).tag(attr=True)
            c = tl.Int(1)
            d = tl.Int(1)

            @podpac.core.node.cache_func("a2", "a")
            def a2(self):
                """a2 docstring"""
                return self.a * 2

            @podpac.core.node.cache_func("b2")
            def b2(self):
                """ b2 docstring """
                return self.b * 2

            @podpac.core.node.cache_func("c2", "c")
            def c2(self):
                """ c2 docstring """
                return self.c * 2

            @podpac.core.node.cache_func("d2")
            def d2(self):
                """ d2 docstring """
                return self.d * 2

        t = Test(cache_ctrl=CacheCtrl([RamCacheStore()]))
        t2 = Test(cache_ctrl=CacheCtrl([RamCacheStore()]))
        t.rem_cache(key="*", coordinates="*")
        t2.rem_cache(key="*", coordinates="*")

        try:
            t.get_cache("a2")
            raise Exception("Cache should be cleared.")
        except podpac.NodeException:
            pass

        assert t.a2() == 2
        assert t.b2() == 2
        assert t.c2() == 2
        assert t.d2() == 2
        assert t2.a2() == 2
        assert t2.b2() == 2
        assert t2.c2() == 2
        assert t2.d2() == 2

        t.a = 2
        assert t.a2() == 4
        t.b = 2
        assert t.b2() == 4  # This happens because the node definition changed
        t.rem_cache(key="*", coordinates="*")
        assert t.c2() == 2  # This forces the cache to update based on the new node definition
        assert t.d2() == 2  # This forces the cache to update based on the new node definition
        t.c = 2
        assert t.c2() == 4  # This happens because of depends
        t.d = 2
        assert t.d2() == 2  # No depends, and doesn't have a tag

        # These should not change
        assert t2.a2() == 2
        assert t2.b2() == 2
        assert t2.c2() == 2
        assert t2.d2() == 2

        t2.a = 2
        assert t2.get_cache("a2") == 4  # This was cached by t
        t2.b = 2
        assert t2.get_cache("c2") == 4  # This was cached by t
        assert t2.get_cache("d2") == 2  # This was cached by t

    def test_cache_func_decorator_with_no_cache(self):
        class Test(podpac.Node):
            a = tl.Int(1).tag(attr=True)
            b = tl.Int(1).tag(attr=True)
            c = tl.Int(1)
            d = tl.Int(1)

            @podpac.core.node.cache_func("a2", "a")
            def a2(self):
                """a2 docstring"""
                return self.a * 2

            @podpac.core.node.cache_func("b2")
            def b2(self):
                """ b2 docstring """
                return self.b * 2

            @podpac.core.node.cache_func("c2", "c")
            def c2(self):
                """ c2 docstring """
                return self.c * 2

            @podpac.core.node.cache_func("d2")
            def d2(self):
                """ d2 docstring """
                return self.d * 2

        t = Test(cache_ctrl=None)
        t2 = Test(cache_ctrl=None)
        t.rem_cache(key="*", coordinates="*")
        t2.rem_cache(key="*", coordinates="*")

        try:
            t.get_cache("a2")
            raise Exception("Cache should be cleared.")
        except podpac.NodeException:
            pass

        assert t.a2() == 2
        assert t.b2() == 2
        assert t.c2() == 2
        assert t.d2() == 2
        assert t2.a2() == 2
        assert t2.b2() == 2
        assert t2.c2() == 2
        assert t2.d2() == 2

        t.a = 2
        assert t.a2() == 4
        t.b = 2
        assert t.b2() == 4  # This happens because the node definition changed
        t.rem_cache(key="*", coordinates="*")
        assert t.c2() == 2  # This forces the cache to update based on the new node definition
        assert t.d2() == 2  # This forces the cache to update based on the new node definition
        t.c = 2
        assert t.c2() == 4  # This happens because of depends
        t.d = 2
        assert t.d2() == 4  # No caching here, so it SHOULD update

        # These should not change
        assert t2.a2() == 2
        assert t2.b2() == 2
        assert t2.c2() == 2
        assert t2.d2() == 2


class TestSerialization(object):
    @classmethod
    def setup_class(cls):
        a = podpac.algorithm.Arange()
        b = podpac.data.Array(source=[10, 20, 30], native_coordinates=podpac.Coordinates([[0, 1, 2]], dims=["lat"]))
        c = podpac.compositor.OrderedCompositor(sources=np.array([a, b]))
        cls.node = podpac.algorithm.Arithmetic(A=a, B=b, C=c, eqn="A + B + C")

        cls.node_file_path = "node.json"
        if os.path.exists(cls.node_file_path):
            os.remove(cls.node_file_path)

    @classmethod
    def teardown_class(cls):
        if os.path.exists(cls.node_file_path):
            os.remove(cls.node_file_path)

    def test_base_ref(self):
        n = Node()
        assert isinstance(n.base_ref, str)

    def test_base_definition(self):
        class N(Node):
            my_attr = tl.Int().tag(attr=True)
            my_node_attr = tl.Instance(Node).tag(attr=True)

        a = Node()
        node = N(my_attr=7, my_node_attr=a)

        d = node.base_definition
        assert isinstance(d, OrderedDict)
        assert "node" in d
        assert isinstance(d["node"], str)
        assert "attrs" in d
        assert isinstance(d["attrs"], OrderedDict)
        assert "my_attr" in d["attrs"]
        assert d["attrs"]["my_attr"] == 7
        assert isinstance(d["lookup_attrs"], OrderedDict)
        assert "my_node_attr" in d["lookup_attrs"]
        assert d["lookup_attrs"]["my_node_attr"] is a

    def test_base_definition_units(self):
        n = Node(units="meters")

        d = n.base_definition
        assert "attrs" in d
        assert isinstance(d["attrs"], OrderedDict)
        assert "units" in d["attrs"]
        assert d["attrs"]["units"] == "meters"

        n = Node()
        d = n.base_definition
        assert "units" not in d

    def test_base_definition_array_attr(self):
        class N(Node):
            my_attr = ArrayTrait().tag(attr=True)

        node = N(my_attr=np.ones((2, 3, 4)))
        d = node.base_definition
        my_attr = np.array(d["attrs"]["my_attr"])
        np.testing.assert_array_equal(my_attr, node.my_attr)

    def test_base_definition_coordinates_attr(self):
        class N(Node):
            my_attr = tl.Instance(podpac.Coordinates).tag(attr=True)

        node = N(my_attr=podpac.Coordinates([[0, 1], [1, 2, 3]], dims=["lat", "lon"]))
        d = node.base_definition
        assert d["attrs"]["my_attr"] == node.my_attr

    def test_base_definition_unserializable(self):
        class N(Node):
            my_attr = tl.Instance(xr.DataArray).tag(attr=True)

        node = N(my_attr=xr.DataArray([0, 1]))
        with pytest.raises(NodeException, match="Cannot serialize attr 'my_attr'"):
            node.base_definition

    def test_definition(self):
        # definition
        d = self.node.definition
        assert isinstance(d, OrderedDict)
        assert len(d) == 4

        # from_definition
        node = Node.from_definition(d)
        assert node is not self.node
        assert node.hash == self.node.hash
        assert isinstance(node, podpac.algorithm.Arithmetic)
        assert isinstance(node.A, podpac.algorithm.Arange)
        assert isinstance(node.B, podpac.data.Array)
        assert isinstance(node.C, podpac.compositor.OrderedCompositor)

    def test_definition_duplicate_base_ref(self):
        n1 = Node()
        n2 = Node()
        n3 = Node()
        n = podpac.compositor.OrderedCompositor(sources=[n1, n2, n3])
        d = n.definition
        assert n1.base_ref == n2.base_ref == n3.base_ref
        assert len(d) == 4

    def test_definition_lookup_attrs(self):
        global MyNodeWithNodeAttr

        class MyNodeWithNodeAttr(Node):
            my_node_attr = tl.Instance(Node).tag(attr=True)

        node = MyNodeWithNodeAttr(my_node_attr=podpac.algorithm.Arange())
        d = node.definition
        assert isinstance(d, OrderedDict)
        assert len(d) == 2

        node2 = Node.from_definition(d)
        assert node2 is not node
        assert node2.hash == node.hash
        assert isinstance(node2, MyNodeWithNodeAttr)
        assert isinstance(node2.my_node_attr, podpac.algorithm.Arange)

    def test_definition_lookup_source(self):
        global MyNodeWithNodeSource

        class MyNodeWithNodeSource(podpac.data.DataSource):
            source = tl.Instance(Node)

        node = MyNodeWithNodeSource(source=podpac.algorithm.Arange())
        d = node.definition
        assert isinstance(d, OrderedDict)
        assert len(d) == 2

        node2 = Node.from_definition(d)
        assert node2 is not node
        assert node2.hash == node.hash
        assert isinstance(node2, MyNodeWithNodeSource)
        assert isinstance(node2.source, podpac.algorithm.Arange)

    def test_json(self):
        # json
        s = self.node.json
        assert isinstance(s, str)
        assert json.loads(s)

        # test from_json
        node = Node.from_json(s)
        assert node is not self.node
        assert node.hash == self.node.hash
        assert isinstance(node, podpac.algorithm.Arithmetic)
        assert isinstance(node.A, podpac.algorithm.Arange)
        assert isinstance(node.B, podpac.data.Array)
        assert isinstance(node.C, podpac.compositor.OrderedCompositor)

    def test_file(self):
        # save
        self.node.save(self.node_file_path)

        assert os.path.exists(self.node_file_path)

        node = Node.load(self.node_file_path)
        assert node is not self.node
        assert node.hash == self.node.hash
        assert isinstance(node, podpac.algorithm.Arithmetic)
        assert isinstance(node.A, podpac.algorithm.Arange)
        assert isinstance(node.B, podpac.data.Array)
        assert isinstance(node.C, podpac.compositor.OrderedCompositor)

    def test_json_pretty(self):
        n = Node()
        s = n.json_pretty
        assert isinstance(s, str)
        json.loads(s)

    def test_hash(self):
        class N(Node):
            my_attr = tl.Int().tag(attr=True)

        class M(Node):
            my_attr = tl.Int().tag(attr=True)

        n1 = N(my_attr=1)
        n2 = N(my_attr=1)
        n3 = N(my_attr=2)
        m1 = M(my_attr=1)

        assert n1.hash == n2.hash
        assert n1.hash != n3.hash
        assert n1.hash != m1.hash

    def test_from_url(self):
        url = (
            r"http://testwms/?map=map&&service={service}&request=GetMap&{layername}={layer}&styles=&format=image%2Fpng"
            r"&transparent=true&version=1.1.1&transparency=true&width=256&height=256&srs=EPSG%3A4326"
            r"&bbox=40,-71,41,70&time=2018-05-19&PARAMS={params}"
        )

        params = ["{}", '{"a":{"node":"algorithm.Arange"}}', "{}", "{}"]

        for service, layername in zip(["WMS", "WCS"], ["LAYERS", "COVERAGE"]):
            for layer, param in zip(
                [
                    "algorithm.SinCoords",
                    "%PARAMS%",
                    # urllib.urlencode({'a':'https://raw.githubusercontent.com/creare-com/podpac/develop/podpac/core/pipeline/test/test.json'})[2:],
                    # urllib.urlencode({'a':'s3://podpac-s3/test/test.json'})[2:]  # Tested locally, works fine. Hard to test with CI
                ],
                params,
            ):
                pipe = Node.from_url(url.format(service=service, layername=layername, layer=layer, params=param))

    def test_pipeline(self):
        n = Node()
        with pytest.warns(DeprecationWarning):
            p = n.pipeline
        assert isinstance(p, podpac.pipeline.Pipeline)


class TestUserDefinition(object):
    def test_empty(self):
        s = "{ }"
        with pytest.raises(ValueError, match="definition cannot be empty"):
            Node.from_json(s)

    def test_no_node(self):
        s = '{"test": { } }'
        with pytest.raises(ValueError, match="'node' property required"):
            Node.from_json(s)

    def test_invalid_node(self):
        # module does not exist
        s = '{"a": {"node": "nonexistent.Arbitrary"} }'
        with pytest.raises(ValueError, match="no module found"):
            Node.from_json(s)

        # node does not exist in module
        s = '{"a": {"node": "core.Nonexistent"} }'
        with pytest.raises(ValueError, match="class 'Nonexistent' not found in module"):
            Node.from_json(s)

    def test_datasource_source(self):
        # basic
        s = """
        {
            "mydata": {
                "node": "data.DataSource",
                "source": "my_data_string"
            }
        }
        """

        node = Node.from_json(s)
        assert isinstance(node, podpac.data.DataSource)
        assert node.source == "my_data_string"

        # not required
        s = """
        {
            "mydata": {
                "node": "data.DataSource"
            }
        }
        """

        node = Node.from_json(s)
        assert isinstance(node, podpac.data.DataSource)

        # incorrect
        s = """
        {
            "mydata": {
                "node": "data.DataSource",
                "attrs": {
                    "source": "my_data_string"
                }
            }
        }
        """

        with pytest.raises(ValueError, match="DataSource 'attrs' cannot have a 'source' property"):
            node = Node.from_json(s)

    def test_datasource_lookup_source(self):
        # sub-node
        s = """
        {
            "mydata": {
                "node": "data.DataSource",
                "source": "my_data_string"
            },
            "double": {
                "node": "algorithm.Arithmetic",
                "inputs": {"A": "mydata"},
                "attrs": { "eqn": "2 * A" }
            },
            "mydata2": {
                "node": "data.DataSource",
                "lookup_source": "double.A.source"
            }
        }
        """

        node = Node.from_json(s)
        assert isinstance(node, podpac.data.DataSource)
        assert node.source == "my_data_string"

        # nonexistent node
        s = """
        {
            "mydata": {
                "node": "data.DataSource",
                "source": "my_data_string"
            },
            "double": {
                "node": "algorithm.Arithmetic",
                "inputs": {"A": "mydata"},
                "attrs": { "eqn": "2 * A" }
            },
            "mydata2": {
                "node": "data.DataSource",
                "lookup_source": "nonexistent.source"
            }
        }
        """

        with pytest.raises(ValueError, match="reference to nonexistent node/attribute"):
            Node.from_json(s)

        # nonexistent subattr
        s = """
        {
            "mydata": {
                "node": "data.DataSource",
                "source": "my_data_string"
            },
            "double": {
                "node": "algorithm.Arithmetic",
                "inputs": {"A": "mydata"},
                "attrs": { "eqn": "2 * A" }
            },
            "mydata2": {
                "node": "data.DataSource",
                "lookup_source": "double.nonexistent.source"
            }
        }
        """

        with pytest.raises(ValueError, match="reference to nonexistent node/attribute"):
            Node.from_json(s)

        # nonexistent subsubattr
        s = """
        {
            "mydata": {
                "node": "data.DataSource",
                "source": "my_data_string"
            },
            "double": {
                "node": "algorithm.Arithmetic",
                "inputs": {"A": "mydata"},
                "attrs": { "eqn": "2 * A" }
            },
            "mydata2": {
                "node": "data.DataSource",
                "lookup_source": "double.A.nonexistent"
            }
        }
        """

        with pytest.raises(ValueError, match="reference to nonexistent node/attribute"):
            Node.from_json(s)

        # in attrs (incorrect)
        s = """
        {
            "mydata": {
                "node": "data.DataSource",
                "attrs": {
                    "lookup_source": "my_data_string"
                }
            }
        }
        """

        with pytest.raises(ValueError, match="DataSource 'attrs' cannot have a 'lookup_source' property"):
            Node.from_json(s)

    def test_reprojected_source_lookup_source(self):
        # NOTE: nonexistent node/attribute references are tested in test_datasource_lookup_source

        # lookup_source
        s = """
        {
            "mysource": {
                "node": "data.DataSource",
                "source": "my_data_string"
            },
            "reprojected": {
                "node": "data.ReprojectedSource",
                "lookup_source": "mysource"
            }
        }
        """

        node = Node.from_json(s)
        assert isinstance(node, podpac.data.ReprojectedSource)
        assert isinstance(node.source, podpac.data.DataSource)
        assert node.source.source == "my_data_string"

        # lookup_source subattr
        s = """
        {
            "mysource": {
                "node": "data.DataSource",
                "source": "my_data_string"
            },
            "double": {
                "node": "algorithm.Arithmetic",
                "inputs": {"A": "mysource"},
                "attrs": { "eqn": "2 * A" }
            },
            "reprojected": {
                "node": "data.ReprojectedSource",
                "lookup_source": "double.A"
            }
        }
        """

        node = Node.from_json(s)
        assert isinstance(node, podpac.data.ReprojectedSource)
        assert isinstance(node.source, podpac.data.DataSource)
        assert node.source.source == "my_data_string"

        # 'source' should fail
        s = """
        {
            "mysource": {
                "node": "data.DataSource",
                "source": "my_data_string"
            },
            "reprojected": {
                "node": "data.ReprojectedSource",
                "source": "mysource"
            }
        }
        """

        with pytest.raises(tl.TraitError):
            Node.from_json(s)

    def test_array_source(self):
        s = """
        {
            "mysource": {
                "node": "data.Array",
                "source": [0, 1, 2]
            }
        }
        """

        node = Node.from_json(s)
        assert isinstance(node, podpac.data.Array)
        np.testing.assert_array_equal(node.source, [0, 1, 2])

    def test_array_lookup_source(self):
        s = """
        {
            "a": {
                "node": "data.Array",
                "source": [0, 1, 2]
            },
            "b": {
                "node": "data.Array",
                "lookup_source": "a.source"
            }
        }
        """

        node = Node.from_json(s)
        assert isinstance(node, podpac.data.Array)
        np.testing.assert_array_equal(node.source, [0, 1, 2])

        # 'source' should fail
        s = """
        {
            "a": {
                "node": "data.Array",
                "source": [0, 1, 2]
            },
            "b": {
                "node": "data.Array",
                "source": "a.source"
            }
        }
        """

        with pytest.raises(ValueError):
            Node.from_json(s)

    def test_algorithm_inputs(self):
        # NOTE: nonexistent node/attribute references are tested in test_datasource_lookup_source

        # basic
        s = """
        {
            "source1": {"node": "algorithm.Arange"},
            "source2": {"node": "algorithm.CoordData"},
            "result": {        
                "node": "algorithm.Arithmetic",
                "inputs": {
                    "A": "source1",
                    "B": "source2"
                },
                "attrs": {
                    "eqn": "A + B"
                }
            }
        }
        """

        node = Node.from_json(s)
        assert isinstance(node, podpac.algorithm.Arithmetic)
        assert isinstance(node.A, podpac.algorithm.Arange)
        assert isinstance(node.B, podpac.algorithm.CoordData)

        # sub-node
        s = """
        {
            "mysource": {"node": "algorithm.Arange"},
            "double": {        
                "node": "algorithm.Arithmetic",
                "inputs": { "A": "mysource" },
                "attrs": { "eqn": "2 * A" }
            },
            "quadruple": {
                "node": "algorithm.Arithmetic",
                "inputs": { "A": "double.A" },
                "attrs": { "eqn": "2 * A" }
            }
        }
        """

        node = Node.from_json(s)
        assert isinstance(node, podpac.algorithm.Arithmetic)
        assert isinstance(node.A, podpac.algorithm.Arange)

        # in attrs (incorrect)
        s = """
        {
            "source1": {"node": "algorithm.Arange"},
            "source2": {"node": "algorithm.CoordData"},
            "result": {        
                "node": "algorithm.Arithmetic",
                "attrs": {
                    "inputs": {
                        "A": "source1",
                        "B": "source2"
                    },
                    "eqn": "A + B"
                }
            }
        }
        """

        with pytest.raises(ValueError, match="Algorithm 'attrs' cannot have an 'inputs' property"):
            Node.from_json(s)

    def test_compositor_sources(self):
        # NOTE: nonexistent node/attribute references are tested in test_datasource_lookup_source

        # basic
        s = """
        {
            "a": {"node": "algorithm.Arange"},
            "b": {"node": "algorithm.CoordData"},
            "c": {
                "node": "compositor.OrderedCompositor",
                "sources": ["a", "b"]
            }
        }
        """

        node = Node.from_json(s)
        assert isinstance(node, podpac.compositor.OrderedCompositor)
        assert isinstance(node.sources[0], podpac.algorithm.Arange)
        assert isinstance(node.sources[1], podpac.algorithm.CoordData)

        # sub-node
        s = """
        {
            "source1": {"node": "algorithm.Arange"},
            "source2": {"node": "algorithm.CoordData"},
            "double": {
                "node": "algorithm.Arithmetic",
                "inputs": { "A": "source1" },
                "attrs": { "eqn": "2 * A" }
            },
            "c": {
                "node": "compositor.OrderedCompositor",
                "sources": ["double.A", "source2"]
            }
        }
        """

        node = Node.from_json(s)
        assert isinstance(node, podpac.compositor.OrderedCompositor)
        assert isinstance(node.sources[0], podpac.algorithm.Arange)
        assert isinstance(node.sources[1], podpac.algorithm.CoordData)

    def test_datasource_interpolation(self):
        s = """
        {
            "mydata": {
                "node": "data.DataSource",
                "source": "my_data_string",
                "interpolation": "nearest"
            }
        }
        """

        node = Node.from_json(s)
        assert isinstance(node, podpac.data.DataSource)
        assert node.interpolation == "nearest"

        # not required
        s = """
        {
            "mydata": {
                "node": "data.DataSource"
            }
        }
        """

        node = Node.from_json(s)
        assert isinstance(node, podpac.data.DataSource)

        # incorrect
        s = """
        {
            "mydata": {
                "node": "data.DataSource",
                "attrs": {
                    "interpolation": "nearest"
                }
            }
        }
        """

        with pytest.raises(ValueError, match="DataSource 'attrs' cannot have an 'interpolation' property"):
            Node.from_json(s)

    def test_compositor_interpolation(self):
        s = """
        {
            "a": {
                "node": "algorithm.Arange"
            },
            "b": {
                "node": "algorithm.Arange"
            },
            "c": {
                "node": "compositor.OrderedCompositor",
                "sources": ["a", "b"],
                "interpolation": "nearest"
            }
        }
        """

        node = Node.from_json(s)
        assert isinstance(node, podpac.compositor.OrderedCompositor)
        assert node.interpolation == "nearest"

        # not required
        s = """
        {
            "a": {
                "node": "algorithm.Arange"
            },
            "b": {
                "node": "algorithm.Arange"
            },
            "c": {
                "node": "compositor.OrderedCompositor",
                "sources": ["a", "b"]
            }
        }
        """

        node = Node.from_json(s)
        assert isinstance(node, podpac.compositor.OrderedCompositor)

        # incorrect
        s = """
        {
            "a": {
                "node": "algorithm.Arange"
            },
            "b": {
                "node": "algorithm.Arange"
            },
            "c": {
                "node": "compositor.OrderedCompositor",
                "sources": ["a", "b"],
                "attrs": {
                    "interpolation": "nearest"
                }
            }
        }
        """

        with pytest.raises(ValueError, match="Compositor 'attrs' cannot have an 'interpolation' property"):
            Node.from_json(s)

    def test_attrs(self):
        s = """
        {
            "sm": {
                "node": "datalib.smap.SMAP",
                "attrs": {
                    "product": "SPL4SMGP"
                }
            }
        }
        """

        node = Node.from_json(s)
        assert isinstance(node, podpac.datalib.smap.SMAP)
        assert node.product == "SPL4SMGP"

    def test_lookup_attrs(self):
        # NOTE: nonexistent node/attribute references are tested in test_datasource_lookup_source

        s = """
        {
            "a": {
                "node": "algorithm.CoordData",
                "attrs": { "coord_name": "lat" }
            },
            "b": {
                "node": "algorithm.CoordData",
                "lookup_attrs": { "coord_name": "a.coord_name" }
            }
        }
        """

        node = Node.from_json(s)
        assert isinstance(node, podpac.algorithm.CoordData)
        assert node.coord_name == "lat"

        # lookup node directly (instead of a sub-attr)
        global MyNodeWithNodeAttr

        class MyNodeWithNodeAttr(Node):
            my_node_attr = tl.Instance(Node).tag(attr=True)

        s = """
        {
            "mysource": {
                "node": "data.DataSource"
            },
            "mynode": {
                "plugin": "test_node",
                "node": "MyNodeWithNodeAttr",
                "lookup_attrs": {
                    "my_node_attr": "mysource"
                }
            }
        }
        """

        node = Node.from_json(s)
        assert isinstance(node, MyNodeWithNodeAttr)
        assert isinstance(node.my_node_attr, podpac.data.DataSource)

        # attrs should not work
        s = """
        {
            "a": {
                "node": "algorithm.CoordData",
                "attrs": { "coord_name": "lat" }
            },
            "b": {
                "node": "algorithm.CoordData",
                "attrs": { "coord_name": "a.coord_name" }
            }
        }
        """

        node = Node.from_json(s)
        assert node.coord_name == "a.coord_name"  # this will fail at evaluation

    def test_invalid_property(self):
        s = """
        {
            "a": {
                "node": "algorithm.Arange",
                "invalid_property": "value"
            }
        }
        """

        with pytest.raises(ValueError, match="unexpected property"):
            Node.from_json(s)

    def test_plugin(self):
        global MyPluginNode

        class MyPluginNode(Node):
            pass

        s = """
        {
            "mynode": {
                "plugin": "test_node",
                "node": "MyPluginNode"
            }
        }
        """

        node = Node.from_json(s)
        assert isinstance(node, MyPluginNode)

        # missing plugin
        s = """
        {
            "mynode": {
                "plugin": "missing",
                "node": "MyPluginNode"
            }
        }
        """

        with pytest.raises(ValueError, match="no module found"):
            Node.from_json(s)


# TODO: remove this - this is currently a placeholder test until we actually have integration tests (pytest will exit with code 5 if no tests found)
@pytest.mark.integration
def tests_node_integration():
    assert True
