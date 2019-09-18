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
        b = podpac.algorithm.CoordData()
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
        assert isinstance(node.B, podpac.algorithm.CoordData)
        assert isinstance(node.C, podpac.compositor.OrderedCompositor)

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
        assert isinstance(node.B, podpac.algorithm.CoordData)
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
        assert isinstance(node.B, podpac.algorithm.CoordData)
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


# TODO: remove this - this is currently a placeholder test until we actually have integration tests (pytest will exit with code 5 if no tests found)
@pytest.mark.integration
def tests_node_integration():
    assert True
