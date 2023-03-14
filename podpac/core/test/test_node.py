from __future__ import division, unicode_literals, print_function, absolute_import

import os
import json
import warnings
import tempfile
from collections import OrderedDict
from copy import deepcopy

try:
    import urllib.parse as urllib
except:  # Python 2.7
    import urlparse as urllib

import six
import pytest
import numpy as np
import xarray as xr
from pint.errors import DimensionalityError, UndefinedUnitError
from pint import UnitRegistry

ureg = UnitRegistry()
import traitlets as tl

import podpac
from podpac.core import common_test_utils as ctu
from podpac.core.utils import ArrayTrait, NodeTrait
from podpac.core.units import UnitsDataArray
from podpac.core.style import Style
from podpac.core.cache import CacheCtrl, RamCacheStore, DiskCacheStore
from podpac.core.node import Node, NodeException, NodeDefinitionError
from podpac.core.node import NoCacheMixin, DiskCacheMixin


class TestNode(object):
    def test_style(self):
        node = Node()
        assert isinstance(node.style, Style)

    def test_units(self):
        node = Node(units="meters")

        with pytest.raises(UndefinedUnitError):
            Node(units="abc")

    def test_outputs(self):
        node = Node()
        assert node.outputs is None

        node = Node(outputs=["a", "b"])
        assert node.outputs == ["a", "b"]

    def test_output(self):
        node = Node()
        assert node.output is None

        node = Node(outputs=["a", "b"])
        assert node.output is None

        node = Node(outputs=["a", "b"], output="b")
        assert node.output == "b"

        # must be one of the outputs
        with pytest.raises(ValueError, match="Invalid output"):
            node = Node(outputs=["a", "b"], output="other")

        # only valid for multiple-output nodes
        with pytest.raises(TypeError, match="Invalid output"):
            node = Node(output="other")

    def test_cache_output(self):
        with podpac.settings:
            podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"] = False
            node = Node()
            assert not node.cache_output

            podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"] = True
            node = Node()
            assert node.cache_output

    def test_cache_ctrl(self):
        # settings
        with podpac.settings:
            podpac.settings["DEFAULT_CACHE"] = ["ram"]
            node = Node()
            assert node.cache_ctrl is not None
            assert len(node.cache_ctrl._cache_stores) == 1
            assert isinstance(node.cache_ctrl._cache_stores[0], RamCacheStore)

            podpac.settings["DEFAULT_CACHE"] = ["ram", "disk"]
            node = Node()
            assert node.cache_ctrl is not None
            assert len(node.cache_ctrl._cache_stores) == 2
            assert isinstance(node.cache_ctrl._cache_stores[0], RamCacheStore)
            assert isinstance(node.cache_ctrl._cache_stores[1], DiskCacheStore)

        # specify
        node = Node(cache_ctrl=["ram"])
        assert node.cache_ctrl is not None
        assert len(node.cache_ctrl._cache_stores) == 1
        assert isinstance(node.cache_ctrl._cache_stores[0], RamCacheStore)

        node = Node(cache_ctrl=["ram", "disk"])
        assert node.cache_ctrl is not None
        assert len(node.cache_ctrl._cache_stores) == 2
        assert isinstance(node.cache_ctrl._cache_stores[0], RamCacheStore)
        assert isinstance(node.cache_ctrl._cache_stores[1], DiskCacheStore)

    def test_tagged_attr_readonly(self):
        class MyNode(Node):
            my_attr = tl.Any().tag(attr=True)

        with podpac.settings:
            podpac.settings["DEBUG"] = False
            node = MyNode()
            assert node.traits()["my_attr"].read_only

            podpac.settings["DEBUG"] = True
            node = MyNode()
            assert not node.traits()["my_attr"].read_only

    def test_trait_is_defined(self):
        node = Node()
        if tl.version_info[0] >= 5:
            assert not node.trait_is_defined("units")
        else:
            assert node.trait_is_defined("units")

    def test_init(self):
        class MyNode(Node):
            init_run = False

            def init(self):
                super(MyNode, self).init()
                self.init_run = True

        node = MyNode()
        assert node.init_run

    def test_attrs(self):
        class MyNode(Node):
            my_attr = tl.Any().tag(attr=True)
            my_trait = tl.Any()

        n = MyNode()
        assert "my_attr" in n.attrs
        assert "my_trait" not in n.attrs

    def test_repr(self):
        n = Node()
        repr(n)

        n = Node(outputs=["a", "b"])
        repr(n)
        assert "outputs=" in repr(n)
        assert "output=" not in repr(n)

        n = Node(outputs=["a", "b"], output="a")
        repr(n)
        assert "outputs=" not in repr(n)
        assert "output=" in repr(n)

    def test_str(self):
        n = Node()
        str(n)

        n = Node(outputs=["a", "b"])
        str(n)
        assert "outputs=" in str(n)
        assert "output=" not in str(n)

        n = Node(outputs=["a", "b"], output="a")
        str(n)
        assert "outputs=" not in str(n)
        assert "output=" in str(n)

    def test_eval_group(self):
        class MyNode(Node):
            def eval(self, coordinates, output=None, selector=None):
                return self.create_output_array(coordinates)

        c1 = podpac.Coordinates([[0, 1], [0, 1]], dims=["lat", "lon"])
        c2 = podpac.Coordinates([[10, 11], [10, 11, 12]], dims=["lat", "lon"])
        g = podpac.coordinates.GroupCoordinates([c1, c2])

        node = MyNode()
        outputs = node.eval_group(g)
        assert isinstance(outputs, list)
        assert len(outputs) == 2
        assert isinstance(outputs[0], UnitsDataArray)
        assert isinstance(outputs[1], UnitsDataArray)
        assert outputs[0].shape == (2, 2)
        assert outputs[1].shape == (2, 3)

        # invalid
        with pytest.raises(Exception):
            node.eval_group(c1)

        with pytest.raises(Exception):
            node.eval(g)

    def test_eval_not_implemented(self):
        node = Node()
        with pytest.raises(NotImplementedError):
            node.eval(podpac.Coordinates([]))

        with pytest.raises(NotImplementedError):
            node.eval(podpac.Coordinates([]), output=None)

    def test_find_coordinates_not_implemented(self):
        node = Node()
        with pytest.raises(NotImplementedError):
            node.find_coordinates()

    def test_get_bounds(self):
        class MyNode(Node):
            def find_coordinates(self):
                return [
                    podpac.Coordinates([[0, 1, 2], [0, 10, 20]], dims=["lat", "lon"], crs="EPSG:2193"),
                    podpac.Coordinates([[3, 4], [30, 40]], dims=["lat", "lon"], crs="EPSG:2193"),
                ]

        node = MyNode()

        with podpac.settings:
            podpac.settings["DEFAULT_CRS"] = "EPSG:4326"

            # specify crs
            bounds, crs = node.get_bounds(crs="EPSG:2193")
            assert bounds == {"lat": (0, 4), "lon": (0, 40)}
            assert crs == "EPSG:2193"

            # default crs
            bounds, crs = node.get_bounds()
            assert bounds == {
                "lat": (-75.81397534013118, -75.81362774074242),
                "lon": (82.92787904584206, 82.9280189659297),
            }
            assert crs == "EPSG:4326"


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

    @pytest.mark.xfail(reason="not yet supported.")
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


class TestNodeEval(object):
    def test_extract_output(self):
        coords = podpac.Coordinates([[0, 1, 2, 3], [0, 1]], dims=["lat", "lon"])

        class MyNode1(Node):
            outputs = ["a", "b", "c"]

            def _eval(self, coordinates, output=None, selector=None):
                return self.create_output_array(coordinates)

        # don't extract when no output field is requested
        node = MyNode1()
        out = node.eval(coords)
        assert out.shape == (4, 2, 3)

        # do extract when an output field is requested
        node = MyNode1(output="b")
        out = node.eval(coords)
        assert out.shape == (4, 2)

        # should still work if the node has already extracted it
        class MyNode2(Node):
            outputs = ["a", "b", "c"]

            def _eval(self, coordinates, output=None, selector=None):
                out = self.create_output_array(coordinates)
                return out.sel(output=self.output)

        node = MyNode2(output="b")
        out = node.eval(coords)
        assert out.shape == (4, 2)

    def test_evaluate_transpose(self):
        class MyNode(Node):
            def _eval(self, coordinates, output=None, selector=None):
                coords = coordinates.transpose("lat", "lon")
                data = np.arange(coords.size).reshape(coords.shape)
                a = self.create_output_array(coords, data=data)
                if output is None:
                    output = a
                else:
                    output[:] = a.transpose(*output.dims)
                return output

        coords = podpac.Coordinates([[0, 1, 2, 3], [0, 1]], dims=["lat", "lon"])

        node = MyNode()
        o1 = node.eval(coords)
        o2 = node.eval(coords.transpose("lon", "lat"))

        # returned output should match the requested coordinates and data should be transposed
        assert o1.dims == ("lat", "lon")
        assert o2.dims == ("lon", "lat")
        np.testing.assert_array_equal(o2.transpose("lat", "lon").data, o1.data)

        # with transposed output
        o3 = node.create_output_array(coords.transpose("lon", "lat"))
        o4 = node.eval(coords, output=o3)

        assert o3.dims == ("lon", "lat")  # stay the same
        assert o4.dims == ("lat", "lon")  # match requested coordinates
        np.testing.assert_equal(o3.transpose("lat", "lon").data, o4.data)

    def test_eval_get_cache(self):
        podpac.settings["RAM_CACHE_ENABLED"] = True

        class MyNode(Node):
            def _eval(self, coordinates, output=None, selector=None):
                coords = coordinates.transpose("lat", "lon")
                data = np.arange(coords.size).reshape(coords.shape)
                a = self.create_output_array(coords, data=data)
                if output is None:
                    output = a
                else:
                    output[:] = a.transpose(*output.dims)
                return output

        coords = podpac.Coordinates([[0, 1, 2, 3], [0, 1]], dims=["lat", "lon"])

        node = MyNode(cache_output=True, cache_ctrl=CacheCtrl([RamCacheStore()])).cache()

        # first eval
        o1 = node.eval(coords)
        assert node._from_cache == False

        # get from cache
        o2 = node.eval(coords)
        assert node._from_cache == True
        np.testing.assert_array_equal(o2, o1)

        # get from cache with output
        o3 = node.eval(coords, output=o1)
        assert node._from_cache == True
        np.testing.assert_array_equal(o3, o1)

        # get from cache with output transposed
        o4 = node.eval(coords, output=o1.transpose("lon", "lat"))
        assert node._from_cache == True
        np.testing.assert_array_equal(o4, o1)

        # get from cache with coords transposed
        o5 = node.eval(coords.transpose("lon", "lat"))
        assert node._from_cache == True
        np.testing.assert_array_equal(o5, o1.transpose("lon", "lat"))

    def test_eval_output_crs(self):
        coords = podpac.Coordinates([[0, 1, 2, 3], [0, 1]], dims=["lat", "lon"])

        node = Node()
        with pytest.raises(ValueError, match="Output coordinate reference system .* does not match"):
            node.eval(coords, output=node.create_output_array(coords.transform("EPSG:2193")))


class TestCaching(object):
    @classmethod
    def setup_class(cls):
        cls._ram_cache_enabled = podpac.settings["RAM_CACHE_ENABLED"]

        podpac.settings["RAM_CACHE_ENABLED"] = True

        class MyNode(Node):
            pass

        cls.node = MyNode(cache_ctrl=CacheCtrl([RamCacheStore()]))
        cls.node.rem_cache(key="*", coordinates="*")

        cls.coords = podpac.Coordinates([0, 0], dims=["lat", "lon"])
        cls.coords2 = podpac.Coordinates([1, 1], dims=["lat", "lon"])

    @classmethod
    def teardown_class(cls):
        cls.node.rem_cache(key="*", coordinates="*")

        podpac.settings["RAM_CACHE_ENABLED"] = cls._ram_cache_enabled

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
            self.node.put_cache(1, "test", overwrite=False)
        assert self.node.get_cache("test") == 0

        self.node.put_cache(1, "test")
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

    def test_put_has_expires(self):
        self.node.put_cache(10, "key1", expires="1,D")
        self.node.put_cache(10, "key2", expires="-1,D")
        assert self.node.has_cache("key1")
        assert not self.node.has_cache("key2")

    def test_put_get_expires(self):
        self.node.put_cache(10, "key1", expires="1,D")
        self.node.put_cache(10, "key2", expires="-1,D")
        assert self.node.get_cache("key1") == 10
        with pytest.raises(NodeException, match="cached data not found"):
            self.node.get_cache("key2")

    # node definition errors
    # this demonstrates both classes of error in the has_cache case, but only one for put/get/rem
    # we could test both classes for put/get/rem as well, but that is not really necessary
    def test_has_cache_unavailable_circular(self):
        class MyNode(Node):
            a = tl.Any().tag(attr=True)

            @tl.default("a")
            def _default_a(self):
                return self.b

            @property
            def b(self):
                self.has_cache("b")
                return 10

        node = MyNode(cache_ctrl=["ram"])
        with pytest.raises(NodeException, match="Cache unavailable, node definition has a circular dependency"):
            assert node.b == 10

    def test_has_cache_unavailable_uninitialized(self):
        class MyNode(Node):
            a = tl.Any().tag(attr=True)

            @tl.validate("a")
            def _validate_a(self, d):
                self.b
                return d["value"]

            @property
            def b(self):
                self.has_cache("key")
                return 10

        with pytest.raises(NodeException, match="Cache unavailable, node is not yet fully initialized"):
            node = MyNode(a=3, cache_ctrl=["ram"])

    def test_put_cache_unavailable_uninitialized(self):
        class MyNode(Node):
            a = tl.Any().tag(attr=True)

            @tl.validate("a")
            def _validate_a(self, d):
                self.b
                return d["value"]

            @property
            def b(self):
                self.put_cache(10, "key")
                return 10

        with pytest.raises(NodeException, match="Cache unavailable"):
            node = MyNode(a=3, cache_ctrl=["ram"])

    def test_get_cache_unavailable_uninitialized(self):
        class MyNode(Node):
            a = tl.Any().tag(attr=True)

            @tl.validate("a")
            def _validate_a(self, d):
                self.b
                return d["value"]

            @property
            def b(self):
                self.get_cache("key")
                return 10

        with pytest.raises(NodeException, match="Cache unavailable"):
            node = MyNode(a=3, cache_ctrl=["ram"])

    def test_rem_cache_unavailable_uninitialized(self):
        class MyNode(Node):
            a = tl.Any().tag(attr=True)

            @tl.validate("a")
            def _validate_a(self, d):
                self.b
                return d["value"]

            @property
            def b(self):
                self.rem_cache("key")
                return 10

        with pytest.raises(NodeException, match="Cache unavailable"):
            node = MyNode(a=3, cache_ctrl=["ram"])


class TestSerialization(object):
    @classmethod
    def setup_class(cls):
        a = podpac.algorithm.Arange()
        b = podpac.data.Array(source=[10, 20, 30], coordinates=podpac.Coordinates([[0, 1, 2]], dims=["lat"]))
        c = podpac.compositor.OrderedCompositor(sources=[a, b])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Insecure evaluation.*")
            cls.node = podpac.algorithm.Arithmetic(A=a, B=b, C=c, eqn="A + B + C")

    def test_base_ref(self):
        node = Node()
        assert isinstance(node.base_ref, six.string_types)

    def test_base_definition(self):
        node = Node()
        d = node._base_definition
        assert "node" in d
        assert isinstance(d["node"], six.string_types)

    def test_base_definition_attrs(self):
        class MyNode(Node):
            my_attr = tl.Int().tag(attr=True)

        node = MyNode(my_attr=7)

        d = node._base_definition
        assert d["attrs"]["my_attr"] == 7

    def test_base_definition_inputs(self):
        class MyNode(Node):
            my_attr = NodeTrait().tag(attr=True)

        a = Node()
        node = MyNode(my_attr=a)

        d = node._base_definition
        assert d["inputs"]["my_attr"] == a

    def test_base_definition_inputs_array(self):
        class MyNode(Node):
            my_attr = ArrayTrait().tag(attr=True)

        a = Node()
        b = Node()
        node = MyNode(my_attr=[a, b])

        d = node._base_definition
        assert d["inputs"]["my_attr"][0] == a
        assert d["inputs"]["my_attr"][1] == b

    def test_base_definition_inputs_dict(self):
        class MyNode(Node):
            my_attr = tl.Dict().tag(attr=True)

        a = Node()
        b = Node()
        node = MyNode(my_attr={"a": a, "b": b})

        d = node._base_definition
        assert d["inputs"]["my_attr"]["a"] == a
        assert d["inputs"]["my_attr"]["b"] == b

    def test_base_definition_style(self):
        node = Node(style=Style(name="test"))
        d = node._base_definition
        assert "style" in node._base_definition

    def test_base_definition_remove_unnecessary_attrs(self):
        node = Node(outputs=["a", "b"], output="a", units="m")
        d = node._base_definition
        assert "outputs" in d["attrs"]
        assert "output" in d["attrs"]
        assert "units" in d["attrs"]

        node = Node()
        d = node._base_definition
        if "attrs" in d:
            assert "outputs" not in d["attrs"]
            assert "output" not in d["attrs"]
            assert "units" not in d["attrs"]

    def test_definition(self):
        # definition
        d = self.node.definition
        assert isinstance(d, OrderedDict)
        assert len(d) == 5

        # from_definition
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Insecure evaluation.*")
            node = Node.from_definition(d)

        assert node is not self.node
        assert node == self.node
        assert isinstance(node, podpac.algorithm.Arithmetic)
        assert isinstance(node.inputs["A"], podpac.algorithm.Arange)
        assert isinstance(node.inputs["B"], podpac.data.Array)
        assert isinstance(node.inputs["C"], podpac.compositor.OrderedCompositor)

    def test_definition_duplicate_base_ref(self):
        n1 = Node(units="m")
        n2 = Node(units="ft")
        n3 = Node(units="in")
        node = podpac.compositor.OrderedCompositor(sources=[n1, n2, n3])
        d = node.definition
        assert n1.base_ref == n2.base_ref == n3.base_ref
        assert len(d) == 5

    def test_definition_inputs_array(self):
        global MyNodeWithArrayInput

        class MyNodeWithArrayInput(Node):
            my_array = ArrayTrait().tag(attr=True)

        node1 = MyNodeWithArrayInput(my_array=[podpac.algorithm.Arange()])
        node2 = Node.from_definition(node1.definition)
        assert node2 is not node1 and node2 == node1

    def test_definition_inputs_dict(self):
        global MyNodeWithDictInput

        class MyNodeWithDictInput(Node):
            my_dict = tl.Dict().tag(attr=True)

        node1 = MyNodeWithDictInput(my_dict={"a": podpac.algorithm.Arange()})
        node2 = Node.from_definition(node1.definition)
        assert node2 is not node1 and node2 == node1

    def test_definition_version(self):
        d = self.node.definition
        assert "podpac_version" in d
        assert d["podpac_version"] == podpac.__version__

    def test_json(self):
        # json
        s = self.node.json
        assert isinstance(s, six.string_types)
        assert json.loads(s)

        # test from_json
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Insecure evaluation.*")
            node = Node.from_json(s)
        assert node is not self.node
        assert node == self.node
        assert isinstance(node, podpac.algorithm.Arithmetic)
        assert isinstance(node.inputs["A"], podpac.algorithm.Arange)
        assert isinstance(node.inputs["B"], podpac.data.Array)
        assert isinstance(node.inputs["C"], podpac.compositor.OrderedCompositor)

    def test_file(self):
        path = tempfile.mkdtemp(prefix="podpac-test-")
        filename = os.path.join(path, "node.json")

        # save
        self.node.save(filename)
        assert os.path.exists(filename)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Insecure evaluation.*")
            node = Node.load(filename)

        assert node is not self.node
        assert node == self.node
        assert isinstance(node, podpac.algorithm.Arithmetic)
        assert isinstance(node.inputs["A"], podpac.algorithm.Arange)
        assert isinstance(node.inputs["B"], podpac.data.Array)
        assert isinstance(node.inputs["C"], podpac.compositor.OrderedCompositor)

    def test_json_pretty(self):
        node = Node()
        s = node.json_pretty
        assert isinstance(s, six.string_types)
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

    def test_hash_preserves_definition(self):
        n = Node()
        d_before = deepcopy(n.definition)
        h = n.hash
        d_after = deepcopy(n.definition)

        assert d_before == d_after

    def test_hash_omit_style(self):
        class N(Node):
            my_attr = tl.Int().tag(attr=True)

        n1 = N(my_attr=1, style=Style(name="a"))
        n2 = N(my_attr=1, style=Style(name="b"))

        # json has style in it
        assert n1.json != n2.json

        # but hash does not
        assert n1.hash == n2.hash

    def test_hash_omit_version(self):
        version = podpac.__version__

        try:
            # actual version
            n1 = Node()
            s1 = n1.json
            h1 = n1.hash

            # spoof different version
            podpac.__version__ = "other"
            n2 = Node()
            s2 = n2.json
            h2 = n2.hash

            # JSON should be different, but hash should be the same
            assert s1 != s2
            assert h1 == h2

        finally:
            # reset version
            podpac.__version__ = version

    def test_eq(self):
        class N(Node):
            my_attr = tl.Int().tag(attr=True)

        class M(Node):
            my_attr = tl.Int().tag(attr=True)

        n1 = N(my_attr=1)
        n2 = N(my_attr=1)
        n3 = N(my_attr=2)
        m1 = M(my_attr=1)

        # eq
        assert n1 == n2
        assert not n1 == n3
        assert not n1 == m1
        assert not n1 == "other"

        # ne
        assert not n1 != n2
        assert n1 != n3
        assert n1 != m1
        assert n1 != "other"

    def test_eq_ignore_style(self):
        class N(Node):
            my_attr = tl.Int().tag(attr=True)

        n1 = N(my_attr=1, style=Style(name="a"))
        n2 = N(my_attr=1, style=Style(name="b"))

        # json has style in it
        assert n1.json != n2.json

        # but == and != don't care
        assert n1 == n2
        assert not n1 != n2

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

    def test_from_url_with_plugin_style_params(self):
        url0 = (
            r"https://mobility-devel.crearecomputing.com/geowatch?&SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&"
            r"LAYERS=Arange&STYLES=&FORMAT=image%2Fpng&TRANSPARENT=true&HEIGHT=256&WIDTH=256"
            r"&CRS=EPSG%3A3857&BBOX=-20037508.342789244,10018754.171394618,-10018754.171394622,20037508.34278071&"
            r'PARAMS={"plugin": "podpac.algorithm"}'
        )
        url1 = (
            r"https://mobility-devel.crearecomputing.com/geowatch?&SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&"
            r"LAYERS=datalib.terraintiles.TerrainTiles&STYLES=&FORMAT=image%2Fpng&TRANSPARENT=true&HEIGHT=256&WIDTH=256&"
            r"TIME=2021-03-01T12%3A00%3A00.000Z&CRS=EPSG%3A3857&BBOX=-10018754.171394622,5009377.08569731,-9392582.035682458,5635549.221409475"
            r'&PARAMS={"style": {"name": "Aspect (Composited 30-90 m)","units": "radians","colormap": "hsv","clim": [0,6.283185307179586]}}'
        )
        node = Node.from_url(url0)
        node = Node.from_url(url1)

    def test_from_name_params(self):
        # Normal
        name = "algorithm.Arange"
        node = Node.from_name_params(name)

        # Normal with params
        name = "algorithm.CoordData"
        params = {"coord_name": "alt"}
        node = Node.from_name_params(name, params)
        assert node.coord_name == "alt"

        # Plugin style
        name = "CoordData"
        params = {"plugin": "podpac.algorithm", "attrs": {"coord_name": "alt"}}
        node = Node.from_name_params(name, params)
        assert node.coord_name == "alt"

    def test_style(self):
        node = podpac.data.Array(
            source=[10, 20, 30],
            coordinates=podpac.Coordinates([[0, 1, 2]], dims=["lat"]),
            style=Style(name="test", units="m"),
        )

        d = node.definition
        assert "style" in d[node.base_ref]

        node2 = Node.from_definition(d)
        assert node2 is not node
        assert isinstance(node2, podpac.data.Array)
        assert node2.style is not node.style
        assert node2.style == node.style
        assert node2.style.name == "test"
        assert node2.style.units == "m"

        # default style
        node = podpac.data.Array(source=[10, 20, 30], coordinates=podpac.Coordinates([[0, 1, 2]], dims=["lat"]))
        d = node.definition
        assert "style" not in d[node.base_ref]

    def test_circular_definition(self):
        # this is admittedly a contrived example in order to demonstrate the most direct case
        class MyNode(Node):
            a = tl.Any().tag(attr=True)

            @tl.default("a")
            def _default_a(self):
                self.definition
                return 10

        node = MyNode()
        with pytest.raises(NodeDefinitionError, match="node definition has a circular dependency"):
            node.a


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

    def test_inputs(self):
        # invalid type
        s = """
        {
            "a": {
                "node": "algorithm.Min",
                "inputs": { "source": 10 }
            }
        }
        """

        with pytest.raises(ValueError, match="Invalid definition for node"):
            Node.from_json(s)

        # nonexistent node
        s = """
        {
            "a": {
                "node": "algorithm.Min",
                "inputs": { "source": "nonexistent" }
            }
        }
        """

        with pytest.raises(ValueError, match="Invalid definition for node"):
            Node.from_json(s)

    def test_lookup_attrs(self):
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

        # invalid type
        s = """
        {
            "a": {
                "node": "algorithm.CoordData",
                "attrs": { "coord_name": "lat" }
            },
            "b": {
                "node": "algorithm.CoordData",
                "lookup_attrs": { "coord_name": 10 }
            }
        }
        """

        with pytest.raises(ValueError, match="Invalid definition for node"):
            Node.from_json(s)

        # nonexistent node
        s = """
        {
            "a": {
                "node": "algorithm.CoordData",
                "attrs": { "coord_name": "lat" }
            },
            "b": {
                "node": "algorithm.CoordData",
                "lookup_attrs": { "coord_name": "nonexistent.coord_name" }
            }
        }
        """

        with pytest.raises(ValueError, match="Invalid definition for node"):
            Node.from_json(s)

        # nonexistent subattr
        s = """
        {
            "a": {
                "node": "algorithm.CoordData",
                "attrs": { "coord_name": "lat" }
            },
            "b": {
                "node": "algorithm.CoordData",
                "lookup_attrs": { "coord_name": "a.nonexistent" }
            }
        }
        """

        with pytest.raises(ValueError, match="Invalid definition for node"):
            Node.from_json(s)

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

    def test_debuggable(self):
        s = """
        {
            "a": {
                "node": "algorithm.Arange"
            },
            "mean": {
                "node": "algorithm.Convolution",
                "lookup_attrs": {"source": "a"},
                "attrs": {"kernel_type": "mean,3", "kernel_dims": ["lat", "lon"]}
            },
            "c": {
                "node": "algorithm.Arithmetic",
                "lookup_attrs": {"A": "a", "B": "mean"},
                "attrs": {"eqn": "a-b"}
            }
        }
        """

        with warnings.catch_warnings(), podpac.settings:
            warnings.filterwarnings("ignore", "Insecure evaluation.*")

            # normally node objects can and should be re-used
            podpac.settings["DEBUG"] = False
            node = Node.from_json(s)
            assert node.inputs["A"] is node.inputs["B"].source

            # when debugging is on, node objects should be unique
            podpac.settings["DEBUG"] = True
            node = Node.from_json(s)
            assert node.inputs["A"] is not node.inputs["B"].source

    def test_from_definition_version_warning(self):
        s = """
        {
            "a": {
                "node": "algorithm.Arange"
            },
            "podpac_version": "other"
        }
        """

        with pytest.warns(UserWarning, match="node definition version mismatch"):
            node = Node.from_json(s)

    def test_from_proper_json(self):
        not_ordered_json = """
        {
            "Arithmetic": {
                "node": "core.algorithm.generic.Arithmetic",
                "attrs": {
                    "eqn": "a+b",
                    "params": {

                    }
                },
                "inputs": {
                    "a": "SinCoords",
                    "b": "Arange"
                }
            },
            "SinCoords": {
                "node": "core.algorithm.utility.SinCoords",
                "style": {
                    "colormap": "jet",
                    "clim": [
                        -1.0,
                        1.0
                    ]
                }
            },
            "Arange": {
                "node": "core.algorithm.utility.Arange"
            },
            "podpac_version": "3.2.0"
        }
        """
        not_ordered_json_2 = """
        {
            "SinCoords": {
                "node": "core.algorithm.utility.SinCoords",
                "style": {
                    "colormap": "jet",
                    "clim": [
                        -1.0,
                        1.0
                    ]
                }
            },
            "Arithmetic": {
                "node": "core.algorithm.generic.Arithmetic",
                "attrs": {
                    "eqn": "a+b",
                    "params": {

                    }
                },
                "inputs": {
                    "a": "SinCoords",
                    "b": "Arange"
                }
            },
            "Arange": {
                "node": "core.algorithm.utility.Arange"
            },
            "podpac_version": "3.2.0"
        }
        """
        ordered_json = """
        {
            "SinCoords": {
                "node": "core.algorithm.utility.SinCoords",
                "style": {
                    "colormap": "jet",
                    "clim": [
                        -1.0,
                        1.0
                    ]
                }
            },
            "Arange": {
                "node": "core.algorithm.utility.Arange"
            },
             "Arithmetic": {
                "node": "core.algorithm.generic.Arithmetic",
                "attrs": {
                    "eqn": "a+b",
                    "params": {

                    }
                },
                "inputs": {
                    "a": "SinCoords",
                    "b": "Arange"
                }
            },
            "podpac_version": "3.2.0"
        }
        """
        # Check that the order doesn't matter. Because .from_json returns the output node, also checks correct output_node is returned
        not_ordered_pipe = Node.from_json(not_ordered_json)
        not_ordered_pipe_2 = Node.from_json(not_ordered_json_2)
        ordered_pipe = Node.from_json(ordered_json)
        assert not_ordered_pipe.definition == ordered_pipe.definition == not_ordered_pipe_2.definition
        assert not_ordered_pipe.hash == ordered_pipe.hash

        # Check that incomplete json will throw ValueError:
        incomplete_json = """
        {
            "Arange": {
                "node": "core.algorithm.utility.Arange"
            },
             "Arithmetic": {
                "node": "core.algorithm.generic.Arithmetic",
                "attrs": {
                    "eqn": "a+b",
                    "params": {

                    }
                },
                "inputs": {
                    "a": "SinCoords",
                    "b": "Arange"
                }
            },
            "podpac_version": "3.2.0"
        }
        """
        with pytest.raises(ValueError):
            Node.from_json(incomplete_json)

    def test_output_node(self):
        included_json = """
        {
            "Arithmetic": {
                "node": "core.algorithm.generic.Arithmetic",
                "attrs": {
                    "eqn": "a+b",
                    "params": {

                    }
                },
                "inputs": {
                    "a": "SinCoords",
                    "b": "Arange"
                }
            },
            "SinCoords": {
                "node": "core.algorithm.utility.SinCoords",
                "style": {
                    "colormap": "jet",
                    "clim": [
                        -1.0,
                        1.0
                    ]
                }
            },
            "Arange": {
                "node": "core.algorithm.utility.Arange"
            },
            "podpac_version": "3.2.0",
            "podpac_output_node": "Arithmetic"
        }
        """
        ordered_json = """
        {
            "SinCoords": {
                "node": "core.algorithm.utility.SinCoords",
                "style": {
                    "colormap": "jet",
                    "clim": [
                        -1.0,
                        1.0
                    ]
                }
            },
            "Arange": {
                "node": "core.algorithm.utility.Arange"
            },
             "Arithmetic": {
                "node": "core.algorithm.generic.Arithmetic",
                "attrs": {
                    "eqn": "a+b",
                    "params": {

                    }
                },
                "inputs": {
                    "a": "SinCoords",
                    "b": "Arange"
                }
            },
            "podpac_version": "3.2.0"
        }
        """
        included_pipe = Node.from_json(included_json)
        ordered_pipe = Node.from_json(ordered_json)
        assert included_pipe.definition == ordered_pipe.definition
        assert included_pipe.hash == ordered_pipe.hash

        wrong_name_json = """
        {
            "SinCoords": {
                "node": "core.algorithm.utility.SinCoords",
                "style": {
                    "colormap": "jet",
                    "clim": [
                        -1.0,
                        1.0
                    ]
                }
            },
            "Arange": {
                "node": "core.algorithm.utility.Arange"
            },
             "Arithmetic": {
                "node": "core.algorithm.generic.Arithmetic",
                "attrs": {
                    "eqn": "a+b",
                    "params": {

                    }
                },
                "inputs": {
                    "a": "SinCoords",
                    "b": "Arange"
                }
            },
            "podpac_version": "3.2.0",
            "podpac_output_node": "Sum"
        }
        """
        with pytest.raises(ValueError):
            Node.from_json(wrong_name_json)


class TestNoCacheMixin(object):
    class NoCacheNode(NoCacheMixin, Node):
        pass

    def test_default_no_cache(self):
        with podpac.settings:
            podpac.settings["DEFAULT_CACHE"] = ["ram"]
            node = self.NoCacheNode()
            assert len(node.cache_ctrl._cache_stores) == 0

    def test_customizable(self):
        podpac.settings["DEFAULT_CACHE"] = ["ram"]
        node = self.NoCacheNode(cache_ctrl=["ram"])
        assert len(node.cache_ctrl._cache_stores) == 1


class TestDiskCacheMixin(object):
    class DiskCacheNode(DiskCacheMixin, Node):
        pass

    def test_default_disk_cache(self):
        with podpac.settings:
            # add disk cache
            podpac.settings["DEFAULT_CACHE"] = ["ram"]
            node = self.DiskCacheNode()
            assert len(node.cache_ctrl._cache_stores) == 2

            # don't add if it is already there
            podpac.settings["DEFAULT_CACHE"] = ["ram", "disk"]
            node = self.DiskCacheNode()
            assert len(node.cache_ctrl._cache_stores) == 2

    def test_customizable(self):
        node = self.DiskCacheNode(cache_ctrl=["ram"])
        assert len(node.cache_ctrl._cache_stores) == 1


# TODO: remove this - this is currently a placeholder test until we actually have integration tests (pytest will exit with code 5 if no tests found)
@pytest.mark.integration
def tests_node_integration():
    assert True
