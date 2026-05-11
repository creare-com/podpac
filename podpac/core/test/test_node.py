from __future__ import division, unicode_literals, print_function, absolute_import

import os
import json
import warnings
import tempfile
from collections import OrderedDict
from copy import deepcopy

import six
import pytest
import numpy as np
from pint.errors import UndefinedUnitError
from pint import UnitRegistry

ureg = UnitRegistry()
import traitlets as tl

import podpac
from podpac.core.utils import ArrayTrait, NodeTrait
from podpac.core.units import UnitsDataArray
from podpac.core.style import Style
from podpac.core.cache import RamCacheStore, DiskCacheStore
from podpac.core.node import Node, NodeException, NodeDefinitionError

_OUTPUTS = "outputs="
_OUTPUT = "output="
_CACHE_UNAVAIL = "Cache unavailable"
_INSECURE_EVAL = "Insecure evaluation.*"
_INVALID_DEF_FOR_NODE = "Invalid definition for node"


class TestNode(object):
    def test_style(self):
        node = Node()
        assert isinstance(node.style, Style)

    def test_units(self):
        Node(units="meters")

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
            Node(outputs=["a", "b"], output="other")

        # only valid for multiple-output nodes
        with pytest.raises(TypeError, match="Invalid output"):
            Node(output="other")

    def test_cache_output(self):
        with podpac.settings:
            podpac.settings["ENABLE_CACHE"] = False
            node = Node()
            assert not node.cache_output

            podpac.settings["ENABLE_CACHE"] = True
            node = Node()
            assert node.cache_output

    def test_cache_ctrl(self):
        # settings
        with podpac.settings:
            podpac.settings["DEFAULT_CACHE"] = ["ram"]
            node = Node().cache()
            assert node.cache_ctrl is not None
            assert len(node.cache_ctrl._cache_stores) == 1
            assert isinstance(node.cache_ctrl._cache_stores[0], RamCacheStore)

            podpac.settings["DEFAULT_CACHE"] = ["ram", "disk"]
            node = Node().cache()
            assert node.cache_ctrl is not None
            assert len(node.cache_ctrl._cache_stores) == 2
            assert isinstance(node.cache_ctrl._cache_stores[0], RamCacheStore)
            assert isinstance(node.cache_ctrl._cache_stores[1], DiskCacheStore)

        # specify
        node = Node().cache(cache_type="ram")
        assert node.cache_ctrl is not None
        assert len(node.cache_ctrl._cache_stores) == 1
        assert isinstance(node.cache_ctrl._cache_stores[0], RamCacheStore)

        node = Node().cache(cache_type=["ram", "disk"])
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

    @pytest.mark.skip("Traitlets behavior changes based on version.")
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
        _ = repr(n)

        n = Node(outputs=["a", "b"])
        _ = repr(n)
        assert _OUTPUTS in repr(n)
        assert _OUTPUT not in repr(n)

        n = Node(outputs=["a", "b"], output="a")
        _ = repr(n)
        assert _OUTPUTS not in repr(n)
        assert _OUTPUT in repr(n)

    def test_str(self):
        n = Node()
        _ = str(n)

        n = Node(outputs=["a", "b"])
        _ = str(n)
        assert _OUTPUTS in str(n)
        assert _OUTPUT not in str(n)

        n = Node(outputs=["a", "b"], output="a")
        _ = str(n)
        assert _OUTPUTS not in str(n)
        assert _OUTPUT in str(n)

    def test_eval_group(self):
        class MyNode(Node):
            def eval(self, coordinates, output=None, selector=None):  # noqa: A003
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
        with pytest.raises(AttributeError):
            node.eval_group(c1)

        with pytest.raises(AttributeError):
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
        podpac.settings["ENABLE_CACHE"] = True

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

        node = MyNode(cache_output=True).cache(cache_type="ram")

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
        cls._ram_cache_enabled = podpac.settings["ENABLE_CACHE"]

        podpac.settings["ENABLE_CACHE"] = True

        class MyNode(Node):
            pass

        cls.node = MyNode().cache(cache_type=["ram"])
        cls.node.rem_cache(key="*", coordinates="*")

        cls.coords = podpac.Coordinates([0, 0], dims=["lat", "lon"])
        cls.coords2 = podpac.Coordinates([1, 1], dims=["lat", "lon"])

    @classmethod
    def teardown_class(cls):
        cls.node.rem_cache(key="*", coordinates="*")

        podpac.settings["ENABLE_CACHE"] = cls._ram_cache_enabled

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
                self.has_property_cache("b")
                return 10

        node = MyNode(cache_ctrl=["ram"]).cache()
        with pytest.raises(NodeException, match="Cache unavailable, node definition has a circular dependency"):
            _ = node.source.b

    def test_has_cache_unavailable_uninitialized(self):
        class MyNode(Node):
            a = tl.Any().tag(attr=True)

            @tl.validate("a")
            def _validate_a(self, d):
                self.b
                return d["value"]

            @property
            def b(self):
                self.has_property_cache("key")
                return 10

        with pytest.raises(NodeException, match="Cache unavailable, node is not yet fully initialized"):
            MyNode(a=3, cache_ctrl=["ram"]).cache()

    def test_put_cache_unavailable_uninitialized(self):
        class MyNode(Node):
            a = tl.Any().tag(attr=True)

            @tl.validate("a")
            def _validate_a(self, d):
                self.b
                return d["value"]

            @property
            def b(self):
                self.put_property_cache(10, "key")  # no longer a relevant test?
                return 10

        with pytest.raises(NodeException, match=_CACHE_UNAVAIL):
            MyNode(a=3, cache_ctrl=["ram"])

    def test_get_cache_unavailable_uninitialized(self):
        class MyNode(Node):
            a = tl.Any().tag(attr=True)

            @tl.validate("a")
            def _validate_a(self, d):
                self.b
                return d["value"]

            @property
            def b(self):
                self.get_property_cache("key")
                return 10

        with pytest.raises(NodeException, match=_CACHE_UNAVAIL):
            MyNode(a=3, cache_ctrl=["ram"])

    def test_rem_cache_unavailable_uninitialized(self):
        class MyNode(Node):
            a = tl.Any().tag(attr=True)

            @tl.validate("a")
            def _validate_a(self, d):
                self.b
                return d["value"]

            @property
            def b(self):
                self.rem_property_cache("key")
                return 10

        with pytest.raises(NodeException, match=_CACHE_UNAVAIL):
            MyNode(a=3, cache_ctrl=["ram"])


class TestSerialization(object):
    @classmethod
    def setup_class(cls):
        a = podpac.algorithm.Arange()
        b = podpac.data.Array(source=[10, 20, 30], coordinates=podpac.Coordinates([[0, 1, 2]], dims=["lat"]))
        c = podpac.compositor.OrderedCompositor(sources=[a, b])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", _INSECURE_EVAL)
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
        node._base_definition
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
            warnings.filterwarnings("ignore", _INSECURE_EVAL)
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
            warnings.filterwarnings("ignore", _INSECURE_EVAL)
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
            warnings.filterwarnings("ignore", _INSECURE_EVAL)
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
        _ = n.hash
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
        assert not n1.__eq__(n3)
        assert not n1.__eq__(m1)
        assert not n1.__eq__("other")

        # ne
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
        assert not n1.__ne__(n2)

    def test_from_url(self):
        url = (
            r"https://testwms/?map=map&&service={service}&request=GetMap&{layername}={layer}&styles=&format=image%2Fpng"
            r"&transparent=true&version=1.1.1&transparency=true&width=256&height=256&srs=EPSG%3A4326"
            r"&bbox=40,-71,41,70&time=2018-05-19&PARAMS={params}"
        )

        params = ["{}", '{"a":{"node":"algorithm.Arange"}}', "{}", "{}"]

        for service, layername in zip(["WMS", "WCS"], ["LAYERS", "COVERAGE"]):
            for layer, param in zip(
                [
                    "algorithm.SinCoords",
                    "%PARAMS%",
                ],
                params,
            ):
                Node.from_url(url.format(service=service, layername=layername, layer=layer, params=param))

    def test_from_url_with_plugin_style_params(self):
        url0 = (
            r"https://mobility-devel.crearecomputing.com/geowatch?&SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&"
            r"LAYERS=Arange&STYLES=&FORMAT=image%2Fpng&TRANSPARENT=true&HEIGHT=256&WIDTH=256"
            r"&CRS=EPSG%3A3857&BBOX=-20037508.342789244,10018754.171394618,-10018754.171394622,20037508.34278071&"
            r'PARAMS={"plugin": "podpac.algorithm"}'
        )

        Node.from_url(url0)

    def test_from_name_params(self):
        # Normal
        name = "algorithm.Arange"
        Node.from_name_params(name)

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
                self.definition()
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

        with pytest.raises(ValueError, match=_INVALID_DEF_FOR_NODE):
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

        with pytest.raises(ValueError, match=_INVALID_DEF_FOR_NODE):
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

        with pytest.raises(ValueError, match=_INVALID_DEF_FOR_NODE):
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

        with pytest.raises(ValueError, match=_INVALID_DEF_FOR_NODE):
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

        with pytest.raises(ValueError, match=_INVALID_DEF_FOR_NODE):
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
            warnings.filterwarnings("ignore", _INSECURE_EVAL)

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
            Node.from_json(s)

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


class TestPropertyCacheCtrlDefault:
    def test_property_cache_ctrl_with_type(self):
        node = Node(property_cache_type="ram")
        assert node.property_cache_ctrl is not None

    def test_property_cache_ctrl_with_list_type(self):
        node = Node(property_cache_type=["ram"])
        assert node.property_cache_ctrl is not None


class TestEvalExtended:
    def test_eval_debug_stores_coordinates_and_output(self):
        class MyNode(Node):
            def _eval(self, coordinates, output=None, _selector=None):
                return self.create_output_array(coordinates)

        coords = podpac.Coordinates([[0, 1, 2]], dims=["lat"])
        with podpac.settings:
            podpac.settings["DEBUG"] = True
            node = MyNode()
            node.eval(coords)
            assert node._requested_coordinates is coords
            assert node._output is not None

    def test_eval_adds_units_to_output(self):
        class MyNode(Node):
            def _eval(self, coordinates, output=None, _selector=None):
                return self.create_output_array(coordinates)

        coords = podpac.Coordinates([[0, 1, 2]], dims=["lat"])
        node = MyNode(units="meters")
        out = node.eval(coords)
        assert "units" in out.attrs

    def test_eval_adds_crs_when_missing_from_data(self):
        from podpac.core.units import UnitsDataArray

        class MyNode(Node):
            def _eval(self, coordinates, output=None, _selector=None):
                arr = self.create_output_array(coordinates)
                new_attrs = {k: v for k, v in arr.attrs.items() if k != "crs"}
                return UnitsDataArray(arr.data, coords=arr.coords, dims=arr.dims, attrs=new_attrs)

        coords = podpac.Coordinates([[0, 1, 2]], dims=["lat"])
        node = MyNode()
        out = node.eval(coords)
        assert "crs" in out.attrs
        assert out.attrs["crs"] == coords.crs


class TestCreateOutputArrayExtended:
    def test_create_output_array_with_existing_attrs(self):
        c = podpac.Coordinates([[0, 1, 2]], dims=["lat"])
        node = Node()
        custom_style = Style(name="custom")
        attrs = {"layer_style": custom_style, "crs": "EPSG:4326"}
        out = node.create_output_array(c, attrs=attrs)
        assert out.attrs["layer_style"] is custom_style
        assert out.attrs["crs"] == "EPSG:4326"

    def test_create_output_array_empty_outputs_treated_as_none(self):
        c = podpac.Coordinates([[0, 1, 2]], dims=["lat"])
        node = Node()
        out = node.create_output_array(c, outputs=[])
        assert "output" not in out.dims

    def test_create_output_array_explicit_outputs(self):
        c = podpac.Coordinates([[0, 1, 2]], dims=["lat"])
        node = Node()
        out = node.create_output_array(c, outputs=["a", "b"])
        assert "output" in out.dims

    def test_create_output_array_with_units_attr(self):
        c = podpac.Coordinates([[0, 1, 2]], dims=["lat"])
        node = Node(units="meters")
        out = node.create_output_array(c)
        assert "units" in out.attrs


class TestNodeConvenienceMethods:
    def test_probe(self):
        node = podpac.algorithm.Arange()
        result = node.probe(lat=0, lon=0)
        assert isinstance(result, dict)

    def test_interpolate(self):
        node = podpac.algorithm.Arange()
        result = node.interpolate()
        assert isinstance(result, podpac.interpolators.Interpolate)
        assert result.source is node

    def test_interpolate_custom_method(self):
        node = podpac.algorithm.Arange()
        result = node.interpolate(interpolation="bilinear")
        assert result.interpolation == "bilinear"

    def test_cache_with_uid(self):
        node = Node()
        cache_node = node.cache(uid="test_uid", cache_type="ram")
        assert cache_node.cache_uid == "test_uid"

    def test_cache_zarr_raises_without_coordinates(self):
        node = Node()
        with pytest.raises(ValueError, match="Cannot use ZarrCache without coordinates"):
            node.cache(node_type="zarr")

    def test_cache_zarr_with_coordinates(self):
        node = podpac.data.Array(source=[1, 2, 3], coordinates=podpac.Coordinates([[0, 1, 2]], dims=["lat"]))
        result = node.cache(node_type="zarr", cache_type="ram")
        assert isinstance(result, podpac.caches.ZarrCache)

    def test_cache_invalid_node_type(self):
        node = Node()
        with pytest.raises(ValueError, match="Invalid cache type"):
            node.cache(node_type="invalid_type")

    def test_cache_with_cache_ctrl_list(self):
        node = Node()
        cache_node = node.cache(cache_type="ram", cache_ctrl=["ram"])
        assert cache_node is not None


class TestPropertyCacheMethods:
    def test_get_property_cache_not_found(self):
        node = Node(property_cache_type="ram")
        with pytest.raises(NodeException, match="cached data not found"):
            node.get_property_cache("missing_key")

    def test_get_put_property_cache(self):
        node = Node(property_cache_type="ram")
        node.put_property_cache("my_data", "key1")
        assert node.get_property_cache("key1") == "my_data"

    def test_has_property_cache_none_ctrl(self):
        node = Node()
        node.property_cache_ctrl = None
        assert not node.has_property_cache("key")

    def test_has_property_cache(self):
        node = Node(property_cache_type="ram")
        assert not node.has_property_cache("key")
        node.put_property_cache("data", "key")
        assert node.has_property_cache("key")

    def test_put_property_cache_none_ctrl(self):
        node = Node()
        node.property_cache_ctrl = None
        node.put_property_cache("data", "key")  # should return without error

    def test_put_property_cache_no_overwrite_raises(self):
        node = Node(property_cache_type="ram")
        node.put_property_cache("data1", "key")
        with pytest.raises(NodeException, match="Cached data already exists"):
            node.put_property_cache("data2", "key", overwrite=False)

    def test_rem_property_cache_none_ctrl(self):
        node = Node()
        node.property_cache_ctrl = None
        node.rem_property_cache("key")  # should return without error

    def test_rem_property_cache(self):
        node = Node(property_cache_type="ram")
        node.put_property_cache("data", "key")
        assert node.has_property_cache("key")
        node.rem_property_cache("key")
        assert not node.has_property_cache("key")


class TestSerializationExtended:
    def test_base_definition_custom_style_subclass(self):
        class CustomStyle(Style):
            pass

        node = Node(style=CustomStyle())
        d = node._base_definition
        assert "style_class" in d

    def test_hash_omit_style_class(self):
        class CustomStyle(Style):
            pass

        n1 = Node(style=CustomStyle())
        n2 = Node()
        assert n1.hash == n2.hash

    def test_from_definition_with_invalid_style_class_module(self):
        s = """
        {
            "a": {
                "node": "algorithm.Arange",
                "style": {"name": "test"},
                "style_class": "nonexistent_module_xyz.Style"
            }
        }
        """
        with pytest.raises(ValueError, match="Invalid definition for style module"):
            Node.from_json(s)

    def test_from_definition_with_invalid_style_class_name(self):
        s = """
        {
            "a": {
                "node": "algorithm.Arange",
                "style": {"name": "test"},
                "style_class": "podpac.core.style.NonExistentStyleClass"
            }
        }
        """
        with pytest.raises(ValueError, match="style class.*not found in style module"):
            Node.from_json(s)

    def test_from_definition_with_valid_style_class(self):
        s = """
        {
            "a": {
                "node": "algorithm.Arange",
                "style": {"name": "test"},
                "style_class": "podpac.core.style.Style"
            }
        }
        """
        node = Node.from_json(s)
        assert node.style.name == "test"


class TestFromUrlExtended:
    def test_from_url_no_params(self):
        url = (
            "https://test/?SERVICE=WMS&REQUEST=GetMap&LAYERS=algorithm.Arange"
            "&WIDTH=256&HEIGHT=256&BBOX=40,-71,41,70&SRS=EPSG:4326"
        )
        node = Node.from_url(url)
        assert isinstance(node, podpac.algorithm.Arange)

    def test_from_url_with_dict_url_and_dict_params(self):
        url = {"SERVICE": "WMS", "LAYERS": "algorithm.Arange", "PARAMS": {}}
        node = Node.from_url(url)
        assert isinstance(node, podpac.algorithm.Arange)


class TestGetUISpec:
    def test_get_ui_spec_basic(self):
        spec = Node.get_ui_spec()
        assert "help" in spec
        assert "module" in spec
        assert "attrs" in spec
        assert "style" in spec

    def test_get_ui_spec_no_docstring(self):
        class MyUndocumentedNode(Node):
            pass

        MyUndocumentedNode.__doc__ = None
        spec = MyUndocumentedNode.get_ui_spec()
        assert spec["help"] == "No help text to display."

    def test_get_ui_spec_union_trait(self):
        class MyNode(Node):
            """Test node"""

            my_union = tl.Union([tl.Int(), tl.Unicode()], default_value=0).tag(attr=True)

        spec = MyNode.get_ui_spec()
        assert "my_union" in spec["attrs"]
        assert isinstance(spec["attrs"]["my_union"]["type"], list)

    def test_get_ui_spec_instance_node_trait(self):
        class MyNode(Node):
            """Test node"""

            my_input = tl.Instance(Node, allow_none=True).tag(attr=True)

        spec = MyNode.get_ui_spec()
        assert spec["attrs"]["my_input"]["type"] == "NodeTrait"

    def test_get_ui_spec_instance_non_node_trait(self):
        from podpac.core.coordinates import Coordinates

        class MyNode(Node):
            """Test node"""

            my_coords = tl.Instance(Coordinates, allow_none=True).tag(attr=True)

        spec = MyNode.get_ui_spec()
        assert spec["attrs"]["my_coords"]["type"] == "Coordinates"

    def test_get_ui_spec_dict_trait(self):
        class MyNode(Node):
            """Test node"""

            my_dict = tl.Dict().tag(attr=True)

        spec = MyNode.get_ui_spec()
        assert spec["attrs"]["my_dict"]["type"] == "Dict"

    def test_get_ui_spec_nan_default(self):
        class MyNode(Node):
            """Test node"""

            my_float = tl.Float(np.nan).tag(attr=True)

        spec = MyNode.get_ui_spec()
        assert spec["attrs"]["my_float"]["default"] == "nan"

    def test_get_ui_spec_with_function_default(self):
        class MyNode(Node):
            """Test node"""

            my_val = tl.Unicode().tag(attr=True)

            @tl.default("my_val")
            def _default_my_val(self):
                return "computed_default"

        spec = MyNode.get_ui_spec()
        assert spec["attrs"]["my_val"]["default"] == "computed_default"

    def test_get_ui_spec_arange(self):
        spec = podpac.algorithm.Arange.get_ui_spec()
        assert isinstance(spec, dict)
        assert "help" in spec


class TestLookupFunctions:
    def test_lookup_attr_list_value(self):
        from podpac.core.node import _lookup_attr

        nodes = OrderedDict()
        nodes["a"] = podpac.algorithm.CoordData(coord_name="lat")
        result = _lookup_attr(nodes, "b", ["a.coord_name"])
        assert result == ["lat"]

    def test_lookup_attr_dict_value(self):
        from podpac.core.node import _lookup_attr

        nodes = OrderedDict()
        nodes["a"] = podpac.algorithm.CoordData(coord_name="lat")
        result = _lookup_attr(nodes, "b", {"key": "a.coord_name"})
        assert result == {"key": "lat"}

    def test_lookup_attr_debug_deepcopy(self):
        from podpac.core.node import _lookup_attr

        nodes = OrderedDict()
        node_a = podpac.algorithm.Arange()
        nodes["a"] = node_a

        with podpac.settings:
            podpac.settings["DEBUG"] = True
            result = _lookup_attr(nodes, "b", "a")
            assert result == node_a
            assert result is not node_a

    def test_lookup_input_debug_deepcopy(self):
        from podpac.core.node import _lookup_input

        arange = podpac.algorithm.Arange()
        nodes = OrderedDict()
        nodes["Arange"] = arange

        with podpac.settings:
            podpac.settings["DEBUG"] = True
            result = _lookup_input(nodes, "test", "Arange", {})
            assert result == arange
            assert result is not arange


@pytest.mark.integration
def tests_node_integration():
    # This is currently a placeholder test until we actually have integration tests (pytest will exit with code 5 if no tests found)
    pass
