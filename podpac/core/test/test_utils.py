from __future__ import division, unicode_literals, print_function, absolute_import

import os
import sys
import json
import datetime
import warnings
from collections import OrderedDict

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import traitlets as tl

import podpac
from podpac.core.utils import common_doc
from podpac.core.utils import trait_is_defined, trait_is_default
from podpac.core.utils import create_logfile
from podpac.core.utils import OrderedDictTrait, ArrayTrait, TupleTrait, NodeTrait
from podpac.core.utils import JSONEncoder, is_json_serializable
from podpac.core.utils import cache_func, cached_property


class TestCommonDocs(object):
    def test_common_docs_does_not_affect_anonymous_functions(self):
        f = lambda x: x
        f2 = common_doc({"key": "value"})(f)
        assert f(42) == f2(42)
        assert f.__doc__ is None


class TestTraitletsHelpers(object):
    def test_trait_is_defined(self):
        class MyClass(tl.HasTraits):
            a = tl.Any()
            b = tl.Any(default_value=0)
            c = tl.Any()

            @tl.default("c")
            def _default_b(self):
                return "test"

        x = MyClass(a=1, b=1, c=1)
        assert trait_is_defined(x, "a")
        assert trait_is_defined(x, "b")
        assert trait_is_defined(x, "c")
        assert not trait_is_defined(x, "other")

        x = MyClass()
        assert trait_is_defined(x, "a")
        assert trait_is_defined(x, "b")
        assert not trait_is_defined(x, "c")

        x.c
        assert trait_is_defined(x, "c")

    def test_trait_is_default(self):
        class MyClass(tl.HasTraits):
            a = tl.Any(default_value=0)

        x = MyClass()
        assert trait_is_default(x, "a")

        x = MyClass(a=0)
        assert trait_is_default(x, "a")

        x = MyClass(a=1)
        assert not trait_is_default(x, "a")

    def test_trait_is_default_array(self):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", module="traitlets", category=DeprecationWarning, message="elementwise comparison failed"
            )

            class MyClass(tl.HasTraits):
                a = tl.Any(default_value=0)

            x = MyClass(a=np.array([0, 1]))
            assert not trait_is_default(x, "a")

            class MyClass(tl.HasTraits):
                a = tl.Any(default_value=np.array([0, 1]))

            x = MyClass()
            assert trait_is_default(x, "a")

            x = MyClass(a=np.array([0, 1]))
            assert trait_is_default(x, "a")

            x = MyClass(a=np.array([0, 1, 2]))
            assert not trait_is_default(x, "a")

            x = MyClass(a=0)
            assert not trait_is_default(x, "a")

    def test_trait_is_default_no_default_value(self):
        class MyClass(tl.HasTraits):
            a = tl.List()

        x = MyClass()
        assert trait_is_default(x, "a") is None


class TestLoggingHelpers(object):
    def test_create_logfile(self):
        create_logfile()


class TestOrderedDictTrait(object):
    def test(self):
        class MyClass(tl.HasTraits):
            d = OrderedDictTrait()

        m = MyClass(d=OrderedDict([("a", 1)]))

        with pytest.raises(tl.TraitError):
            MyClass(d=[])

    @pytest.mark.skipif(sys.version < "3.6", reason="python < 3.6")
    def test_dict_python36(self):
        class MyClass(tl.HasTraits):
            d = OrderedDictTrait()

        m = MyClass(d={"a": 1})

    @pytest.mark.skipif(sys.version >= "3.6", reason="python >= 3.6")
    def test_dict_python2(self):
        class MyClass(tl.HasTraits):
            d = OrderedDictTrait()

        with pytest.raises(tl.TraitError):
            m = MyClass(d={"a": 1})

        # empty is okay, will be converted
        m = MyClass(d={})


class TestArrayTrait(object):
    def test(self):
        class MyClass(tl.HasTraits):
            a = ArrayTrait()

        # basic usage
        o = MyClass(a=np.array([0, 4]))
        assert isinstance(o.a, np.ndarray)
        np.testing.assert_equal(o.a, [0, 4])

        # coerce
        o = MyClass(a=[0, 4])
        assert isinstance(o.a, np.ndarray)
        np.testing.assert_equal(o.a, [0, 4])

    def test_ndim(self):
        class MyClass(tl.HasTraits):
            a = ArrayTrait(ndim=2)

        MyClass(a=np.array([[0, 4]]))
        MyClass(a=[[0, 4]])

        # invalid
        with pytest.raises(tl.TraitError):
            MyClass(a=[4, 5])

    def test_shape(self):
        class MyClass(tl.HasTraits):
            a = ArrayTrait(shape=(2, 2))

        MyClass(a=np.array([[0, 1], [2, 3]]))
        MyClass(a=[[0, 1], [2, 3]])

        # invalid
        with pytest.raises(tl.TraitError):
            MyClass(a=np.array([0, 1, 2, 3]))

    def test_dtype(self):
        class MyClass(tl.HasTraits):
            a = ArrayTrait(dtype=float)

        m = MyClass(a=np.array([0.0, 1.0]))
        assert m.a.dtype == float

        m = MyClass(a=[0.0, 1.0])
        assert m.a.dtype == float

        # astype
        m = MyClass(a=[0, 1])
        assert m.a.dtype == float

        # invalid
        with pytest.raises(tl.TraitError):
            MyClass(a=np.array(["a", "b"]))

    def test_args(self):
        # shape and ndim must match
        t = ArrayTrait(ndim=2, shape=(2, 2))

        with pytest.raises(ValueError):
            ArrayTrait(ndim=1, shape=(2, 2))

        # dtype lookup
        t = ArrayTrait(dtype="datetime64")
        assert t.dtype == np.datetime64

        # invalid dtype
        with pytest.raises(ValueError):
            ArrayTrait(dtype="notatype")


class TestNodeTrait(object):
    def test(self):
        class MyClass(tl.HasTraits):
            node = NodeTrait()

        t = MyClass(node=podpac.Node())

        with pytest.raises(tl.TraitError):
            MyClass(node=0)

    def test_debug(self):
        class MyClass(tl.HasTraits):
            node = NodeTrait()

        node = podpac.Node()

        with podpac.settings:
            podpac.settings["DEBUG"] = False
            t = MyClass(node=node)
            assert t.node is node

            podpac.settings["DEBUG"] = True
            t = MyClass(node=node)
            assert t.node is not node


@pytest.mark.skip("TODO")
class TestTupleTrait(object):
    def test_trait(self):
        class MyClass(tl.HasTraits):
            t = TupleTrait(trait=int)

        MyClass(t=(1, 2, 3))

        with pytest.raises(TypeError):
            MyClass(t=("a", "b", "c"))


class TestJSONEncoder(object):
    def test_coordinates(self):
        coordinates = podpac.coordinates.Coordinates([0], dims=["time"])
        json.dumps(coordinates, cls=JSONEncoder)

    def test_node(self):
        node = podpac.Node()
        json.dumps(node, cls=JSONEncoder)

    def test_style(self):
        style = podpac.core.style.Style()
        json.dumps(style, cls=JSONEncoder)

    def test_interpolation(self):
        interpolation = podpac.data.Interpolation()
        json.dumps(interpolation, cls=JSONEncoder)

    def test_interpolator(self):
        kls = podpac.data.INTERPOLATORS[0]
        json.dumps(kls, cls=JSONEncoder)

    def test_units(self):
        units = podpac.core.units.ureg.Unit("meters")
        json.dumps(units, cls=JSONEncoder)

    def test_datetime64(self):
        dt = np.datetime64()
        json.dumps(dt, cls=JSONEncoder)

    def test_timedelta64(self):
        td = np.timedelta64()
        json.dumps(td, cls=JSONEncoder)

    def test_datetime(self):
        now = datetime.datetime.now()
        json.dumps(now, cls=JSONEncoder)

    def test_date(self):
        today = datetime.date.today()
        json.dumps(today, cls=JSONEncoder)

    def test_dataframe(self):
        df = pd.DataFrame()
        json.dumps(df, cls=JSONEncoder)

    def test_array_datetime64(self):
        a = np.array(["2018-01-01", "2018-01-02"]).astype(np.datetime64)
        json.dumps(a, cls=JSONEncoder)

    def test_array_timedelta64(self):
        a = np.array([np.timedelta64(1, "D"), np.timedelta64(1, "D")])
        json.dumps(a, cls=JSONEncoder)

    def test_array_numerical(self):
        a = np.array([0.0, 1.0, 2.0])
        json.dumps(a, cls=JSONEncoder)

    def test_array_node(self):
        a = np.array([podpac.Node(), podpac.Node()])
        json.dumps(a, cls=JSONEncoder)

    def test_array_unserializable(self):
        class MyClass(object):
            pass

        a = np.array([MyClass()])
        with pytest.raises(TypeError, match="Cannot serialize numpy array"):
            json.dumps(a, cls=JSONEncoder)

    def test_unserializable(self):
        value = xr.DataArray([])
        with pytest.raises(TypeError, match="Object of type DataArray is not JSON serializable"):
            json.dumps(value, cls=JSONEncoder)

    def test_is_json_serializable(self):
        assert is_json_serializable("test")
        assert not is_json_serializable(xr.DataArray([]))


def test_cache_func():
    class Test(podpac.Node):
        a = tl.Int(1).tag(attr=True)
        b = tl.Int(1).tag(attr=True)
        c = tl.Int(1)
        d = tl.Int(1)

        @cache_func("a2", "a")
        def a2(self):
            """a2 docstring"""
            return self.a * 2

        @cache_func("b2")
        def b2(self):
            """ b2 docstring """
            return self.b * 2

        @cache_func("c2", "c")
        def c2(self):
            """ c2 docstring """
            return self.c * 2

        @cache_func("d2")
        def d2(self):
            """ d2 docstring """
            return self.d * 2

    t = Test(cache_ctrl=podpac.core.cache.CacheCtrl([podpac.core.cache.RamCacheStore()]))
    t2 = Test(cache_ctrl=podpac.core.cache.CacheCtrl([podpac.core.cache.RamCacheStore()]))
    t.rem_cache(key="*", coordinates="*")
    t2.rem_cache(key="*", coordinates="*")

    with pytest.raises(podpac.NodeException):
        t.get_cache("a2")

    assert t.a2() == 2
    assert t.b2() == 2
    assert t.c2() == 2
    assert t.d2() == 2
    assert t2.a2() == 2
    assert t2.b2() == 2
    assert t2.c2() == 2
    assert t2.d2() == 2

    t.set_trait("a", 2)
    assert t.a2() == 4
    t.set_trait("b", 2)
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

    t2.set_trait("a", 2)
    assert t2.get_cache("a2") == 4  # This was cached by t
    t2.set_trait("b", 2)
    assert t2.get_cache("c2") == 4  # This was cached by t
    assert t2.get_cache("d2") == 2  # This was cached by t


def test_cache_func_with_no_cache():
    class Test(podpac.Node):
        a = tl.Int(1).tag(attr=True)
        b = tl.Int(1).tag(attr=True)
        c = tl.Int(1)
        d = tl.Int(1)

        @cache_func("a2", "a")
        def a2(self):
            """a2 docstring"""
            return self.a * 2

        @cache_func("b2")
        def b2(self):
            """ b2 docstring """
            return self.b * 2

        @cache_func("c2", "c")
        def c2(self):
            """ c2 docstring """
            return self.c * 2

        @cache_func("d2")
        def d2(self):
            """ d2 docstring """
            return self.d * 2

    t = Test(cache_ctrl=None)
    t2 = Test(cache_ctrl=None)
    t.rem_cache(key="*", coordinates="*")
    t2.rem_cache(key="*", coordinates="*")

    with pytest.raises(podpac.NodeException):
        t.get_cache("a2")
        raise Exception("Cache should be cleared")

    assert t.a2() == 2
    assert t.b2() == 2
    assert t.c2() == 2
    assert t.d2() == 2
    assert t2.a2() == 2
    assert t2.b2() == 2
    assert t2.c2() == 2
    assert t2.d2() == 2

    t.set_trait("a", 2)
    assert t.a2() == 4
    t.set_trait("b", 2)
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


def test_cached_property():
    class MyNode(podpac.Node):
        my_property_called = 0
        my_cached_property_called = 0
        my_ram_cached_property_called = 0

        @property
        def my_property(self):
            self.my_property_called += 1
            return 10

        @cached_property
        def my_cached_property(self):
            self.my_cached_property_called += 1
            return 20

        @cached_property(use_cache_ctrl=True)
        def my_ram_cached_property(self):
            self.my_ram_cached_property_called += 1
            return 30

    with podpac.settings:
        podpac.utils.clear_cache()

        a = MyNode(cache_ctrl=["ram"])
        b = MyNode(cache_ctrl=["ram"])
        c = MyNode(cache_ctrl=[])

        # normal property should be called every time
        assert a.my_property_called == 0
        assert a.my_property == 10
        assert a.my_property_called == 1
        assert a.my_property == 10
        assert a.my_property == 10
        assert a.my_property_called == 3

        assert b.my_property_called == 0
        assert b.my_property == 10
        assert b.my_property_called == 1
        assert b.my_property == 10
        assert b.my_property == 10
        assert b.my_property_called == 3

        assert c.my_property_called == 0
        assert c.my_property == 10
        assert c.my_property_called == 1
        assert c.my_property == 10
        assert c.my_property == 10
        assert c.my_property_called == 3

        # cached property should only be called when it is accessed
        assert a.my_cached_property_called == 0
        assert a.my_cached_property == 20
        assert a.my_cached_property_called == 1
        assert a.my_cached_property == 20
        assert a.my_cached_property == 20
        assert a.my_cached_property_called == 1

        assert b.my_cached_property_called == 0
        assert b.my_cached_property == 20
        assert b.my_cached_property_called == 1
        assert b.my_cached_property == 20
        assert b.my_cached_property == 20
        assert b.my_cached_property_called == 1

        assert c.my_cached_property_called == 0
        assert c.my_cached_property == 20
        assert c.my_cached_property_called == 1
        assert c.my_cached_property == 20
        assert c.my_cached_property == 20
        assert c.my_cached_property_called == 1

        # cache_ctrl cached property should only be called in the first node that accessses it
        assert a.my_ram_cached_property_called == 0
        assert a.my_ram_cached_property == 30
        assert a.my_ram_cached_property_called == 1
        assert a.my_ram_cached_property == 30
        assert a.my_ram_cached_property == 30
        assert a.my_ram_cached_property_called == 1

        assert b.my_ram_cached_property_called == 0
        assert b.my_ram_cached_property == 30
        assert b.my_ram_cached_property_called == 0
        assert b.my_ram_cached_property == 30
        assert b.my_ram_cached_property == 30
        assert b.my_ram_cached_property_called == 0

        # but only if a cache_ctrl exists for the Node
        assert c.my_ram_cached_property_called == 0
        assert c.my_ram_cached_property == 30
        assert c.my_ram_cached_property_called == 1
        assert c.my_ram_cached_property == 30
        assert c.my_ram_cached_property == 30
        assert c.my_ram_cached_property_called == 1


def test_cached_property_invalid_argument():
    with pytest.raises(TypeError, match="cached_property decorator does not accept keyword argument"):
        cached_property(other=True)

    with pytest.raises(TypeError, match="cached_property decorator does not accept any positional arguments"):

        class MyNode(podpac.Node):
            @cached_property(True)
            def my_property(self):
                return 10
