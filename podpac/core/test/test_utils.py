from __future__ import division, unicode_literals, print_function, absolute_import
from io import StringIO

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
from podpac.core.utils import trait_is_defined
from podpac.core.utils import create_logfile
from podpac.core.utils import OrderedDictTrait, ArrayTrait, TupleTrait, NodeTrait
from podpac.core.utils import JSONEncoder, is_json_serializable
from podpac.core.utils import cached_property
from podpac.core.utils import ind2slice
from podpac.core.utils import probe_node
from podpac.core.utils import align_xarray_dict


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
        if tl.version_info[0] >= 5:
            assert not trait_is_defined(x, "a")
            assert not trait_is_defined(x, "b")
            assert not trait_is_defined(x, "c")
        else:
            assert trait_is_defined(x, "a")
            assert trait_is_defined(x, "b")
            assert not trait_is_defined(x, "c")

        x.c
        assert trait_is_defined(x, "c")


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


class TestTupleTrait(object):
    def test_trait(self):
        class MyClass(tl.HasTraits):
            t = TupleTrait(trait=tl.Int())

        MyClass(t=(1, 2, 3))

        with pytest.raises(tl.TraitError):
            MyClass(t=("a", "b", "c"))

    def test_tuple(self):
        class MyClass(tl.HasTraits):
            t = TupleTrait(trait=tl.Int())

        a = MyClass(t=(1, 2, 3))
        assert isinstance(a.t, tuple)

        a = MyClass(t=[1, 2, 3])
        assert isinstance(a.t, tuple)


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
        interpolation = podpac.core.interpolation.interpolation.Interpolate()
        json.dumps(interpolation, cls=JSONEncoder)

    def test_interpolator(self):
        kls = podpac.core.interpolation.INTERPOLATORS[0]
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
        with pytest.raises(TypeError, match="not JSON serializable"):
            json.dumps(value, cls=JSONEncoder)

    def test_is_json_serializable(self):
        assert is_json_serializable("test")
        assert not is_json_serializable(xr.DataArray([]))


class TestCachedPropertyDecorator(object):
    def test_cached_property(self):
        class MyNode(podpac.Node):
            my_property_called = 0
            my_cached_property_called = 0
            my_cache_ctrl_property_called = 0

            @property
            def my_property(self):
                self.my_property_called += 1
                return 10

            @cached_property
            def my_cached_property(self):
                self.my_cached_property_called += 1
                return 20

            @cached_property(use_cache_ctrl=True)
            def my_cache_ctrl_property(self):
                self.my_cache_ctrl_property_called += 1
                return 30

        a = MyNode(property_cache_type="ram").cache(
            node_type="hash", cache_type=["ram"]
        )  # when caching properties, include the "property_cache_type" when instantiating nodes. Worth documenting.
        b = MyNode(property_cache_type="ram").cache(node_type="hash", cache_type=["ram"])
        c = MyNode().cache(node_type="hash", cache_type=None)

        a.rem_cache(key="*")
        b.rem_cache(key="*")
        c.rem_cache(key="*")

        # normal property should be called every time
        assert a.source.my_property_called == 0
        assert a.source.my_property == 10
        assert a.source.my_property_called == 1
        assert a.source.my_property == 10
        assert a.source.my_property == 10
        assert a.source.my_property_called == 3

        assert b.source.my_property_called == 0
        assert b.source.my_property == 10
        assert b.source.my_property_called == 1
        assert b.source.my_property == 10
        assert b.source.my_property == 10
        assert b.source.my_property_called == 3

        assert c.source.my_property_called == 0
        assert c.source.my_property == 10
        assert c.source.my_property_called == 1
        assert c.source.my_property == 10
        assert c.source.my_property == 10
        assert c.source.my_property_called == 3

        # cached property should only be called when it is accessed
        assert a.source.my_cached_property_called == 0
        assert a.source.my_cached_property == 20
        assert a.source.my_cached_property_called == 1
        assert a.source.my_cached_property == 20
        assert a.source.my_cached_property == 20
        assert a.source.my_cached_property_called == 1

        assert b.source.my_cached_property_called == 0
        assert b.source.my_cached_property == 20
        assert b.source.my_cached_property_called == 1
        assert b.source.my_cached_property == 20
        assert b.source.my_cached_property == 20
        assert b.source.my_cached_property_called == 1

        assert c.source.my_cached_property_called == 0
        assert c.source.my_cached_property == 20
        assert c.source.my_cached_property_called == 1
        assert c.source.my_cached_property == 20
        assert c.source.my_cached_property == 20
        assert c.source.my_cached_property_called == 1

        # cache_ctrl cached property should only be called in the first node that accessses it
        assert a.source.my_cache_ctrl_property_called == 0
        assert a.source.my_cache_ctrl_property == 30
        assert a.source.my_cache_ctrl_property_called == 1
        assert a.source.my_cache_ctrl_property == 30
        assert a.source.my_cache_ctrl_property == 30
        assert a.source.my_cache_ctrl_property_called == 1

        assert b.source.my_cache_ctrl_property_called == 0
        assert b.source.my_cache_ctrl_property == 30
        assert b.source.my_cache_ctrl_property_called == 0
        assert b.source.my_cache_ctrl_property == 30
        assert b.source.my_cache_ctrl_property == 30
        assert b.source.my_cache_ctrl_property_called == 0

        # but only if a cache_ctrl exists for the Node
        assert c.source.my_cache_ctrl_property_called == 0
        assert c.source.my_cache_ctrl_property == 30
        assert c.source.my_cache_ctrl_property_called == 1
        assert c.source.my_cache_ctrl_property == 30
        assert c.source.my_cache_ctrl_property == 30
        assert c.source.my_cache_ctrl_property_called == 1

    def test_cached_property_expires(self):
        class MyNode(podpac.Node):
            expires_tomorrow_called = 0
            expired_yesterday_called = 0

            @cached_property(use_cache_ctrl=True, expires="1,D")
            def expires_tomorrow(self):
                self.expires_tomorrow_called += 1
                return 10

            @cached_property(use_cache_ctrl=True, expires="-1,D")
            def expired_yesterday(self):
                self.expired_yesterday_called += 1
                return 20

        a = MyNode(property_cache_type=["ram"])
        b = MyNode(property_cache_type=["ram"])

        # not expired, b uses cached version
        assert a.expires_tomorrow_called == 0
        assert b.expires_tomorrow_called == 0

        assert a.expires_tomorrow == 10
        assert a.expires_tomorrow == 10
        assert b.expires_tomorrow == 10
        assert b.expires_tomorrow == 10

        assert a.expires_tomorrow_called == 1
        assert b.expires_tomorrow_called == 0  # cache was used!

        # expired, b can't use cached version
        assert a.expired_yesterday_called == 0
        assert b.expired_yesterday_called == 0

        assert a.expired_yesterday == 20
        assert a.expired_yesterday == 20
        assert b.expired_yesterday == 20
        assert b.expired_yesterday == 20

        assert a.expired_yesterday_called == 1  # note the expiration only applies to fetching from the cache
        assert b.expired_yesterday_called == 1  # cache was not used!

    def test_invalid_argument(self):
        with pytest.raises(TypeError, match="cached_property decorator does not accept keyword argument"):
            cached_property(other=True)

        with pytest.raises(TypeError, match="cached_property decorator does not accept any positional arguments"):
            cached_property(True)


class TestInd2Slice(object):
    def test_slice(self):
        assert ind2slice((slice(1, 4),)) == (slice(1, 4),)

    def test_integer(self):
        assert ind2slice((1,)) == (1,)

    def test_integer_array(self):
        assert ind2slice(([1, 2, 4],)) == (slice(1, 5),)

    def test_boolean_array(self):
        assert ind2slice(([False, True, True, False, True, False],)) == (slice(1, 5),)

    def test_stepped(self):
        assert ind2slice(([1, 3, 5],)) == (slice(1, 7, 2),)
        assert ind2slice(([False, True, False, True, False, True],)) == (slice(1, 7, 2),)

    def test_multiindex(self):
        I = (slice(1, 4), 1, [1, 2, 4], [False, True, False], [1, 3, 5])
        assert ind2slice(I) == (slice(1, 4), 1, slice(1, 5), 1, slice(1, 7, 2))

    def test_nontuple(self):
        assert ind2slice(slice(1, 4)) == slice(1, 4)
        assert ind2slice(1) == 1
        assert ind2slice([1, 2, 4]) == slice(1, 5)
        assert ind2slice([False, True, True, False, True, False]) == slice(1, 5)
        assert ind2slice([1, 3, 5]) == slice(1, 7, 2)


class AnotherOne(podpac.algorithm.Algorithm):
    def algorithm(self, inputs, coordinates):
        return self.create_output_array(coordinates, data=1)


class TestNodeProber(object):
    coords = podpac.Coordinates([podpac.clinspace(0, 2, 3, "lat"), podpac.clinspace(0, 2, 3, "lon")])
    one = podpac.data.Array(
        source=np.ones((3, 3)), coordinates=coords, style=podpac.style.Style(name="one_style", units="o")
    )
    two = podpac.data.Array(
        source=np.ones((3, 3)) * 2, coordinates=coords, style=podpac.style.Style(name="two_style", units="t")
    )
    arange = podpac.algorithm.Arange()
    nan = podpac.data.Array(source=np.ones((3, 3)) * np.nan, coordinates=coords)
    another_one = AnotherOne()

    def test_single_prober(self):
        expected = {
            "Array": {
                "active": True,
                "value": 1,
                "units": "o",
                "inputs": [],
                "name": "one_style",
                "node_hash": self.one.hash,
            }
        }
        out = probe_node(self.one, lat=1, lon=1)
        assert out == expected

    def test_serial_prober(self):
        with podpac.settings:
            podpac.settings.set_unsafe_eval(True)
            a = podpac.algorithm.Arithmetic(one=self.one, eqn="one * 2")
            b = podpac.algorithm.Arithmetic(a=a, eqn="a*3", style=podpac.style.Style(name="six_style", units="m"))
            expected = {
                "Array": {
                    "active": True,
                    "value": 1.0,
                    "units": "o",
                    "inputs": [],
                    "name": "one_style",
                    "node_hash": self.one.hash,
                },
                "Arithmetic": {
                    "active": True,
                    "value": 2.0,
                    "units": "",
                    "inputs": ["Array"],
                    "name": "Arithmetic",
                    "node_hash": a.hash,
                },
                "Arithmetic_1": {
                    "active": True,
                    "value": 6.0,
                    "units": "m",
                    "inputs": ["Arithmetic"],
                    "name": "six_style",
                    "node_hash": b.hash,
                },
            }
            out = probe_node(b, lat=1, lon=1)
            assert out == expected

    def test_parallel_prober(self):
        with podpac.settings:
            podpac.settings.set_unsafe_eval(True)
            a = podpac.algorithm.Arithmetic(one=self.one, two=self.two, eqn="one * two")
            expected = {
                "Array": {
                    "active": True,
                    "value": 1.0,
                    "units": "o",
                    "inputs": [],
                    "name": "one_style",
                    "node_hash": self.one.hash,
                },
                "Array_1": {
                    "active": True,
                    "value": 2.0,
                    "units": "t",
                    "inputs": [],
                    "name": "two_style",
                    "node_hash": self.two.hash,
                },
                "Arithmetic": {
                    "active": True,
                    "value": 2.0,
                    "units": "",
                    "inputs": ["Array", "Array_1"],
                    "name": "Arithmetic",
                    "node_hash": a.hash,
                },
            }
            out = probe_node(a, lat=1, lon=1)
            assert out == expected

    def test_composited_prober(self):
        a = podpac.compositor.OrderedCompositor(sources=[self.one, self.arange])
        expected = {
            "Array": {
                "active": True,
                "value": 1.0,
                "units": "o",
                "inputs": [],
                "name": "one_style",
                "node_hash": self.one.hash,
            },
            "Arange": {
                "active": False,
                "value": 0.0,
                "units": "",
                "inputs": [],
                "name": "Arange",
                "node_hash": self.arange.hash,
            },
            "OrderedCompositor": {
                "active": True,
                "value": 1.0,
                "units": "",
                "inputs": ["Array", "Arange"],
                "name": "OrderedCompositor",
                "node_hash": a.hash,
            },
        }
        out = probe_node(a, lat=1, lon=1)
        assert out == expected

        a = podpac.compositor.OrderedCompositor(sources=[self.nan, self.two])
        expected = {
            "Array": {
                "active": False,
                "value": "nan",
                "units": "",
                "inputs": [],
                "name": "Array",
                "node_hash": self.nan.hash,
            },
            "Array_1": {
                "active": True,
                "value": 2.0,
                "units": "t",
                "inputs": [],
                "name": "two_style",
                "node_hash": self.two.hash,
            },
            "OrderedCompositor": {
                "active": True,
                "value": 2.0,
                "units": "",
                "inputs": ["Array", "Array_1"],
                "name": "OrderedCompositor",
                "node_hash": a.hash,
            },
        }
        out = probe_node(a, lat=1, lon=1)
        for k in out:
            if np.isnan(out[k]["value"]):
                out[k]["value"] = "nan"
        assert out == expected

        a = podpac.compositor.OrderedCompositor(sources=[self.nan, self.one, self.another_one])
        expected = {
            "Array": {
                "active": False,
                "value": "nan",
                "units": "",
                "inputs": [],
                "name": "Array",
                "node_hash": self.nan.hash,
            },
            "Array_1": {
                "active": True,
                "value": 1.0,
                "units": "o",
                "inputs": [],
                "name": "one_style",
                "node_hash": self.one.hash,
            },
            "AnotherOne": {
                "active": False,
                "value": 1.0,
                "units": "",
                "inputs": [],
                "name": "AnotherOne",
                "node_hash": self.another_one.hash,
            },
            "OrderedCompositor": {
                "active": True,
                "value": 1.0,
                "units": "",
                "inputs": ["Array", "Array_1", "AnotherOne"],
                "name": "OrderedCompositor",
                "node_hash": a.hash,
            },
        }
        out = probe_node(a, lat=1, lon=1)
        for k in out:
            if np.isnan(out[k]["value"]):
                out[k]["value"] = "nan"
        assert out == expected

    def test_composited_prober_nested(self):
        a = podpac.compositor.OrderedCompositor(
            sources=[self.one, self.arange], style=podpac.style.Style(name="composited", units="c")
        )
        expected = {
            "name": "composited",
            "value": "1.0 c",
            "active": True,
            "node_id": a.hash,
            "params": {},
            "inputs": {
                "inputs": [
                    {
                        "name": "one_style",
                        "value": "1.0 o",
                        "active": True,
                        "node_id": self.one.hash,
                        "params": {},
                        "inputs": {},
                    },
                    {
                        "name": "Arange",
                        "value": "0.0",
                        "active": False,
                        "node_id": self.arange.hash,
                        "params": {},
                        "inputs": {},
                    },
                ]
            },
        }
        out = probe_node(a, lat=1, lon=1, nested=True)
        assert out == expected

    def test_prober_with_enumerated_legends(self):
        enumeration_style = podpac.style.Style(
            name="composited",
            units="my_units",
            enumeration_legend={0: "dirt", 1: "sand"},
            enumeration_colors={0: (0, 0, 0), 1: (0.5, 0.5, 0.5)},
        )
        nan = podpac.data.Array(source=np.ones((3, 3), int) * np.nan, coordinates=self.coords, style=enumeration_style)
        one = podpac.data.Array(source=np.ones((3, 3), int), coordinates=self.coords, style=enumeration_style)
        zero = podpac.data.Array(source=np.zeros((3, 3), int), coordinates=self.coords, style=enumeration_style)
        a = podpac.compositor.OrderedCompositor(sources=[nan, one, zero], style=enumeration_style)

        expected = {
            "name": "composited",
            "value": "1 (sand) my_units",
            "active": True,
            "node_id": a.hash,
            "params": {},
            "inputs": {
                "inputs": [
                    {
                        "name": "composited",
                        "value": "nan (unknown) my_units",
                        "active": False,
                        "node_id": nan.hash,
                        "params": {},
                        "inputs": {},
                    },
                    {
                        "name": "composited",
                        "value": "1 (sand) my_units",
                        "active": True,
                        "node_id": one.hash,
                        "params": {},
                        "inputs": {},
                    },
                    {
                        "name": "composited",
                        "value": "0 (dirt) my_units",
                        "active": False,
                        "node_id": zero.hash,
                        "params": {},
                        "inputs": {},
                    },
                ]
            },
        }
        out = probe_node(a, lat=1, lon=1, nested=True, add_enumeration_labels=True)
        assert out == expected

def test_align_xarray_dict():
    data_a = np.random.random((20,15))
    a = podpac.UnitsDataArray(
        np.array(data_a),
        coords={"lat": np.linspace(0,5,20), "lon": np.linspace(10,20,15)},
        dims=["lat", "lon"],
    )

    data_b = np.random.random((20,15))
    b = podpac.UnitsDataArray(
        np.array(data_b),
        coords={"lat": np.linspace(0.2,5.2,20), "lon": np.linspace(10.1,20.1,15)},
        dims=["lat", "lon"],
    )

    data_c = np.random.random((20,15))
    c = podpac.UnitsDataArray(
        np.array(data_c),
        coords={"lat": np.linspace(-44,18,20), "lon": np.linspace(-101,-90,15)},
        dims=["lat", "lon"],
    )
    
    inputs = {'A':a,
              'B':b,
              'C':c}
    

    inputs = align_xarray_dict(inputs)

    for k in ['lat','lon']:
        assert(np.all(inputs['A'][k]==inputs['B'][k]))
        assert(np.all(inputs['A'][k]==inputs['C'][k]))
    assert(np.all(inputs['A'].data==data_a))
    assert(np.all(inputs['B'].data==data_b))
    assert(np.all(inputs['C'].data==data_c))
