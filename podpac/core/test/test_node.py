from __future__ import division, unicode_literals, print_function, absolute_import

import os
from collections import OrderedDict
import json
import six

import pytest
import numpy as np
import xarray as xr
from pint.errors import DimensionalityError
from pint import UnitRegistry; ureg = UnitRegistry()
import traitlets as tl

import podpac
from podpac.core import common_test_utils as ctu
from podpac.core.units import UnitsDataArray
from podpac.core.node import Node, NodeException
    
class TestNode(object):
    def test_base_ref(self):
        n = Node()
        assert isinstance(n.base_ref, str)

    def test_base_definition(self):
        class N(Node):
            my_attr = tl.Int().tag(attr=True)
            my_node_attr = tl.Instance(Node).tag(attr=True)
        
        a = Node()
        n = N(my_attr=7, my_node_attr=a)

        d = n.base_definition
        assert isinstance(d, OrderedDict)
        assert 'node' in d
        assert isinstance(d['node'], str)
        assert 'attrs' in d
        assert isinstance(d['attrs'], OrderedDict)
        assert 'my_attr' in d['attrs']
        assert d['attrs']['my_attr'] == 7
        assert isinstance(d['lookup_attrs'], OrderedDict)
        assert 'my_node_attr' in d['lookup_attrs']
        assert d['lookup_attrs']['my_node_attr'] is a

    def test_base_definition_array_attr(self):
        class N(Node):
            my_attr = tl.Instance(np.ndarray).tag(attr=True)

        node = N(my_attr=np.ones((2, 3, 4)))
        d = node.base_definition
        my_attr = np.array(d['attrs']['my_attr'])
        np.testing.assert_array_equal(my_attr, node.my_attr)

    def test_base_definition_coordinates_attr(self):
        class N(Node):
            my_attr = tl.Instance(podpac.Coordinates).tag(attr=True)

        node = N(my_attr=podpac.Coordinates([[0, 1], [1, 2, 3]], dims=['lat', 'lon']))
        d = node.base_definition
        my_attr = podpac.Coordinates.from_definition(d['attrs']['my_attr'])
        
        # TODO this shouldn't raise an exception an more once __eq__ is merged in
        with pytest.raises(AssertionError):
            assert my_attr == node.my_attr

    def test_base_definition_unserializable(self):
        class N(Node):
            my_attr = tl.Instance(xr.DataArray).tag(attr=True)

        node = N(my_attr=xr.DataArray([0, 1]))
        with pytest.raises(NodeException, match="Cannot serialize attr 'my_attr'"):
            node.base_definition

    def test_definition(self):
        n = Node()
        d = n.definition
        assert isinstance(d, OrderedDict)
        assert list(d.keys()) == ['nodes']

    def test_make_pipeline_definition(self):
        a = podpac.algorithm.Arange()
        b = podpac.algorithm.CoordData()
        c = podpac.compositor.OrderedCompositor(sources=np.array([a, b]))
        
        node = podpac.algorithm.Arithmetic(A=a, B=b, C=c, eqn="A + B + C")
        definition = node.definition

        # make sure it is a valid pipeline
        pipeline = podpac.pipeline.Pipeline(definition=definition)

        assert isinstance(pipeline.node.A, podpac.algorithm.Arange)
        assert isinstance(pipeline.node.B, podpac.algorithm.CoordData)
        assert isinstance(pipeline.node.C, podpac.compositor.OrderedCompositor)
        assert isinstance(pipeline.node, podpac.algorithm.Arithmetic)

        assert isinstance(pipeline.output, podpac.pipeline.NoOutput)
        assert pipeline.output.name == node.base_ref

    def test_make_pipeline_definition_duplicate_ref(self):
        a = podpac.algorithm.Arange()
        b = podpac.algorithm.Arange()
        c = podpac.algorithm.Arange()
        
        node = podpac.compositor.OrderedCompositor(sources=np.array([a, b, c]))
        definition = node.definition

        # make sure it is a valid pipeline
        pipeline = podpac.pipeline.Pipeline(definition=definition)

        # check that the arange refs are unique
        assert len(pipeline.definition['nodes']) == 4
        
    def test_pipeline(self):
        n = Node()
        p = n.pipeline
        assert isinstance(p, podpac.pipeline.Pipeline)
    
    def test_json(self):
        n = Node()

        s = n.json
        assert isinstance(s, str)
        json.loads(s)

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

        c1 = podpac.Coordinates([[0, 1], [0, 1]], dims=['lat', 'lon'])
        c2 = podpac.Coordinates([[10, 11], [10, 11, 12]], dims=['lat', 'lon'])
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

class TestCreateOutputArray(object):
    @classmethod
    def setup_class(cls):
        cls.c1 = podpac.Coordinates([podpac.clinspace((0, 0), (1, 1), 10), [0, 1, 2]], dims=['lat_lon', 'time'])
        cls.c2 = podpac.Coordinates([podpac.clinspace((0.5, 0.1), (1.5, 1.1), 15)], dims=['lat_lon'])
        cls.crds = [cls.c1, cls.c2]

    def test_create_output_array_default(self):
        node = Node()

        for crd in self.crds:
            output = node.create_output_array(crd)
            assert isinstance(output, UnitsDataArray)
            assert output.shape == crd.shape
            assert output.dtype == node.dtype
            assert np.all(np.isnan(output))

    def test_create_output_array_data(self):
        node = Node()

        output = node.create_output_array(self.c1, data=0)
        assert isinstance(output, UnitsDataArray)
        assert output.shape == self.c1.shape
        assert output.dtype == node.dtype
        assert np.all(output == 0.0)

    def test_create_output_array_dtype(self):
        node = Node(dtype=bool)

        output = node.create_output_array(self.c1, data=0)
        assert isinstance(output, UnitsDataArray)
        assert output.shape == self.c1.shape
        assert output.dtype == node.dtype
        assert np.all(~output)

# @pytest.mark.skip("TODO")
class TestCaching(object):
    @classmethod
    def setup_class(cls):
        class MyNode(Node):
            pass

        cls.node = MyNode(cache_type='disk')
        cls.node.rem_cache()

        cls.coords = podpac.Coordinates([0, 0], dims=['lat', 'lon'])
        cls.coords2 = podpac.Coordinates([1, 1], dims=['lat', 'lon'])

    @classmethod
    def teardown_class(cls):
        cls.node.rem_cache()

    def setup_method(self, method):
        self.node.rem_cache()

    def teardown_method(self, method):
        self.node.rem_cache()

    def test_has_cache(self):
        assert not self.node.has_cache('test')

        self.node.put_cache(0, 'test')
        assert self.node.has_cache('test')
        assert not self.node.has_cache('test', coordinates=self.coords)

    def test_has_coordinates(self):
        assert not self.node.has_cache('test', coordinates=self.coords)

        self.node.put_cache(0, 'test', coordinates=self.coords)

        assert not self.node.has_cache('test')
        assert self.node.has_cache('test', coordinates=self.coords)
        assert not self.node.has_cache('test', coordinates=self.coords2)

    def test_get_put_cache(self):
        with pytest.raises(NodeException):
            self.node.get_cache('test')

        self.node.put_cache(0, 'test')
        assert self.node.get_cache('test') == 0

    def test_get_put_coordinates(self):
        with pytest.raises(NodeException):
            self.node.get_cache('test')
        with pytest.raises(NodeException):
            self.node.get_cache('test', coordinates=self.coords)
        with pytest.raises(NodeException):
            self.node.get_cache('test', coordinates=self.coords2)

        self.node.put_cache(0, 'test')
        self.node.put_cache(1, 'test', coordinates=self.coords)
        self.node.put_cache(2, 'test', coordinates=self.coords2)

        assert self.node.get_cache('test') == 0
        assert self.node.get_cache('test', coordinates=self.coords) == 1
        assert self.node.get_cache('test', coordinates=self.coords2) == 2

    def test_put_overwrite(self):
        self.node.put_cache(0, 'test')
        assert self.node.get_cache('test') == 0

        with pytest.raises(NodeException):
            self.node.put_cache(1, 'test')

        self.node.put_cache(1, 'test', overwrite=True)
        assert self.node.get_cache('test') == 1

    def test_rem_all(self):
        self.node.put_cache(0, 'a')
        self.node.put_cache(0, 'b')
        self.node.put_cache(0, 'a', coordinates=self.coords)
        self.node.put_cache(0, 'c', coordinates=self.coords)
        self.node.put_cache(0, 'c', coordinates=self.coords2)
        self.node.put_cache(0, 'd', coordinates=self.coords)

        self.node.rem_cache()
        assert not self.node.has_cache('a')
        assert not self.node.has_cache('b')
        assert not self.node.has_cache('a', coordinates=self.coords)
        assert not self.node.has_cache('c', coordinates=self.coords)
        assert not self.node.has_cache('c', coordinates=self.coords2)
        assert not self.node.has_cache('d', coordinates=self.coords)

    @pytest.mark.skip('BUG: Need to fix this.')
    def test_rem_key(self):
        self.node.put_cache(0, 'a')
        self.node.put_cache(0, 'b')
        self.node.put_cache(0, 'a', coordinates=self.coords)
        self.node.put_cache(0, 'c', coordinates=self.coords)
        self.node.put_cache(0, 'c', coordinates=self.coords2)
        self.node.put_cache(0, 'd', coordinates=self.coords)

        self.node.rem_cache(key='a')

        assert not self.node.has_cache('a')
        assert not self.node.has_cache('a', coordinates=self.coords)
        assert self.node.has_cache('b')
        assert self.node.has_cache('c', coordinates=self.coords)
        assert self.node.has_cache('c', coordinates=self.coords2)
        assert self.node.has_cache('d', coordinates=self.coords)

    @pytest.mark.skip('BUG: Need to fix this.')
    def test_rem_coordinates(self):
        self.node.put_cache(0, 'a')
        self.node.put_cache(0, 'b')
        self.node.put_cache(0, 'a', coordinates=self.coords)
        self.node.put_cache(0, 'c', coordinates=self.coords)
        self.node.put_cache(0, 'c', coordinates=self.coords2)
        self.node.put_cache(0, 'd', coordinates=self.coords)

        self.node.rem_cache(coordinates=self.coords)

        assert self.node.has_cache('a')
        assert not self.node.has_cache('a', coordinates=self.coords)
        assert self.node.has_cache('b')
        assert not self.node.has_cache('c', coordinates=self.coords)
        assert self.node.has_cache('c', coordinates=self.coords2)
        assert not self.node.has_cache('d', coordinates=self.coords)

    def test_rem_key_coordinates(self):
        self.node.put_cache(0, 'a')
        self.node.put_cache(0, 'b')
        self.node.put_cache(0, 'a', coordinates=self.coords)
        self.node.put_cache(0, 'c', coordinates=self.coords)
        self.node.put_cache(0, 'c', coordinates=self.coords2)
        self.node.put_cache(0, 'd', coordinates=self.coords)

        self.node.rem_cache(key='a', coordinates=self.coords)

        assert self.node.has_cache('a')
        assert not self.node.has_cache('a', coordinates=self.coords)
        assert self.node.has_cache('b')
        assert self.node.has_cache('c', coordinates=self.coords)
        assert self.node.has_cache('c', coordinates=self.coords2)
        assert self.node.has_cache('d', coordinates=self.coords)

class TestCachePropertyDecorator(object):
    def test_cache_property_decorator(self):
        class Test(podpac.Node):
            a = tl.Int(1).tag(attr=True)
            b = tl.Int(1).tag(attr=True)
            c = tl.Int(1)
            d = tl.Int(1)

            @podpac.core.node.cache_func('a2', 'a')
            def a2(self):
                """a2 docstring"""
                return self.a * 2

            @podpac.core.node.cache_func('b2')
            def b2(self):
                """ b2 docstring """
                return self.b * 2

            @podpac.core.node.cache_func('c2', 'c')
            def c2(self):
                """ c2 docstring """
                return self.c * 2

            @podpac.core.node.cache_func('d2')
            def d2(self):
                """ d2 docstring """
                return self.d * 2
            
        t = Test(cache_type='disk')
        t2 = Test(cache_type='disk')
        t.rem_cache()
        t2.rem_cache()

        try: 
            t.get_cache('a2')
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
        t.rem_cache()
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
        assert t2.get_cache('a2') == 4  # This was cached by t
        t2.b = 2
        assert t2.get_cache('c2') == 4  # This was cached by t
        assert t2.get_cache('d2') == 2  # This was cached by t
        
    def test_cache_func_decorator_with_no_cache(self):
        class Test(podpac.Node):
            a = tl.Int(1).tag(attr=True)
            b = tl.Int(1).tag(attr=True)
            c = tl.Int(1)
            d = tl.Int(1)

            @podpac.core.node.cache_func('a2', 'a')
            def a2(self):
                """a2 docstring"""
                return self.a * 2

            @podpac.core.node.cache_func('b2')
            def b2(self):
                """ b2 docstring """
                return self.b * 2

            @podpac.core.node.cache_func('c2', 'c')
            def c2(self):
                """ c2 docstring """
                return self.c * 2

            @podpac.core.node.cache_func('d2')
            def d2(self):
                """ d2 docstring """
                return self.d * 2
            
        t = Test(cache_type=None)
        t2 = Test(cache_type=None)
        t.rem_cache()
        t2.rem_cache()

        try: 
            t.get_cache('a2')
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
        t.rem_cache()
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

# TODO: remove this - this is currently a placeholder test until we actually have integration tests (pytest will exit with code 5 if no tests found)
@pytest.mark.integration
def tests_node_integration():
    assert True
