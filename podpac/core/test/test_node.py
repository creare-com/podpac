from __future__ import division, unicode_literals, print_function, absolute_import

import os
from collections import OrderedDict
import json
import six

import pytest
import numpy as np
from pint.errors import DimensionalityError
from pint import UnitRegistry; ureg = UnitRegistry()
import traitlets as tl

import podpac
from podpac.core import common_test_utils as ctu
from podpac.core.units import UnitsDataArray
from podpac.core.node import Node, NodeException
    
class TestInit(object):
    pass # TODO

class TestNodeProperties(object):
    def test_base_ref(self):
        n = Node()
        assert isinstance(n.base_ref, str)

    def test_base_definition(self):
        class N(Node):
            attr = tl.Int().tag(attr=True)
        n = N(attr=7)

        d = n.base_definition
        assert isinstance(d, OrderedDict)
        assert 'node' in d
        assert isinstance(d['node'], str)
        assert 'attrs' in d
        assert isinstance(d['attrs'], OrderedDict)
        assert 'attr' in d['attrs']
        assert d['attrs']['attr'] == 7

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

        assert isinstance(pipeline.nodes[a.base_ref], podpac.algorithm.Arange)
        assert isinstance(pipeline.nodes[b.base_ref], podpac.algorithm.CoordData)
        assert isinstance(pipeline.nodes[c.base_ref], podpac.compositor.OrderedCompositor)
        assert isinstance(pipeline.nodes[node.base_ref], podpac.algorithm.Arithmetic)
        assert isinstance(pipeline.pipeline_output, podpac.pipeline.NoOutput)

        assert pipeline.pipeline_output.node is pipeline.nodes[node.base_ref]
        assert pipeline.pipeline_output.name == node.base_ref

    def test_make_pipeline_definition_duplicate_ref(self):
        a = podpac.algorithm.Arange()
        b = podpac.algorithm.Arange()
        c = podpac.algorithm.Arange()
        
        node = podpac.compositor.OrderedCompositor(sources=np.array([a, b, c]))
        definition = node.definition

        # make sure it is a valid pipeline
        pipeline = podpac.pipeline.Pipeline(definition=definition)

        # check that the arange refs are unique
        assert len(pipeline.nodes) == 4
    
    def test_pipeline(self):
        n = Node()
        p = n.pipeline
        assert isinstance(p, podpac.pipeline.Pipeline)
    
    def test_json(self):
        n = Node()
        s = n.json
        assert isinstance(s, str)
        json.loads(s)

    def test_hash(self):
        class N(Node):
            attr = tl.Int().tag(attr=True)

        class M(Node):
            attr = tl.Int().tag(attr=True)

        n1 = N(attr=1)
        n2 = N(attr=1)
        n3 = N(attr=2)
        m1 = M(attr=1)

        assert n1.hash == n2.hash
        assert n1.hash != n3.hash
        assert n1.hash != m1.hash

class TestNotImplementedMethods(object):
    def test_eval(self):
        n = Node()
        with pytest.raises(NotImplementedError):
            n.eval(None)

        with pytest.raises(NotImplementedError):
            n.eval(None, output=None)

    def test_find_coordinates(self):
        n = Node()
        with pytest.raises(NotImplementedError):
            n.find_coordinates()

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

@pytest.mark.skip("TODO")
class TestCaching(object):
    @classmethod
    def setup_class(cls):
        class MyNode(Node):
            pass

        cls.node = MyNode()
        cls.node.del_cache()

        cls.coords = podpac.Coordinates([0, 0], dims=['lat', 'lon'])
        cls.coords2 = podpac.Coordinates([1, 1], dims=['lat', 'lon'])

    @classmethod
    def teardown_class(cls):
        cls.node.del_cache()

    def setup_method(self, method):
        self.node.del_cache()

    def teardown_method(self, method):
        self.node.del_cache()

    def test_has(self):
        assert not self.node.has('test')

        self.node.put(0, 'test')
        assert self.node.has('test')
        assert not self.node.has('test', coordinates=self.coords)

    def test_has_coordinates(self):
        assert not self.node.has('test', coordinates=self.coords)

        self.node.put(0, 'test', coordinates=self.coords)

        assert not self.node.has('test')
        assert self.node.has('test', coordinates=self.coords)
        assert not self.node.has('test', coordinates=self.coords2)

    def test_get_put(self):
        with pytest.raises(NodeException):
            self.node.get('test')

        self.node.put(0, 'test')
        assert self.node.get('test') == 0

    def test_get_put_coordinates(self):
        with pytest.raises(NodeException):
            self.node.get('test')
        with pytest.raises(NodeException):
            self.node.get('test', coordinates=self.coords)
        with pytest.raises(NodeException):
            self.node.get('test', coordinates=self.coords2)

        self.node.put(0, 'test')
        self.node.put(1, 'test', coordinates=self.coords)
        self.node.put(2, 'test', coordinates=self.coords2)

        assert self.node.get('test') == 0
        assert self.node.get('test', coordinates=self.coords) == 1
        assert self.node.get('test', coordinates=self.coords2) == 2

    def test_put_overwrite(self):
        self.node.put(0, 'test')
        assert self.node.get('test') == 0

        with pytest.raises(NodeException):
            self.node.put('test', 1)

        self.node.put(1, 'test', overwrite=True)
        assert self.node.get('test') == 1

    def test_del_all(self):
        self.node.put(0, 'a')
        self.node.put(0, 'b')
        self.node.put(0, 'a', coordinates=self.coords)
        self.node.put(0, 'c', coordinates=self.coords)
        self.node.put(0, 'c', coordinates=self.coords2)
        self.node.put(0, 'd', coordinates=self.coords)

        self.node.del_cache()
        assert not self.has_cache('a')
        assert not self.has_cache('b')
        assert not self.has_cache('a', coordinates=self.coords)
        assert not self.has_cache('c', coordinates=self.coords)
        assert not self.has_cache('c', coordinates=self.coords2)
        assert not self.has_cache('d', coordinates=self.coords)

    def test_del_key(self):
        self.node.put(0, 'a')
        self.node.put(0, 'b')
        self.node.put(0, 'a', coordinates=self.coords)
        self.node.put(0, 'c', coordinates=self.coords)
        self.node.put(0, 'c', coordinates=self.coords2)
        self.node.put(0, 'd', coordinates=self.coords)

        self.node.del_cache(key='a')

        assert not self.has_cache('a')
        assert not self.has_cache('a', coordinates=self.coords)
        assert self.has_cache('b')
        assert self.has_cache('c', coordinates=self.coords)
        assert self.has_cache('c', coordinates=self.coords2)
        assert self.has_cache('d', coordinates=self.coords)

    def test_del_coordinates(self):
        self.node.put(0, 'a')
        self.node.put(0, 'b')
        self.node.put(0, 'a', coordinates=self.coords)
        self.node.put(0, 'c', coordinates=self.coords)
        self.node.put(0, 'c', coordinates=self.coords2)
        self.node.put(0, 'd', coordinates=self.coords)

        self.node.del_cache(coordinates=self.coords)

        assert self.has_cache('a')
        assert not self.has_cache('a', coordinates=self.coords)
        assert self.has_cache('b')
        assert not self.has_cache('c', coordinates=self.coords)
        assert self.has_cache('c', coordinates=self.coords2)
        assert not self.has_cache('d', coordinates=self.coords)

    def test_del_key_coordinates(self):
        self.node.put(0, 'a')
        self.node.put(0, 'b')
        self.node.put(0, 'a', coordinates=self.coords)
        self.node.put(0, 'c', coordinates=self.coords)
        self.node.put(0, 'c', coordinates=self.coords2)
        self.node.put(0, 'd', coordinates=self.coords)

        self.node.del_cache(key='a', cordinates=self.coords)

        assert self.has_cache('a')
        assert not self.has_cache('a', coordinates=self.coords)
        assert self.has_cache('b')
        assert self.has_cache('c', coordinates=self.coords)
        assert self.has_cache('c', coordinates=self.coords2)
        assert self.has_cache('d', coordinates=self.coords)

class TestDeprecatedMethods(object):
    def setup_method(self):
        self.paths_to_remove = []

    def teardown_method(self):
        for path in self.paths_to_remove:
            try:
                os.remove(path)
            except:
                pass

    def test_write(self):
        n = Node()
        c = podpac.Coordinates([0, 1], dims=['lat', 'lon'])
        n._requested_coordinates = c # hack instead of evaluating the node
        n._output = UnitsDataArray([0, 1])
        p = n.write('temp_test')
        self.paths_to_remove.append(p)
        
        assert os.path.exists(p)
    
    def test_load(self):
        c = podpac.Coordinates([0, 1], dims=['lat', 'lon'])
        fn = 'temp_test'
        
        n1 = Node()
        n1._output = UnitsDataArray([0, 1])
        n1._requested_coordinates = c # hack instead of evaluating the node
        p1 = n1.write(fn)
        self.paths_to_remove.append(p1)

        n2 = Node()
        p2 = n2.load(fn, c)
        
        assert p1 == p2
        np.testing.assert_array_equal(n1._output.data, n2._output.data)

    def test_cache_dir(self):
        n = Node()
        assert isinstance(n.cache_dir, six.string_types)
        assert n.cache_dir.endswith('Node')
        assert 'cache' in n.cache_dir

    def test_cache_path(self):
        n = Node()
        with pytest.raises(AttributeError):
            n.cache_path('testfile')
        with pytest.raises(AttributeError):
            n.cache_obj('testObject', 'testFileName')
        with pytest.raises(AttributeError):
            n.load_cached_obj('testFileName')    
    
    @pytest.mark.skip()
    def test_clear_disk_cache(self):
        class N(Node):
            source = 'test'

        n = N()
        with pytest.raises(AttributeError):
            n.clear_disk_cache()
        n.clear_disk_cache(all_cache=True)
        with pytest.raises(AttributeError):
            n.clear_disk_cache(node_cache=True)

# TODO: remove this - this is currently a placeholder test until we actually have integration tests (pytest will exit with code 5 if no tests found)
@pytest.mark.integration
def tests_node_integration():
    assert True
