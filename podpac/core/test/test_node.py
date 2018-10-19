from __future__ import division, unicode_literals, print_function, absolute_import

import os

import pytest
import numpy as np
from pint.errors import DimensionalityError
from pint import UnitRegistry; ureg = UnitRegistry()
import traitlets as tl

import podpac
from podpac.core import common_test_utils as ctu
from podpac.core.units import UnitsDataArray
from podpac.core.node import Node, NodeException
        
class TestNodeProperties(object):
    def test_base_ref(self):
        # Just make sure this doesn't error out
        Node().base_ref
        
    def test_latlon_bounds_str(self):
        n = Node(requested_coordinates=podpac.Coordinates([[0, 0.5, 1], [0, 0.5, 1]], dims=['lat', 'lon']))
        assert(n.latlon_bounds_str == '0.0_0.0_x_1.0_1.0')
        
    def test_cache_dir(self):
        d = Node().cache_dir
        assert(d.endswith('Node'))
        assert('cache' in d)
        
        
class TestNotImplementedMethods(object):
    def test_eval(self):
        with pytest.raises(NotImplementedError):
            Node().eval(None)
    
    def test_definition(self):
        with pytest.raises(NotImplementedError):
            Node().definition()
    
    def test_pipeline_definition(self):
        with pytest.raises(NotImplementedError):
            Node().pipeline_definition
        
    def test_pipeline_json(self):
        with pytest.raises(NotImplementedError):
            Node().pipeline_json
    
    def test_pipeline(self):
        with pytest.raises(NotImplementedError):
            Node().pipeline
    
class TestNodeMethods(object):
    @classmethod
    def setup_class(cls):
        cls.c1 = podpac.Coordinates([podpac.clinspace((0, 0), (1, 1), 10), [0, 1, 2]], dims=['lat_lon', 'time'])
        cls.c2 = podpac.Coordinates([podpac.clinspace((0.5, 0.1), (1.5, 1.1), 15)], dims=['lat_lon'])
        cls.crds = [cls.c1, cls.c2]

    def test_get_output_coords(self):
        # TODO
        pass

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

class TestPipelineDefinition(object):
    def test_base_definition(self):
        class N(Node):
            attr = tl.Int().tag(attr=True)
        n = N(attr=7)
        bd = n.base_definition()
        assert(bd.get('node', '') == 'N')
        assert(bd.get('attrs', {}).get('attr', {}) == 7)

class TestFilesAndCaching(object):
    def test_get_hash(self):
        # TODO attrs should result in different hashes
        crds1 = podpac.Coordinates([1], dims=['lat'])
        crds2 = podpac.Coordinates([2], dims=['lat'])
        crds3 = podpac.Coordinates([1], dims=['lon'])
        n1 = Node()
        n2 = Node()
        assert(n1.get_hash(crds1) == n2.get_hash(crds1))
        assert(n1.get_hash(crds2) != n2.get_hash(crds1))
        assert(n1.get_hash(crds3) != n2.get_hash(crds1))
        
    def test_evaluated_hash(self):
        n = Node()
        with pytest.raises(NodeException):
            n.evaluated_hash
        n.requested_coordinates = podpac.Coordinates([0], dims=['lat'])
        n.evaluated_hash
        
    def test_get_output_path(self):
        p = Node().get_output_path('testfilename.txt')
        assert(p.endswith('testfilename.txt'))
        assert(os.path.exists(os.path.dirname(p)))
        
    def test_write_file(self):
        n = Node()
        n.requested_coordinates = podpac.Coordinates([0, 1], dims=['lat', 'lon'])
        fn = 'temp_test'
        p = n.write(fn)
        assert(os.path.exists(p))
        os.remove(p)
        with pytest.raises(NotImplementedError):
            n.write(fn, format='notARealFormat')
    
    @pytest.mark.skip(reason="spec changes")
    def test_load_file(self):
        c = podpac.Coordinates([0, 1], dims=['lat', 'lon'])
        n = Node()
        n.requested_coordinates = c
        fn = 'temp_test'
        p = n.write(fn)
        o = n.output
        _ = n.load(fn, c)
        np.testing.assert_array_equal(o, n.output.data)
        os.remove(p)
        
    def test_cache_path(self):
        with pytest.raises(AttributeError):
            p = Node().cache_path('testfile')
        with pytest.raises(AttributeError):
            p = Node().cache_obj('testObject', 'testFileName')
        with pytest.raises(AttributeError):
            p = Node().load_cached_obj('testFileName')    
    
    @pytest.mark.skip("This doesn't really work without self.source")
    def test_clear_cache(self):
        n = Node()
        with pytest.raises(AttributeError):
            n.clear_disk_cache()
        n.clear_disk_cache(all_cache=True)
        with pytest.raises(AttributeError):
            n.clear_disk_cache(node_cache=True)

@pytest.mark.skip("spec has changed")
class TestNodeOutputCoordinates(object):
    @pytest.mark.xfail(reason="This defines part of the node spec, which still needs to be implemented")
    def test_node_output_coordinates(self):
        coords_list = ctu.make_coordinate_combinations()
        kwargs = {}
        kwargs['lat'] = [-1, 0, 1]
        kwargs['lon'] = [-1, 0, 1]
        kwargs['alt'] = [-1, 0, 1]
        kwargs['time'] = ['2000-01-01T00:00:00', '2000-02-01T00:00:00']
        nc = ctu.make_coordinate_combinations(**kwargs)
        
        node = Node()
        for coords in coords_list.values():
            for n in nc.values():
                node.native_coordinates = n
                
                # The request must contain all the dimensions in the native coordinates
                allcovered = True
                for d in n.dims:
                    if d not in coords.dims:
                        allcovered = False
                if allcovered: # If request contains all dimensions, the order should be in the evaluated coordinates
                    c = node.get_output_coords(coords)
                    i = 0
                    for d in coords.dims:
                        if d in n.dims:
                            print (d, c.dims, i, coords, n)
                            assert(d == c.dims[i])
                            i += 1
                else:  # We throw an exception
                    with pytest.raises(Exception):
                        c = node.get_output_coords(coords)
                    
                    

# TODO: remove this - this is currently a placeholder test until we actually have integration tests (pytest will exit with code 5 if no tests found)
@pytest.mark.integration
def tests_node_integration():
    assert True
