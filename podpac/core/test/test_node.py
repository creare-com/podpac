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
    @pytest.mark.xfail(reason="get_output_shape removed, pending node refactor")
    def test_shape_not_Valid(self):
        n = Node()
        with pytest.raises(NodeException):
            n.get_output_shape()

    @pytest.mark.xfail(reason="get_output_shape removed, pending node refactor")
    def test_shape_no_nc(self):
        n = Node()
        
        lat = podpac.clinspace(0.5, 1.5, 15)
        lon = podpac.clinspace(0.1, 1.1, 15)
        time = podpac.clinspace(0, 1, 2)

        # lat, lon, time
        coords = podpac.Coordinates([lat, lon, time], dims=['lat', 'lon', 'time'])
        np.testing.assert_array_equal(coords.shape, n.get_output_shape(coords))

        # lat_lon
        coords = podpac.Coordinates([[lat, lon]], dims=['lat_lon', 'time'])
        np.testing.assert_array_equal(coords.shape, n.get_output_shape(coords))
        
        # lat_lon, time
        coords = podpac.Coordinates([[lat, lon], time], dims=['lat_lon', 'time'])
        np.testing.assert_array_equal(coords.shape, n.get_output_shape(coords))
    
    @pytest.mark.xfail(reason="get_output_shape removed, pending node refactor")
    def test_shape_with_nc(self):
        lat_lon = podpac.clinspace((0.5, 0.1), (1.5, 1.1), 15)
        time = podpac.clinspace(0, 1, 2)

        crd_fine = podpac.Coordinates(lat_lon, dims=['lat_lon'])
        crd_coarse = podpac.Coordinates(lat_lon[::3], dims=['lat_lon'])
        crd_time = podpac.Coordinates(time, dims=['time'])
        crd_coarse_time = podpac.Coordinates([lat_lon[::3], crd_time], dims=['lat_lon', 'time'])
        
        n = Node(native_coordinates=crd_fine)
        np.testing.assert_array_equal(crd_coarse.shape, n.get_output_shape(crd_coarse))
        
        # WE SHOULD FIX THE SPEC: This really should be [3, 5] # TODO JXM
        # TODO actually this should fail
        # TODO also, remove __add__? it's weird
        n = Node(native_coordinates=crd_coarse_time)
        np.testing.assert_array_equal([5, 3], n.get_output_shape(crd_time))
        np.testing.assert_array_equal(n.native_coordinates.shape, n.shape)
    
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
        c1 = podpac.Coordinates([podpac.clinspace((0, 0), (1, 1), 10), [0, 1, 2]], dims=['lat_lon', 'time'])
        c2 = podpac.Coordinates([podpac.clinspace((0.5, 0.1), (1.5, 1.1), 15)], dims=['lat_lon'])
        cls.crds = [c1, c2]
    
    @pytest.mark.xfail(reason="get_output_shape removed, pending node refactor")
    def test_get_output_dims(self):
        n1 = Node()
        n2 = Node(native_coordinates=podpac.Coordinates([[0, .5, 1.]], dims=['alt']))
        n3 = Node()
        for crd in self.crds:
            np.testing.assert_array_equal(n1.get_output_dims(crd), crd.dims)
            np.testing.assert_array_equal(n2.get_output_dims(crd), ['alt'])
            n3.requested_coordinates = crd
            np.testing.assert_array_equal(n3.get_output_dims(), crd.dims)
            assert(n1.get_output_dims(OrderedDict([('lat',0)])) == ['lat'])
        
@pytest.mark.skip(reason="pending node refactor")
class TestNodeOutputArrayCreation(object):
    @classmethod
    def setup_class(cls):
        cls.c1 = podpac.Coordinates([podpac.clinspace((0, 0), (1, 1), 10), [0, 1, 2]], dims=['lat_lon', 'time'])
        cls.c2 = podpac.Coordinates([podpac.clinspace((0.5, 0.1), (1.5, 1.1), 15)], dims=['lat_lon'])
        cls.crds = [cls.c1, cls.c2]
        cls.init_types = ['empty', 'nan', 'zeros', 'ones', 'full', 'data']
    
    def test_copy_output_array(self):
        crd = self.crds[0]
        n1 = Node(native_coordinates=crd)
        np.testing.assert_array_equal(n1.copy_output_array(), n1.output)
        assert(id(n1.output) != id(n1.copy_output_array()))
        # Just run through the different creating methods
        for init_type in self.init_types[:4]:
            n1.copy_output_array(init_type)
        
        with pytest.raises(ValueError):
            n1.copy_output_array('notValidInitType')
                
    def test_default_output_native_coordinates(self):
        n = Node(native_coordinates=self.c1)
        o = n.output
        np.testing.assert_array_equal(o.shape, self.c1.shape)
        assert(np.all(np.isnan(o)))

    def test_default_output_requested_coordinates(self):
        n = Node(requested_coordinates=self.c1)
        o = n.output
        np.testing.assert_array_equal(o.shape, self.c1.shape)
        assert(np.all(np.isnan(o)))

    def test_output_creation_stacked_native(self):
        n = Node(native_coordinates=self.c1)
        s1 = n.initialize_output_array().shape
        assert((10, 2) == s1)
        n.requested_coordinates = self.c2
        s2 = n.initialize_output_array().shape
        assert((15, 2) == s2)
        n.requested_coordinates = self.c2.unstack()
        s3 = n.initialize_output_array().shape
        assert((15, 15, 2) == s3)
        
    def test_output_creation_unstacked_native(self):
        n = Node(native_coordinates=self.c1.unstack())
        s1 = n.initialize_output_array().shape
        assert((10, 10, 2) == s1)
        n.requested_coordinates = self.c2
        s2 = n.initialize_output_array().shape
        assert((15, 2) == s2)
        n.requested_coordinates = self.c2.unstack()
        s3 = n.initialize_output_array().shape
        assert((15, 15, 2) == s3)    
        
    def test_init_array_types(self):
        n1 = Node(native_coordinates=self.crds[0])
        crdvals = list(self.crds[0].coords.values())
        for init_type in self.init_types[:4]:
            n1.initialize_array(init_type, coords=crdvals)
        o = n1.initialize_array(init_type='full', fillval=3.14159, coords=crdvals)
        assert(np.all(o.data == 3.14159))
        o = n1.initialize_array(init_type='data', fillval=o.data, units='m', coords=crdvals)
        assert(np.all(o.data == 3.14159))
        with pytest.raises(ValueError):
            n1.initialize_array(init_type="doesn't exist", coords=crdvals)

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
        n = Node(native_coordinates=podpac.Coordinates([0, 1], dims=['lat', 'lon']))
        n.requested_coordinates = n.native_coordinates
        fn = 'temp_test'
        p = n.write(fn)
        assert(os.path.exists(p))
        os.remove(p)
        with pytest.raises(NotImplementedError):
            n.write(fn, format='notARealFormat')
    
    def test_load_file(self):
        n = Node(native_coordinates=podpac.Coordinates([0, 1], dims=['lat', 'lon']))
        n.requested_coordinates = n.native_coordinates
        fn = 'temp_test'
        p = n.write(fn)
        o = n.output
        _ = n.load(fn, n.native_coordinates)
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
        

@pytest.mark.xfail(reason="not sure why this is failing")
class TestGetImage(object):
    def test_get_image(self):
        nc = podpac.Coordinates([podpac.clinspace(0, 1, 3), podpac.clinspace(0, 1, 5)], dims=['lat', 'lon'])
        n = Node(native_coordinates=nc)
        n.output[:] = 1
        im = n.get_image()
        assert im == b'iVBORw0KGgoAAAANSUhEUgAAAAUAAAADCAYAAABbNsX4AAAABHNCSVQICAgIfAhkiAAAABVJREFUCJljdGEM+c+ABpjQBXAKAgBgJgGe5UsCaQAAAABJRU5ErkJggg=='

class TestNodeOutputCoordinates(object):
    @pytest.mark.xfail(reason="This defines part of the node spec, which still needs to be implemented")
    def test_node_output_coordinates(self):
        ev = ctu.make_coordinate_combinations()
        kwargs = {}
        kwargs['lat'] = [-1, 0, 1]
        kwargs['lon'] = [-1, 0, 1]
        kwargs['alt'] = [-1, 0, 1]
        kwargs['time'] = ['2000-01-01T00:00:00', '2000-02-01T00:00:00']
        nc = ctu.make_coordinate_combinations(**kwargs)
        
        node = Node()
        for e in ev.values():
            for n in nc.values():
                node.native_coordinates = n
                
                # The request must contain all the dimensions in the native coordinates
                allcovered = True
                for d in n.dims:
                    if d not in e.dims:
                        allcovered = False
                if allcovered: # If request contains all dimensions, the order should be in the evaluated coordinates
                    c = node.get_output_coords(e)
                    i = 0
                    for d in e.dims:
                        if d in n.dims:
                            print (d, c.dims, i, e, n)
                            assert(d == c.dims[i])
                            i += 1
                else:  # We throw an exception
                    with pytest.raises(Exception):
                        c = node.get_output_coords(e)
                    
                    

# TODO: remove this - this is currently a placeholder test until we actually have integration tests (pytest will exit with code 5 if no tests found)
@pytest.mark.integration
def tests_node_integration():
    assert True
