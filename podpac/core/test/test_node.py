from __future__ import division, unicode_literals, print_function, absolute_import

import pytest
import numpy as np
from pint.errors import DimensionalityError
from pint import UnitRegistry
ureg = UnitRegistry()
import traitlets as tl

from podpac.core.node import *
from podpac.core.units import UnitsDataArray

class TestStyleCreation(object):
    def test_basic_creation(self):
        s = Style()
    def test_create_with_node(self):
        s = Style(Node())
    def test_get_default_cmap(self):
        Style().cmap
        

class TestNodeProperties(object):
    @classmethod
    def setup_class(cls):
        from podpac import Coordinate
        cls.crds = [Coordinate(lat_lon=((0, 1), (0, 1), 10), time=(0, 1, 2),
                            order=['lat_lon', 'time']),
                    Coordinate(lat_lon=((0.5, 1.5), (0.1, 1.1), 15))
                    ]

    def test_shape_no_nc(self):
        n = Node()
        for crd in self.crds:
            np.testing.assert_array_equal(crd.shape, n.get_output_shape(crd))
    
    def test_shape_with_nc(self):
        crd1 = Coordinate(lat_lon=((0.5, 1.5), (0.1, 1.1), 5))
        n = Node(native_coordinates=Coordinate(lat_lon=((0.5, 1.5), (0.1, 1.1), 15)))
        np.testing.assert_array_equal(crd1.shape,
                                      n.get_output_shape(crd1))
        crd2 = Coordinate(time=(0, 1, 3))
        n.native_coordinates = crd1 + crd2
        # WE SHOULD FIX THE SPEC: This really should be [3, 5]
        np.testing.assert_array_equal([5, 3],
                                      n.get_output_shape(crd2))
        np.testing.assert_array_equal(n.native_coordinates.shape,
                                      n.shape)
    
    def test_base_ref(self):
        # Just make sure this doesn't error out
        Node().base_ref
        
    def test_latlon_bounds_str(self):
        n = Node(evaluated_coordinates=Coordinate(lat=(0, 1, 3), lon=(0, 1, 3)))
        assert(n.latlon_bounds_str == '0.0_0.0_x_1.0_1.0')
        
    def test_cache_dir(self):
        d = Node().cache_dir
        assert(d.endswith('Node'))
        assert('cache' in d)
        

class TestNodeDefaults(object):
    def test_node_defaults(self):
        class N2(Node):
            node_defaults={'interpolation': 'testDefault'}
        
        n = N2(interpolation='testInput')
        assert(n.interpolation == 'testInput')
        n = N2()
        assert(n.interpolation == 'testDefault')
    
    def test_node_defaults_as_input(self):
        n = Node(node_defaults={'interpolation': 'testDefault'}, 
                 interpolation="testInput")
        assert(n.interpolation == 'testInput')
        
class TestNotImplementedMethods(object):
    def test_execute(self):
        with pytest.raises(NotImplementedError):
            Node().execute(None)
    
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
        from podpac import Coordinate
        cls.crds = [Coordinate(lat_lon=((0, 1), (0, 1), 10), time=(0, 1, 2),
                            order=['lat_lon', 'time']),
                    Coordinate(lat_lon=((0.5, 1.5), (0.1, 1.1), 15))
                    ]
    
    def test_get_output_dims(self):
        n1 = Node()
        n2 = Node(native_coordinates=Coordinate(alt=(0, 1, 3)))
        n3 = Node()
        for crd in self.crds:
            np.testing.assert_array_equal(n1.get_output_dims(crd), 
                                          crd.dims)
            np.testing.assert_array_equal(n2.get_output_dims(crd), 
                                          ['alt'])            
            n3.evaluated_coordinates = crd
            np.testing.assert_array_equal(n3.get_output_dims(), 
                                          crd.dims)            
        

class TestNodeOutputArrayCreation(object):
    @classmethod
    def setup_class(cls):
        from podpac import Coordinate
        cls.c1 = Coordinate(lat_lon=((0, 1), (0, 1), 10), time=(0, 1, 2),
                            order=['lat_lon', 'time'])
        cls.c2 = Coordinate(lat_lon=((0.5, 1.5), (0.1, 1.1), 15))
        cls.crds = [Coordinate(lat_lon=((0, 1), (0, 1), 10), time=(0, 1, 2),
                            order=['lat_lon', 'time']),
                    Coordinate(lat_lon=((0.5, 1.5), (0.1, 1.1), 15))
                    ]        
        cls.init_types = ['empty', 'nan', 'zeros', 'ones', 'full', 'data']
    
    def test_copy_output_array(self):
            crd = self.crds[0]
            n1 = Node(native_coordinates=crd)
            np.testing.assert_array_equal(n1.copy_output_array(),
                                          n1.output)
            assert(id(n1.output) != id(n1.copy_output_array()))
            # Just run through the different creating methods
            for init_type in self.init_types[:4]:
                n1.copy_output_array(init_type)
                
    def test_default_output_native_coordinates(self):
        n = Node(native_coordinates=self.c1)
        o = n.output
        np.testing.assert_array_equal(o.shape, self.c1.shape)
        assert(np.all(np.isnan(o)))

    def test_default_output_evaluated_coordinates(self):
        n = Node(evaluated_coordinates=self.c1)
        o = n.output
        np.testing.assert_array_equal(o.shape, self.c1.shape)
        assert(np.all(np.isnan(o)))

    def test_output_creation_stacked_native(self):
        n = Node(native_coordinates=self.c1)
        s1 = n.initialize_output_array().shape
        assert((10, 2) == s1)
        n.evaluated_coordinates = self.c2
        s2 = n.initialize_output_array().shape
        assert((15, 2) == s2)
        n.evaluated_coordinates = self.c2.unstack()
        s3 = n.initialize_output_array().shape
        assert((15, 15, 2) == s3)
        
    def test_output_creation_unstacked_native(self):
        n = Node(native_coordinates=self.c1.unstack())
        s1 = n.initialize_output_array().shape
        assert((10, 10, 2) == s1)
        n.evaluated_coordinates = self.c2
        s2 = n.initialize_output_array().shape
        assert((15, 2) == s2)
        n.evaluated_coordinates = self.c2.unstack()
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
            param = tl.Float().tag(param=True)
        n = N(param=1.1, attr=7)
        bd = n.base_definition()
        assert(bd.get('node', '') == 'N')
        assert(bd.get('params', {}).get('param', {}) == 1.1)
        assert(bd.get('attrs', {}).get('attr', {}) == 7)

class TestExecuteParams(object):
    def test_default_params(self):
        class N(Node):
            param1 = tl.Int(7).tag(param=True)
            param2 = tl.Float(3.14).tag(param=True)
        n = N()
        p = n.get_params()
        assert(p.get('param1') == 7)
        assert(p.get('param2') == 3.14)
    def test_runtime_params(self):
        class N(Node):
            param1 = tl.Int(7).tag(param=True)
            param2 = tl.Float(3.14).tag(param=True)
        n = N()
        p = n.get_params(dict(param1=-13))
        assert(p.get('param1') == -13)
        assert(p.get('param2') == 3.14)  
        assert(n.param1 == 7)

class TestFilesAndCaching(object):
    def test_get_hash(self):
        crds1 = Coordinate(lat=1)
        crds2 = Coordinate(lat=2)
        crds3 = Coordinate(lon=1)
        n1 = Node()
        n2 = Node()
        params = {'param1': 1}
        assert(n1.get_hash(crds1, params) == n2.get_hash(crds1, params))
        assert(n1.get_hash(crds1, params) != n2.get_hash(crds1, {}))
        assert(n1.get_hash(crds2, params) != n2.get_hash(crds1, params))
        assert(n1.get_hash(crds3, params) != n2.get_hash(crds1, params))
        
    def test_evaluated_hash(self):
        n = Node()
        with pytest.raises(NodeException):
            n.evaluated_hash
        n.evaluated_coordinates = Coordinate(lat=0)
        n.evaluated_hash
        
    def test_get_output_path(self):
        p = Node().get_output_path('testfilename.txt')
        assert(p.endswith('testfilename.txt'))
        assert(os.path.exists(os.path.dirname(p)))
        
    def test_write_file(self):
        n = Node(native_coordinates=Coordinate(lat=0, lon=1))
        n.evaluated_coordinates = n.native_coordinates
        fn = 'temp_test'
        p = n.write(fn)
        assert(os.path.exists(p))
        os.remove(p)
        with pytest.raises(NotImplementedError):
            n.write(fn, format='notARealFormat')
    
    def test_load_file(self):
        n = Node(native_coordinates=Coordinate(lat=0, lon=1))
        n.evaluated_coordinates = n.native_coordinates
        fn = 'temp_test'
        p = n.write(fn)
        o = n.output
        _ = n.load(fn, n.native_coordinates, {})
        np.testing.assert_array_equal(o, n.output.data)
        os.remove(p)
        
    def test_cache_path(self):
        with pytest.raises(AttributeError):
            p = Node().cache_path('testfile')
        with pytest.raises(AttributeError):
            p = Node().cache_obj('testObject', 'testFileName')
        with pytest.raises(AttributeError):
            p = Node().load_cached_obj('testFileName')    
    
    @pytest.skip("This doesn't really work without self.source")
    def test_clear_cache(self):
        n = Node()
        with pytest.raises(AttributeError):
            n.clear_disk_cache()
        n.clear_disk_cache(all_cache=True)
        with pytest.raises(AttributeError):
            n.clear_disk_cache(node_cache=True)
        

class TestGetImage(object):
    def test_get_image(self):
        n = Node(native_coordinates=Coordinate(lat=(0, 1, 3), lon=(0, 1, 5)))
        n.output[:] = 1
        im = n.get_image()
        assert(im == b'iVBORw0KGgoAAAANSUhEUgAAAAUAAAADCAYAAABbNsX4AAAABHNCSVQICAgIfAhkiAAAABVJREFUCJljdGEM+c+ABpjQBXAKAgBgJgGe5UsCaQAAAABJRU5ErkJggg==')
        
        
        

# TODO: remove this - this is currently a placeholder test until we actually have integration tests (pytest will exit with code 5 if no tests found)
@pytest.mark.integration
def tests_node_integration():
    assert True
