
from __future__ import division, unicode_literals, print_function, absolute_import

import os
from json import JSONDecoder
from collections import OrderedDict
import numpy as np
import pytest

import podpac
from podpac.core.coordinate import Coordinate
from podpac.core.data.type import NumpyArray
from podpac.core.algorithm.algorithm import Arange
from podpac.core.pipeline import NoOutput, FileOutput, FTPOutput, AWSOutput, ImageOutput
from podpac.core.pipeline import Pipeline, PipelineNode, PipelineError, make_pipeline_definition

coords = Coordinate(lat=(0, 1, 10), lon=(0, 1, 10), order=['lat', 'lon'])
node = Arange()
node.execute(coords)

class RandomData(NumpyArray):
    source = np.random.random(coords.shape)
    native_coordinates = coords

class TestNoOutput(object):
    def test(self):
        output = NoOutput(node=node, name='test')
        output.write()

class TestFileOutput(object):
    def _test(self, format):
        output = FileOutput(node=node, name='test', outdir='.', format=format)
        output.write()

        assert output.path != None
        assert os.path.isfile(output.path)
        os.remove(output.path)
        
    def test_pickle(self):
        self._test('pickle')

    def test_png(self):
        # self._test('png')

        output = FileOutput(node=node, name='test', outdir='.', format='png')
        with pytest.raises(NotImplementedError):
            output.write()

    def test_geotif(self):
        # self._test('geotif')

        output = FileOutput(node=node, name='test', outdir='.', format='geotif')
        with pytest.raises(NotImplementedError):
            output.write()

class TestFTPOutput(object):
    def test(self):
        output = FTPOutput(node=node, name='test', url='none', user='none')
        with pytest.raises(NotImplementedError):
            output.write()

class TestAWSOutput(object):
    def test(self):
        output = AWSOutput(node=node, name='test', user='none', bucket='none')
        with pytest.raises(NotImplementedError):
            output.write()

class TestImageOutput(object):
    def _test(self, format):
        output = ImageOutput(node=node, name='test', format=format)
        output.write()
        assert output.image != None

    def test_png(self):
        self._test('png')

class TestPipeline(object):
    
    # Note: these tests are designed somewhat with the upcoming pipeline node refactor in mind

    def test_load_from_file(self):
        path = os.path.join(os.path.abspath(podpac.__path__[0]), 'core', 'test', 'test.json')
        pipeline = Pipeline(path)
        assert pipeline.path == path
        assert pipeline.definition['nodes']
        assert pipeline.definition['outputs']
        assert isinstance(pipeline.nodes['a'], podpac.core.algorithm.algorithm.Arange)
        assert len(pipeline.outputs) == 1
        assert isinstance(pipeline.outputs[0], ImageOutput)

    def test_parse_node_invalid_node(self):
        # module does not exist
        s = '''
        {
            "nodes": {"a": {"node": "nonexistent.Arbitrary"} },
            "outputs": [{"mode": "none"}]
        }
        '''

        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        
        with pytest.raises(PipelineError):
            Pipeline(d)

        # node does not exist in module
        s = '''
        {
            "nodes": {"a": {"node": "core.Nonexistent"} },
            "outputs": [{"mode": "none"}]
        }
        '''

        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        
        with pytest.raises(PipelineError):
            Pipeline(d)

    def test_parse_node_algorithm_inputs(self):
        # translate node references
        s = '''
        {
            "nodes": {
                "source1": {"node": "core.algorithm.algorithm.Arange"},
                "source2": {"node": "core.algorithm.algorithm.Arange"},
                "result": {        
                    "node": "Arithmetic",
                    "inputs": {
                        "A": "source1",
                        "B": "source2"
                    }
                }
            },
            "outputs": [{"mode": "none"}]
        }
        '''

        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d)

        assert pipeline.nodes['result'].A is pipeline.nodes['source1']
        assert pipeline.nodes['result'].B is pipeline.nodes['source2']

        # nonexistent node
        s = '''
        {
            "nodes": {
                "source2": {"node": "core.algorithm.algorithm.Arange"},
                "result": {        
                    "node": "Arithmetic",
                    "inputs": {
                        "A": "source1",
                        "B": "source2"
                    }
                }
            },
            "outputs": [{"mode": "none"}]
        }
        '''

        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        with pytest.raises(PipelineError):
            Pipeline(d)

    def test_parse_node_compositor_sources(self):
        s = '''
        {
            "nodes": {
                "a": {
                    "node": "core.algorithm.algorithm.Arange"
                },
                "b": {
                    "node": "core.algorithm.algorithm.Arange"
                },
                "c": {
                    "node": "core.compositor.OrderedCompositor",
                    "sources": ["a", "b"]
                }
            },
            "outputs": [{"mode": "none"}]
        }
        '''
        
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d)
        assert pipeline.nodes['c'].sources[0] is pipeline.nodes['a']
        assert pipeline.nodes['c'].sources[1] is pipeline.nodes['b']

        s = '''
        {
            "nodes": {
                "a": {
                    "node": "core.algorithm.algorithm.Arange"
                },
                "c": {
                    "node": "core.compositor.OrderedCompositor",
                    "sources": ["a", "b"]
                }
            },
            "outputs": [{"mode": "none"}]
        }
        '''
        
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        with pytest.raises(PipelineError):
            Pipeline(d)

    def test_parse_node_attrs(self):
        s = '''
        {
            "nodes": {
                "sm": {
                    "node": "datalib.smap.SMAP",
                    "attrs": {
                        "product": "SPL4SMGP.003",
                        "interpolation": "bilinear"
                    }
                }
            },
            "outputs": [
                {
                    "mode": "none"
                }
            ]
        }
        '''

        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d)
        assert pipeline.nodes['sm'].product == "SPL4SMGP.003"
        assert pipeline.nodes['sm'].interpolation == "bilinear"

    def test_parse_node_params(self):
        s = '''
        {
            "nodes": {
                "a": {
                    "node": "core.algorithm.algorithm.Arange",
                    "params": {
                        "param_a": 0,
                        "param_b": "test"
                    }
                }
            },
            "outputs": [{"mode": "none"}]
        }
        '''

        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d)
        assert pipeline.params['a']['param_a'] == 0
        assert pipeline.params['a']['param_b'] == 'test'

    def test_parse_node_evaluate(self):
        s = '''
        {
            "nodes": {
                "a": {
                    "node": "core.algorithm.algorithm.Arange",
                    "evaluate": true
                },
                "b": {
                    "node": "core.algorithm.algorithm.Arange",
                    "evaluate": false
                },
                "c": {
                    "node": "core.algorithm.algorithm.Arange",
                    "evaluate": false
                },
                "d": {
                    "node": "core.algorithm.algorithm.Arange"
                }
            },
            "outputs": [{"mode": "none"}]
        }
        '''

        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d)
        assert pipeline.skip_evaluate == ['b', 'c']

    def test_parse_node_invalid_property(self):
        s = '''
        {
            "nodes": {
                "a": {
                    "node": "core.algorithm.algorithm.Arange",
                    "invalid_property": "value"
                }
            },
            "outputs": [{"mode": "none"}]
        }
        '''

        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        with pytest.raises(PipelineError):
            Pipeline(d)

    def test_parse_node_plugin(self):
        pass
    
    def test_parse_output_none(self):
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "outputs": [{
                "mode": "none"
            }]
        }
        '''
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d)
        assert isinstance(pipeline.outputs[0], NoOutput)

    def test_parse_output_file(self):
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "outputs": [{
                "node": "a",
                "mode": "file",
                "format": "pickle",
                "outdir": "my_directory"
            }]
        }
        '''
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d)
        assert isinstance(pipeline.outputs[0], FileOutput)
        assert pipeline.outputs[0].node is pipeline.nodes['a']
        assert pipeline.outputs[0].name == 'a'
        assert pipeline.outputs[0].format == 'pickle'
        assert pipeline.outputs[0].outdir == 'my_directory'

    def test_parse_output_image(self):
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "outputs": [{
                "node": "a",
                "mode": "image",
                "format": "png"
            }]
        }
        '''
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d)
        assert isinstance(pipeline.outputs[0], ImageOutput)
        assert pipeline.outputs[0].node is pipeline.nodes['a']
        assert pipeline.outputs[0].name == 'a'
        assert pipeline.outputs[0].format == 'png'

    def test_parse_output_aws(self):
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "outputs": [{
                "node": "a",
                "mode": "aws",
                "user": "my_user",
                "bucket": "my_bucket"
            }]
        }
        '''
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d)
        assert isinstance(pipeline.outputs[0], AWSOutput)
        assert pipeline.outputs[0].node is pipeline.nodes['a']
        assert pipeline.outputs[0].name == 'a'
        assert pipeline.outputs[0].user == 'my_user'
        assert pipeline.outputs[0].bucket == 'my_bucket'

    def test_parse_output_ftp(self):
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "outputs": [{
                "node": "a",
                "mode": "ftp",
                "url": "my_url",
                "user": "my_user"
            }]
        }
        '''
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d)
        assert isinstance(pipeline.outputs[0], FTPOutput)
        assert pipeline.outputs[0].node is pipeline.nodes['a']
        assert pipeline.outputs[0].name == 'a'
        assert pipeline.outputs[0].user == 'my_user'
        assert pipeline.outputs[0].url == 'my_url'

    def test_parse_output_invalid_mode(self):
        # invalid mode
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "outputs": [{"mode": "nonexistent_mode"}]
        }
        '''
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        with pytest.raises(PipelineError):
            Pipeline(d)

        # no mode
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "outputs": [{ }]
        }
        '''
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        with pytest.raises(PipelineError):
            Pipeline(d)

    def test_parse_output_nonexistent_node(self):
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "outputs": [{
                "node": "b",
                "mode": "file",
                "format": "pickle",
                "outdir": "my_directory"
            }]
        }
        '''
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        with pytest.raises(PipelineError):
            Pipeline(d)

    def test_execute(self):
        s = '''
        {
            "nodes": {
                "a": {
                    "node": "core.algorithm.algorithm.Arange"
                }
            },
            "outputs": [{
                "node": "a",
                "mode": "image",
                "format": "png"
            }]
        }
        '''

        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d)
        pipeline.execute(coords)
        assert pipeline.outputs[0].image is not None

class TestPipelineNode(object):
    def test_pipeline_node(self):
        path = os.path.join(os.path.abspath(podpac.__path__[0]), 'core', 'test', 'test.json')
        node = PipelineNode(path=path)
        node.execute(coords)

def test_make_pipeline_definition():
    a = podpac.core.algorithm.algorithm.Arange()
    b = podpac.core.algorithm.algorithm.CoordData()
    c = podpac.core.compositor.OrderedCompositor(sources=np.array([a, b]))
    d = podpac.core.algorithm.algorithm.Arithmetic(A=a, B=b, C=c, eqn="A + B + C")
    
    definition = make_pipeline_definition(d)

    # make sure it is a valid pipeline
    pipeline = Pipeline(definition)

    assert isinstance(pipeline.nodes[a.base_ref], podpac.core.algorithm.algorithm.Arange)
    assert isinstance(pipeline.nodes[b.base_ref], podpac.core.algorithm.algorithm.CoordData)
    assert isinstance(pipeline.nodes[c.base_ref], podpac.core.compositor.OrderedCompositor)
    assert isinstance(pipeline.nodes[d.base_ref], podpac.core.algorithm.algorithm.Arithmetic)
    assert isinstance(pipeline.outputs[0], ImageOutput)
    assert pipeline.outputs[0].node == pipeline.nodes[d.base_ref]

def test_make_pipeline_definition_duplicate_base_ref():
    a = podpac.core.algorithm.algorithm.Arange()
    b = podpac.core.algorithm.algorithm.Arange()
    c = podpac.core.algorithm.algorithm.Arange()
    d = podpac.core.compositor.OrderedCompositor(sources=np.array([a, b, c]))
    
    definition = make_pipeline_definition(d)

    # make sure it is a valid pipeline
    pipeline = Pipeline(definition)

    assert len(pipeline.nodes) == 4
    assert pipeline.outputs[0].node == pipeline.nodes[d.base_ref]