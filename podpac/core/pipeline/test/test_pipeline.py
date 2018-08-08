
from __future__ import division, unicode_literals, print_function, absolute_import

import os
from json import JSONDecoder
from collections import OrderedDict
import warnings

import numpy as np
import pytest

import podpac
from podpac.core.coordinate import Coordinate
from podpac.core.data.type import NumpyArray
from podpac.core.algorithm.algorithm import Arange
from podpac.core.pipeline.output import NoOutput, FileOutput, FTPOutput, S3Output
from podpac.core.pipeline.pipeline import Pipeline, PipelineNode, PipelineError

coords = Coordinate(lat=(0, 1, 10), lon=(0, 1, 10), order=['lat', 'lon'])
node = Arange()
node.execute(coords)

class TestPipeline(object):
    def test_load_from_file(self):
        path = os.path.join(os.path.abspath(podpac.__path__[0]), 'core', 'pipeline', 'test', 'test.json')
        pipeline = Pipeline(path)
        assert pipeline.path == path
        assert pipeline.definition['nodes']
        assert pipeline.definition['output']
        assert isinstance(pipeline.nodes['a'], podpac.core.algorithm.algorithm.Arange)
        assert isinstance(pipeline.output, FileOutput)

    def test_parse_no_nodes(self):
        s = '{ }'
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        with pytest.raises(PipelineError):
            Pipeline(d)

        s = '{"nodes": { } }'
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        with pytest.raises(PipelineError):
            Pipeline(d)

    def test_parse_node_invalid_node(self):
        # module does not exist
        s = '{"nodes": {"a": {"node": "nonexistent.Arbitrary"} } }'
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        with pytest.raises(PipelineError):
            Pipeline(d)

        # node does not exist in module
        s = '{"nodes": {"a": {"node": "core.Nonexistent"} } }'
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
            }
        }
        '''

        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d, warn=False)

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
            }
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
            }
        }
        '''
        
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d, warn=False)
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
            }
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
            }
        }
        '''

        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d, warn=False)
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
            }
        }
        '''

        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d, warn=False)
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
            }
        }
        '''

        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d, warn=False)
        assert pipeline.skip_evaluate == ['b', 'c']

    def test_parse_node_invalid_property(self):
        s = '''
        {
            "nodes": {
                "a": {
                    "node": "core.algorithm.algorithm.Arange",
                    "invalid_property": "value"
                }
            }
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
            "output": {"node": "a", "mode": "none"}
        }
        '''
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d)
        assert isinstance(pipeline.output, NoOutput)
        assert pipeline.output.node is pipeline.nodes['a']
        assert pipeline.output.name == 'a'

    def test_parse_output_file(self):
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "output": {
                "node": "a",
                "mode": "file",
                "format": "pickle",
                "outdir": "my_directory"
            }
        }
        '''
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d)
        assert isinstance(pipeline.output, FileOutput)
        assert pipeline.output.node is pipeline.nodes['a']
        assert pipeline.output.name == 'a'
        assert pipeline.output.format == 'pickle'
        assert pipeline.output.outdir == 'my_directory'

    def test_parse_output_s3(self):
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "output": {
                "node": "a",
                "mode": "s3",
                "user": "my_user",
                "bucket": "my_bucket"
            }
        }
        '''
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d)
        assert isinstance(pipeline.output, S3Output)
        assert pipeline.output.node is pipeline.nodes['a']
        assert pipeline.output.name == 'a'
        assert pipeline.output.user == 'my_user'
        assert pipeline.output.bucket == 'my_bucket'

    def test_parse_output_ftp(self):
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "output": {
                "node": "a",
                "mode": "ftp",
                "url": "my_url",
                "user": "my_user"
            }
        }
        '''
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d)
        assert isinstance(pipeline.output, FTPOutput)
        assert pipeline.output.node is pipeline.nodes['a']
        assert pipeline.output.name == 'a'
        assert pipeline.output.user == 'my_user'
        assert pipeline.output.url == 'my_url'
        # TODO password

    def test_parse_output_invalid_mode(self):
        # invalid mode
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "output": {"mode": "nonexistent_mode"}
        }
        '''
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        with pytest.raises(PipelineError):
            Pipeline(d, warn=False)

    def test_parse_output_implicit_mode(self):
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "output": {"node": "a"}
        }
        '''
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d)
        assert isinstance(pipeline.output, NoOutput)
        assert pipeline.output.node is pipeline.nodes['a']
        assert pipeline.output.name == 'a'

    def test_parse_output_nonexistent_node(self):
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "output": {
                "node": "b",
                "mode": "file",
                "format": "pickle",
                "outdir": "my_directory"
            }
        }
        '''
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        with pytest.raises(PipelineError):
            Pipeline(d)

    def test_parse_output_implicit_node(self):
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
            "output": {
                "mode": "none"
            }
        }
        '''
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        with pytest.warns(UserWarning, match="No output node provided"):
            pipeline = Pipeline(d)
        assert pipeline.output.node is pipeline.nodes['result']

    def test_parse_output_implicit(self):
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} }
        }
        '''
        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        with pytest.warns(UserWarning, match="No output node provided"):
            pipeline = Pipeline(d)
        assert isinstance(pipeline.output, NoOutput)
        assert pipeline.output.node is pipeline.nodes['a']
        assert pipeline.output.name == 'a'

    def test_unused_node_warning(self):
        s = '''
        {
            "nodes": {
                "a": { "node": "core.algorithm.algorithm.Arange" },
                "b": { "node": "core.algorithm.algorithm.Arange" }
            }
        }
        '''

        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        with pytest.warns(UserWarning, match="Unused pipeline node 'a'"):
            pipeline = Pipeline(d)

    def test_execute(self):
        s = '''
        {
            "nodes": {
                "a": {
                    "node": "core.algorithm.algorithm.Arange"
                }
            }
        }
        '''

        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d, warn=False)
        pipeline.execute(coords)

    def test_execute_output(self):
        path = os.path.join(os.path.abspath(podpac.__path__[0]), 'core', 'pipeline', 'test')

        s = '''
        {
            "nodes": {
                "a": {
                    "node": "core.algorithm.algorithm.Arange"
                }
            },
            "output": {
                "node": "a",
                "mode": "file",
                "format": "pickle",
                "outdir": "."
            }
        }
        '''

        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d)
        pipeline.execute(coords)
        assert pipeline.output.path is not None
        assert os.path.isfile(pipeline.output.path)
        os.remove(pipeline.output.path)

    def test_execute_params(self):
        s = '''
        {
            "nodes": {
                "source": {"node": "core.algorithm.algorithm.Arange"},
                "result": {        
                    "node": "Arithmetic",
                    "inputs": {
                        "A": "source"
                    },
                    "params": {
                        "eqn": "2 * A"
                    }
                }
            }
        }
        '''

        d = JSONDecoder(object_pairs_hook=OrderedDict).decode(s)
        pipeline = Pipeline(d, warn=False)
        
        a = Arange()
        aout = a.execute(coords)

        # no params argument
        pipeline.execute(coords)
        np.testing.assert_array_equal(2 * aout, pipeline.nodes['result'].output)
        
        # empty params argument
        pipeline.execute(coords, params={})
        np.testing.assert_array_equal(2 * aout, pipeline.nodes['result'].output)
        
        # None params argument
        pipeline.execute(coords, params=None)
        np.testing.assert_array_equal(2 * aout, pipeline.nodes['result'].output)
        
        # set params argument
        pipeline.execute(coords, params={'result': {'eqn': "3 * A"}})
        np.testing.assert_array_equal(3 * aout, pipeline.nodes['result'].output)
        
        # nonexistent node
        with pytest.raises(PipelineError):
            pipeline.execute(coords, params={'a': {'test': 0}})
            

class TestPipelineNode(object):
    def test_pipeline_node(self):
        path = os.path.join(os.path.abspath(podpac.__path__[0]), 'core', 'pipeline', 'test', 'test.json')
        node = PipelineNode(path=path)
        # node.execute(coords)