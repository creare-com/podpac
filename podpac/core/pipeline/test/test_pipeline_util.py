
from __future__ import division, unicode_literals, print_function, absolute_import

import json
from collections import OrderedDict
import numpy as np
import pytest

import podpac
from podpac.core.pipeline.pipeline import Pipeline
from podpac.core.pipeline.output import NoOutput, FTPOutput, S3Output, FileOutput, ImageOutput
from podpac.core.pipeline.util import PipelineError
from podpac.core.pipeline.util import parse_pipeline_definition

class TestParsePipelineDefinition(object):
    def test_empty(self):
        s = '{ }'
        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError):
            parse_pipeline_definition(d)

    def test_no_nodes(self):
        s = '{"nodes": { } }'
        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError):
            parse_pipeline_definition(d)

    def test_invalid_node(self):
        # module does not exist
        s = '{"nodes": {"a": {"node": "nonexistent.Arbitrary"} } }'
        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError):
            parse_pipeline_definition(d)

        # node does not exist in module
        s = '{"nodes": {"a": {"node": "core.Nonexistent"} } }'
        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError):
            parse_pipeline_definition(d)

    def test_algorithm_inputs(self):
        # translate node references
        s = '''
        {
            "nodes": {
                "source1": {"node": "core.algorithm.algorithm.Arange"},
                "source2": {"node": "core.algorithm.algorithm.Arange"},
                "result": {        
                    "node": "algorithm.Arithmetic",
                    "inputs": {
                        "A": "source1",
                        "B": "source2"
                    },
                    "attrs": {
                        "eqn": "A + B"
                    }
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        nodes, output = parse_pipeline_definition(d)

        assert nodes['result'].A is nodes['source1']
        assert nodes['result'].B is nodes['source2']

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
                    },
                    "attrs": {
                        "eqn": "A + B"
                    }
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError):
            parse_pipeline_definition(d)

    def test_compositor_sources(self):
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
        
        d = json.loads(s, object_pairs_hook=OrderedDict)
        nodes, output = parse_pipeline_definition(d)
        assert nodes['c'].sources[0] is nodes['a']
        assert nodes['c'].sources[1] is nodes['b']

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
        
        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError):
            parse_pipeline_definition(d)

    def test_attrs(self):
        s = '''
        {
            "nodes": {
                "sm": {
                    "node": "datalib.smap.SMAP",
                    "attrs": {
                        "product": "SPL4SMGP",
                        "interpolation": "bilinear"
                    }
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        nodes, output = parse_pipeline_definition(d)
        assert nodes['sm'].product == "SPL4SMGP"
        assert nodes['sm'].interpolation == "bilinear"

    def test_invalid_property(self):
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

        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError):
            parse_pipeline_definition(d)

    def test_plugin(self):
        pass

    def test_parse_output_none(self):
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "output": {"node": "a", "mode": "none"}
        }
        '''
        d = json.loads(s, object_pairs_hook=OrderedDict)
        nodes, output = parse_pipeline_definition(d)
        assert isinstance(output, NoOutput)
        assert output.node is nodes['a']
        assert output.name == 'a'

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
        d = json.loads(s, object_pairs_hook=OrderedDict)
        nodes, output = parse_pipeline_definition(d)
        assert isinstance(output, FileOutput)
        assert output.node is nodes['a']
        assert output.name == 'a'
        assert output.format == 'pickle'
        assert output.outdir == 'my_directory'

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
        d = json.loads(s, object_pairs_hook=OrderedDict)
        nodes, output = parse_pipeline_definition(d)
        assert isinstance(output, S3Output)
        assert output.node is nodes['a']
        assert output.name == 'a'
        assert output.user == 'my_user'
        assert output.bucket == 'my_bucket'

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
        d = json.loads(s, object_pairs_hook=OrderedDict)
        nodes, output = parse_pipeline_definition(d)
        assert isinstance(output, FTPOutput)
        assert output.node is nodes['a']
        assert output.name == 'a'
        assert output.user == 'my_user'
        assert output.url == 'my_url'
        # TODO password

    def test_parse_output_image(self):
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "output": {
                "node": "a",
                "mode": "image"
            }
        }
        '''
        d = json.loads(s, object_pairs_hook=OrderedDict)
        nodes, output = parse_pipeline_definition(d)
        assert isinstance(output, ImageOutput)
        assert output.node is nodes['a']
        assert output.name == 'a'

    def test_parse_output_invalid_mode(self):
        # invalid mode
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "output": {"mode": "nonexistent_mode"}
        }
        '''
        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError):
            parse_pipeline_definition(d)

    def test_parse_output_implicit_mode(self):
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "output": {"node": "a"}
        }
        '''
        d = json.loads(s, object_pairs_hook=OrderedDict)
        nodes, output = parse_pipeline_definition(d)
        assert isinstance(output, NoOutput)
        assert output.node is nodes['a']
        assert output.name == 'a'

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
        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError):
            parse_pipeline_definition(d)

    def test_parse_output_implicit_node(self):
        s = '''
        {
            "nodes": {
                "source1": {"node": "core.algorithm.algorithm.Arange"},
                "source2": {"node": "core.algorithm.algorithm.Arange"},
                "result": {        
                    "node": "algorithm.Arithmetic",
                    "inputs": {
                        "A": "source1",
                        "B": "source2"
                    },
                    "attrs": {
                        "eqn": "A + B"
                    }
                }
            },
            "output": {
                "mode": "none"
            }
        }
        '''
        d = json.loads(s, object_pairs_hook=OrderedDict)
        nodes, output = parse_pipeline_definition(d)
        assert output.node is nodes['result']

    def test_parse_output_implicit(self):
        s = '''
        {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} }
        }
        '''
        d = json.loads(s, object_pairs_hook=OrderedDict)
        nodes, output = parse_pipeline_definition(d)
        assert isinstance(output, NoOutput)
        assert output.node is nodes['a']
        assert output.name == 'a'

    def test_parse_custom_output(self):
        s = ''' {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "output": {
                "plugin": "podpac.core.pipeline.output",
                "output": "ImageOutput"
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        nodes, output = parse_pipeline_definition(d)
        assert isinstance(output, ImageOutput)

        s = ''' {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "output": {
                "plugin": "podpac",
                "output": "core.pipeline.output.ImageOutput"
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        nodes, output = parse_pipeline_definition(d)
        assert isinstance(output, ImageOutput)

    def test_parse_custom_output_invalid(self):
        # no module
        s = ''' {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "output": {
                "plugin": "nonexistent_module",
                "output": "arbitrary"
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError):
            parse_pipeline_definition(d)

        # module okay, but no such class
        s = ''' {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "output": {
                "plugin": "podpac.core.pipeline.output",
                "output": "Nonexistent"
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError):
            parse_pipeline_definition(d)

        # module okay, class found, could not create
        s = ''' {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "output": {
                "plugin": "numpy",
                "output": "ndarray"
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError):
            parse_pipeline_definition(d)

        # module okay, class found, incorrect type
        s = ''' {
            "nodes": {"a": {"node": "core.algorithm.algorithm.Arange"} },
            "output": {
                "plugin": "collections",
                "output": "OrderedDict"
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError):
            parse_pipeline_definition(d)
        
    def test_unused_node_warning(self):
        s = '''
        {
            "nodes": {
                "a": { "node": "core.algorithm.algorithm.Arange" },
                "b": { "node": "core.algorithm.algorithm.Arange" }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.warns(UserWarning, match="Unused pipeline node 'a'"):
            parse_pipeline_definition(d)