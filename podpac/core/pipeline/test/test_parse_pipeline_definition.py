
from __future__ import division, unicode_literals, print_function, absolute_import

import json
from collections import OrderedDict
import numpy as np
import traitlets as tl
import pytest

import podpac
from podpac.core.pipeline.pipeline import Pipeline, PipelineError, parse_pipeline_definition
from podpac.core.pipeline.output import NoOutput, FTPOutput, S3Output, FileOutput, ImageOutput

class TestParsePipelineDefinition(object):
    def test_empty(self):
        s = '{ }'
        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError, match="Pipeline definition requires 'nodes' property"):
            parse_pipeline_definition(d)

    def test_no_nodes(self):
        s = '{"nodes": { } }'
        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError, match="'nodes' property cannot be empty"):
            parse_pipeline_definition(d)

    def test_invalid_node(self):
        # module does not exist
        s = '{"nodes": {"a": {"node": "nonexistent.Arbitrary"} } }'
        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError, match='No module found'):
            parse_pipeline_definition(d)

        # node does not exist in module
        s = '{"nodes": {"a": {"node": "core.Nonexistent"} } }'
        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError, match="Node 'Nonexistent' not found"):
            parse_pipeline_definition(d)

    def test_datasource_source(self):
        # basic
        s = '''
        {
            "nodes": {
                "mydata": {
                    "node": "data.DataSource",
                    "source": "my_data_string"
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        assert output.node.source == "my_data_string"

        # not required
        s = '''
        {
            "nodes": {
                "mydata": {
                    "node": "data.DataSource"
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)

        # incorrect
        s = '''
        {
            "nodes": {
                "mydata": {
                    "node": "data.DataSource",
                    "attrs": {
                        "source": "my_data_string"
                    }
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError, match="The 'source' property cannot be in attrs"):
            parse_pipeline_definition(d)

    def test_datasource_lookup_source(self):
        # sub-node
        s = '''
        {
            "nodes": {
                "mydata": {
                    "node": "data.DataSource",
                    "source": "my_data_string"
                },
                "double": {
                    "node": "algorithm.Arithmetic",
                    "inputs": {"A": "mydata"},
                    "attrs": { "eqn": "2 * A" }
                },
                "mydata2": {
                    "node": "data.DataSource",
                    "lookup_source": "double.A.source"
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        assert output.node.source == 'my_data_string'

        # nonexistent node
        s = '''
        {
            "nodes": {
                "mydata": {
                    "node": "data.DataSource",
                    "source": "my_data_string"
                },
                "double": {
                    "node": "algorithm.Arithmetic",
                    "inputs": {"A": "mydata"},
                    "attrs": { "eqn": "2 * A" }
                },
                "mydata2": {
                    "node": "data.DataSource",
                    "lookup_source": "nonexistent.source"
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError, match="'mydata2' references nonexistent node/attribute"):
            parse_pipeline_definition(d)

        # nonexistent subattr
        s = '''
        {
            "nodes": {
                "mydata": {
                    "node": "data.DataSource",
                    "source": "my_data_string"
                },
                "double": {
                    "node": "algorithm.Arithmetic",
                    "inputs": {"A": "mydata"},
                    "attrs": { "eqn": "2 * A" }
                },
                "mydata2": {
                    "node": "data.DataSource",
                    "lookup_source": "double.nonexistent.source"
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError, match="'mydata2' references nonexistent node/attribute"):
            parse_pipeline_definition(d)

        # nonexistent subsubattr
        s = '''
        {
            "nodes": {
                "mydata": {
                    "node": "data.DataSource",
                    "source": "my_data_string"
                },
                "double": {
                    "node": "algorithm.Arithmetic",
                    "inputs": {"A": "mydata"},
                    "attrs": { "eqn": "2 * A" }
                },
                "mydata2": {
                    "node": "data.DataSource",
                    "lookup_source": "double.A.nonexistent"
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError, match="'mydata2' references nonexistent node/attribute"):
            parse_pipeline_definition(d)

        # in attrs (incorrect)
        s = '''
        {
            "nodes": {
                "mydata": {
                    "node": "data.DataSource",
                    "attrs": {
                        "lookup_source": "my_data_string"
                    }
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError, match="The 'lookup_source' property cannot be in attrs"):
            parse_pipeline_definition(d)

    def test_reprojected_source_lookup_source(self):
        # source doesn't work
        s = '''
        {
            "nodes": {
                "mysource": {
                    "node": "data.DataSource",
                    "source": "my_data_string"
                },
                "reprojected": {
                    "node": "data.ReprojectedSource",
                    "source": "mysource"
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(tl.TraitError):
            parse_pipeline_definition(d)

        # lookup_source
        s = '''
        {
            "nodes": {
                "mysource": {
                    "node": "data.DataSource",
                    "source": "my_data_string"
                },
                "reprojected": {
                    "node": "data.ReprojectedSource",
                    "lookup_source": "mysource"
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        assert output.node.source
        assert output.node.source.source == 'my_data_string'
        
        # lookup_source subattr
        s = '''
        {
            "nodes": {
                "mysource": {
                    "node": "data.DataSource",
                    "source": "my_data_string"
                },
                "double": {
                    "node": "algorithm.Arithmetic",
                    "inputs": {"A": "mysource"},
                    "attrs": { "eqn": "2 * A" }
                },
                "reprojected": {
                    "node": "data.ReprojectedSource",
                    "lookup_source": "double.A"
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        assert output.node.source
        assert output.node.source.source == "my_data_string"

        # nonexistent node/attribute references are tested in test_datasource_lookup_source

    def test_array_source(self):
        s = '''
        {
            "nodes": {
                "mysource": {
                    "node": "data.Array",
                    "source": [0, 1, 2]
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        np.testing.assert_array_equal(output.node.source, [0, 1, 2])

    def test_array_lookup_source(self):
        # source doesn't work
        s = '''
        {
            "nodes": {
                "a": {
                    "node": "data.Array",
                    "source": [0, 1, 2]
                },
                "b": {
                    "node": "data.Array",
                    "source": "a.source"
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(ValueError):
            parse_pipeline_definition(d)

        # lookup_source does work
        s = '''
        {
            "nodes": {
                "a": {
                    "node": "data.Array",
                    "source": [0, 1, 2]
                },
                "b": {
                    "node": "data.Array",
                    "lookup_source": "a.source"
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        np.testing.assert_array_equal(output.node.source, [0, 1, 2])

    def test_algorithm_inputs(self):
        # basic
        s = '''
        {
            "nodes": {
                "source1": {"node": "algorithm.Arange"},
                "source2": {"node": "algorithm.CoordData"},
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
        output = parse_pipeline_definition(d)

        assert isinstance(output.node.A, podpac.algorithm.Arange)
        assert isinstance(output.node.B, podpac.algorithm.CoordData)

        # sub-node
        s = '''
        {
            "nodes": {
                "mysource": {"node": "algorithm.Arange"},
                "double": {        
                    "node": "algorithm.Arithmetic",
                    "inputs": { "A": "mysource" },
                    "attrs": { "eqn": "2 * A" }
                },
                "quadruple": {
                    "node": "algorithm.Arithmetic",
                    "inputs": { "A": "double.A" },
                    "attrs": { "eqn": "2 * A" }
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)

        assert isinstance(output.node.A, podpac.algorithm.Arange)

        # nonexistent node/attribute references are tested in test_datasource_lookup_source

    def test_compositor_sources(self):
        # basic
        s = '''
        {
            "nodes": {
                "a": {"node": "algorithm.Arange"},
                "b": {"node": "algorithm.CoordData"},
                "c": {
                    "node": "compositor.OrderedCompositor",
                    "sources": ["a", "b"]
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        assert isinstance(output.node.sources[0], podpac.algorithm.Arange)
        assert isinstance(output.node.sources[1], podpac.algorithm.CoordData)

        # sub-node
        s = '''
        {
            "nodes": {
                "source1": {"node": "algorithm.Arange"},
                "source2": {"node": "algorithm.CoordData"},
                "double": {
                    "node": "algorithm.Arithmetic",
                    "inputs": { "A": "source1" },
                    "attrs": { "eqn": "2 * A" }
                },
                "c": {
                    "node": "compositor.OrderedCompositor",
                    "sources": ["double.A", "source2"]
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        assert isinstance(output.node.sources[0], podpac.algorithm.Arange)
        assert isinstance(output.node.sources[1], podpac.algorithm.CoordData)

        # nonexistent node/attribute references are tested in test_datasource_lookup_source

    def test_datasource_interpolation(self):
        s = '''
        {
            "nodes": {
                "mydata": {
                    "node": "data.DataSource",
                    "source": "my_data_string",
                    "interpolation": "nearest"
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        assert output.node.interpolation == "nearest"

        # not required
        s = '''
        {
            "nodes": {
                "mydata": {
                    "node": "data.DataSource"
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)

        # incorrect
        s = '''
        {
            "nodes": {
                "mydata": {
                    "node": "data.DataSource",
                    "attrs": {
                        "interpolation": "nearest"
                    }
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError, match="The 'interpolation' property cannot be in attrs"):
            parse_pipeline_definition(d)

    def test_compositor_interpolation(self):
        s = '''
        {
            "nodes": {
                "a": {
                    "node": "algorithm.Arange"
                },
                "b": {
                    "node": "algorithm.Arange"
                },
                "c": {
                    "node": "compositor.OrderedCompositor",
                    "sources": ["a", "b"],
                    "interpolation": "nearest"
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        assert output.node.interpolation == "nearest"


    def test_attrs(self):
        s = '''
        {
            "nodes": {
                "sm": {
                    "node": "datalib.smap.SMAP",
                    "attrs": {
                        "product": "SPL4SMGP"
                    }
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        assert output.node.product == "SPL4SMGP"

    def test_lookup_attrs(self):
        # attrs doesn't work
        s = '''
        {
            "nodes": {
                "a": {
                    "node": "algorithm.CoordData",
                    "attrs": { "coord_name": "lat" }
                },
                "b": {
                    "node": "algorithm.CoordData",
                    "attrs": { "coord_name": "a.coord_name" }
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        with pytest.raises(AssertionError):
            assert output.node.coord_name == 'lat'

        # but lookup_attrs does
        s = '''
        {
            "nodes": {
                "a": {
                    "node": "algorithm.CoordData",
                    "attrs": { "coord_name": "lat" }
                },
                "b": {
                    "node": "algorithm.CoordData",
                    "lookup_attrs": { "coord_name": "a.coord_name" }
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        assert output.node.coord_name == 'lat'

        # NOTE: no nodes currently have a Node as an attr
        # # lookup node directly (instead of a sub-attr)
        # s = '''
        # {
        #     "nodes": {
        #         "mysource": {
        #             "node": "data.DataSource"
        #         },
        #         "mynode": {
        #             "node": "MyNode",
        #             "lookup_attrs": {
        #                 "my_node_attr": "mysource"
        #             }
        #         }
        #     }
        # }
        # '''

        # d = json.loads(s, object_pairs_hook=OrderedDict)
        # output = parse_pipeline_definition(d)
        # assert isinstance(output.node.my_node_attr, DataSource)

        # nonexistent node/attribute references are tested in test_datasource_lookup_source

    def test_invalid_property(self):
        s = '''
        {
            "nodes": {
                "a": {
                    "node": "algorithm.Arange",
                    "invalid_property": "value"
                }
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError, match="node 'a' has unexpected property"):
            parse_pipeline_definition(d)

    def test_plugin(self):
        pass

    def test_parse_output_none(self):
        s = '''
        {
            "nodes": {"a": {"node": "algorithm.Arange"} },
            "output": {"node": "a", "mode": "none"}
        }
        '''
        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        assert isinstance(output, NoOutput)
        assert isinstance(output.node, podpac.algorithm.Arange)
        assert output.name == 'a'

    def test_parse_output_file(self):
        s = '''
        {
            "nodes": {"a": {"node": "algorithm.Arange"} },
            "output": {
                "node": "a",
                "mode": "file",
                "format": "pickle",
                "outdir": "my_directory"
            }
        }
        '''
        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        assert isinstance(output, FileOutput)
        assert isinstance(output.node, podpac.algorithm.Arange)
        assert output.name == 'a'
        assert output.format == 'pickle'
        assert output.outdir == 'my_directory'

    def test_parse_output_s3(self):
        s = '''
        {
            "nodes": {"a": {"node": "algorithm.Arange"} },
            "output": {
                "node": "a",
                "mode": "s3",
                "user": "my_user",
                "bucket": "my_bucket"
            }
        }
        '''
        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        assert isinstance(output, S3Output)
        assert isinstance(output.node, podpac.algorithm.Arange)
        assert output.name == 'a'
        assert output.user == 'my_user'
        assert output.bucket == 'my_bucket'

    def test_parse_output_ftp(self):
        s = '''
        {
            "nodes": {"a": {"node": "algorithm.Arange"} },
            "output": {
                "node": "a",
                "mode": "ftp",
                "url": "my_url",
                "user": "my_user"
            }
        }
        '''
        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        assert isinstance(output, FTPOutput)
        assert isinstance(output.node, podpac.algorithm.Arange)
        assert output.name == 'a'
        assert output.user == 'my_user'
        assert output.url == 'my_url'
        # TODO password

    def test_parse_output_image(self):
        s = '''
        {
            "nodes": {"a": {"node": "algorithm.Arange"} },
            "output": {
                "node": "a",
                "mode": "image"
            }
        }
        '''
        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        assert isinstance(output, ImageOutput)
        assert isinstance(output.node, podpac.algorithm.Arange)
        assert output.name == 'a'

    def test_parse_output_invalid_mode(self):
        # invalid mode
        s = '''
        {
            "nodes": {"a": {"node": "algorithm.Arange"} },
            "output": {"mode": "nonexistent_mode"}
        }
        '''
        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError, match='output has unexpected mode'):
            parse_pipeline_definition(d)

    def test_parse_output_implicit_mode(self):
        s = '''
        {
            "nodes": {"a": {"node": "algorithm.Arange"} },
            "output": {"node": "a"}
        }
        '''
        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        assert isinstance(output, NoOutput)
        assert isinstance(output.node, podpac.algorithm.Arange)
        assert output.name == 'a'

    def test_parse_output_nonexistent_node(self):
        s = '''
        {
            "nodes": {"a": {"node": "algorithm.Arange"} },
            "output": {
                "node": "b",
                "mode": "file",
                "format": "pickle",
                "outdir": "my_directory"
            }
        }
        '''
        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError, match="output' references nonexistent node"):
            parse_pipeline_definition(d)

    def test_parse_output_implicit_node(self):
        s = '''
        {
            "nodes": {
                "source1": {"node": "algorithm.Arange"},
                "source2": {"node": "algorithm.Arange"},
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
        output = parse_pipeline_definition(d)
        assert isinstance(output.node, podpac.algorithm.Arithmetic)

    def test_parse_output_implicit(self):
        s = '''
        {
            "nodes": {"a": {"node": "algorithm.Arange"} }
        }
        '''
        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        assert isinstance(output, NoOutput)
        assert isinstance(output.node, podpac.algorithm.Arange)
        assert output.name == 'a'

    def test_parse_custom_output(self):
        s = ''' {
            "nodes": {"a": {"node": "algorithm.Arange"} },
            "output": {
                "plugin": "podpac.core.pipeline.output",
                "output": "ImageOutput"
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        assert isinstance(output, ImageOutput)

        s = ''' {
            "nodes": {"a": {"node": "algorithm.Arange"} },
            "output": {
                "plugin": "podpac",
                "output": "core.pipeline.output.ImageOutput"
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        output = parse_pipeline_definition(d)
        assert isinstance(output, ImageOutput)

    def test_parse_custom_output_invalid(self):
        # no module
        s = ''' {
            "nodes": {"a": {"node": "algorithm.Arange"} },
            "output": {
                "plugin": "nonexistent_module",
                "output": "arbitrary"
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError, match="No module found"):
            parse_pipeline_definition(d)

        # module okay, but no such class
        s = ''' {
            "nodes": {"a": {"node": "algorithm.Arange"} },
            "output": {
                "plugin": "podpac.core.pipeline.output",
                "output": "Nonexistent"
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError, match="Output 'Nonexistent' not found"):
            parse_pipeline_definition(d)

        # module okay, class found, could not create
        s = ''' {
            "nodes": {"a": {"node": "algorithm.Arange"} },
            "output": {
                "plugin": "numpy",
                "output": "ndarray"
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        with pytest.raises(PipelineError, match='Could not create custom output'):
            parse_pipeline_definition(d)

        # module okay, class found, incorrect type
        s = ''' {
            "nodes": {"a": {"node": "algorithm.Arange"} },
            "output": {
                "plugin": "collections",
                "output": "OrderedDict"
            }
        }
        '''

        d = json.loads(s, object_pairs_hook=OrderedDict)
        m = "Custom output 'collections.OrderedDict' must subclass 'podpac.core.pipeline.output.Output'"
        with pytest.raises(PipelineError, match=m):
            parse_pipeline_definition(d)