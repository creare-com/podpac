
from __future__ import division, unicode_literals, print_function, absolute_import

import os
import json
from collections import OrderedDict
import warnings

import numpy as np
import pytest

import podpac
from podpac.core.algorithm.algorithm import Arange
from podpac.core.pipeline.pipeline import Pipeline, PipelineError
from podpac.core.pipeline.output import FileOutput

coords = podpac.Coordinates([[0, 1, 2], [10, 20, 30]], dims=['lat', 'lon'])
node = Arange()
node.eval(coords)

class TestPipeline(object):
    def test_init_path(self):
        path = os.path.join(os.path.abspath(podpac.__path__[0]), 'core', 'pipeline', 'test', 'test.json')
        pipeline = Pipeline(path=path)
        
        assert pipeline.json
        assert pipeline.definition
        assert pipeline.output

    def test_init_json(self):
        s = '''
        {
            "nodes": {
                "a": {
                    "node": "core.algorithm.algorithm.Arange"
                }
            }
        }
        '''

        pipeline = Pipeline(json=s)
        assert pipeline.json
        assert pipeline.definition
        assert pipeline.output

    def test_init_definition(self):
        s = '''
        {
            "nodes": {
                "a": {
                    "node": "core.algorithm.algorithm.Arange"
                }
            }
        }
        '''
        
        d = json.loads(s, object_pairs_hook=OrderedDict)

        pipeline = Pipeline(definition=d)
        assert pipeline.json
        assert pipeline.definition
        assert pipeline.output

    def test_init_error(self):
        pass

    def test_eval(self):
        s = '''
        {
            "nodes": {
                "a": {
                    "node": "core.algorithm.algorithm.Arange"
                }
            }
        }
        '''

        pipeline = Pipeline(json=s)
        pipeline.eval(coords)
        
        pipeline.units
        pipeline.dtype
        pipeline.cache_type
        pipeline.style

    def test_eval_output(self):
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

        pipeline = Pipeline(json=s)
        pipeline.eval(coords)
        assert pipeline.output.path is not None
        assert os.path.isfile(pipeline.output.path)
        os.remove(pipeline.output.path)

    def test_eval_no_output(self):
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

        pipeline = Pipeline(json=s, do_write_output=False)
        pipeline.eval(coords)
        if pipeline.output.path is not None and os.path.isfile(pipeline.output.path):
            os.remove(pipeline.output.path)
        assert pipeline.output.path is None