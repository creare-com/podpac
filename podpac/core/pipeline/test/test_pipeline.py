
from __future__ import division, unicode_literals, print_function, absolute_import

import os
import json
from collections import OrderedDict
import warnings

import numpy as np
import pytest

import podpac
from podpac.core.algorithm.algorithm import Arange
from podpac.core.pipeline.pipeline import Pipeline
from podpac.core.pipeline.output import FileOutput
from podpac.core.pipeline.util import PipelineError

coords = podpac.Coordinates([[0, 1, 2], [10, 20, 30]], dims=['lat', 'lon'])
node = Arange()
node.eval(coords)

class TestPipeline(object):
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

        d = json.loads(s, object_pairs_hook=OrderedDict)
        pipeline = Pipeline(definition=d)
        pipeline.eval(coords)

        pipeline.native_coordinates
        pipeline.evaluated
        pipeline.units
        pipeline.dtype
        pipeline.cache_type
        pipeline.interpolation
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

        d = json.loads(s, object_pairs_hook=OrderedDict)
        pipeline = Pipeline(definition=d)
        pipeline.eval(coords)
        assert pipeline.pipeline_output.path is not None
        assert os.path.isfile(pipeline.pipeline_output.path)
        os.remove(pipeline.pipeline_output.path)

    def test_eval_from_file(self):
        path = os.path.join(os.path.abspath(podpac.__path__[0]), 'core', 'pipeline', 'test', 'test.json')
        pipeline = Pipeline(path=path)
        pipeline.eval(coords)

        assert pipeline.definition['nodes']
        assert pipeline.definition['output']
        assert isinstance(pipeline.nodes['a'], podpac.core.algorithm.algorithm.Arange)
        assert isinstance(pipeline.pipeline_output, FileOutput)
        assert pipeline.pipeline_output.path is not None
        assert os.path.isfile(pipeline.pipeline_output.path)
        os.remove(pipeline.pipeline_output.path)

    def test_eval_from_json(self):
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