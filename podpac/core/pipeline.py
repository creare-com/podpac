from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
import json
import copy
import importlib
import warnings
import numpy as np
import traitlets as tl
import matplotlib.colors, matplotlib.cm
import matplotlib.pyplot as plt
from pint import UnitRegistry
ureg = UnitRegistry()

from podpac.core.coordinate import Coordinate
from podpac.core.node import Node, UnitsDataArray

class PipelineError(Exception):
    pass

class Output(tl.HasTraits):
    pass

class Pipeline():
    path = tl.Unicode(allow_none=True, help="Path to the JSON definition")
    definition = tl.Dict(allow_none=False, help="pipeline definition")
    nodes = tl.Dict(trait=tl.Instance(Node), allow_none=False, help="pipeline nodes")
    params = tl.Dict(trait=tl.Dict(), allow_none=False, help="default parameter overrides")
    outputs = tl.List(trait=tl.Instance(Output), allow_none=False, help="pipeline outputs")
    
    def __init__(self, source):
        if type(source) is dict:
            self.definition = source
        
        else:
            self.path = source
            with open(self.path) as f:
                self.definition = json.load(f, object_pairs_hook=OrderedDict)

        # parse nodes and default params
        self.nodes = OrderedDict()
        self.params = {}
        for key, d in self.definition['nodes'].items():
            key = key.encode()
            self.nodes[key] = self.parse_node(key, d)
            self.params[key] = d.get('params', {})

        # parse outputs
        self.outputs = []
        for d in self.definition['outputs']:
            self.outputs.append(self.parse_output(d))

    def parse_node(self, name, d):
        # get node class
        node_string = d['node']
        if '.' in node_string:
            submodule_name, node_name = node_string.rsplit('.', 1)
            module_name = 'podpac.%s' % submodule_name
        else:
            module_name = 'podpac'
            node_name = node_string
        module = importlib.import_module(module_name)
        node_class = getattr(module, node_name)
        
        # parse and configure kwargs
        skip = ['node', 'params', 'sources', 'inputs']
        kwargs = {k:v for k, v in d.items() if k not in skip}
        kwargs['implicit_pipeline_evaluation'] = False
        
        try:
            if 'sources' in d:
                kwargs['sources'] = [self.nodes[source] for source in d['sources']]
            if 'inputs' in d:
                kwargs.update({k:self.nodes[v] for k, v in d['inputs'].items()})
        except KeyError as e:
            raise PipelineError(
                "node '%s' definition references nonexistent node '%s'" % (name, e))

        # return node info
        return node_class(**kwargs)

    def parse_output(self, d):
        raise NotImplementedError

    def info(self):
        raise NotImplementedError

    def check_params(self, params):
        for node in params:
            if node not in self.nodes:
                raise PipelineError(
                    "params reference nonexistent node '%s'" % node)

    def execute(self, coordinates, params):
        self.check_params(params)

        for key in self.nodes:
            node = self.nodes[key]
            d = copy.deepcopy(self.params[key])
            d.update(params.get(key, {}))
            
            if node.evaluated_coordinates == coordinates and node.params == d:
                continue
            
            node.execute(coordinates, params=d)

if __name__ == '__main__':
    import argparse
    import os
    from collections import defaultdict
    import podpac
    
    def parse_param(item):
        try:
            key, value = item.split('=')
            layer, param = key.split('.')
        except:
            raise ValueError("Invalid params argument '%s', "
                             "expected <layer>.<param>=<value>" % item)
        
        try:
            value = json.loads(value)
        except ValueError:
            pass # leave as string

        return layer, param, value

    def parse_params(l):
        if len(l) == 1 and os.path.isfile(l[0]):
            with open(l) as f:
                d = json.load(f)

        else:
            d = defaultdict(dict)
            for item in l:
                layer, param, value = parse_param(item)
                d[layer][param] = value
            d = dict(d)

        return d

    podpac_path = os.path.abspath(podpac.__path__[0])
    test_pipeline = os.path.join(podpac_path, 'core', 'test', 'test.json')

    parser = argparse.ArgumentParser()
    parser.add_argument('pipeline', nargs='?', default=test_pipeline,
                        help='path to JSON pipeline definition')
    parser.add_argument('--params', type=str, nargs='+', default=[])
    parser.add_argument('-d', '--dry-run', action='store_true')
    args = parser.parse_args()

    # TODO coordinate arguments and coordinate file path argument
    coords = Coordinate(
        lat=[45., 66., 50],
        lon=[-80., -70., 20],
        time='2015-04-11T06:00:00',
        order=['lat', 'lon', 'time'])
    params = parse_params(args.params)

    pipeline = Pipeline(args.pipeline)

    print('\npipeline path      \t', pipeline.path)
    print('\npipeline definition\t', pipeline.definition)
    print('\npipeline nodes     \t', pipeline.nodes)
    print('\npipeline params    \t', pipeline.params)
    print('\npipeline outputs   \t', pipeline.outputs)
    print()
    print('\ncoords\t', coords)
    print('\nparams\t', params)
    
    if args.dry_run:
        pipeline.check_params(params)
    else:
        pipeline.execute(coords, params)
