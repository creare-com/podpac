from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
import json
import copy
import importlib
import inspect
import warnings
import numpy as np
import traitlets as tl
import matplotlib.colors, matplotlib.cm
import matplotlib.pyplot as plt
from pint import UnitRegistry
ureg = UnitRegistry()

from podpac.core.coordinate import Coordinate
from podpac.core.node import Node, UnitsDataArray
from podpac.core.data.data import DataSource
from podpac.core.algorithm.algorithm import Algorithm
from podpac.core.compositor import Compositor

class PipelineError(Exception):
    pass

class Output(tl.HasTraits):
    node = tl.Instance(Node)
    name = tl.Unicode()

    def save(self):
        raise NotImplementedError

class FileOutput(Output):
    outdir = tl.Unicode()
    format = tl.CaselessStrEnum(values=['pickle', 'geotif', 'png'], default='pickle')

    def save(self):
        self.node.write(self.name, outdir=self.outdir, format=self.format)

class FTPOutput(Output):
    url = tl.Unicode()
    user = tl.Unicode()

class AWSOutput(Output):
    user = tl.Unicode()
    bucket = tl.Unicode()

class Pipeline(tl.HasTraits):
    path = tl.Unicode(allow_none=True, help="Path to the JSON definition")
    definition = tl.Dict(help="pipeline definition")
    nodes = tl.Dict(trait=tl.Instance(Node), help="pipeline nodes")
    params = tl.Dict(trait=tl.Dict(), help="default parameter overrides")
    outputs = tl.List(trait=tl.Instance(Output), help="pipeline outputs")
    
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
            self.params[key] = d.get('params', OrderedDict())

        # parse outputs
        self.outputs = []
        for d in self.definition['outputs']:
            self.outputs += self.parse_output(d)

        # check execution graph for unused nodes
        self.check_execution_graph()

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
        kwargs = {}
        whitelist = ['node', 'attrs', 'params']
        
        try:
            parents = inspect.getmro(node_class)
            
            if DataSource in parents:
                if 'source' in d:
                    kwargs['source'] = d['source']
                    whitelist.append('source')
            elif Compositor in parents:
                if 'sources' in d:
                    kwargs['sources'] = [self.nodes[source] for source in d['sources']]
                    whitelist.append('sources')
            elif Algorithm in parents:
                if 'inputs' in d:
                    kwargs.update({k:self.nodes[v] for k, v in d['inputs'].items()})
                    whitelist.append('inputs')
            else:
                raise PipelineError("node '%s' is not a DataSource, Compositor, or Algorithm" % name)
        except KeyError as e:
            raise PipelineError(
                "node '%s' definition references nonexistent node '%s'" % (name, e))
        
        kwargs['implicit_pipeline_evaluation'] = False
       
        if 'params' in d:
            kwargs['params'] = d['params']
            
        if 'attrs' in d:
            kwargs.update(d['attrs'])

        for key in d:
            if key not in whitelist:
                raise PipelineError("node '%s' definition has unexpected property '%s'" % (name, key))

        # return node info
        return node_class(**kwargs)

    def parse_output(self, d):
        # modes
        if 'mode' not in d:
            raise PipelineError("output definition requires 'mode' property")
        elif d['mode'] == 'file':
            output_class = FileOutput
            kwargs = {'outdir': d.get('outdir'), 'format': d['format']}
        elif d['mode'] == 'ftp':
            output_class = FTPOutput
            kwargs = {'url': d['url'], 'user': d['user']}
        elif d['mode'] == 'aws':
            output_class = AWSOutput
            kwargs = {'user': d['user'], 'bucket': d['bucken']}
        else:
            raise PipelineError("output definition has unexpected mode '%s'" % d['mode'])

        # node rerfences
        if 'node' in d and 'nodes' in d:
            raise PipelineError("output definition expects 'node' or 'nodes' property, not both")
        elif 'node' in d:
            refs = [d['node']]
        elif 'nodes' in d:
            refs = d['nodes']
        else:
            raise PipelineError("output definition requires 'node' or 'nodes' property")
        
        # nodes
        try:
            nodes = [self.nodes[ref] for ref in refs]
        except KeyError as e:
            raise PipelineError("output definition references nonexistent node '%s'" % (e))

        # outputs
        return [output_class(node=node, name=ref, **kwargs) for ref, node in zip(refs, nodes)]

    def check_execution_graph(self):
        used = {ref:False for ref in self.nodes}
        
        def f(base_ref):
            if used[base_ref]: return

            used[base_ref] = True

            d = self.definition['nodes'][base_ref]
            for ref in d.get('sources', []) + d.get('inputs', {}).values():
                f(ref)

        for output in self.outputs:
            f(output.name)

        for ref in self.nodes:
            if not used[ref]:
                raise PipelineError("Unused node '%s'" % ref)

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
            d.update(params.get(key, OrderedDict()))
            
            if node.evaluated_coordinates == coordinates and node.params == d:
                continue
            
            node.execute(coordinates, params=d)

        for output in self.outputs:
            output.save()

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

    print('Done')
