from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
import os
import json
import copy
import importlib
import inspect
import re
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

    def write(self):
        raise NotImplementedError

class FileOutput(Output):
    outdir = tl.Unicode()
    format = tl.CaselessStrEnum(values=['pickle', 'geotif', 'png'], default='pickle')

    def write(self):
        self.node.write(self.name, outdir=self.outdir, format=self.format)

class FTPOutput(Output):
    url = tl.Unicode()
    user = tl.Unicode()

class AWSOutput(Output):
    user = tl.Unicode()
    bucket = tl.Unicode()

class ImageOutput(Output):
    format = tl.CaselessStrEnum(values=['png'], default='png')
    image = tl.Bytes()

    def write(self):
        self.image = self.node.get_image(format=self.format)

class Pipeline(tl.HasTraits):
    path = tl.Unicode(allow_none=True, help="Path to the JSON definition")
    definition = tl.Instance(OrderedDict, help="pipeline definition")
    nodes = tl.Instance(OrderedDict, help="pipeline nodes")
    params = tl.Dict(trait=tl.Instance(OrderedDict), help="default parameter overrides")
    outputs = tl.List(trait=tl.Instance(Output), help="pipeline outputs")
    skip_evaluate = tl.List(trait=tl.Unicode, help="nodes to skip")
    
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
        whitelist = ['node', 'attrs', 'params', 'evaluate']
        
        try:
            parents = inspect.getmro(node_class)
            
            if DataSource in parents:
                if 'source' in d:
                    kwargs['source'] = d['source']
                    whitelist.append('source')
            if Compositor in parents:
                if 'sources' in d:
                    kwargs['sources'] = [self.nodes[source] for source in d['sources']]
                    whitelist.append('sources')
            if Algorithm in parents:
                if 'inputs' in d:
                    kwargs.update({k:self.nodes[v] for k, v in d['inputs'].items()})
                    whitelist.append('inputs')
            if DataSource not in parents and\
                   Compositor not in parents and\
                   Algorithm not in parents:
                raise PipelineError("node '%s' is not a DataSource, Compositor, or Algorithm" % name)
        except KeyError as e:
            raise PipelineError(
                "node '%s' definition references nonexistent node '%s'" % (name, e))
        
        kwargs['implicit_pipeline_evaluation'] = False
       
        if 'params' in d:
            kwargs['params'] = d['params']

        if 'attrs' in d:
            kwargs.update(d['attrs'])

        if d.get('evaluate') is False:
            self.skip_evaluate.append(name)

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
        elif d['mode'] == 'image':
            output_class = ImageOutput
            kwargs = {'format': d.get('image', 'png')}
        else:
            raise PipelineError("output definition has unexpected mode '%s'" % d['mode'])

        # node references
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
            for ref in d.get('sources', []) + list(d.get('inputs', {}).values()):
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
            if key in self.skip_evaluate:
                continue

            node = self.nodes[key]

            d = copy.deepcopy(self.params[key])
            d.update(params.get(key, OrderedDict()))
            
            if node.evaluated_coordinates == coordinates and node.params == d:
                continue
            
            print("executing node", key)
            node.execute(coordinates, params=d)

        for output in self.outputs:
            output.write()

def make_pipeline_definition(main_node):
    """
    Make a pipeline definition, including the flattened node definitions and a
    default file output for the input node.
    """

    nodes = []
    refs = []
    definitions = []

    def add_node(node):
        if node in nodes:
            return refs[nodes.index(node)]

        # get definition
        d = node.definition
        
        # replace nodes with references, adding nodes depth first
        if 'inputs' in d:
            for key, input_node in d['inputs'].items():
                d['inputs'][key] = add_node(input_node)
        if 'sources' in d:
            for i, source_node in enumerate(d['sources']):
                d['sources'][i] = add_node(source_node)

        # unique ref
        ref = node.base_ref
        if ref in refs:
            if re.search('_[1-9][0-9]*$', ref):
                ref, i = ref.rsplit('_', 1)
                i = int(i)
            else:
                i = 0
            ref = '%s_%d' % (ref, i+1)
        
        nodes.append(node)
        refs.append(ref)
        definitions.append(d)

        return ref

    add_node(main_node)

    d = OrderedDict()
    d['nodes'] = OrderedDict(zip(refs, definitions))
    d['outputs'] = OrderedDict()
    d['outputs']['mode'] = 'file'
    d['outputs']['format'] = 'cPickle'
    d['outputs']['outdir'] = os.path.join(os.getcwd(), 'out')
    d['outputs']['nodes'] = [refs[-1]]
    return d

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
        lat=[43.759843545782765, 43.702536630730286, 64],
        lon=[-72.3940658569336, -72.29999542236328, 32],
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

    print('\nrebuilt pipeline definition:')
    print(pipeline.nodes.values()[-1].pipeline_json)
    
    if args.dry_run:
        pipeline.check_params(params)
    else:
        pipeline.execute(coords, params)

    print('Done')
