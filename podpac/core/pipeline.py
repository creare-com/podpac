from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
import os
import json
import copy
import importlib
import inspect
import re
import numpy as np
import traitlets as tl

from podpac.core.coordinate import Coordinate
from podpac.core.node import Node
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

class NoOutput(Output):
    def write(self):
        pass

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
    vmax = tl.CFloat(allow_none=True, default_value=np.nan)
    vmin = tl.CFloat(allow_none=True, default_value=np.nan)
    image = tl.Bytes()

    def write(self):
        try:
            self.image = self.node.get_image(format=self.format,
                                             vmin=self.vmin,
                                             vmax=self.vmax)
        except:
            pass

class Pipeline(tl.HasTraits):
    path = tl.Unicode(allow_none=True, help="Path to the JSON definition")
    definition = tl.Instance(OrderedDict, help="pipeline definition")
    nodes = tl.Instance(OrderedDict, help="pipeline nodes")
    params = tl.Dict(trait=tl.Instance(OrderedDict), help="default parameter overrides")
    outputs = tl.List(trait=tl.Instance(Output), help="pipeline outputs")
    skip_evaluate = tl.List(trait=tl.Unicode, help="nodes to skip")
    implicit_pipeline_evaluation = tl.Bool(False)
    
    def __init__(self, source, implicit_pipeline_evaluation=False):
        self.implicit_pipeline_evaluation = implicit_pipeline_evaluation
        if isinstance(source, dict):
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
        module_root = d.get('plugin', 'podpac')
        node_string = '%s.%s' % (module_root, d['node'])
        module_name, node_name = node_string.rsplit('.', 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            raise PipelineError("No module found '%s'" % module_name)
        try:
            node_class = getattr(module, node_name)
        except AttributeError:
            raise PipelineError("Node '%s' not found in module '%s'" % (
                node_name, module_name))
        
        # parse and configure kwargs
        kwargs = {}
        whitelist = ['node', 'attrs', 'params', 'evaluate', 'plugin']
        
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
            if PipelineNode in parents:
                if 'pipeline_json' in d:
                    kwargs['pipeline_json'] = d['pipeline_json']
                    whitelist.append('pipeline_json')
              
            if DataSource not in parents and\
                   Compositor not in parents and\
                   Algorithm not in parents and\
                   PipelineNode not in parents:
                raise PipelineError("node '%s' is not a DataSource, Compositor, or Algorithm" % name)
        except KeyError as e:
            raise PipelineError(
                "node '%s' definition references nonexistent node '%s'" % (name, e))
        
        kwargs['implicit_pipeline_evaluation'] = self.implicit_pipeline_evaluation
       
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
        kwargs = {}
        # modes
        if 'mode' not in d:
            raise PipelineError("output definition requires 'mode' property")
        elif d['mode'] == 'none':
            output_class = NoOutput
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
            kwargs = {'format': d.get('image', 'png'),
                      'vmin': d.get('vmin', np.nan),
                      'vmax': d.get('vmax', np.nan)}
        else:
            raise PipelineError("output definition has unexpected mode '%s'" % d['mode'])

        # node references
        if 'node' in d and 'nodes' in d:
            raise PipelineError("output definition expects 'node' or 'nodes' property, not both")
        elif 'node' in d:
            refs = [d['node']]
        elif 'nodes' in d:
            refs = d['nodes']
        elif d['mode'] == 'none':
            nodes = self.nodes
            refs = list(nodes.keys()) 
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
        if params is None:
            params = {}

        for node in params:
            if node not in self.nodes:
                raise PipelineError(
                    "params reference nonexistent node '%s'" % node)

    def execute(self, coordinates, params=None):
        if params is None:
            params = {}

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

class PipelineNode(Node):
    """
    Wraps a pipeline into a Node.

    Todo: shape, native_coordinates, etc
    """

    path = tl.Unicode(allow_none=True, help="Path to the JSON definition")
    implicit_pipeline_evaluation = tl.Bool(False)

    output_node = tl.Unicode()
    @tl.default('output_node')
    def _output_node_default(self):
        return self.definition['outputs'][0]['nodes'][0]

    pipeline_json = tl.Unicode(help="pipeline json definition")
    @tl.default('pipeline_json')
    def _pipeline_json_default(self):
        with open(self.path) as f:
            pipeline_json = f.read()
        return pipeline_json

    definition = tl.Instance(OrderedDict, help="pipeline definition")
    @tl.default('definition')
    def _definition_default(self):
        return json.loads(self.pipeline_json, object_pairs_hook=OrderedDict)

    source_pipeline = tl.Instance(Pipeline, allow_none=False)
    @tl.default('source_pipeline')
    def _source_pipeline_default(self):
        return Pipeline(source=self.definition,
                        implicit_pipeline_evaluation=self.implicit_pipeline_evaluation)

    @tl.default('native_coordinates')
    def get_native_coordinates(self):
        return self.source_pipeline.nodes[self.output_node].native_coordinates

    def execute(self, coordinates, params=None, output=None):
        self.coordinates = coordinates
        self.params = params
        self.output = output

        self.source_pipeline.execute(coordinates, params)
        
        out = self.source_pipeline.nodes[self.output_node].output
        if self.output is None:
            self.output = out
        else:
            self.output[:] = out
        
        return self.output

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
                if input_node is not None:
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

    output = OrderedDict()
    #output['mode'] = 'file'
    #output['format'] = 'pickle'
    #output['outdir'] = os.path.join(os.getcwd(), 'out')
    #output['nodes'] = [refs[-1]]

    output['mode'] = 'image'
    output['format'] = 'png'
    output['vmin'] = -1.2
    output['vmax'] = 1.2
    output['nodes'] = [refs[-1]]


    d = OrderedDict()
    d['nodes'] = OrderedDict(zip(refs, definitions))
    d['outputs'] = [output]
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
    print(list(pipeline.nodes.values())[-1].pipeline_json)
    rebuilt_pipeline = Pipeline(list(pipeline.nodes.values())[-1].pipeline_definition)
    
    if args.dry_run:
        pipeline.check_params(params)
    else:
        pipeline.execute(coords, params)

    print('Done')
