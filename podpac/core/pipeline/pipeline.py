"""
Pipeline Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict, defaultdict
import os
import json
import copy
import importlib
import inspect
import warnings

import numpy as np
import traitlets as tl

from podpac.core.coordinate import Coordinate
from podpac.core.node import Node, NodeException
from podpac.core.data.data import DataSource
from podpac.core.algorithm.algorithm import Algorithm
from podpac.core.compositor import Compositor

from podpac.core.pipeline.output import Output, NoOutput, FileOutput, ImageOutput
from podpac.core.pipeline.util import make_pipeline_definition

class PipelineError(NodeException):
    """Summary
    """

    pass

class Pipeline(tl.HasTraits):
    """Summary

    Attributes
    ----------
    definition : TYPE
        Description
    implicit_pipeline_evaluation : TYPE
        Description
    nodes : TYPE
        Description
    output : list
        Description
    params : dict
        Description
    path : TYPE
        Description
    skip_evaluate : TYPE
        Description
    """

    path = tl.Unicode(allow_none=True, help="Path to the JSON definition")
    definition = tl.Instance(OrderedDict, help="pipeline definition")
    nodes = tl.Instance(OrderedDict, help="pipeline nodes")
    params = tl.Dict(trait=tl.Instance(OrderedDict), help="default parameter overrides")
    output = tl.Instance(Output, help="pipeline output")
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


        # parse node definitions and default params
        self.nodes = OrderedDict()
        self.params = {}
        
        if 'nodes' not in self.definition:
            raise PipelineError("Pipeline definition requires 'nodes' property")

        if len(self.definition['nodes']) == 0:
            raise PipelineError("Pipeline definition 'nodes' property cannot be empty")
        
        for key, d in self.definition['nodes'].items():
            self.nodes[key] = self._parse_node_definition(key, d)
            self.params[key] = d.get('params', OrderedDict())

        # parse output definition
        self.output = self._parse_output_definition(self.definition.get('output'))

        # check execution graph for unused nodes
        self._check_execution_graph()

    def _parse_node_definition(self, name, d):
        """Summary

        Parameters
        ----------
        name : TYPE
            Description
        d : TYPE
            Description

        Returns
        -------
        TYPE
            Description

        Raises
        ------
        PipelineError
            Description
        """

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
            raise PipelineError("node '%s' definition references nonexistent node '%s'" % (name, e))

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

    def _parse_output_definition(self, d):
        if d is None:
            d = {}

        # node (uses last node by default)
        if 'node' in d:
            name = d['node']
        else:
            name = list(self.nodes.keys())[-1]
            warnings.warn("No output node provided, using last node '%s'" % name)

        try:
            node = self.nodes[name]
        except KeyError as e:
            raise PipelineError("output definition references nonexistent node '%s'" % (e))

        # mode (uses NoOutput by default)
        mode = d.get('mode', 'none')
        if mode == 'none':
            output_class = NoOutput
        elif mode == 'file':
            output_class = FileOutput
        # elif mode == 'ftp':
        #     output_class = FTPOutput
        # elif mode == 's3':
        #     output_class = S3Output
        elif mode == 'image':
            output_class = ImageOutput
        else:
            raise PipelineError("output definition has unexpected mode '%s'" % mode)

        # config
        config = {k:v for k, v in d.items() if k not in ['node', 'mode']}
        
        # output
        output = output_class(node=node, name=name, **config)
        return output

    def _check_execution_graph(self):
        used = {ref:False for ref in self.nodes}

        def f(base_ref):
            if used[base_ref]:
                return

            used[base_ref] = True

            d = self.definition['nodes'][base_ref]
            for ref in d.get('sources', []) + list(d.get('inputs', {}).values()):
                f(ref)

        f(self.output.name)

        for ref in self.nodes:
            if not used[ref]:
                warnings.warning("Unused pipeline node '%s'" % ref, UserWarning)

    def _check_params(self, params):
        if params is None:
            params = {}

        for node in params:
            if node not in self.nodes:
                raise PipelineError("params reference nonexistent node '%s'" % node)

    def execute(self, coordinates, params=None):
        """Execute the pipeline, writing the output if one is defined.

        Parameters
        ----------
        coordinates : TYPE
            Description
        params : None, optional
            Description
        """
        if params is None:
            params = {}

        self._check_params(params)

        for key in self.nodes:
            if key in self.skip_evaluate:
                continue

            node = self.nodes[key]

            d = copy.deepcopy(self.params[key])
            d.update(params.get(key, OrderedDict()))

            if node.evaluated_coordinates == coordinates and node.params == d:
                continue

            node.execute(coordinates, params=d)

        self.output.write()

class PipelineNode(Node):
    """
    Wraps a pipeline into a Node.

    Todo: shape, native_coordinates, etc

    Attributes
    ----------
    coordinates : TYPE
        Description
    definition : TYPE
        Description
    implicit_pipeline_evaluation : TYPE
        Description
    output : TYPE
        Description
    output_node : TYPE
        Description
    params : TYPE
        Description
    path : TYPE
        Description
    pipeline_json : TYPE
        Description
    source_pipeline : TYPE
        Description
    """

    path = tl.Unicode(allow_none=True, help="Path to the JSON definition")
    implicit_pipeline_evaluation = tl.Bool(False)

    output_node = tl.Unicode()
    @tl.default('output_node')
    def _output_node_default(self):
        return self.definition['output']['node']

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
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.source_pipeline.nodes[self.output_node].native_coordinates

    def execute(self, coordinates, params=None, output=None):
        """Summary

        Parameters
        ----------
        coordinates : TYPE
            Description
        params : None, optional
            Description
        output : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
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

if __name__ == '__main__':
    import argparse
    import podpac

    def parse_param(item):
        """Summary

        Parameters
        ----------
        item : TYPE
            Description

        Returns
        -------
        TYPE
            Description

        Raises
        ------
        ValueError
            Description
        """
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
        """Summary

        Parameters
        ----------
        l : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
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
    print('\npipeline output   \t', pipeline.output)
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
