
"""
Pipeline utils
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import importlib
import inspect
import warnings
from collections import OrderedDict

import numpy as np
from podpac.core.node import NodeException
from podpac.core.data.datasource import DataSource
from podpac.core.algorithm.algorithm import Algorithm
from podpac.core.compositor import Compositor
from podpac.core.pipeline.output import Output, NoOutput, FileOutput, S3Output, FTPOutput, ImageOutput

class PipelineError(NodeException):
    """Summary
    """
    pass

def parse_pipeline_definition(definition):
    if 'nodes' not in definition:
        raise PipelineError("Pipeline definition requires 'nodes' property")

    if len(definition['nodes']) == 0:
        raise PipelineError("'nodes' property cannot be empty")

    # parse node definitions
    nodes = OrderedDict()
    for key, d in definition['nodes'].items():
        nodes[key] = _parse_node_definition(nodes, key, d)

    # parse output definition
    output = _parse_output_definition(nodes, definition.get('output', {}))

    _check_evaluation_graph(definition, nodes, output)

    return nodes, output

def _parse_node_definition(nodes, name, d):
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
        raise PipelineError("Node '%s' not found in module '%s'" % (node_name, module_name))

    # parse and configure kwargs
    kwargs = {}
    whitelist = ['node', 'attrs', 'evaluate', 'plugin']

    parents = inspect.getmro(node_class)
    
    # DataSource, Compositor, and Algorithm specific properties
    if DataSource in parents:
        if 'source' in d:
            kwargs['source'] = d['source']
            whitelist.append('source')

        if 'attrs' in d and 'source' in d['attrs']:
            raise PipelineError("The DataSource 'source' property cannot be in attrs")

    if Compositor in parents:
        if 'sources' in d:
            try:
                sources = [nodes[source] for source in d['sources']] # translate node references
            except KeyError as e:
                raise PipelineError("node '%s' references nonexistent node %s" % (name, e))
            kwargs['sources'] = np.array(sources)
            whitelist.append('sources')
            
    if Algorithm in parents:
        if 'inputs' in d:
            try:
                inputs = {k:nodes[v] for k, v in d['inputs'].items()} # translate node references
            except KeyError as e:
                raise PipelineError("node '%s' references nonexistent node %s" % (name, e))
            kwargs.update(inputs)
            whitelist.append('inputs')

    if 'attrs' in d:
        kwargs.update(d['attrs'])

    for key in d:
        if key not in whitelist:
            raise PipelineError("node '%s' has unexpected property %s" % (name, key))

    # return node info
    return node_class(**kwargs)

def _parse_output_definition(nodes, d):
    # node (uses last node by default)
    if 'node' in d:
        name = d['node']
    else:
        name = list(nodes.keys())[-1]

    try:
        node = nodes[name]
    except KeyError as e:
        raise PipelineError("output references nonexistent node %s" % (e))

    # output parameters
    config = {k:v for k, v in d.items() if k not in ['node', 'mode', 'plugin', 'output']}

    # get output class from mode
    if 'plugin' not in d:
        # core output (from mode)
        mode = d.get('mode', 'none')
        if mode == 'none':
            output_class = NoOutput
        elif mode == 'file':
            output_class = FileOutput
        elif mode == 'ftp':
            output_class = FTPOutput
        elif mode == 's3':
            output_class = S3Output
        elif mode == 'image':
            output_class = ImageOutput
        else:
            raise PipelineError("output has unexpected mode '%s'" % mode)

        output = output_class(node=node, name=name, **config)

    else:
        # custom output (from plugin)
        custom_output = '%s.%s' % (d['plugin'], d['output'])
        module_name, class_name = custom_output.rsplit('.', 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            raise PipelineError("No module found '%s'" % module_name)
        try:
            output_class = getattr(module, class_name)
        except AttributeError:
            raise PipelineError("Output '%s' not found in module '%s'" % (class_name, module_name))

        try:
            output = output_class(node=node, name=name, **config)
        except Exception as e:
            raise PipelineError("Could not create custom output '%s': %s" % (custom_output, e))

        if not isinstance(output, Output):
            raise PipelineError("Custom output '%s' must subclass 'podpac.core.pipeline.output.Output'" % custom_output)

    return output

def _check_evaluation_graph(definition, nodes, output):
    used = {ref:False for ref in nodes}

    def f(base_ref):
        if used[base_ref]:
            return

        used[base_ref] = True

        d = definition['nodes'][base_ref]
        for ref in d.get('sources', []):
            f(ref)

        for ref in d.get('inputs', {}).values():
            f(ref)

    f(output.name)

    for ref in nodes:
        if not used[ref]:
            warnings.warn("Unused pipeline node '%s'" % ref, UserWarning)