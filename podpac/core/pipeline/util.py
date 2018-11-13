
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
from podpac.core.data.types import ReprojectedSource, Array
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

    return output

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
    whitelist = ['node', 'attrs', 'lookup_attrs', 'plugin']

    # DataSource, Compositor, and Algorithm specific properties
    parents = inspect.getmro(node_class)

    if DataSource in parents:
        if 'attrs' in d:
            if 'source' in d['attrs']:
                raise PipelineError("The 'source' property cannot be in attrs")

            if 'lookup_source' in d['attrs']:
                raise PipelineError("The 'lookup_source' property cannot be in attrs")

        if 'source' in d:
            if Array in parents:
                kwargs['source'] = np.array(d['source'])
            else:
                kwargs['source'] = d['source']
            whitelist.append('source')

        elif 'lookup_source' in d:
            kwargs['source'] = _get_subattr(nodes, name, d['lookup_source'])
            whitelist.append('lookup_source')

    if Compositor in parents:
        if 'sources' in d:
            sources = [_get_subattr(nodes, name, source) for source in d['sources']]
            kwargs['sources'] = np.array(sources)
            whitelist.append('sources')

    if DataSource in parents or Compositor in parents:
        if 'attrs' in d and 'interpolation' in d['attrs']:
            raise PipelineError("The 'interpolation' property cannot be in attrs")

        if 'interpolation' in d:
            kwargs['interpolation'] = d['interpolation']
            whitelist.append('interpolation')
            
    if Algorithm in parents:
        if 'inputs' in d:
            inputs = {k:_get_subattr(nodes, name, v) for k, v in d['inputs'].items()}
            kwargs.update(inputs)
            whitelist.append('inputs')

    for k, v in d.get('attrs', {}).items():
        kwargs[k] = v

    for k, v in d.get('lookup_attrs', {}).items():
        kwargs[k] = _get_subattr(nodes, name, v)

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

    node = _get_subattr(nodes, 'output', name)
    
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

def _get_subattr(nodes, name, ref):
    refs = ref.split('.')
    
    try:
        attr = nodes[refs[0]]
        for _name in refs[1:]:
            attr = getattr(attr, _name)
    except (KeyError, AttributeError):
        raise PipelineError("'%s' references nonexistent node/attribute '%s'" % (name, ref))
    
    return attr