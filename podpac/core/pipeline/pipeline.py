"""
Pipeline Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
import json

import traitlets as tl

from podpac.core.node import Node

from podpac.core.pipeline.output import Output
from podpac.core.pipeline.util import parse_pipeline_definition

class Pipeline(Node):
    """Summary

    Attributes
    ----------
    path : string
        path to pipeline JSON definition
    json : string
        pipeline JSON definition
    definition : OrderedDict
        pipeline definition
    nodes : OrderedDict
        dictionary of pipeline nodes
    pipeline_output : Output
        pipeline output
    """

    path = tl.Unicode(allow_none=True, help="Path to the JSON definition")
    json = tl.Unicode(allow_none=True, help="JSON definition")
    definition = tl.Instance(OrderedDict, help="pipeline definition")
    nodes = tl.Instance(OrderedDict, help="pipeline nodes")
    pipeline_output = tl.Instance(Output, help="pipeline output")
    implicit_pipeline_evaluation = tl.Bool(True)
    
    @property
    def units(self):
        return self.pipeline_output.node.units
    
    @property
    def dtype(self):
        return self.pipeline_output.node.dtype
    
    @property
    def cache_type(self):
        return self.pipeline_output.node.cache_type
    
    @property
    def style(self):
        return self.pipeline_output.node.style

    @tl.validate('json')
    def _json_validate(self, proposal):
        s = proposal['value']
        definition = json.loads(s, object_pairs_hook=OrderedDict)
        self.nodes, self.pipeline_output = parse_pipeline_definition(definition)
        self.definition = definition
        return s

    @tl.validate('path')
    def _path_validate(self, proposal):
        path = proposal['value']
        with open(path) as f:
            definition = json.load(f, object_pairs_hook=OrderedDict)
        self.nodes, self.pipeline_output = parse_pipeline_definition(definition)
        self.definition = definition
        return path

    @tl.validate('definition')
    def _validate_definition(self, proposal):
        definition = proposal['value']
        self.nodes, self.pipeline_output = parse_pipeline_definition(definition)
        return definition

    def eval(self, coordinates, output=None):
        """Evaluate the pipeline, writing the output if one is defined.

        Parameters
        ----------
        coordinates : TYPE
            Description
        """
        
        output = self.pipeline_output.node.eval(coordinates, output)
        self.pipeline_output.write(output, coordinates)
        
        # debugging
        self._requested_coordinates = coordinates
        self._output_coordinates = self.pipeline_output.node._output_coordinates
        self._output = output

        return output