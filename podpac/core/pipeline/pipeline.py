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
    definition : OrderedDict
        pipeline definition
    nodes : OrderedDict
        dictionary of pipeline nodes
    pipeline_output : Output
        pipeline output
    do_write_output : Bool
        True to call pipeline_output.write() on execute, false otherwise.
    """

    path = tl.Unicode(allow_none=True, help="Path to the JSON definition")
    json = tl.Unicode(allow_none=True, help="JSON definition")
    definition = tl.Instance(OrderedDict, help="pipeline definition")
    nodes = tl.Instance(OrderedDict, help="pipeline nodes")
    pipeline_output = tl.Instance(Output, help="pipeline output")
    implicit_pipeline_evaluation = tl.Bool(True)
    do_write_output = tl.Bool(True)

    @property
    def native_coordinates(self):
        return self.pipeline_output.node.native_coordinates

    @property
    def output(self):
        return self.pipeline_output.node.output

    @property
    def evaluated(self):
        return self.pipeline_output.node.evaluated

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
    def interpolation(self):
        return self.pipeline_output.node.interpolation

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

    def execute(self, coordinates, output=None):
        """Execute the pipeline, writing the output if one is defined.

        Parameters
        ----------
        coordinates : TYPE
            Description
        """

        if self.implicit_pipeline_evaluation:
            self.pipeline_output.node.execute(coordinates, output)

        self.pipeline_output.write()

        return self.output
