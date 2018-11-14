"""
Pipeline Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
import json

import traitlets as tl

from podpac.core.node import Node
from podpac.core.utils import OrderedDictTrait
from podpac.core.pipeline.output import Output
from podpac.core.pipeline.util import parse_pipeline_definition

class Pipeline(Node):
    """Summary

    Attributes
    ----------
    json : string
        pipeline JSON definition
    definition : OrderedDict
        pipeline definition
    output : Output
        pipeline output
    node : Node
        pipeline output node
    do_write_output : Bool
        True to call output.write() on execute, false otherwise.
    """

    definition = OrderedDictTrait(readonly=True, help="pipeline definition")
    json = tl.Unicode(readonly=True, help="JSON definition")
    output = tl.Instance(Output, readonly=True, help="pipeline output")
    do_write_output = tl.Bool(True)

    def _first_init(self, path=None, **kwargs):
        if (path is not None) + ('definition' in kwargs) + ('json' in kwargs) != 1:
            raise TypeError("Pipeline requires exactly one 'path', 'json', or 'definition' argument")

        if path is not None:
            with open(path) as f:
                kwargs['definition'] = json.load(f, object_pairs_hook=OrderedDict)

        return kwargs

    @tl.validate('json')
    def _json_validate(self, proposal):
        s = proposal['value']
        definition = json.loads(s)
        parse_pipeline_definition(definition)
        return json.dumps(json.loads(s)) # standardize

    @tl.validate('definition')
    def _validate_definition(self, proposal):
        definition = proposal['value']
        parse_pipeline_definition(definition)
        return definition

    @tl.default('json')
    def _json_from_definition(self):
        return json.dumps(self.definition)

    @tl.default('definition')
    def _definition_from_json(self):
        print("definition from json")
        return json.loads(self.json, object_pairs_hook=OrderedDict)

    @tl.default('output')
    def _parse_definition(self):
        return parse_pipeline_definition(self.definition)

    def eval(self, coordinates, output=None):
        """Evaluate the pipeline, writing the output if one is defined.

        Parameters
        ----------
        coordinates : TYPE
            Description
        """

        self._requested_coordinates = coordinates

        output = self.output.node.eval(coordinates, output)
        if self.do_write_output:
            self.output.write(output, coordinates)

        self._output = output
        return output

    # -----------------------------------------------------------------------------------------------------------------
    # properties, forwards output node
    # -----------------------------------------------------------------------------------------------------------------

    @property
    def node(self):
        return self.output.node

    @property
    def units(self):
        return self.node.units

    @property
    def dtype(self):
        return self.node.dtype

    @property
    def cache_type(self):
        return self.node.cache_type

    @property
    def style(self):
        return self.node.style