"""
Pipeline Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import inspect
import warnings
import importlib
from collections import OrderedDict
import json
from copy import deepcopy

import traitlets as tl
import numpy as np

from podpac.core.settings import settings
from podpac.core.utils import OrderedDictTrait, JSONEncoder
from podpac.core.node import Node, NodeException
from podpac.core.data.datasource import DataSource
from podpac.core.data.types import ReprojectedSource, Array
from podpac.core.algorithm.algorithm import Algorithm
from podpac.core.compositor import Compositor

from podpac.core.pipeline.output import Output, NoOutput, FileOutput, S3Output, FTPOutput, ImageOutput


class PipelineError(NodeException):
    """
    Raised when parsing a Pipeline definition fails.
    """

    pass


class Pipeline(Node):
    """Deprecated. See Node.definition and Node.from_definition."""

    definition = OrderedDictTrait(readonly=True, help="pipeline definition")
    json = tl.Unicode(readonly=True, help="JSON definition")
    output = tl.Instance(Output, readonly=True, help="pipeline output")
    do_write_output = tl.Bool(True)

    def _first_init(self, path=None, **kwargs):
        warnings.warn(
            "Pipelines are deprecated and will be removed in podpac 2.0. See Node.definition and "
            "Node.from_definition for Node serialization.",
            DeprecationWarning,
        )

        if (path is not None) + ("definition" in kwargs) + ("json" in kwargs) != 1:
            raise TypeError("Pipeline requires exactly one 'path', 'json', or 'definition' argument")

        if path is not None:
            with open(path) as f:
                kwargs["definition"] = json.load(f, object_pairs_hook=OrderedDict)

        return kwargs

    @tl.validate("json")
    def _json_validate(self, proposal):
        s = proposal["value"]
        definition = json.loads(s, object_pairs_hook=OrderedDict)
        parse_pipeline_definition(definition)
        return json.dumps(
            json.loads(s, object_pairs_hook=OrderedDict), separators=(",", ":"), cls=JSONEncoder
        )  # standardize

    @tl.validate("definition")
    def _validate_definition(self, proposal):
        definition = proposal["value"]
        parse_pipeline_definition(definition)
        return definition

    @tl.default("json")
    def _json_from_definition(self):
        return json.dumps(self.definition, separators=(",", ":"), cls=JSONEncoder)

    @tl.default("definition")
    def _definition_from_json(self):
        return json.loads(self.json, object_pairs_hook=OrderedDict)

    @tl.default("output")
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
    def cache_ctrl(self):
        return self.node.cache_ctrl

    @property
    def style(self):
        return self.node.style


# ---------------------------------------------------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------------------------------------------------


def parse_pipeline_definition(definition):
    if "nodes" not in definition:
        raise PipelineError("Pipeline definition requires 'nodes' property")

    if len(definition["nodes"]) == 0:
        raise PipelineError("'nodes' property cannot be empty")

    # parse node definitions
    nodes = OrderedDict()
    for key, d in definition["nodes"].items():
        nodes[key] = _parse_node_definition(nodes, key, d)

    # parse output definition
    output = _parse_output_definition(nodes, definition.get("output", {}))

    return output


def _parse_node_definition(nodes, name, d):
    # get node class
    module_root = d.get("plugin", "podpac")
    node_string = "%s.%s" % (module_root, d["node"])
    module_name, node_name = node_string.rsplit(".", 1)
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
    whitelist = ["node", "attrs", "lookup_attrs", "plugin"]

    # DataSource, Compositor, and Algorithm specific properties
    parents = inspect.getmro(node_class)

    if DataSource in parents:
        if "attrs" in d:
            if "source" in d["attrs"]:
                raise PipelineError("The 'source' property cannot be in attrs")

            if "lookup_source" in d["attrs"]:
                raise PipelineError("The 'lookup_source' property cannot be in attrs")

        if "source" in d:
            kwargs["source"] = d["source"]
            whitelist.append("source")

        elif "lookup_source" in d:
            kwargs["source"] = _get_subattr(nodes, name, d["lookup_source"])
            whitelist.append("lookup_source")

    if Compositor in parents:
        if "sources" in d:
            sources = [_get_subattr(nodes, name, source) for source in d["sources"]]
            kwargs["sources"] = np.array(sources)
            whitelist.append("sources")

    if DataSource in parents or Compositor in parents:
        if "attrs" in d and "interpolation" in d["attrs"]:
            raise PipelineError("The 'interpolation' property cannot be in attrs")

        if "interpolation" in d:
            kwargs["interpolation"] = d["interpolation"]
            whitelist.append("interpolation")

    if Algorithm in parents:
        if "inputs" in d:
            inputs = {k: _get_subattr(nodes, name, v) for k, v in d["inputs"].items()}
            kwargs.update(inputs)
            whitelist.append("inputs")

    for k, v in d.get("attrs", {}).items():
        kwargs[k] = v

    for k, v in d.get("lookup_attrs", {}).items():
        kwargs[k] = _get_subattr(nodes, name, v)

    for key in d:
        if key not in whitelist:
            raise PipelineError("node '%s' has unexpected property %s" % (name, key))

    # return node info
    return node_class(**kwargs)


def _parse_output_definition(nodes, d):
    # node (uses last node by default)
    if "node" in d:
        name = d["node"]
    else:
        name = list(nodes.keys())[-1]

    node = _get_subattr(nodes, "output", name)

    # output parameters
    config = {k: v for k, v in d.items() if k not in ["node", "mode", "plugin", "output"]}

    # get output class from mode
    if "plugin" not in d:
        # core output (from mode)
        mode = d.get("mode", "none")
        if mode == "none":
            output_class = NoOutput
        elif mode == "file":
            output_class = FileOutput
        elif mode == "ftp":
            output_class = FTPOutput
        elif mode == "s3":
            output_class = S3Output
        elif mode == "image":
            output_class = ImageOutput
        else:
            raise PipelineError("output has unexpected mode '%s'" % mode)

        output = output_class(node, name, **config)

    else:
        # custom output (from plugin)
        custom_output = "%s.%s" % (d["plugin"], d["output"])
        module_name, class_name = custom_output.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            raise PipelineError("No module found '%s'" % module_name)
        try:
            output_class = getattr(module, class_name)
        except AttributeError:
            raise PipelineError("Output '%s' not found in module '%s'" % (class_name, module_name))

        try:
            output = output_class(node, name, **config)
        except Exception as e:
            raise PipelineError("Could not create custom output '%s': %s" % (custom_output, e))

        if not isinstance(output, Output):
            raise PipelineError("Custom output '%s' must subclass 'podpac.core.pipeline.output.Output'" % custom_output)

    return output


def _get_subattr(nodes, name, ref):
    refs = ref.split(".")

    try:
        attr = nodes[refs[0]]
        for _name in refs[1:]:
            attr = getattr(attr, _name)
    except (KeyError, AttributeError):
        raise PipelineError("'%s' references nonexistent node/attribute '%s'" % (name, ref))

    if settings["DEBUG"]:
        attr = deepcopy(attr)

    return attr
