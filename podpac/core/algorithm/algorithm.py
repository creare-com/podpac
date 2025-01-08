"""
Base class for Algorithm Nodes
"""

from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
import inspect

import numpy as np
import xarray as xr
import traitlets as tl

# Internal dependencies
from podpac.core.coordinates import Coordinates, union
from podpac.core.units import UnitsDataArray
from podpac.core.node import Node, NodeException, COMMON_NODE_DOC
from podpac.core.utils import common_doc, NodeTrait, align_xarray_dict
from podpac.core.settings import settings
from podpac.core.managers.multi_threading import thread_manager

COMMON_DOC = COMMON_NODE_DOC.copy()


class BaseAlgorithm(Node):
    """Base class for algorithm nodes.

    Note: developers should generally use one of the Algorithm or UnaryAlgorithm child classes.
    """

    @property
    def inputs(self):
        # gettattr(self, ref) can take a long time, so we inspect trait.klass instead
        return {
            ref: getattr(self, ref)
            for ref, trait in self.traits().items()
            if hasattr(trait, "klass") and Node in inspect.getmro(trait.klass) and getattr(self, ref) is not None
        }

    def find_coordinates(self):
        """
        Get the available coordinates for the inputs to the Node.

        Returns
        -------
        coords_list : list
            list of available coordinates (Coordinate objects)
        """

        return [c for node in self.inputs.values() for c in node.find_coordinates()]


class Algorithm(BaseAlgorithm):
    """Base class for computation nodes with a custom algorithm.

    Attributes
    ----------
    xarray_floating_point_correction: bool
        if true, ensures that all input coordinates match during _eval calls 

    Notes
    ------
    Developers of new Algorithm nodes need to implement the `algorithm` method.
    """

    # not the best solution... hard to check for these attrs
    # abstract = tl.Bool(default_value=True, allow_none=True).tag(attr=True, required=False, hidden=True)

    xarray_floating_point_correction = tl.Bool(allow_none=False).tag(attr=True)

    @tl.default("xarray_floating_point_correction")
    def _default_xarray_floating_point_correction(self):
        return settings["ALGORITHM_XARRAY_FLOATING_POINT_CORRECTION"]

    def algorithm(self, inputs, coordinates):
        """
        Arguments
        ----------
        inputs : dict
            Evaluated outputs of the input nodes. The keys are the attribute names. Each item is a `UnitsDataArray`.
        coordinates : podpac.Coordinates
            Requested coordinates.
            Note that the ``inputs`` may contain different coordinates than the requested coordinates
        """

        raise NotImplementedError

    @common_doc(COMMON_DOC)
    def _eval(self, coordinates, output=None, _selector=None):
        """Evalutes this nodes using the supplied coordinates.

        Parameters
        ----------
        coordinates : podpac.Coordinates
            {requested_coordinates}
        output : podpac.UnitsDataArray, optional
            {eval_output}
        _selector: callable(coordinates, request_coordinates)
            {eval_selector}

        Returns
        -------
        {eval_return}
        """

        self._requested_coordinates = coordinates

        inputs = {}

        if settings["MULTITHREADING"]:
            n_threads = thread_manager.request_n_threads(len(self.inputs))
            if n_threads == 1:
                thread_manager.release_n_threads(n_threads)
        else:
            n_threads = 0

        if settings["MULTITHREADING"] and n_threads > 1:
            # Create a function for each thread to execute asynchronously
            def f(node):
                return node.eval(coordinates, _selector=_selector)

            # Create pool of size n_threads, note, this may be created from a sub-thread (i.e. not the main thread)
            pool = thread_manager.get_thread_pool(processes=n_threads)

            # Evaluate nodes in parallel/asynchronously
            results = [pool.apply_async(f, [node]) for node in self.inputs.values()]

            # Collect the results in dictionary
            for key, res in zip(self.inputs.keys(), results):
                inputs[key] = res.get()

            # This prevents any more tasks from being submitted to the pool, and will close the workers once done
            pool.close()

            # Release these number of threads back to the thread pool
            thread_manager.release_n_threads(n_threads)
            self._multi_threaded = True
        else:
            # Evaluate nodes in serial
            for key, node in self.inputs.items():
                inputs[key] = node.eval(coordinates, output=output, _selector=_selector)
            self._multi_threaded = False

        if self.xarray_floating_point_correction:
            inputs = align_xarray_dict(inputs)
        
        result = self.algorithm(inputs, coordinates)

        if not isinstance(result, xr.DataArray):
            raise NodeException("algorithm returned unsupported type '%s'" % type(result))

        if "output" in result.dims and self.output is not None:
            result = result.sel(output=self.output)

        if output is not None:
            missing = [dim for dim in result.dims if dim not in output.dims]
            if any(missing):
                raise NodeException("provided output is missing dims %s" % missing)

            output_dims = output.dims
            output = output.transpose(..., *result.dims)
            output[:] = result.data
            output = output.transpose(*output_dims)
        elif isinstance(result, UnitsDataArray):
            output = result
        else:
            output_coordinates = Coordinates.from_xarray(result)
            output = self.create_output_array(output_coordinates, data=result.data)

        return output


class UnaryAlgorithm(BaseAlgorithm):
    """
    Base class for computation nodes that take a single source and transform it.

    Attributes
    ----------
    source : Node
        The source node

    Notes
    ------
    Developers of new Algorithm nodes need to implement the `eval` method.
    """

    source = NodeTrait().tag(attr=True, required=True)

    # list of attribute names, used by __repr__ and __str__ to display minimal info about the node
    _repr_keys = ["source"]

    @tl.default("outputs")
    def _default_outputs(self):
        return self.source.outputs

    @tl.default("style")
    def _default_style(self):  # Pass through source style by default
        return self.source.style
