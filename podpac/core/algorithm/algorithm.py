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
from podpac.core.node import Node, NodeException, node_eval, COMMON_NODE_DOC
from podpac.core.utils import common_doc, NodeTrait
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
        Get the available native coordinates for the inputs to the Node.

        Returns
        -------
        coords_list : list
            list of available coordinates (Coordinate objects)
        """

        return [c for node in self.inputs.values() for c in node.find_coordinates()]


class Algorithm(BaseAlgorithm):
    """Base class for computation nodes with a custom algorithm.
    
    Notes
    ------
    Developers of new Algorithm nodes need to implement the `algorithm` method. 
    """

    def algorithm(self, inputs):
        """
        Arguments
        ----------
        inputs : dict
            Evaluated outputs of the input nodes. The keys are the attribute names.
        
        Raises
        ------
        NotImplementedError
            Description
        """
        raise NotImplementedError

    @common_doc(COMMON_DOC)
    @node_eval
    def eval(self, coordinates, output=None):
        """Evalutes this nodes using the supplied coordinates. 
        
        Parameters
        ----------
        coordinates : podpac.Coordinates
            {requested_coordinates}
        output : podpac.UnitsDataArray, optional
            {eval_output}
        
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
                return node.eval(coordinates)

            # Create pool of size n_threads, note, this may be created from a sub-thread (i.e. not the main thread)
            pool = thread_manager.get_thread_pool(processes=n_threads)

            # Evaluate nodes in parallel/asynchronously
            results = [pool.apply_async(f, [node]) for node in self.inputs.values()]

            # Collect the results in dictionary
            for key, res in zip(self.inputs.keys(), results):
                inputs[key] = res.get()

            # This prevents any more tasks from being submitted to the pool, and will close the workers one done
            pool.close()

            # Release these number of threads back to the thread pool
            thread_manager.release_n_threads(n_threads)
            self._multi_threaded = True
        else:
            # Evaluate nodes in serial
            for key, node in self.inputs.items():
                inputs[key] = node.eval(coordinates)
            self._multi_threaded = False

        # accumulate output coordinates
        coords_list = [Coordinates.from_xarray(a.coords, crs=a.attrs.get("crs")) for a in inputs.values()]
        output_coordinates = union([coordinates] + coords_list)

        result = self.algorithm(inputs)
        if isinstance(result, UnitsDataArray):
            if output is None:
                output = result
            else:
                output[:] = result
        elif isinstance(result, xr.DataArray):
            if output is None:
                output = self.create_output_array(
                    Coordinates.from_xarray(result.coords, crs=result.attrs.get("crs")), data=result.data
                )
            else:
                output[:] = result.data
        elif isinstance(result, np.ndarray):
            if output is None:
                output = self.create_output_array(output_coordinates, data=result)
            else:
                output.data[:] = result
        else:
            raise NodeException

        if "output" in output.dims and self.output is not None:
            output = output.sel(output=self.output)

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

    source = NodeTrait().tag(attr=True)

    @tl.default("outputs")
    def _default_outputs(self):
        self.traits()["outputs"].default = self.source.outputs
        return self.source.outputs
