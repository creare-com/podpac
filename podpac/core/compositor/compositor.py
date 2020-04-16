"""
Compositor Summary
"""


from __future__ import division, unicode_literals, print_function, absolute_import

import copy

import numpy as np
import traitlets as tl

# Internal imports
from podpac.core.settings import settings
from podpac.core.coordinates import Coordinates
from podpac.core.utils import common_doc, NodeTrait
from podpac.core.node import COMMON_NODE_DOC, node_eval, Node
from podpac.core.data.datasource import COMMON_DATA_DOC
from podpac.core.interpolation.interpolation import InterpolationTrait
from podpac.core.managers.multi_threading import thread_manager

COMMON_COMPOSITOR_DOC = COMMON_DATA_DOC.copy()  # superset of COMMON_NODE_DOC


@common_doc(COMMON_COMPOSITOR_DOC)
class BaseCompositor(Node):
    """A base class for compositor nodes.
    
    Attributes
    ----------
    sources : list
        Source nodes.
    source_coordinates : :class:`podpac.Coordinates`
        Coordinates that make each source unique. Must the same size as ``sources`` and single-dimensional. Optional.
    interpolation : str, dict, optional
        {interpolation}
    
    Notes
    -----
    Developers of compositor subclasses nodes need to implement the `composite` method.

    Multitheading::
      * When MULTITHREADING is False, the compositor stops evaluated sources once the output is completely filled.
      * When MULTITHREADING is True, the compositor must evaluate every source.
        The result is the same, but note that because of this, disabling multithreading could sometimes be faster,
        especially if the number of threads is low.
      * NASA data servers seem to have a hard limit of 10 simultaneous requests, so a max of 10 threads is recommend
        for most use-cases.
    """

    sources = tl.List(trait=NodeTrait()).tag(attr=True)
    interpolation = InterpolationTrait(allow_none=True, default_value=None).tag(attr=True)
    source_coordinates = tl.Instance(Coordinates, allow_none=True, default_value=None).tag(attr=True)

    auto_outputs = tl.Bool(False)

    # debug traits
    _eval_sources = tl.Any()

    @tl.validate("sources")
    def _validate_sources(self, d):
        sources = d["value"]

        n = np.sum([source.outputs is None for source in sources])
        if not (n == 0 or n == len(sources)):
            raise ValueError(
                "Cannot composite standard sources with multi-output sources. "
                "The sources must all be standard single-output nodes or all multi-output nodes."
            )

        # copy so that interpolation trait of the input source is not overwritten
        return [copy.deepcopy(source) for source in sources]

    @tl.validate("source_coordinates")
    def _validate_source_coordinates(self, d):
        if d["value"] is None:
            return None

        if d["value"].ndim != 1:
            raise ValueError("Invalid source_coordinates, invalid ndim (%d != 1)" % d["value"].ndim)

        if d["value"].size != len(self.sources):
            raise ValueError(
                "Invalid source_coordinates, source and source_coordinates size mismatch (%d != %d)"
                % (d["value"].size, len(self.sources))
            )

        return d["value"]

    @tl.default("outputs")
    def _default_outputs(self):
        if not self.auto_outputs:
            return None

        # autodetect outputs from sources
        if all(source.outputs is None for source in self.sources):
            outputs = None

        elif all(source.outputs is not None and source.output is None for source in self.sources):
            outputs = []
            for source in self.sources:
                for output in source.outputs:
                    if output not in outputs:
                        outputs.append(output)

            if len(outputs) == 0:
                outputs = None

        else:
            raise RuntimeError(
                "Compositor sources were not validated correctly. "
                "Cannot composite standard sources with multi-output sources."
            )

        return outputs

    def select_sources(self, coordinates):
        """Select and prepare sources based on requested coordinates.
        
        Parameters
        ----------
        coordinates : :class:`podpac.Coordinates`
            Coordinates to evaluate at compositor sources
        
        Returns
        -------
        sources : :class:`np.ndarray`
            Array of sources

        Notes
        -----
         * If :attr:`source_coordinates` is defined, only sources that intersect the requested coordinates are selected.
         * Sets sources :attr:`interpolation`.
        """

        # select intersecting sources, if possible
        if self.source_coordinates is None:
            sources = self.sources
        else:
            try:
                _, I = self.source_coordinates.intersect(coordinates, outer=True, return_indices=True)
            except:
                # Likely non-monotonic coordinates
                _, I = self.source_coordinates.intersect(coordinates, outer=False, return_indices=True)
            i = I[0]
            sources = np.array(self.sources)[i].tolist()

        # set the interpolation properties for sources
        if self.trait_is_defined("interpolation") and self.interpolation is not None:
            for s in sources:
                if s.has_trait("interpolation"):
                    s.set_trait("interpolation", self.interpolation)

        return sources

    def composite(self, coordinates, data_arrays, result=None):
        """Implements the rules for compositing multiple sources together. Must be implemented by child classes.
        
        Parameters
        ----------
        coordinates : :class:`podpac.Coordinates`
            {requested_coordinates}
        data_arrays : list
            Evaluated data from the sources.
        result : UnitDataArray, optional
            An optional pre-filled array may be supplied, otherwise the output will be allocated.

        Returns
        -------
        {eval_return} 
        """

        raise NotImplementedError()

    def iteroutputs(self, coordinates):
        """Summary
        
        Parameters
        ----------
        coordinates : :class:`podpac.Coordinates`
            Coordinates to evaluate at compositor sources
        
        Yields
        ------
        :class:`podpac.core.units.UnitsDataArray`
            Output from source node eval method
        """

        # get sources, potentially downselected
        sources = self.select_sources(coordinates)

        if settings["DEBUG"]:
            self._eval_sources = sources

        if len(sources) == 0:
            yield self.create_output_array(coordinates)
            return

        if settings["MULTITHREADING"]:
            n_threads = thread_manager.request_n_threads(len(sources))
            if n_threads == 1:
                thread_manager.release_n_threads(n_threads)
        else:
            n_threads = 0

        if settings["MULTITHREADING"] and n_threads > 1:
            # evaluate nodes in parallel using thread pool
            self._multi_threaded = True
            pool = thread_manager.get_thread_pool(processes=n_threads)
            outputs = pool.map(lambda src: src.eval(coordinates), sources)
            pool.close()
            thread_manager.release_n_threads(n_threads)
            for output in outputs:
                yield output

        else:
            # evaluate nodes serially
            self._multi_threaded = False
            for src in sources:
                yield src.eval(coordinates)

    @node_eval
    @common_doc(COMMON_COMPOSITOR_DOC)
    def eval(self, coordinates, output=None):
        """Evaluates this nodes using the supplied coordinates. 

        Parameters
        ----------
        coordinates : :class:`podpac.Coordinates`
            {requested_coordinates}
        output : podpac.UnitsDataArray, optional
            {eval_output}
            
        Returns
        -------
        {eval_return}
        """

        self._requested_coordinates = coordinates
        outputs = self.iteroutputs(coordinates)
        output = self.composite(coordinates, outputs, output)
        return output

    def find_coordinates(self):
        """
        Get the available native coordinates for the Node.

        Returns
        -------
        coords_list : list
            available coordinates from all of the sources.
        """

        return [coords for source in self.sources for coords in source.find_coordinates()]

    @property
    def _repr_keys(self):
        """list of attribute names, used by __repr__ and __str__ to display minimal info about the node"""
        keys = []
        if self.trait_is_defined("sources"):
            keys.append("sources")
        return keys
