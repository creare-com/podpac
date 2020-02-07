"""
Compositor Summary
"""


from __future__ import division, unicode_literals, print_function, absolute_import

import copy

import numpy as np
import traitlets as tl

# Internal imports
from podpac.core.settings import settings
from podpac.core.coordinates import Coordinates, merge_dims
from podpac.core.utils import common_doc, ArrayTrait, trait_is_defined
from podpac.core.units import UnitsDataArray
from podpac.core.node import COMMON_NODE_DOC, node_eval, Node
from podpac.core.data.datasource import COMMON_DATA_DOC
from podpac.core.data.interpolation import interpolation_trait
from podpac.core.managers.multi_threading import thread_manager

COMMON_COMPOSITOR_DOC = COMMON_DATA_DOC.copy()  # superset of COMMON_NODE_DOC


@common_doc(COMMON_COMPOSITOR_DOC)
class Compositor(Node):
    """Compositor
    
    Attributes
    ----------
    cache_native_coordinates : Bool
        Default is True. If native_coordinates are requested by the user, it may take a long time to calculate if the
        Compositor points to many sources. The result is relatively small and is cached by default. Caching may not be
        desired if the datasource change or is updated.
    interpolation : str, dict, optional
        {interpolation}
    is_source_coordinates_complete : Bool
        Default is False. The source_coordinates do not have to completely describe the source. For example, the source
        coordinates could include the year-month-day of the source, but the actual source also has hour-minute-second
        information. In that case, source_coordinates is incomplete. This flag is used to automatically construct
        native_coordinates.
    shared_coordinates : :class:`podpac.Coordinates`, optional
        Coordinates that are shared amongst all of the composited sources
    source : str
        The source is used for a unique name to cache composited products.
    source_coordinates : :class:`podpac.Coordinates`
        Coordinates that make each source unique. Much be single-dimensional the same size as ``sources``. Optional.
    sources : :class:`np.ndarray`
        An array of sources. This is a numpy array as opposed to a list so that boolean indexing may be used to
        subselect the nodes that will be evaluated.
    source_coordinates : :class:`podpac.Coordinates`, optional
        Coordinates that make each source unique. This is used for subsetting which sources to evaluate based on the
        user-requested coordinates. It is an optimization.
    strict_source_outputs : bool
        Default is False. When compositing multi-output sources, combine the outputs from all sources. If True, do not
        allow sources with different outputs (an exception will be raised if the sources contain different outputs).

    Notes
    -----
    Developers of new Compositor nodes need to implement the `composite` method.

    Multitheading::
      * When MULTITHREADING is False, the compositor stops evaluated sources once the output is completely filled.
      * When MULTITHREADING is True, the compositor must evaluate every source.
        The result is the same, but note that because of this, disabling multithreading could sometimes be faster,
        especially if the number of threads is low.
      * NASA data servers seem to have a hard limit of 10 simultaneous requests, so a max of 10 threads is recommend
        for most use-cases.
    """

    shared_coordinates = tl.Instance(Coordinates, allow_none=True)
    source_coordinates = tl.Instance(Coordinates, allow_none=True)
    is_source_coordinates_complete = tl.Bool(
        False,
        help=(
            "This allows some optimizations but assumes that a node's "
            "native_coordinates=source_coordinate + shared_coordinate "
            "IN THAT ORDER"
        ),
    )

    sources = ArrayTrait(ndim=1)
    cache_native_coordinates = tl.Bool(True)
    interpolation = interpolation_trait(default_value=None)
    strict_source_outputs = tl.Bool(False)

    @tl.default("source_coordinates")
    def _source_coordinates_default(self):
        return self.get_source_coordinates()

    @tl.validate("sources")
    def _validate_sources(self, d):
        # The following line sets up a infinite recursion in the TerrainTiles node.
        # self.outputs  # check for consistent outputs
        return np.array([copy.deepcopy(source) for source in d["value"]])

    @tl.default("outputs")
    def _default_outputs(self):
        if all(source.outputs is None for source in self.sources):
            return None

        elif all(source.outputs is not None and source.output is None for source in self.sources):
            if self.strict_source_outputs:
                outputs = self.sources[0].outputs
                if any(source.outputs != outputs for source in self.sources):
                    raise ValueError(
                        "Source outputs mismatch, and strict_source_outputs is True. "
                        "The sources must all contain the same outputs if strict_source_outputs is True. "
                    )
                return outputs
            else:
                outputs = []
                for source in self.sources:
                    for output in source.outputs:
                        if output not in outputs:
                            outputs.append(output)
                if len(outputs) == 0:
                    outputs = None
                return outputs

        else:
            raise ValueError(
                "Cannot composite standard sources with multi-output sources. "
                "The sources must all be stardard single-output nodes or all multi-output nodes."
            )

    @tl.validate("source_coordinates")
    def _validate_source_coordinates(self, d):
        if d["value"] is not None:
            if d["value"].ndim != 1:
                raise ValueError("Invalid source_coordinates, invalid ndim (%d != 1)" % d["value"].ndim)

            if d["value"].size != self.sources.size:
                raise ValueError(
                    "Invalid source_coordinates, source and source_coordinates size mismatch (%d != %d)"
                    % (d["value"].size, self.sources.size)
                )

        return d["value"]

    # default representation
    def __repr__(self):
        source_name = str(self.__class__.__name__)

        rep = "{}".format(source_name)
        rep += "\n\tsource: {}".format("_".join(str(source) for source in self.sources[:3]))
        rep += "\n\tinterpolation: {}".format(self.interpolation)

        return rep

    def get_source_coordinates(self):
        """
        Returns the coordinates describing each source.
        This may be implemented by derived classes, and is an optimization that allows evaluation subsets of source.
        
        Returns
        -------
        :class:`podpac.Coordinates`
            Coordinates describing each source.
        """
        return None

    @tl.default("shared_coordinates")
    def _shared_coordinates_default(self):
        return self.get_shared_coordinates()

    def get_shared_coordinates(self):
        """Coordinates shared by each source.
        
        Raises
        ------
        NotImplementedError
            Description
        """
        raise NotImplementedError()

    def select_sources(self, coordinates):
        """Downselect compositor sources based on requested coordinates.
        
        This is used during the :meth:`eval` process as an optimization
        when :attr:`source_coordinates` are not pre-defined.
        
        Parameters
        ----------
        coordinates : :class:`podpac.Coordinates`
            Coordinates to evaluate at compositor sources
        
        Returns
        -------
        :class:`np.ndarray`
            Array of downselected sources
        """

        # if source coordinates are defined, use intersect
        if self.source_coordinates is not None:
            # intersecting sources only
            try:
                _, I = self.source_coordinates.intersect(coordinates, outer=True, return_indices=True)

            except:  # Likely non-monotonic coordinates
                _, I = self.source_coordinates.intersect(coordinates, outer=False, return_indices=True)
            i = I[0]
            src_subset = self.sources[i]

        # no downselection possible - get all sources compositor
        else:
            src_subset = self.sources

        return src_subset

    def composite(self, coordinates, outputs, result=None):
        """Implements the rules for compositing multiple sources together.
        
        Parameters
        ----------
        outputs : list
            A list of outputs that need to be composited together
        result : UnitDataArray, optional
            An optional pre-filled array may be supplied, otherwise the output will be allocated.
        
        Raises
        ------
        NotImplementedError
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
        # downselect sources based on coordinates
        src_subset = self.select_sources(coordinates)

        if len(src_subset) == 0:
            yield self.create_output_array(coordinates)
            return

        # Set the interpolation properties for sources
        if self.interpolation is not None:
            for s in src_subset.ravel():
                if trait_is_defined(self, "interpolation"):
                    s.set_trait("interpolation", self.interpolation)

        # Optimization: if coordinates complete and source coords is 1D,
        # set native_coordinates unless they are set already
        # WARNING: this assumes
        #              native_coords = source_coords + shared_coordinates
        #         NOT  native_coords = shared_coords + source_coords
        if self.is_source_coordinates_complete and self.source_coordinates.ndim == 1:
            coords_subset = list(self.source_coordinates.intersect(coordinates, outer=True).coords.values())[0]
            coords_dim = list(self.source_coordinates.dims)[0]
            for s, c in zip(src_subset, coords_subset):
                nc = merge_dims([Coordinates(np.atleast_1d(c), dims=[coords_dim]), self.shared_coordinates])

                if trait_is_defined(s, "native_coordinates") is False:
                    s.set_trait("native_coordinates", nc)

        if settings["MULTITHREADING"]:
            n_threads = thread_manager.request_n_threads(len(src_subset))
            if n_threads == 1:
                thread_manager.release_n_threads(n_threads)
        else:
            n_threads = 0

        if settings["MULTITHREADING"] and n_threads > 1:
            # evaluate nodes in parallel using thread pool
            self._multi_threaded = True
            pool = thread_manager.get_thread_pool(processes=n_threads)
            outputs = pool.map(lambda src: src.eval(coordinates), src_subset)
            pool.close()
            thread_manager.release_n_threads(n_threads)
            for output in outputs:
                yield output

        else:
            # evaluate nodes serially
            self._multi_threaded = False
            for src in src_subset:
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
            list of available coordinates (Coordinate objects)
        """

        raise NotImplementedError("TODO")

    @property
    @common_doc(COMMON_COMPOSITOR_DOC)
    def base_definition(self):
        """Base node defintion for Compositor nodes. 
        
        Returns
        -------
        {definition_return}
        """
        d = super(Compositor, self).base_definition
        d["sources"] = self.sources
        d["interpolation"] = self.interpolation
        return d


class OrderedCompositor(Compositor):
    """Compositor that combines sources based on their order in self.sources. Once a request contains no
    nans, the result is returned. 
    """

    @common_doc(COMMON_COMPOSITOR_DOC)
    def composite(self, coordinates, data_arrays, result=None):
        """Composites data_arrays in order that they appear.
        
        Parameters
        ----------
        coordinates : :class:`podpac.Coordinates`
            {requested_coordinates}
        data_arrays : generator
            Generator that gives UnitDataArray's with the source values.
        result : podpac.UnitsDataArray, optional
            {eval_output}
        
        Returns
        -------
        {eval_return} This composites the sources together until there are no nans or no more sources.
        """

        if result is None:
            result = self.create_output_array(coordinates)
        else:
            result[:] = np.nan

        mask = UnitsDataArray.create(coordinates, outputs=self.outputs, data=0, dtype=bool)
        for data in data_arrays:
            if self.outputs is None:
                data = data.transpose(*result.dims)
                self._composite(result, data, mask)
            else:
                for name in data["output"]:
                    self._composite(result.sel(output=name), data.sel(output=name), mask.sel(output=name))

            # stop if the results are full
            if np.all(mask):
                break

        return result

    @staticmethod
    def _composite(result, data, mask):
        source_mask = np.isfinite(data.data)
        b = ~mask & source_mask
        result.data[b.data] = data.data[b.data]
        mask |= source_mask
