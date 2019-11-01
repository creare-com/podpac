"""
Compositor Summary
"""


from __future__ import division, unicode_literals, print_function, absolute_import

from multiprocessing.pool import ThreadPool
import numpy as np
import traitlets as tl

# Internal imports
from podpac.core.settings import settings
from podpac.core.coordinates import Coordinates, merge_dims
from podpac.core.node import Node
from podpac.core.utils import common_doc
from podpac.core.utils import ArrayTrait
from podpac.core.node import COMMON_NODE_DOC
from podpac.core.node import node_eval
from podpac.core.data.datasource import COMMON_DATA_DOC
from podpac.core.data.interpolation import interpolation_trait
from podpac.core.utils import trait_is_defined

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
        Description
    sources : :class:`np.ndarray`
        An array of sources. This is a numpy array as opposed to a list so that boolean indexing may be used to
        subselect the nodes that will be evaluated.
    source_coordinates : :class:`podpac.Coordinates`, optional
        Coordinates that make each source unique. This is used for subsetting which sources to evaluate based on the
        user-requested coordinates. It is an optimization.
    
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

    source = tl.Unicode().tag(attr=True)
    sources = ArrayTrait(ndim=1)
    cache_native_coordinates = tl.Bool(True)

    interpolation = interpolation_trait(default_value=None)

    @tl.default("source")
    def _source_default(self):
        source = []
        for s in self.sources[:3]:
            source.append(str(s))
        return "_".join(source)

    @tl.default("source_coordinates")
    def _source_coordinates_default(self):
        return self.get_source_coordinates()

    # default representation
    def __repr__(self):
        source_name = str(self.__class__.__name__)

        rep = "{}".format(source_name)
        rep += "\n\tsource: {}".format(self.source)
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

            src_subset = self.sources[I]

        # no downselection possible - get all sources compositor
        else:
            src_subset = self.sources

        return src_subset

    def composite(self, outputs, result=None):
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
                    s.native_coordinates = nc

        if settings["MULTITHREADING"]:
            # TODO pool of pre-allocated scratch space
            # TODO: docstring?
            def f(src):
                return src.eval(coordinates)

            pool = ThreadPool(processes=settings.get("N_THREADS", 10))
            results = [pool.apply_async(f, [src]) for src in src_subset]

            for src, res in zip(src_subset, results):
                yield res.get()
                # src._output = None # free up memory

        else:
            output = None  # scratch space
            for src in src_subset:
                output = src.eval(coordinates, output)
                yield output
                # output[:] = np.nan

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
        output = self.composite(outputs, output)
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
    def composite(self, outputs, result=None):
        """Composites outputs in order that they appear.
        
        Parameters
        ----------
        outputs : generator
            Generator that gives UnitDataArray's with the source values.
        result : None, optional
            Description
        
        Returns
        -------
        {eval_return} This composites the sources together until there are no nans or no more sources.
        """
        if result is None:
            # consume the first source output
            result = next(outputs).copy()

        # initialize the mask
        # if result is None, probably this is all false
        mask = np.isfinite(result.data)
        if np.all(mask):
            return result

        # loop through remaining outputs
        for output in outputs:
            output = output.transpose(*result.dims)
            source_mask = np.isfinite(output.data)
            b = ~mask & source_mask
            result.data[b] = output.data[b]
            mask |= source_mask

            # stop if the results are full
            if np.all(mask):
                break

        return result
