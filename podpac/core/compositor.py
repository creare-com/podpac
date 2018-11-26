"""
Compositor Summary
"""


from __future__ import division, unicode_literals, print_function, absolute_import

from multiprocessing.pool import ThreadPool
import numpy as np
import traitlets as tl

# Internal imports
from podpac.core.coordinates import Coordinates, union, merge_dims
from podpac.core.node import Node
from podpac.core.utils import common_doc
from podpac.core.node import COMMON_NODE_DOC 

COMMON_DOC = COMMON_NODE_DOC.copy()

class Compositor(Node):
    """The base class for all Nodes, which defines the common interface for everything.

    Attributes
    ----------
    shared_coordinates : podpac.Coordinates, optional
        Coordinates that are shared amongst all of the composited sources
    source_coordinates = podpac.Coordinates, optional
        Coordinates that make each source unique. This is used for subsetting which sources to evaluate based on the 
        user-requested coordinates. It is an optimization. 
    is_source_coordinates_complete : Bool
        Default is False. The source_coordinates do not have to completely describe the source. For example, the source
        coordinates could include the year-month-day of the source, but the actual source also has hour-minute-second
        information. In that case, source_coordinates is incomplete. This flag is used to automatically construct 
        native_coordinates
    source: str
        The source is used for a unique name to cache composited products. 
    sources : np.ndarray
        An array of sources. This is a numpy array as opposed to a list so that boolean indexing may be used to 
        subselect the nodes that will be evaluated.
    cache_native_coordinates : True
        Default is True. If native_coordinates are requested by the user, it may take a long time to calculate if the 
        Compositor points to many sources. The result is relatively small and is cached by default. Caching may not be
        desired if the datasource change or is updated. 
    interpolation : str
        Indicates the interpolation type. This gets passed down to the DataSources as part of the compositor. 
    threaded : bool, optional
        Default if False.
        When threaded is False, the compositor stops evaluated sources once the output is completely filled.
        When threaded is True, the compositor must evaluate every source.
        The result is the same, but note that because of this, threaded=False could be faster than threaded=True,
        especially if n_threads is low. For example, threaded with n_threads=1 could be much slower than non-threaded
        if the output is completely filled after the first few sources.
    n_threads : int
        Default is 10 -- used when threaded is True. 
        NASA data servers seem to have a hard limit of 10 simultaneous requests, which determined the default value.
        
    Notes
    ------
    Developers of new Compositor nodes need to implement the `composite` method.
    """
    shared_coordinates = tl.Instance(Coordinates, allow_none=True)
    source_coordinates = tl.Instance(Coordinates, allow_none=True)
    is_source_coordinates_complete = tl.Bool(False,
        help=("This allows some optimizations but assumes that a node's "
              "native_coordinates=source_coordinate + shared_coordinate "
              "IN THAT ORDER"))

    source = tl.Unicode().tag(attr=True)
    sources = tl.Instance(np.ndarray)
    cache_native_coordinates = tl.Bool(True)
    
    interpolation = tl.Unicode('').tag(attr=True)
   
    threaded = tl.Bool(False)
    n_threads = tl.Int(10)
    
    @tl.default('source')
    def _source_default(self):
        source = []
        for s in self.sources[:3]:
            source.append(str(s))
        return '_'.join(source)
    
    @tl.default('source_coordinates')
    def _source_coordinates_default(self):
        return self.get_source_coordinates()

    def get_source_coordinates(self):
        """
        Returns the coordinates describing each source. 
        This may be implemented by derived classes, and is an optimization that allows evaluation subsets of source.
        
        Returns
        -------
        podpac.Coordinates
            Coordinates describing each source.
        """
        return None
        
    @tl.default('shared_coordinates')
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
            Description
        """
        raise NotImplementedError()
    
    def iteroutputs(self, coordinates, method=None):
        """Summary
        
        Parameters
        ----------
        coordinates : TYPE
            Description
        
        Yields
        ------
        TYPE
            Description
        """
        # determine subset of sources needed
        if self.source_coordinates is None:
            src_subset = self.sources # all
        else:
            # intersecting sources only
            try:
                _, I = self.source_coordinates.intersect(coordinates, outer=True, return_indices=True)
            except: # Likely non-monotonic coordinates
                _, I = self.source_coordinates.intersect(coordinates, outer=False, return_indices=True)
            src_subset = self.sources[I]

        if len(src_subset) == 0:
            yield self.create_output_array(coordinates)
            return

        # Set the interpolation properties for sources
        if self.interpolation:
            for s in src_subset.ravel():
                if hasattr(s, 'interpolation'):
                    s.interpolation = self.interpolation

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
                # Switching from _trait_values to hasattr because "native_coordinates"
                # sometimes not showing up in _trait_values in other locations
                # Not confirmed here
                #if 'native_coordinates' not in s._trait_values:
                if hasattr(s,'native_coordinates') is False:
                    s.native_coordinates = nc

        if self.threaded:
            # TODO pool of pre-allocated scratch space
            # TODO: docstring?
            def f(src):
                return src.eval(coordinates, method=method)
            pool = ThreadPool(processes=self.n_threads)
            results = [pool.apply_async(f, [src]) for src in src_subset]
            
            for src, res in zip(src_subset, results):
                yield res.get()
                #src._output = None # free up memory

        else:
            output = None # scratch space
            for src in src_subset:
                output = src.eval(coordinates, output, method)
                yield output
                output[:] = np.nan

    @common_doc(COMMON_DOC)
    @node_eval
    def eval(self, coordinates, output=None, method=None):
        """Evaluates this nodes using the supplied coordinates. 

        Parameters
        ----------
        coordinates : podpac.Coordinates
            {requested_coordinates}
        output : podpac.UnitsDataArray, optional
            {eval_output}
        method : str, optional
            {eval_method}
            
        Returns
        -------
        {eval_return}
        """
        
        outputs = self.iteroutputs(coordinates, method=method)
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
    @common_doc(COMMON_DOC)
    def base_definition(self):
        """Base node defintion for Compositor nodes. 
        
        Returns
        -------
        {definition_return}
        """
        d = super(Compositor, self).base_definition
        d['sources'] = self.sources
        return d


class OrderedCompositor(Compositor):
    """Compositor that combines sources based on their order in self.sources. Once a request contains no
    nans, the result is returned. 
    """
    @common_doc(COMMON_DOC)
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
            mask &= source_mask

            # stop if the results are full
            if np.all(mask):
                break

        return result
