from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
from multiprocessing.pool import ThreadPool
import numpy as np
import traitlets as tl

# Internal imports
from podpac.core.coordinate import Coordinate
from podpac.core.node import Node

class Compositor(Node):

    shared_coordinates = tl.Instance(Coordinate, allow_none=True)
    source_coordinates = tl.Instance(Coordinate, allow_none=True)
    is_source_coordinates_complete = tl.Bool(False,
        help=("This allows some optimizations but assumes that a node's " 
        "native_coordinates=source_coordinate + shared_coordinate "
        "IN THAT ORDER"))

    sources = tl.Instance(np.ndarray)
    cache_native_coordinates = tl.Bool(True)
    
    interpolation = tl.Unicode('')

    # When threaded is False, the compositor stops executing sources once the
    # output is completely filled for efficiency. When threaded is True, the
    # compositor must execute every source. The result is the same, but note
    # that because of this, threaded=False could be faster than threaded=True,
    # especially if n_threads is low. For example, threaded with n_threads=1
    # could be much slower than non-threaded if the output is completely filled
    # after the first few sources.

    # NASA data servers seem to have a hard limit of 10 simultaneous requests,
    # so the default for now is 10.
    
    threaded = tl.Bool(False)
    n_threads = tl.Int(10)
    
    @tl.default('source_coordinates')
    def _source_coordinates_default(self):
        return self.get_source_coordinates()
    def get_source_coordinates(self):
        raise NotImplementedError()
        
    @tl.default('shared_coordinates')
    def _shared_coordinates_default(self):
        return self.get_shared_coordinates()
    def get_shared_coordinates(self):
        raise NotImplementedError()
    
    
    @tl.default('native_coordinates')
    def _native_coordinates_default(self):
        return self.get_native_coordinates()
    
    def get_native_coordinates(self):
        """
        This one is tricky... you can have multi-level compositors
        One for a folder described by a date
        One fo all the folders over all dates. 
        The single folder one has time coordinates that are actually
        more accurate than just the folder time coordinate, so you want
        to replace the time coordinate in native coordinate -- does this 
        rule hold? `
        """
        try: 
            return self.load_cached_obj('native.coordinates')
        except: 
            pass
        if self.shared_coordinates is not None and self.is_source_coordinates_complete:
            crds = self.source_coordinates + self.shared_coordinates
        else:
            crds = self.sources[0].native_coordinates
            for s in self.sources[1:]:
                crds = crds + s.native_coordinates
        if self.cache_native_coordinates:
            self.cache_obj(crds, 'native.coordinates')
        return crds
    
    def iteroutputs(self, coordinates, params):
        # determine subset of sources needed
        if self.source_coordinates is None:
            src_subset = self.sources # all
        else:
            # intersecting sources only
            I = self.source_coordinates.intersect(coordinates, pad=1, ind=True)
            src_subset = self.sources[I]

        if len(src_subset) == 0:
            yield self.initialize_coord_array(coordinates, init_type='nan')
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
        if self.is_source_coordinates_complete \
                and len(self.source_coordinates.shape) == 1:
            coords_subset = list(self.source_coordinates.intersect(coordinates,
                    pad=1).coords.values())[0]
            coords_dim = list(self.source_coordinates.dims)[0]
            for s, c in zip(src_subset, coords_subset):
                nc = Coordinate(**{coords_dim: c}) + self.shared_coordinates
                if 'native_coordinates' not in s._trait_values:
                    s.native_coordinates = nc

        if self.threaded:
            # TODO pool of pre-allocated scratch space
            def f(src):
                return src.execute(coordinates, params)
            pool = ThreadPool(processes=self.n_threads)
            results = [pool.apply_async(f, [src]) for src in src_subset]
            
            for src, res in zip(src_subset, results):
                yield res.get()
                src.output = None # free up memory

        else:
            output = None # scratch space
            for src in src_subset:
                output = src.execute(coordinates, params, output)
                yield output
                output[:] = np.nan

    def execute(self, coordinates, params=None, output=None):
        self.evaluated_coordinates = coordinates
        self.params = params
        self.output = output
        
        outputs = self.iteroutputs(coordinates, params)
        self.output = self.composite(outputs, self.output)
        self.evaluated = True

        return self.output

    @property
    def definition(self):
        return NotImplementedError

        # TODO test

        d = OrderedDict()
        d['node'] = self.podpac_path
        d['sources'] = self.sources

        if self.interpolation:
            d['attrs'] = OrderedCompositor()
            d['attrs']['interpolation'] = self.interpolation
        return d


class OrderedCompositor(Compositor):

    def composite(self, outputs, result=None):
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