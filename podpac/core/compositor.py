from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
import numpy as np
import traitlets as tl
import matplotlib.colors, matplotlib.cm
import matplotlib.pyplot as plt
from pint import UnitRegistry
ureg = UnitRegistry()

# Internal imports
from podpac.core.coordinate import Coordinate
from podpac.core.node import Node, UnitsDataArray

class Compositor(Node):

    shared_coordinates = tl.Instance(Coordinate, allow_none=True)
    source_coordinates = tl.Instance(Coordinate, allow_none=True)
    is_source_coordinates_complete = tl.Bool(False)
    sources = tl.Instance(np.ndarray)
    cache_native_coordinates = tl.Bool(True)
    
    interpolation = tl.Unicode('')
    
    @tl.default('native_coordinates')
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
    
    def execute(self, coordinates, params=None, output=None):
        coords, params, out = \
                self._execute_common(coordinates, params, None, 
                                     initialize_output=False)
        out = output
        
        # Decide which sources need to be evaluated
        if self.source_coordinates is None: # Do them all
            src_subset = self.sources
        else:
            coords_subset_slc = \
                self.source_coordinates.intersect_ind_slice(coordinates, pad=0)
            src_subset = self.sources[coords_subset_slc]
            
        #set the interpolation properties for sources
        if self.interpolation:
            for s in src_subset:
                if hasattr(s, 'interpolation'):
                    s.interpolation = self.interpolation
            
        if len(src_subset) == 0:
            return self.initialize_coord_array(coordinates, init_type='nan')

        if output is None:
            self.output = self.composite(src_subset, coordinates,
                                         params, output)
        else: 
            out[:] = self.composite(src_subset, coordinates, params, output)
            self.output = out
        self.evaluated = True

        return self.output

    @property
    def definition(self):
        return NotImplementedError

        # currently doesn't work because sources is a list of np.ndarray
        # instead of a list of nodes

        d = OrderedDict()
        d['node'] = self.podpac_path
        d['sources'] = self.sources

        if self.interpolation:
            d['attrs'] = OrderedCompositor()
            d['attrs']['interpolation'] = interpolation
        return d


class OrderedCompositor(Compositor):

    def composite(self, src_subset, coordinates, params, output):
        start = 0
        # The compositor doesn't actually know what dimensions are 
        # in the source, so we rely on the first node to create the output
        # output array if it has not been given. 
        if output is None: 
            output = src_subset[0].execute(coordinates, params)
            start = 1

        o = output.copy()  # Create the dataset once
        I = np.isfinite(o.data)  # Create the masks once
        Id = I.copy()
        for src in src_subset[start:]:  # This could be a parfor (threaded)
            if np.all(I):
                break
            o = src.execute(coordinates, params, o).transpose(*output.dims)
            Id[:] = np.isfinite(o.data)
            output.data[~I & Id] = o.data[~I & Id]
            I &= Id
        return output
