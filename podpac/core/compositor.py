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
    source_coordinates = tl.Instance(Coordinate)

    sources = tl.Instance(np.ndarray)
    
    @tl.default('native_coordinates')
    def set_native_coordinates(self):
        if self.shared_coordinates is not None:
            return self.source_coordinates + self.shared_coordinates
        else:
            crds = self.sources[0].native_coordinates
            for s in self.sources[1:]:
                crds = crds + s.native_coordinates
            return crds
    
    def execute(self, coordinates, params=None, output=None):
        coords, params, out = \
                self._execute_common(coordinates, params, output)

        # Decide which sources need to be evaluated
        coords_subset_slc = \
                self.source_coordinates.intersect_ind_slice(coordinates, pad=0)
        src_subset = self.sources[coords_subset_slc]
        if len(src_subset) == 0:
            return self.initialize_coord_array(coordinates, init_type='nan')

        if output is None:
            self.output = self.composite(src_subset, coordinates,
                                         params, output)
        else: 
            out[:] = self.composite(src_subset, coordintes, params, output)
            self.output = out
        self.evaluated = True

        return self.output


class GridCompositor(Compositor):

    def composite(self, src_subset, coordintes, params, output):
        pass
