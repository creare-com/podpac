from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
import numpy as np
import xarray as xr
import numexpr as ne
import traitlets as tl
import matplotlib.colors, matplotlib.cm
import matplotlib.pyplot as plt
from pint import UnitRegistry
ureg = UnitRegistry()

from podpac.core.coordinate import Coordinate
from podpac.core.node import Node, UnitsDataArray

class Algorithm(Node):
    def execute(self, coordinates, params=None, output=None):
        coordinates, params, out = \
            self._execute_common(coordinates, params, output)
        
        kwargs = OrderedDict()
        for name in self.trait_names():
            node = getattr(self, name)
            if isinstance(node, Node):
                if self.implicit_pipeline_evaluation:
                    node.execute(coordinates, params, output)
                kwargs[name] = node.output

        if output is None:
            res = self.algorithm(**kwargs)
            if isinstance(res, UnitsDataArray):
                self.output = res
            else:
                self.output = xr.align(*kwargs.values())[0]
                self.output[:] = res
        else:
            output[:] = self.algorithm(**kwargs)
            self.output = output
            
        self.evaluted = True
        return self.output
        
    def algorithm(self, **kwargs):
        """
        """
        raise NotImplementedError
        
class SinCoords(Algorithm):
    def algorithm(self):
        out = self.initialize_output_array('ones')
        crds = np.meshgrid(*out.coords.values()[::-1])
        for crd in crds:
            out *= np.sin(np.pi * crd / 90.0)
        return out

class Arithmetic(Algorithm):
    A = tl.Instance(Node)
    B = tl.Instance(Node)
    eqn = tl.Unicode(default_value='A+B')
    
    def algorithm(self, A, B):
        if 'eqn' not in self.params:
            eqn = self.eqn
        else: 
            eqn = self.params['eqn']
        return ne.evaluate(eqn)
        
if __name__ == "__main__":
    a = SinCoords()
    coords = Coordinate(lat=[-90, 90, 1.], lon=[-180, 180, 1.], 
                        order=['lat', 'lon'])
    o = a.execute(coords)
    a2 = Arithmetic(A=a, B=a)
    o2 = a2.execute(coords, params={'eqn': '2*abs(A) - B'})

    print ("Done")        
