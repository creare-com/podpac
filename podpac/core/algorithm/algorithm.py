from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
import inspect
import numpy as np
import xarray as xr
try: 
    import numexpr as ne
except: 
    ne = None
import traitlets as tl

from podpac.core.units import UnitsDataArray
from podpac.core.coordinate import Coordinate
from podpac.core.node import Node

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

    @property
    def definition(self):
        d = OrderedDict()
        d['node'] = self.podpac_path
        
        # this first version is nicer, but the gettattr(self, ref) can take a
        # a long time if it is has a default value or is a property

        # d['inputs'] = {
        #     ref:getattr(self, ref)
        #     for ref in self.trait_names()
        #     if isinstance(getattr(self, ref, None), Node)
        # }
        
        d['inputs'] = {
            ref:getattr(self, ref)
            for ref, trait in self.traits().items()
            if hasattr(trait, 'klass') and Node in inspect.getmro(trait.klass)
        }
        
        if self.params:
            d['params'] = self.params
            
        return d
        
class SinCoords(Algorithm):
    def algorithm(self):
        out = self.initialize_output_array('ones')
        crds = np.meshgrid(*out.coords.values()[::-1])
        for crd in crds:
            out *= np.sin(np.pi * crd / 90.0)
        return out

class Arithmetic(Algorithm):
    A = tl.Instance(Node)
    B = tl.Instance(Node, allow_none=True)
    C = tl.Instance(Node, allow_none=True)
    D = tl.Instance(Node, allow_none=True)
    E = tl.Instance(Node, allow_none=True)
    F = tl.Instance(Node, allow_none=True)
    G = tl.Instance(Node, allow_none=True)
    eqn = tl.Unicode(default_value='A+B+C+D+E+F+G')
    
    def algorithm(self, A, B=None, C=None, D=None, E=None, F=None, G=None):
        if 'eqn' not in self.params:
            eqn = self.eqn
        else: 
            eqn = self.params['eqn']
        f_locals = locals()
        fields = [f_locals[f] for f in 'ABCDEFG' if f_locals[f] is not None]
        keys = [f for f in 'ABCDEFG' if f_locals[f] is not None]
        res = xr.broadcast(*fields)
        for key, r in zip(keys, res):
            f_locals[key] = r
        
        out = A.copy()
        if ne is None:
            out.data[:] = eval(eqn.format(**self.params),
                                              local_dict=f_locals)            
        else:
            out.data[:] = ne.evaluate(eqn.format(**self.params),
                                  local_dict=f_locals)
        return out

    @property
    def definition(self):
        d = super(Arithmetic, self).definition
        
        if 'eqn' not in self.params:
            d['params']['eqn'] = self.eqn

        return d
        
if __name__ == "__main__":
    a = SinCoords()
    coords = Coordinate(lat=[-90, 90, 1.], lon=[-180, 180, 1.], 
                        order=['lat', 'lon'])
    o = a.execute(coords)
    a2 = Arithmetic(A=a, B=a)
    o2 = a2.execute(coords, params={'eqn': '2*abs(A) - B'})

    print ("Done")        
