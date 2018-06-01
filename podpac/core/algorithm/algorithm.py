"""
Algorithm Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import inspect
from collections import OrderedDict
import numpy as np
import xarray as xr
import traitlets as tl

try:
    import numexpr as ne
except: 
    ne = None

from podpac.core.coordinate import Coordinate, convert_xarray_to_podpac
from podpac.core.node import Node


class Algorithm(Node):
    """Summary
    
    Attributes
    ----------
    evaluated : bool
        Description
    evaluated_coordinates : TYPE
        Description
    output : TYPE
        Description
    params : TYPE
        Description
    """
    
    def execute(self, coordinates, params=None, output=None):
        """Summary
        
        Parameters
        ----------
        coordinates : TYPE
            Description
        params : None, optional
            Description
        output : None, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        self.evaluated_coordinates = coordinates
        self.params = params
        self.output = output

        coords = None
        for name in self.trait_names():
            node = getattr(self, name)
            if isinstance(node, Node):
                if self.implicit_pipeline_evaluation:
                    if params is None:
                        node.execute(coordinates, params)
                    else:
                        node.execute(coordinates, params.get(name, {}))
                # accumulate coordinates
                if coords is None:
                    coords = convert_xarray_to_podpac(node.output.coords)
                else:
                    coords = coords.add_unique(
                        convert_xarray_to_podpac(node.output.coords))
        if coords is None:
            coords = coordinates

        result = self.algorithm()
        if isinstance(result, np.ndarray):
            if self.output is None:
                self.output = self.initialize_coord_array(coords)
            self.output.data[:] = result
        else:
            dims = [d for d in self.evaluated_coordinates.dims if d in result.dims]
            if self.output is None:
                coords = convert_xarray_to_podpac(result.coords)
                self.output = self.initialize_coord_array(coords)
            self.output[:] = result
            self.output = self.output.transpose(*dims) # split into 2nd line to avoid broadcasting issues with slice [:]
        self.evaluated = True
        return self.output
        
    def algorithm(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
            Description
        
        Raises
        ------
        NotImplementedError
            Description
        """
        raise NotImplementedError

    @property
    def definition(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        d = self._base_definition()
        
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
            if hasattr(trait, 'klass') and Node in inspect.getmro(trait.klass) and getattr(self, ref) is not None
        }
        
        if self.params:
            d['params'] = self.params
        
        return d

class Arange(Algorithm):
    '''A simple test node 
    '''

    def algorithm(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        out = self.initialize_output_array('ones')
        return out * np.arange(out.size).reshape(out.shape)
      

class CoordData(Algorithm):
    """Summary
    
    Attributes
    ----------
    coord_name : TYPE
        Description
    """
    
    coord_name = tl.Unicode('')

    def algorithm(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if self.params:
            coord_name = self.params.get('coord_name', self.coord_name)
        else: coord_name = self.coord_name
        ec = self.evaluated_coordinates
        if coord_name not in ec.dims:
            return xr.DataArray([1]).min()
       
        c = ec[coord_name]
        data = c.coordinates
        coords = Coordinate(OrderedDict(**{coord_name:c}))
        return self.initialize_coord_array(coords, init_type='data', fillval=data)


class SinCoords(Algorithm):
    """Summary
    """
    
    def algorithm(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        out = self.initialize_output_array('ones')
        crds = list(out.coords.values())
        try:
            i_time = list(out.coords.keys()).index('time')
            crds[i_time] = crds[i_time].astype('datetime64[h]').astype(float)
        except ValueError:
            pass
        
        crds = np.meshgrid(*crds, indexing='ij')
        for crd in crds:
            out *= np.sin(np.pi * crd / 90.0)
        return out


class Arithmetic(Algorithm):
    """Summary
    
    Attributes
    ----------
    A : TYPE
        Description
    B : TYPE
        Description
    C : TYPE
        Description
    D : TYPE
        Description
    E : TYPE
        Description
    eqn : TYPE
        Description
    F : TYPE
        Description
    G : TYPE
        Description
    """
    
    A = tl.Instance(Node)
    B = tl.Instance(Node, allow_none=True)
    C = tl.Instance(Node, allow_none=True)
    D = tl.Instance(Node, allow_none=True)
    E = tl.Instance(Node, allow_none=True)
    F = tl.Instance(Node, allow_none=True)
    G = tl.Instance(Node, allow_none=True)
    eqn = tl.Unicode(default_value='A+B+C+D+E+F+G')
    
    def algorithm(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if self.params:
            eqn = self.params.get('eqn', self.eqn)
            eqn = eqn.format(**self.params)
        else: eqn = self.eqn
        
        fields = [f for f in 'ABCDEFG' if getattr(self, f) is not None]
          
        res = xr.broadcast(*[getattr(self, f).output for f in fields])
        f_locals = dict(zip(fields, res))

        if ne is None:
            result = eval(eqn, f_locals)
        else:
            result = ne.evaluate(eqn, f_locals)
        res = res[0].copy()  # Make an xarray object with correct dimensions
        res[:] = result
        return res

    @property
    def definition(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        d = super(Arithmetic, self).definition
        
        if self.params and 'eqn' not in self.params:
            d['params']['eqn'] = self.eqn

        return d
        
if __name__ == "__main__":
    import podpac
    a = SinCoords()
    coords = podpac.coordinate(lat=[-90, 90, 1.], lon=[-180, 180, 1.], order=['lat', 'lon'])
    o = a.execute(coords)
    a2 = Arithmetic(A=a, B=a)
    o2 = a2.execute(coords, params={'eqn': '2*abs(A) - B'})

    print ("Done")
