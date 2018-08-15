"""
Algorithm Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import inspect
import numpy as np
import xarray as xr
import traitlets as tl

try:
    import numexpr as ne
except: 
    ne = None

from podpac.core.coordinate import Coordinate, convert_xarray_to_podpac
from podpac.core.node import Node
from podpac.core.node import COMMON_NODE_DOC
from podpac.core.utils import common_doc

COMMON_DOC = COMMON_NODE_DOC.copy()

class Algorithm(Node):
    """Base node for any algorithm or computation node. 
    
    Notes
    ------
    Developers of new Algorithm nodes need to implement the `algorithm` method. 
    """
    
    @common_doc(COMMON_DOC)
    def execute(self, coordinates, output=None, method=None):
        """Executes this nodes using the supplied coordinates. 
        
        Parameters
        ----------
        coordinates : podpac.Coordinate
            {evaluated_coordinates}
        output : podpac.UnitsDataArray, optional
            {execute_out}
        method : str, optional
            {execute_method}
        
        Returns
        -------
        {execute_return}
        """
        self.evaluated_coordinates = coordinates
        self.output = output

        coords = None
        for name in self.trait_names():
            node = getattr(self, name)
            if isinstance(node, Node):
                if self.implicit_pipeline_evaluation:
                    node.execute(coordinates, method)
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
            Key-word arguments for the algorithm
        
        Raises
        ------
        NotImplementedError
            Description
        """
        raise NotImplementedError

    @property
    def definition(self):
        """Pipeline node definition. 

        Returns
        -------
        OrderedDict
            Extends base description by adding 'inputs'
        """
        d = self.base_definition()
        
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
        
        return d

class Arange(Algorithm):
    '''A simple test node that gives each value in the output a number.
    '''

    def algorithm(self):
        """Uses np.arange to give each value in output a unique number
        
        Returns
        -------
        UnitsDataArray
            A row-majored numbered array of the requested size. 
        """
        out = self.initialize_output_array('ones')
        return out * np.arange(out.size).reshape(out.shape)
      

class CoordData(Algorithm):
    """Extracts the coordinates from a request and makes it available as a data
    
    Attributes
    ----------
    coord_name : str
        Name of coordinate to extract (one of lat, lon, time, alt)
    """
    
    coord_name = tl.Unicode('').tag(attr=True)

    def algorithm(self):
        """Extract coordinate from request and makes data available.
        
        Returns
        -------
        UnitsDataArray
            The coordinates as data for the requested coordinate.
        """
        coord_name = self.coord_name
        ec = self.evaluated_coordinates
        if coord_name not in ec.dims:
            raise ValueError('Coordinate name not in evaluated coordinates')
       
        c = ec[coord_name]
        data = c.coordinates
        coords = Coordinate(order=[coord_name], **{coord_name: c})
        return self.initialize_coord_array(coords, init_type='data', fillval=data)


class SinCoords(Algorithm):
    """A simple test node that creates a data based on coordinates and trigonometric (sin) functions. 
    """
    
    def algorithm(self):
        """Computes sinusoids of all the coordinates. 
        
        Returns
        -------
        UnitsDataArray
            Sinusoids of a certain period for all of the requested coordinates
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
    """Create a simple point-by-point computation of up to 7 different input nodes.
    
    Attributes
    ----------
    A : podpac.Node
        An input node that can be used in a computation. 
    B : podpac.Node
        An input node that can be used in a computation. 
    C : podpac.Node
        An input node that can be used in a computation. 
    D : podpac.Node
        An input node that can be used in a computation. 
    E : podpac.Node
        An input node that can be used in a computation. 
    F : podpac.Node
        An input node that can be used in a computation. 
    G : podpac.Node
        An input node that can be used in a computation. 
    eqn : str
        An equation stating how the datasources can be combined. 
        Parameters may be specified in {}'s
        
    Examples
    ----------
    a = SinCoords()
    b = Arange()
    arith = Arithmetic(A=a, B=b, eqn = 'A * B + {offset}', params={'offset': 1})
    """
    
    A = tl.Instance(Node)
    B = tl.Instance(Node, allow_none=True)
    C = tl.Instance(Node, allow_none=True)
    D = tl.Instance(Node, allow_none=True)
    E = tl.Instance(Node, allow_none=True)
    F = tl.Instance(Node, allow_none=True)
    G = tl.Instance(Node, allow_none=True)
    eqn = tl.Unicode().tag(attr=True)
    params = tl.Dict().tag(attr=True)
    
    def algorithm(self):
        """Summary
        
        Returns
        -------
        UnitsDataArray
            Description
        """
        
        if self.eqn == '':
            raise ValueError("Cannot evaluate Arithmetic node: 'eqn' attribute missing or empty")

        eqn = self.eqn.format(**self.params)        
        
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

