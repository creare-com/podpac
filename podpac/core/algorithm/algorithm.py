from __future__ import division, unicode_literals, print_function, absolute_import

import warnings
from collections import OrderedDict
from operator import mul
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
        self.evaluated_coordinates = coordinates
        self.params = params
        self.output = output

        if self.output is None:
            self.output = self.initialize_output_array()

        if self.implicit_pipeline_evaluation:
            for name in self.trait_names():
                node = getattr(self, name)
                if isinstance(node, Node):
                    node.execute(coordinates, params)

        result = self.algorithm()
        self.output[:] = result #.transpose(*self.output.dims) # is this necessary?
        self.evaluated = True
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
            if hasattr(trait, 'klass') and
               Node in inspect.getmro(trait.klass) and
               getattr(self, ref) is not None
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
    
    def algorithm(self):
        eqn = self.params.get('eqn', self.eqn)
        eqn = eqn.format(**self.params)
        
        fields = [f for f in 'ABCDEFG' if getattr(self, f) is not None]
        res = xr.broadcast(*[getattr(self, f).output for f in fields])
        f_locals = dict(zip(fields, res))

        if ne is None:
            result = eval(eqn, f_locals)
        else:
            result = ne.evaluate(eqn, f_locals)

        return result

    @property
    def definition(self):
        d = super(Arithmetic, self).definition
        
        if 'eqn' not in self.params:
            d['params']['eqn'] = self.eqn

        return d

class Reduce(Algorithm):
    input_coordinates = tl.Instance(Coordinate)
    input_node = tl.Instance(Node)

    @property
    def dims(self):
        """
        Validates and translates requested reduction dimensions.
        """

        input_dims = self.input_coordinates.dims

        if self.params is None or 'dims' not in self.params:
            return input_dims

        valid_dims = []
        if 'time' in input_dims:
            valid_dims.append('time')
        if 'lat_lon' in input_dims:
            valid_dims.append('lat_lon')
        if 'lat' in input_dims and 'lon' in input_dims:
            valid_dims.append('lat_lon')
        if 'alt' in input_dims:
            valid_dims.append('alt')

        params_dims = self.params['dims']
        if not isinstance(params_dims, (list, tuple)):
            params_dims = [params_dims]

        dims = []
        for dim in params_dims:
            if dim not in valid_dims:
                raise ValueError("Invalid Reduce dimension: %s" % dim)    
            elif dim in input_dims:
                dims.append(dim)
            elif dim == 'lat_lon':
                dims.append('lat')
                dims.append('lon')
        
        return dims

    def get_reduced_coordinates(self, coordinates):
        dims = self.dims
        kwargs = {}
        order = []
        for dim in coordinates.dims:
            if dim in self.dims:
                continue
            
            kwargs[dim] = coordinates[dim]
            order.append(dim)
        
        if order:
            kwargs['order'] = order

        return Coordinate(**kwargs)

    @property
    def chunk_size(self):
        if 'chunk_size' not in self.params:
            # return self.input_coordinates.shape
            return None

        if self.params['chunk_size'] == 'auto':
            return 1024**2 # TODO

        return self.params['chunk_size']

    @property
    def chunk_shape(self):
        if self.chunk_size is None:
            # return self.input_coordinates.shape
            return None

        chunk_size = self.chunk_size
        coords = self.input_coordinates
        
        d = {k:coords[k].size for k in coords.dims if k not in self.dims}
        s = reduce(mul, d.values(), 1)
        for dim in self.dims:
            n = chunk_size // s
            if n == 0:
                d[dim] = 1
            elif n < coords[dim].size:
                d[dim] = n
            else:
                d[dim] = coords[dim].size
            s *= d[dim]

        return [d[dim] for dim in coords.dims]

    def iteroutputs(self):
        for chunk in self.input_coordinates.iterchunks(self.chunk_shape):
            yield self.input_node.execute(chunk, self.params)

    def execute(self, coordinates, params=None, output=None):
        self.input_coordinates = coordinates
        self.params = params
        self.output = output

        self.evaluated_coordinates = self.get_reduced_coordinates(coordinates)

        if self.output is None:
            self.output = self.initialize_output_array()
            
        if self.chunk_size and self.chunk_size < reduce(mul, coordinates.shape, 1):
            result = self.reduce_chunked(self.iteroutputs())
        else:
            if self.implicit_pipeline_evaluation:
                self.input_node.execute(coordinates, params)
            result = self.reduce(self.input_node.output)

        if self.output.shape is (): # or self.evaluated_coordinates is None
            self.output.data = result
        else:
            self.output[:] = result #.transpose(*self.output.dims) # is this necessary?
        
        self.evaluated = True

        return self.output

    def reduce(self, x):
        """
        Reduce a full array, e.g. x.mean(self.dims).

        Must be defined in each child.
        """

        raise NotImplementedError

    def reduce_chunked(self, xs):
        """
        Reduce a list of xs with a memory-effecient iterative algorithm.

        Optionally defined in each child.
        """

        warnings.warn("No reduce_chunked method defined, using one-step reduce")
        x = self.input_node.execute(self.input_coordinates, self.params)
        return self.reduce(x)

    @property
    def evaluated_hash(self):
        if self.input_coordinates is None:
            raise Exception("node not evaluated")
            
        return self.get_hash(self.input_coordinates, self.params)

    @property
    def latlon_bounds_str(self):
        return self.input_coordinates.latlon_bounds_str

class Mean(Reduce):
    def reduce(self, x):
        return x.mean(dim=self.dims)

    def reduce_chunked(self, xs):
        s = xr.zeros_like(self.output) # alt: s = np.zeros(self.shape)
        n = xr.zeros_like(self.output)
        for x in xs:
            # TODO efficency
            s += x.sum(dim=self.dims)
            n += np.isfinite(x).sum(dim=self.dims)
        output = s / n
        return output

class Sum(Reduce):
    def reduce(self, x):
        return x.sum(dim=self.dims)

    def reduce_chunked(self, xs):
        s = xr.zeros_like(self.output)
        for x in xs:
            s += x.sum(dim=self.dims)
        return s

class Count(Reduce):
    def reduce(self, x):
        return np.isfinite(x).sum(dim=self.dims)

    def reduce_chunked(self, xs):
        sn= xr.zeros_like(self.output)
        for x in xs:
            n += np.isfinite(x).sum(dim=self.dims)

        
if __name__ == "__main__":
    a = SinCoords()
    coords = Coordinate(lat=[-90, 90, 1.], lon=[-180, 180, 1.], 
                        order=['lat', 'lon'])
    o = a.execute(coords)
    a2 = Arithmetic(A=a, B=a)
    o2 = a2.execute(coords, params={'eqn': '2*abs(A) - B'})

    from time import time
    from podpac.datalib.smap import SMAP

    smap = SMAP(product='SPL4SMAU.003')
    coords = smap.native_coordinates
    
    coords = Coordinate(
        time=coords.coords['time'][:5],
        lat=[45., 66., 50], lon=[-80., -70., 20],
        order=['time', 'lat', 'lon'])

    smap_mean = Mean(input_node=SMAP(product='SPL4SMAU.003'))
    
    print("lat_lon means")
    mll = smap_mean.execute(coords, {'dims':'lat_lon'})
    mll2 = smap_mean.execute(coords, {'dims':'lat_lon', 'chunk_size':'auto'})
    mll3 = smap_mean.execute(coords, {'dims':'lat_lon', 'chunk_size': 2000})
    mll4 = smap_mean.execute(coords, {'dims':'lat_lon', 'chunk_size': 6000})
    
    print("time means")
    mt = smap_mean.execute(coords, {'dims':'time'})
    mt2 = smap_mean.execute(coords, {'dims':'time', 'chunk_size':'auto'})
    mt3 = smap_mean.execute(coords, {'dims':'time', 'chunk_size': 2000})

    print ("full means")
    mllt = smap_mean.execute(coords, {'dims':['lat_lon', 'time']})
    mllt2 = smap_mean.execute(coords, {'dims': ['lat_lon', 'time'], 'chunk_size': 1000})
    mllt3 = smap_mean.execute(coords, {})

    print("lat_lon counts")
    smap_count = Count(input_node=SMAP(product='SPL4SMAU.003'))
    mll = smap_count.execute(coords, {'dims':'lat_lon'})
    mll2 = smap_count.execute(coords, {'dims':'lat_lon', 'chunk_size':'auto'})
    mll3 = smap_count.execute(coords, {'dims':'lat_lon', 'chunk_size': 2000})
    mll4 = smap_count.execute(coords, {'dims':'lat_lon', 'chunk_size': 6000})

    print ("Done")
