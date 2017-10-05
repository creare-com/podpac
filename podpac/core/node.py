from __future__ import division, print_function, absolute_import

import os
import operator
from collections import OrderedDict
import xarray as xr
import numpy as np
import traitlets as tl
import matplotlib.colors, matplotlib.cm
import matplotlib.pyplot as plt
from pint import UnitRegistry
from pint.unit import _Unit
ureg = UnitRegistry()

try:
    import cPickle  # Python 2.7
except:
    import _pickle as cPickle

#import podpac.core.coordinate
from podpac import settings

class UnitsNode(tl.TraitType):      
    info_text = "A UnitDataArray with specified dimensionality"
    def validate(self, obj, value):
        if isinstance(value, Node):
            if 'units' in self.metadata and value.units is not None:
                u = ureg.check(self.metadata['units'])(lambda x: x)(value.units)
                return value
        self.error(obj, value)
 
class Units(tl.TraitType):
    info_text = "A pint Unit"
    #default_value = None
    def validate(self, obj, value):
        if isinstance(value, _Unit):
            return value
        self.error(obj, value)

class UnitsDataArray(xr.DataArray):
    """Like xarray.DataArray, but transfers units
     """
    def __array_wrap__(self, obj, context=None):
        new_var = super(UnitsDataArray, self).__array_wrap__(obj, context)
        if self.attrs.get("units"):
            new_var.attrs["units"] = context[0](ureg.Quantity(1, self.attrs.get("units"))).u
        return new_var

    def _apply_binary_op_to_units(self, func, other, x):
        if self.attrs.get("units", None) or getattr(other, 'units', None):
            x.attrs["units"] = func(ureg.Quantity(1, getattr(self, "units", "1")),
                                    ureg.Quantity(1, getattr(other, "units", "1"))).u
        return x

    def _get_unit_multiplier(self, other):
        multiplier = 1
        if self.attrs.get("units", None) or getattr(other, 'units', None):
            otheru = ureg.Quantity(1, getattr(other, "units", "1"))
            myu = ureg.Quantity(1, getattr(self, "units", "1"))
            multiplier = otheru.to(myu.u).magnitude
        return multiplier

    # pow is different because resulting unit depends on argument, not on
    # unit of argument (which must be unitless)
    def __pow__(self, other):
        x = super(UnitsDataArray, self).__pow__(other)
        if self.attrs.get("units"):
            x.attrs["units"] = pow(
                ureg.Quantity(1, getattr(self, "units", "1")),
                ureg.Quantity(other, getattr(other, "units", "1"))
                ).u
        return x
    
    def _copy_units(self, x):
        if self.attrs.get("units", None):
            x.attrs["units"] = self.attrs.get('units')
        return x
    
    def to(self, unit):
        x = self.copy()
        if self.attrs.get("units", None):
            myu = ureg.Quantity(1, getattr(self, "units", "1"))
            multiplier = myu.to(unit).magnitude
            x = x * multiplier
            x.attrs['units'] = unit
        return x
    
    def to_base_units(self):
        if self.attrs.get("units", None):
            myu = ureg.Quantity(1, getattr(self, "units", "1")).to_base_units()        
            return self.to(myu.u)
        else:
            return self.copy()
        
for tp in ("mul", "matmul", "truediv", "div"):
    meth = "__{:s}__".format(tp)
    def func(self, other, meth=meth, tp=tp):
        x = getattr(super(UnitsDataArray, self), meth)(other)
        return self._apply_binary_op_to_units(getattr(operator, tp), other, x)
    func.__name__ = meth
    setattr(UnitsDataArray, meth, func)
for tp in ("add", "sub", "mod", "floordiv"): #, "divmod", ):
    meth = "__{:s}__".format(tp)
    def func(self, other, meth=meth, tp=tp):
        multiplier = self._get_unit_multiplier(other)
        x = getattr(super(UnitsDataArray, self), meth)(other * multiplier)
        return self._apply_binary_op_to_units(getattr(operator, tp), other, x)
    func.__name__ = meth
    setattr(UnitsDataArray, meth, func)
for tp in ("lt", "le", "eq", "ne", "gt", "ge"):
    meth = "__{:s}__".format(tp)
    def func(self, other, meth=meth, tp=tp):
        multiplier = self._get_unit_multiplier(other)
        return getattr(super(UnitsDataArray, self), meth)(other * multiplier)
    func.__name__ = meth
    setattr(UnitsDataArray, meth, func)    
for tp in ("mean", 'min', 'max'):
        def func(self, *args, **kwargs):
            x = getattr(super(UnitsDataArray, self), tp)(*args, **kwargs)
            return self._copy_units(x)
        func.__name__ = tp
        setattr(UnitsDataArray, tp, func)        
del func

class Style(tl.HasTraits):
    node = tl.Instance('podpac.core.node.Node', allow_none=False)
    name = tl.Unicode()
    @tl.default('name')
    def _name_default(self):
        return self.node.__class__.__name__
        
    units = Units(allow_none=True)
    @tl.default('units')
    def _units_default(self):
        return self.node.units
    
    is_enumerated = tl.Bool(default_value=False)
    enumeration_legend = tl.Tuple(trait=tl.Unicode)
    enumeration_colors = tl.Tuple(trait=tl.Tuple)
    
    clim = tl.List(default_value=[None, None])
    cmap = tl.Instance(matplotlib.colors.Colormap)
    tl.default('cmap')
    def _cmap_default(self):
        return matplotlib.cm.get_cmap('viridis')
    
class Node(tl.HasTraits):
    output = tl.Instance(UnitsDataArray, allow_none=True, default_value=None)
    @tl.default('output')
    def _output_default(self):
        return self.initialize_output_array('nan')
    
    native_coordinates = tl.Instance('podpac.core.coordinate.Coordinate',
                                     allow_none=True)
    evaluted = tl.Bool(default_value=False)
    implicit_pipeline_evaluation = tl.Bool(default_value=True,
        help="Evaluate the pipeline implicitly (True, Default)")
    evaluated_coordinates = tl.Instance('podpac.core.coordinate.Coordinate', 
                                        allow_none=True)
    params = tl.Dict(default_value=None, allow_none=True)
    units = Units(default_value=None, allow_none=True)
    dtype = tl.Any(default_value=float)
    cache_type = tl.Enum([None, 'disk', 'ram'], allow_none=True)    
    
    style = tl.Instance(Style)
    @tl.default('style')
    def _style_default(self):
        return Style(node=self)
    
    @property
    def shape(self):
        # Changes here likely will also require changes in initialize_output_array
        ev = self.evaluated_coordinates
        nv = self.native_coordinates
        if ev is not None:
            stacked = ev.stacked_coords
            stack_dict = OrderedDict([(c, True) for c, v in ev._coords.items() if v.stacked != 1])
            if nv is not None:
                shape = []
                for c in nv.coords:
                    if c in ev.coords:
                        shape.append(ev[c].size)
                    elif c in stacked and stack_dict[stacked[c]]:
                        shape.append(ev[stacked[c]].size)
                        stack_dict[stacked[c]] = False
                    elif c not in stacked:
                        shape.append(nv[c].size)
            else:
                shape = ev.shape
            return shape
        else:
            return nv.shape    
    
    def __init__(self, *args, **kwargs):
        """ Do not overwrite me """
        targs, tkwargs = self._first_init(*args, **kwargs)
        super(Node, self).__init__(*targs, **tkwargs)
        self.init()

    def _first_init(*args, **kwargs):
        """ Only overwrite me if absolutely necessary """
        return args, kwargs

    def init(self):
        pass

    def execute(self, coordinates, params=None, output=None):
        """ This is the common interface used for ALL nodes. Pipelines only
        understand this and get_description. 
        """
        raise NotImplementedError

    def _execute_common(self, coordinates, params=None, output=None):
        """ 
        Common input sanatization etc for when executing a node 
        """
        self.evaluated_coordinates = coordinates
        self.params = params
        out = None
        if output is not None:
            # This should be a reference, not a copy
            # subselect if neccessary
            out = output.loc[coordinates.coords] 
            self.output[:] = out
        else:
            self.output = self.initialize_output_array()

        return coordinates, params, out

    def get_description(self):
        """
        This is to get the pipeline lineage or provenance
        """
        raise NotImplementedError
    
    def initialize_output_array(self, init_type='nan', fillval=0, style=None,
                              no_style=False, shape=None, coords=None,
                              dims=None, units=None, dtype=np.float, **kwargs):
        # Changes here likely will also require changes in shape
        if coords is None: 
            coords = self.evaluated_coordinates.coords
            stacked_coords = self.evaluated_coordinates.stacked_coords
        else: 
            stacked_coords = Coordinate.get_stacked_coord_dict(coords)
        if not isinstance(coords, dict): coords = dict(coords)
        if dims is None:
            if self.native_coordinates is not None:
                dims = self.native_coordinates.dims
                dims = [stacked_coords.get(d, d) for d in dims]
                dims = [d for i, d in enumerate(dims) if d not in dims[:i]]
            else:
                dims = coords.keys()
        if self.native_coordinates is not None:
            crds = OrderedDict()
            for c in self.native_coordinates.coords:
                if c in coords:
                    crds[c] = coords[c]
                elif c in stacked_coords:
                    crds[stacked_coords[c]] = coords[stacked_coords[c]]
                else:
                    crds[c] = self.native_coordinates.coords[c]
        else:
            crds = coords        
        return self.initialize_array(init_type, fillval, style, no_style, shape,
                                     crds, dims, units, dtype, **kwargs)
    
    def initialize_coord_array(self, coords, init_type='nan', fillval=0, 
                               style=None, no_style=False, units=None,
                               dtype=np.float, **kwargs):
        return self.initialize_array(init_type, fillval, style, no_style, 
                                     coords.shape, coords.coords, coords.dims,
                                     units, dtype, **kwargs)
    

    def initialize_array(self, init_type='nan', fillval=0, style=None,
                              no_style=False, shape=None, coords=None,
                              dims=None, units=None, dtype=np.float,  **kwargs):
        """Initialize output data array

        Parameters
        -----------
        init_type : str, optional
            How to initialize the array. Options are:
                nan: uses np.full(..., np.nan) (Default option)
                empty: uses np.empty 
                zeros: uses np.zeros()
                ones: uses np.ones
                full: uses np.full(..., fillval)
                data: uses the fillval as the input array
        fillval : number, optional
            used if init_type=='full' or 'data', default = 0
        style : Style, optional
            The style to use for plotting. Uses self.style by default
        no_style : bool, optional
            Default is False. If True, self.style will not be assigned to 
            arr.attr['layer_style']
        shape : tuple, optional
            Shape of array. Uses self.shape by default.
        coords : dict/list, optional
            input to UnitsDataArray, uses self.coords by default
        dims : list(str), optional
            input to UnitsDataArray, uses self.native_coords.dims by default
        units : pint.unit.Unit, optional
            Default is self.units The Units for the data contained in the 
            DataArray
        dtype : type, optional
            Default is np.float. Datatype used by default
        kwargs : kwargs
            other keyword arguments passed to UnitsDataArray

        Returns
        -------
        arr : UnitsDataArray
            Unit-aware xarray DataArray of the desired size initialized using 
            the method specified
        """
        if style is None: style = self.style
        if shape is None: shape = self.shape
        if units is None: units = self.units
        if not isinstance(coords, dict): coords = dict(coords)

        if init_type == 'empty':
            data = np.empty(shape)
        elif init_type == 'nan':
            data = np.full(shape, np.nan)
        elif init_type == 'zeros':
            data = np.zeros(shape)
        elif init_type == 'ones':
            data = np.ones(shape)
        elif init_type == 'full':
            data = np.full(shape, fillval)
        elif init_type == 'data':
            data = fillval
        else:
            raise ValueError("Unknown init_type=%" % init_type)
        
        x = UnitsDataArray(data, coords=coords, dims=dims, **kwargs)
        
        if not no_style:
            x.attrs['layer_style'] = style
        if units is not None:
            x.attrs['units'] = units
        x.attrs['params'] = self.params
        return x

    def plot(self, show=True, interpolation='none', **kwargs):
        """
        Plot function to display the output
        
        TODO: Improve this substantially please
        """
        if kwargs:
            plt.imshow(self.output.data, cmap=self.style.cmap,
                       interpolation=interpolation, **kwargs)
        else:
            self.output.plot()
        if show:
            plt.show()
            
    @property
    def cache_dir(self):
        basedir = settings.CACHE_DIR
        subdir = str(self.__class__)[8:-2].split('.')
        dirs = [basedir] + subdir
        return os.path.join(*dirs)
    
    def cache_path(self, filename):
        pre = str(self.source).replace('/', '_').replace('\\', '_').replace(':', '_')
        return os.path.join(self.cache_dir, pre  + '_' + filename)
    
    def cache_obj(self, obj, filename):
        path = self.cache_path(filename)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        with open(path, 'wb') as fid:
            cPickle.dump(obj, fid)#, protocol=cPickle.HIGHEST_PROTOCOL)
            
    def load_cached_obj(self, filename):
        path = self.cache_path(filename)
        with open(path, 'rb') as fid:
            obj = cPickle.load(fid)
        return obj

if __name__ == "__main__":
    a1 = UnitsDataArray(np.ones((4,3)), dims=['lat', 'lon'],
                           attrs={'units': ureg.meter})
    a2 = UnitsDataArray(np.ones((4,3)), dims=['lat', 'lon'],
                           attrs={'units': ureg.kelvin}) 

    np.mean(a1)    
    np.std(a1)
    print ("Done")
