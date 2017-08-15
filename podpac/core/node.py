from __future__ import division, print_function, absolute_import

import operator
import xarray as xr
import numpy as np
import traitlets as tl
from pint import UnitRegistry
ureg = UnitRegistry()

from podpac.core.coordinate import Coordinate

class UnitDataArray(xr.DataArray):
    """Like xarray.DataArray, but transfers units
     """
    def __array_wrap__(self, obj, context=None):
        new_var = super().__array_wrap__(obj, context)
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
        x = super().__pow__(other)
        if self.attrs.get("units"):
            x.attrs["units"] = pow(
                ureg.Quantity(1, getattr(self, "units", "1")),
                ureg.Quantity(other, getattr(other, "units", "1"))
                ).u
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
        x = getattr(super(UnitDataArray, self), meth)(other)
        return self._apply_binary_op_to_units(getattr(operator, tp), other, x)
    func.__name__ = meth
    setattr(UnitDataArray, meth, func)
for tp in ("add", "sub", "mod", "floordiv"): #, "divmod", ):
    meth = "__{:s}__".format(tp)
    def func(self, other, meth=meth, tp=tp):
        multiplier = self._get_unit_multiplier(other)
        x = getattr(super(UnitDataArray, self), meth)(other * multiplier)
        return self._apply_binary_op_to_units(getattr(operator, tp), other, x)
    func.__name__ = meth
    setattr(UnitDataArray, meth, func)
for tp in ("lt", "le", "eq", "ne", "gt", "ge"):
    meth = "__{:s}__".format(tp)
    def func(self, other, meth=meth, tp=tp):
        multiplier = self._get_unit_multiplier(other)
        return getattr(super(UnitDataArray, self), meth)(other * multiplier)
    func.__name__ = meth
    setattr(UnitDataArray, meth, func)    
    
del func

class Node(tl.HasTraits):

    output = tl.Instance(xr.Dataset, allow_none=True)
    native_coordinates = tl.Instance(Coordinate)
    evaluted = tl.Bool(default_value=False)
    evaluated_coordinates = tl.Instance(Coordinate)
    params = tl.Dict(default_value=None, allow_none=True)


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
        if output is not None:
            # This should be a reference, not a copy
            # subselect if neccessary
            out = output[coordinates.get_coord] 

        return coordinates, params, out


    def get_description(self):
        """
        This is to get the pipeline lineage or provenance
        """
        raise NotImplementedError

    def get_intersecting_coordinates(self, evaluated=None, native=None):
        """ Helper function to get the reqions where the requested and
        native coordinates intersect.

        Parameters
        -------------
        evaluated: Coordinate
            Coordinates where the Node should be evaluated
        native: Coordinate
            The Node's native Coordinates

        Returns
        ---------
        en_intersect: Coordinate
            The coordinates of the overlap at the resolution/projection/scale
            of the evaluated Coordinate object
        ne_intersect: Coordinate
            Like en_intersect, but at the resolution/projection/scale of the
            native coordinates
        """
        if evaluated is None and self.evaluated:
            evaluated = self.evaluated_coordinates
        if native is None:
            native = self.native_coordinates

        intersect = native.intersect(evaluated)

        return intersect

    def initialize_dataset(self, initial_value=0, dtype=np.float):
        pass

if __name__ == "__main__":
    
    print ("Done")