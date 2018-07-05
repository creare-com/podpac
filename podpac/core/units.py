"""Units module summary

Attributes
----------
ureg : TYPE
    Description
"""

from copy import deepcopy
from numbers import Number
import operator

import numpy as np
import xarray as xr
import traitlets as tl
from pint import UnitRegistry
from pint.unit import _Unit
ureg = UnitRegistry()


class UnitsNode(tl.TraitType):
    """UnitsNode Summary

    Attributes
    ----------
    info_text : str
        Description
    """
    info_text = "A UnitDataArray with specified dimensionality"

    def validate(self, obj, value):
        """validate summary

        Parameters
        ----------
        obj : TYPE
            Description
        value : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        import podpac.core.node as node
        if isinstance(value, node.Node):
            if 'units' in self.metadata and value.units is not None:
                u = ureg.check(self.metadata['units'])(lambda x: x)(value.units)
                return value
        self.error(obj, value)


class Units(tl.TraitType):
    """Units Summary

    Attributes
    ----------
    info_text : str
        Description
    """
    info_text = "A pint Unit"
    #default_value = None

    def validate(self, obj, value):
        """Summary

        Parameters
        ----------
        obj : TYPE
            Description
        value : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
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
        """Summary

        Parameters
        ----------
        unit : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        x = self.copy()
        if self.attrs.get("units", None):
            myu = ureg.Quantity(1, getattr(self, "units", "1"))
            multiplier = myu.to(unit).magnitude
            x = x * multiplier
            x.attrs['units'] = unit
        return x

    def to_base_units(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        if self.attrs.get("units", None):
            myu = ureg.Quantity(1, getattr(self, "units", "1")).to_base_units()
            return self.to(myu.u)
        else:
            return self.copy()

    def __getitem__(self, key):
        # special cases when key is also a DataArray
        # and has only one dimension
        if isinstance(key, xr.DataArray) and len(key.dims) == 1:
            # transpose with shared dims first
            shared_dims = [dim for dim in self.dims if dim in key.dims]
            missing_dims = [dim for dim in self.dims if dim not in key.dims]
            xT = self.transpose(*shared_dims + missing_dims)

            # index
            outT = xT[key.data]

            # transpose back to original dimensions
            out = outT.transpose(*self.dims)
            return out

        return super(UnitsDataArray, self).__getitem__(key)

    def part_transpose(self, new_dims):
        """Summary

        Parameters
        ----------
        new_dims : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        shared_dims = [dim for dim in new_dims if dim in self.dims]
        self_only_dims = [dim for dim in self.dims if dim not in new_dims]

        return self.transpose(*shared_dims+self_only_dims)

    def set(self, value, mask):
        """ Set the UnitsDataArray data to have a particular value, possibly using a mask
        in general, want to handle cases where value is a single value, an array,
        or a UnitsDataArray, and likewise for mask to be None, ndarray, or UnitsDataArray
        For now, focus on case where value is a single value and mask is a UnitsDataArray

        Parameters
        ----------
        value : TYPE
            Description
        mask : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """

        if isinstance(mask,UnitsDataArray) and isinstance(value,Number):
            orig_dims = deepcopy(self.dims)   

            # find out status of all dims
            shared_dims = [dim for dim in mask.dims if dim in self.dims]
            self_only_dims = [dim for dim in self.dims if dim not in mask.dims]
            mask_only_dims = [dim for dim in mask.dims if dim not in self.dims]

            # don't handle case where there are mask_only_dims
            if len(mask_only_dims) > 0:
                return

            # transpose self to have same order of dims as mask so those shared dims
            # come first and in the same order in both cases
            self = self.transpose(*shared_dims+self_only_dims)

            # set the values approved by ok_mask to be value
            self.values[mask.values, ...] = value

            # set self to have the same dims (and same order) as when first started
            self = self.transpose(*orig_dims)

for tp in ("mul", "matmul", "truediv", "div"):
    meth = "__{:s}__".format(tp)

    def func(self, other, meth=meth, tp=tp):
        """Summary

        Parameters
        ----------
        other : TYPE
            Description
        meth : TYPE, optional
            Description
        tp : TYPE, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        x = getattr(super(UnitsDataArray, self), meth)(other)
        return self._apply_binary_op_to_units(getattr(operator, tp), other, x)

    func.__name__ = meth
    setattr(UnitsDataArray, meth, func)


for tp in ("add", "sub", "mod", "floordiv"): #, "divmod", ):
    meth = "__{:s}__".format(tp)

    def func(self, other, meth=meth, tp=tp):
        """Summary

        Parameters
        ----------
        other : TYPE
            Description
        meth : TYPE, optional
            Description
        tp : TYPE, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        multiplier = self._get_unit_multiplier(other)
        x = getattr(super(UnitsDataArray, self), meth)(other * multiplier)
        return self._apply_binary_op_to_units(getattr(operator, tp), other, x)

    func.__name__ = meth
    setattr(UnitsDataArray, meth, func)


for tp in ("lt", "le", "eq", "ne", "gt", "ge"):
    meth = "__{:s}__".format(tp)

    def func(self, other, meth=meth, tp=tp):
        """Summary

        Parameters
        ----------
        other : TYPE
            Description
        meth : TYPE, optional
            Description
        tp : TYPE, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        multiplier = self._get_unit_multiplier(other)
        return getattr(super(UnitsDataArray, self), meth)(other * multiplier)

    func.__name__ = meth
    setattr(UnitsDataArray, meth, func)


for tp in ("mean", 'min', 'max', 'sum', 'cumsum'):

    def func(self, tp=tp, *args, **kwargs):
        """Summary

        Parameters
        ----------
        tp : TYPE, optional
            Description
        *args
            Description
        **kwargs
            Description

        Returns
        -------
        TYPE
            Description
        """
        x = getattr(super(UnitsDataArray, self), tp)(*args, **kwargs)
        return self._copy_units(x)

    func.__name__ = tp
    setattr(UnitsDataArray, tp, func)

del func

