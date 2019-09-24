"""Units module summary

Attributes
----------
ureg : TYPE
    Description
"""

from copy import deepcopy
from numbers import Number
import operator
from six import string_types
import json

from io import BytesIO
import base64

try:
    import cPickle  # Python 2.7
except:
    import _pickle as cPickle

import numpy as np
import xarray as xr
import traitlets as tl
from pint import UnitRegistry
from pint.unit import _Unit

ureg = UnitRegistry()

import podpac
from podpac.core.settings import settings
from podpac.core.utils import JSONEncoder
from podpac.core.style import Style


class UnitsDataArray(xr.DataArray):
    """Like xarray.DataArray, but transfers units
    """

    def __init__(self, *args, **kwargs):
        super(UnitsDataArray, self).__init__(*args, **kwargs)
        self.deserialize()

    def __array_wrap__(self, obj, context=None):
        new_var = super(UnitsDataArray, self).__array_wrap__(obj, context)
        if self.attrs.get("units"):
            if context and settings["ENABLE_UNITS"]:
                new_var.attrs["units"] = context[0](ureg.Quantity(1, self.attrs.get("units"))).u
            elif settings["ENABLE_UNITS"]:
                new_var = self._copy_units(new_var)
        return new_var

    def _apply_binary_op_to_units(self, func, other, x):
        if (self.attrs.get("units", None) or getattr(other, "units", None)) and settings["ENABLE_UNITS"]:
            x.attrs["units"] = func(
                ureg.Quantity(1, getattr(self, "units", "1")), ureg.Quantity(1, getattr(other, "units", "1"))
            ).u
        return x

    def _get_unit_multiplier(self, other):
        multiplier = 1
        if (self.attrs.get("units", None) or getattr(other, "units", None)) and settings["ENABLE_UNITS"]:
            otheru = ureg.Quantity(1, getattr(other, "units", "1"))
            myu = ureg.Quantity(1, getattr(self, "units", "1"))
            multiplier = otheru.to(myu.u).magnitude
        return multiplier

    # pow is different because resulting unit depends on argument, not on
    # unit of argument (which must be unitless)
    def __pow__(self, other):
        x = super(UnitsDataArray, self).__pow__(other)
        if self.attrs.get("units") and settings["ENABLE_UNITS"]:
            x.attrs["units"] = pow(
                ureg.Quantity(1, getattr(self, "units", "1")), ureg.Quantity(other, getattr(other, "units", "1"))
            ).u
        return x

    def _copy_units(self, x):
        if self.attrs.get("units", None):
            x.attrs["units"] = self.attrs.get("units")
        return x

    def to(self, unit):
        """Converts the UnitsDataArray units to the requested unit

        Parameters
        ----------
        unit : pint.UnitsRegistry unit
            The desired unit from podpac.unit

        Returns
        -------
        UnitsDataArray
            The array converted to the desired unit
            
        Raises
        --------
        DimensionalityError
            If the requested unit is not dimensionally consistent with the original unit.
        """
        x = self.copy()
        if self.attrs.get("units", None):
            myu = ureg.Quantity(1, getattr(self, "units", "1"))
            multiplier = myu.to(unit).magnitude
            x = x * multiplier
            x.attrs["units"] = unit
        return x

    def to_base_units(self):
        """Converts the UnitsDataArray units to the base SI units.

        Returns
        -------
        UnitsDataArray
            The units data array converted to the base SI units
        """
        if self.attrs.get("units", None):
            myu = ureg.Quantity(1, getattr(self, "units", "1")).to_base_units()
            return self.to(myu.u)
        else:
            return self.copy()

    def to_netcdf(self, *args, **kwargs):
        o = self
        for d in self.dims:
            if "_" in d and "dim" not in d:  # This it is stacked
                o = o.reset_index(d)
        o.serialize()
        r = super(UnitsDataArray, o).to_netcdf(*args, **kwargs)
        self.deserialize()
        return r

    def to_format(self, format, *args, **kwargs):
        """
        Helper function for converting Node outputs to alternative formats.
        
        Parameters
        -----------
        format: str
            Format to which output should be converted. This is uses the to_* functions provided by xarray
        *args: *list
            Extra arguments for a particular output function
        **kwargs: **dict
            Extra keyword arguments for a particular output function
            
        Returns
        --------
        io.BytesIO()
            In-memory version of the file or Python object. Note, depending on the input arguments, the file may instead
            be saved to disk.
            
        Notes
        ------
        This is a helper function for accessing existing to_* methods provided by the base xarray.DataArray object, with 
        a few additional formats supported: 
            * json
            * png, jpg, jpeg
            * tiff (GEOtiff)
        """
        self.serialize()
        if format in ["netcdf", "nc", "hdf5", "hdf"]:
            r = self.to_netcdf(*args, **kwargs)
        elif format in ["json", "dict"]:
            r = self.to_dict()
            if format == "json":
                r = json.dumps(r, cls=JSONEncoder)
        elif format in ["png", "jpg", "jpeg"]:
            r = get_image(self, format, *args, **kwargs)
        elif format.upper() in ["TIFF", "TIF", "GEOTIFF"]:
            raise NotImplementedError("Format {} is not implemented.".format(format))
        elif format in ["pickle", "pkl"]:
            r = cPickle.dumps(self)
        else:
            try:
                getattr(self, "to_" + format)(*args, **kwargs)
            except:
                raise NotImplementedError("Format {} is not implemented.".format(format))
        self.deserialize()
        return r

    def serialize(self):
        if self.attrs.get("units"):
            self.attrs["units"] = str(self.attrs["units"])
        if self.attrs.get("layer_style") and not isinstance(self.attrs["layer_style"], string_types):
            self.attrs["layer_style"] = self.attrs["layer_style"].json

    def deserialize(self):
        # Deserialize units
        if self.attrs.get("units") and isinstance(self.attrs["units"], string_types):
            self.attrs["units"] = ureg(self.attrs["units"]).u

        # Deserialize layer_stylers
        if self.attrs.get("layer_style") and isinstance(self.attrs["layer_style"], string_types):
            self.attrs["layer_style"] = podpac.core.style.Style.from_json(self.attrs["layer_style"])

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

    #     def reduce(self, func, *args, **kwargs):
    #         new_var = super(UnitsDataArray, self).reduce(func, *args, **kwargs)
    #         if self.attrs.get("units", None):
    #            new_var.attrs['units'] = self.units
    #         return new_var

    def part_transpose(self, new_dims):
        """Partially transpose the UnitsDataArray based on the input dimensions. The remaining
        dimensions will have their original order, and will be included at the end of the 
        transpose.

        Parameters
        ----------
        new_dims : list
            List of dimensions in the order they should be transposed

        Returns
        -------
        UnitsDataArray
            The UnitsDataArray transposed according to the user inputs
        """
        shared_dims = [dim for dim in new_dims if dim in self.dims]
        self_only_dims = [dim for dim in self.dims if dim not in new_dims]

        return self.transpose(*shared_dims + self_only_dims)

    def set(self, value, mask):
        """ Set the UnitsDataArray data to have a particular value, possibly using a mask
        in general, want to handle cases where value is a single value, an array,
        or a UnitsDataArray, and likewise for mask to be None, ndarray, or UnitsDataArray
        For now, focus on case where value is a single value and mask is a UnitsDataArray

        Parameters
        ----------
        value : Number
            A constant number that will replace the masked values.
        mask : UnitsDataArray
            A UnitsDataArray representing a boolean index.
            
        Notes
        ------
        This function modifies the UnitsDataArray inplace
        """

        if isinstance(mask, UnitsDataArray) and isinstance(value, Number):
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
            self = self.transpose(*shared_dims + self_only_dims)

            # set the values approved by ok_mask to be value
            self.values[mask.values, ...] = value

            # set self to have the same dims (and same order) as when first started
            self = self.transpose(*orig_dims)


for tp in ("mul", "matmul", "truediv", "div"):
    meth = "__{:s}__".format(tp)

    def make_func(meth, tp):
        def func(self, other):
            x = getattr(super(UnitsDataArray, self), meth)(other)
            return self._apply_binary_op_to_units(getattr(operator, tp), other, x)

        return func

    func = make_func(meth, tp)
    func.__name__ = meth
    setattr(UnitsDataArray, meth, func)


for tp in ("add", "sub", "mod", "floordiv"):  # , "divmod", ):
    meth = "__{:s}__".format(tp)

    def make_func(meth, tp):
        def func(self, other):
            multiplier = self._get_unit_multiplier(other)
            x = getattr(super(UnitsDataArray, self), meth)(other * multiplier)
            return self._apply_binary_op_to_units(getattr(operator, tp), other, x)

        return func

    func = make_func(meth, tp)
    func.__name__ = meth
    setattr(UnitsDataArray, meth, func)


for tp in ("lt", "le", "eq", "ne", "gt", "ge"):
    meth = "__{:s}__".format(tp)

    def make_func(meth):
        def func(self, other):
            multiplier = self._get_unit_multiplier(other)
            return getattr(super(UnitsDataArray, self), meth)(other * multiplier)

        return func

    func = make_func(meth)
    func.__name__ = meth
    setattr(UnitsDataArray, meth, func)


for tp in ("mean", "min", "max", "sum", "cumsum"):

    def make_func(tp):
        def func(self, *args, **kwargs):
            x = getattr(super(UnitsDataArray, self), tp)(*args, **kwargs)
            return self._copy_units(x)

        return func

    func = make_func(tp)
    func.__name__ = tp
    setattr(UnitsDataArray, tp, func)

del func


# ---------------------------------------------------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------------------------------------------------


def create_data_array(c, data=np.nan, outputs=None, dtype=float, **kwargs):
    """
    Initialize a units data array with the given coordinates and data.

    Parameters
    ----------
    c : Coordinates
        Podpac Coordinates containing the desired array dimensions and coords
    data : None, number, array-like (optional)
        Initial array data value(s)
    outputs : list (optional)
        Data output names (for multi-output nodes).
    dtype : type (optional)
        Array data type (default float)
    **kwargs
        Dictioary of any additional keyword arguments that will be passed to UnitsDataArray.
    """

    if not isinstance(c, podpac.Coordinates):
        raise TypeError("create_data_array expected Coordinates object, not '%s'" % type(c))

    # data array
    if np.shape(data) == ():
        shape = c.shape
        if outputs is not None:
            shape = shape + (len(outputs),)

        if data is None:
            data = np.empty(shape, dtype=dtype)
        elif data == 0:
            data = np.zeros(shape, dtype=dtype)
        elif data == 1:
            data = np.ones(shape, dtype=dtype)
        else:
            data = np.full(shape, data, dtype=dtype)
    else:
        if outputs is not None and len(outputs) != data.shape[-1]:
            raise ValueError(
                "data with shape %s does not match provided outputs %s (%d != %d)"
                % (data.shape, outputs, data.shape[-1], len(outputs))
            )
        data = data.astype(dtype)

    # coords and dims
    coords = c.coords
    dims = c.idims
    if outputs is not None:
        dims = dims + ("output",)
        coords["output"] = outputs

    # crs attr
    if "attrs" in kwargs:
        if "crs" not in kwargs["attrs"]:
            kwargs["attrs"]["crs"] = c.crs
    else:
        kwargs["attrs"] = {"crs": c.crs}

    return UnitsDataArray(data, coords=coords, dims=dims, **kwargs)


def get_image(data, format="png", vmin=None, vmax=None, return_base64=False):
    """Return a base64-encoded image of the data

    Parameters
    ----------
    data : array-like
        data to output, usually a UnitsDataArray
    format : str, optional
        Default is 'png'. Type of image. 
    vmin : number, optional
        Minimum value of colormap
    vmax : vmax, optional
        Maximum value of colormap
    return_base64: bool, optional
        Default is False. Normally this returns an io.BytesIO, but if True, will return a base64 encoded string.
        

    Returns
    -------
    BytesIO/str
        Binary or Base64 encoded image. 
    """

    import matplotlib
    import matplotlib.cm
    from matplotlib.image import imsave

    matplotlib.use("agg")

    if format != "png":
        raise ValueError("Invalid image format '%s', must be 'png'" % format)

    style = None
    if isinstance(data, xr.DataArray):
        style = data.attrs.get("layer_style", None)
        if isinstance(style, string_types):
            style = Style.from_json(style)
        dims = data.squeeze().dims
        y = data.coords[dims[0]]
        x = data.coords[dims[1]]
        data = data.data
        if y[1] > y[0]:
            data = data[::-1, :]
        if x[1] < x[0]:
            data = data[:, ::1]

    data = data.squeeze()

    if not np.any(np.isfinite(data)):
        vmin = 0
        vmax = 1
    else:
        if vmin is None or np.isnan(vmin):
            if style is not None and style.clim[0] != None:
                vmin = style.clim[0]
            else:
                vmin = np.nanmin(data)
        if vmax is None or np.isnan(vmax):
            if style is not None and style.clim[1] != None:
                vmax = style.clim[1]
            else:
                vmax = np.nanmax(data)
    if vmax == vmin:
        vmax += 1e-15

    # get the colormap
    if style is None:
        cmap = matplotlib.cm.viridis
    else:
        cmap = style.cmap

    c = (data - vmin) / (vmax - vmin)
    i = cmap(c, bytes=True)
    i[np.isnan(c), 3] = 0
    im_data = BytesIO()
    imsave(im_data, i, format=format)
    im_data.seek(0)
    if return_base64:
        return base64.b64encode(im_data.getvalue())
    else:
        return im_data
