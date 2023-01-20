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
import warnings

from io import BytesIO
import base64
import logging

try:
    import cPickle  # Python 2.7
except:
    import _pickle as cPickle

import numpy as np
import xarray as xr
import traitlets as tl
from pint import UnitRegistry

ureg = UnitRegistry()

import podpac
from podpac import Coordinates
from podpac.core.settings import settings
from podpac.core.utils import JSONEncoder
from podpac.core.style import Style

# Optional dependencies
from lazy_import import lazy_module, lazy_class

rasterio = lazy_module("rasterio")
affine = lazy_module("affine")

# Set up logging
_logger = logging.getLogger(__name__)


class UnitsDataArray(xr.DataArray):
    """Like xarray.DataArray, but transfers units"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super(UnitsDataArray, self).__init__(*args, **kwargs)
        self = self._pp_deserialize()

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
                try:
                    o = o.reset_index(d)
                except KeyError:
                    pass  # This is fine, actually didn't need to reset because not a real dim
        o._pp_serialize()
        r = super(UnitsDataArray, o).to_netcdf(*args, **kwargs)
        self._pp_deserialize()
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
        self._pp_serialize()
        if format in ["netcdf", "nc", "hdf5", "hdf"]:
            r = self.to_netcdf(*args, **kwargs)
        elif format in ["json", "dict"]:
            r = self.to_dict()
            if format == "json":
                r = json.dumps(r, cls=JSONEncoder)
        elif format in ["png", "jpg", "jpeg"]:
            r = self.to_image(format, *args, **kwargs)
        elif format.upper() in ["TIFF", "TIF", "GEOTIFF"]:
            r = self.to_geotiff(*args, **kwargs)

        elif format in ["pickle", "pkl"]:
            r = cPickle.dumps(self)
        elif format == "zarr_part":
            from podpac.core.data.zarr_source import Zarr
            import zarr

            if "part" in kwargs:
                part = kwargs.pop("part")
                part = tuple([slice(*sss) for sss in part])
            else:
                part = slice(None)

            zn = Zarr(source=kwargs.pop("source"))
            store = zn._get_store()

            zf = zarr.open(store, *args, **kwargs)

            if "output" in self.dims:
                for key in self.coords["output"].data:
                    zf[key][part] = self.sel(output=key).data
            else:
                data_key = kwargs.get("data_key", "data")
                zf[data_key][part] = self.data
            r = zn.source
        else:
            try:
                getattr(self, "to_" + format)(*args, **kwargs)
            except:
                raise NotImplementedError("Format {} is not implemented.".format(format))
        self._pp_deserialize()
        return r

    def to_image(self, format="png", vmin=None, vmax=None, return_base64=False):
        """Return a base64-encoded image of the data.

        Parameters
        ----------
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
        return to_image(self, format, vmin, vmax, return_base64)

    def to_geotiff(self, fp=None, geotransform=None, crs=None, **kwargs):
        """
        For documentation, see `core.units.to_geotiff`
        """
        return to_geotiff(fp, self, geotransform=geotransform, crs=crs, **kwargs)

    def _pp_serialize(self):
        if self.attrs.get("units"):
            self.attrs["units"] = str(self.attrs["units"])
        if self.attrs.get("layer_style") and not isinstance(self.attrs["layer_style"], string_types):
            self.attrs["layer_style"] = self.attrs["layer_style"].json
        if self.attrs.get("bounds"):
            if isinstance(self.attrs["bounds"], dict) and "time" in self.attrs["bounds"]:
                time_bounds = self.attrs["bounds"]["time"]
                new_bounds = []
                for tb in time_bounds:
                    if isinstance(tb, np.datetime64):
                        new_bounds.append(str(tb))
                    else:
                        new_bounds.append(tb)
                self.attrs["bounds"]["time"] = tuple(new_bounds)
            self.attrs["bounds"] = json.dumps(self.attrs["bounds"])
        if self.attrs.get("boundary_data") is not None and not isinstance(self.attrs["boundary_data"], string_types):
            self.attrs["boundary_data"] = json.dumps(self.attrs["boundary_data"])

    def _pp_deserialize(self):
        # Deserialize units
        if self.attrs.get("units") and isinstance(self.attrs["units"], string_types):
            self.attrs["units"] = ureg(self.attrs["units"]).u

        # Deserialize layer_stylers
        if self.attrs.get("layer_style") and isinstance(self.attrs["layer_style"], string_types):
            self.attrs["layer_style"] = podpac.core.style.Style.from_json(self.attrs["layer_style"])

        if self.attrs.get("bounds") and isinstance(self.attrs["bounds"], string_types):
            self.attrs["bounds"] = json.loads(self.attrs["bounds"])
            if "time" in self.attrs["bounds"]:
                time_bounds = self.attrs["bounds"]["time"]
                new_bounds = []
                for tb in time_bounds:
                    if isinstance(tb, string_types):
                        new_bounds.append(np.datetime64(tb))
                    else:
                        new_bounds.append(tb)
                self.attrs["bounds"]["time"] = tuple(new_bounds)
        if self.attrs.get("boundary_data") and isinstance(self.attrs["boundary_data"], string_types):
            self.attrs["boundary_data"] = json.loads(self.attrs["boundary_data"])

        # Deserialize the multi-index
        for dim in self.dims:
            if dim in self.coords or "-" in dim:  # The "-" is for multi-dimensional stacked coordinates
                continue
            try:
                self = self.set_index(**{dim: dim.split("-")[0].split("_")})
            except ValueError as e:
                _logger.warning("Tried to rebuild stacked coordinates but failed with error: {}".format(e))
        return self

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
        """This function sets the values of the dataarray equal to 'value' where ever mask is True.
        This operation happens in-place.

        Set the UnitsDataArray data to have a particular value, possibly using a mask
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

        if isinstance(mask, xr.DataArray) and isinstance(value, Number):
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

    @classmethod
    def open(cls, *args, **kwargs):
        """
        Open an :class:`podpac.UnitsDataArray` from a file or file-like object containing a single data variable.

        This is a wrapper around :func:`xarray.open_datarray`.
        The inputs to this function are passed directly to :func:`xarray.open_datarray`.
        See http://xarray.pydata.org/en/stable/generated/xarray.open_dataarray.html#xarray.open_dataarray.

        The DataArray passed back from :func:`xarray.open_datarray` is used to create a units data array using :func:`creare_dataarray`.

        Returns
        -------
        :class:`podpac.UnitsDataArray`
        """
        da = xr.open_dataarray(*args, **kwargs)
        coords = Coordinates.from_xarray(da)

        # pass in kwargs to constructor
        uda_kwargs = {"attrs": da.attrs}
        if "output" in da.dims:
            uda_kwargs.update({"outputs": da.coords["output"]})
        return cls.create(coords, data=da.data, **uda_kwargs)

    @classmethod
    def create(cls, c, data=np.nan, outputs=None, dtype=float, **kwargs):
        """Shortcut to create :class:`podpac.UnitsDataArray`

        Parameters
        ----------
        c : :class:`podpac.Coordinates`
            PODPAC Coordinates
        data : np.ndarray, optional
            Data to fill in. Defaults to np.nan.
        dtype : type, optional
            Data type. Defaults to float.
        **kwargs
            keyword arguments to pass to :class:`podpac.UnitsDataArray` constructor

        Returns
        -------
        :class:`podpac.UnitsDataArray`
        """
        if not isinstance(c, podpac.Coordinates):
            raise TypeError("`UnitsDataArray.create` expected Coordinates object, not '%s'" % type(c))

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
        coords = c.xcoords
        dims = c.xdims

        if outputs is not None:
            dims = dims + ("output",)
            coords["output"] = outputs

        # crs attr
        if "attrs" in kwargs:
            if "crs" not in kwargs["attrs"]:
                kwargs["attrs"]["crs"] = c.crs
        else:
            kwargs["attrs"] = {"crs": c.crs}

        return cls(data, coords=coords, dims=dims, **kwargs)


for tp in ("mul", "matmul", "truediv", "div"):
    meth = "__{:s}__".format(tp)

    def make_func(meth, tp):
        def func(self, other):
            x = getattr(super(UnitsDataArray, self), meth)(other)
            x2 = self._apply_binary_op_to_units(getattr(operator, tp), other, x)
            units = x2.attrs.get("units")
            x2.attrs = self.attrs
            if units is not None:
                x2.attrs["units"] = units
            return x2

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
            x2 = self._apply_binary_op_to_units(getattr(operator, tp), other, x)
            x2.attrs = self.attrs
            return x2

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


def to_image(data, format="png", vmin=None, vmax=None, return_base64=False):
    """Return a base64-encoded image of data

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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


def to_geotiff(fp, data, geotransform=None, crs=None, **kwargs):
    """Export a UnitsDataArray to a Geotiff

    Params
    -------
    fp:  str, file object or pathlib.Path object
        A filename or URL, a file object opened in binary ('rb') mode, or a Path object. If not supplied, the results will
        be written to a memfile object
    data: UnitsDataArray, xr.DataArray, np.ndarray
        The data to be saved. If there is more than 1 band, this should be the last dimension of the array.
        If given a np.ndarray, ensure that the 'lat' dimension is aligned with the rows of the data, with an appropriate
        geotransform.
    geotransform: tuple, optional
        The geotransform that describes the input data. If not given, will look for data.attrs['geotransform']
    crs: str, optional
        The coordinate reference system for the data
    kwargs: **dict
        Additional key-word arguments that overwrite defaults used in the `rasterio.open` function. This function
        populates the following defaults:
                drive="GTiff"
                height=data.shape[0]
                width=data.shape[1]
                count=data.shape[2]
                dtype=data.dtype
                mode="w"

    Returns
    --------
    MemoryFile, list
        If fp is given, results a list of the results for writing to each band r.append(dst.write(data[..., i], i + 1))
        If fp is None, returns the MemoryFile object
    """

    # This only works for data that essentially has lat/lon only
    dims = list(data.coords.keys())
    if "lat" not in dims or "lon" not in dims:
        raise NotImplementedError("Cannot export GeoTIFF for dataset with lat/lon coordinates.")
    if "time" in dims and len(data.coords["time"]) > 1:
        raise NotImplemented("Cannot export GeoTIFF for dataset with multiple times,")
    if "alt" in dims and len(data.coords["alt"]) > 1:
        raise NotImplemented("Cannot export GeoTIFF for dataset with multiple altitudes.")

    # TODO: add proper checks, etc. to make sure we handle edge cases and throw errors when we cannot support
    #       i.e. do work to remove this warning.
    _logger.warning("GeoTIFF export assumes data is in a uniform, non-rotated coordinate system.")

    # Get the crs and geotransform that describes the coordinates
    if crs is None:
        crs = data.attrs.get("crs")
    if crs is None:
        raise ValueError(
            "The `crs` of the data needs to be provided to save as GeoTIFF. If supplying a UnitsDataArray, created "
            " through a PODPAC Node, the crs should be automatically populated. If not, please file an issue."
        )
    if geotransform is None:
        geotransform = data.attrs.get("geotransform")
        # Geotransform should ALWAYS be defined as (lon_origin, lon_dj, lon_di, lat_origin, lat_dj, lat_di)
        # if isinstance(data, xr.DataArray) and data.dims.index('lat') > data.dims.index('lon'):
        # geotransform = geotransform[3:] + geotransform[:3]

    if geotransform is None:
        try:
            geotransform = Coordinates.from_xarray(data).geotransform
        except (TypeError, AttributeError):
            raise ValueError(
                "The `geotransform` of the data needs to be provided to save as GeoTIFF. If the geotransform attribute "
                "wasn't automatically populated as part of the dataset, it means that the data is in a non-uniform "
                "coordinate system. This can sometimes happen when the data is transformed to a different CRS than the "
                "native CRS, which can cause the coordinates to seems non-uniform due to floating point precision. "
            )

    # Make all types into a numpy array
    if isinstance(data, xr.DataArray):
        data = data.data

    # Get the data
    dtype = kwargs.get("dtype", np.float32)
    data = data.astype(dtype).squeeze()

    if len(data.shape) == 2:
        data = data[:, :, None]

    geotransform = affine.Affine.from_gdal(*geotransform)

    # Update the kwargs that rasterio will use. Anything added by the user will take priority.
    kwargs2 = dict(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=data.shape[2],
        dtype=data.dtype,
        crs=crs,
        transform=geotransform,
    )
    kwargs2.update(kwargs)

    # Write the file
    if fp is None:
        # Write to memory file
        r = rasterio.io.MemoryFile()
        with r.open(**kwargs2) as dst:
            for i in range(data.shape[2]):
                dst.write(data[..., i], i + 1)
    else:
        r = []
        kwargs2["mode"] = "w"
        with rasterio.open(fp, **kwargs2) as dst:
            for i in range(data.shape[2]):
                r.append(dst.write(data[..., i], i + 1))

    return r
