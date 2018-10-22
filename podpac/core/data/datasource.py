"""
Generic Data Source Class

DataSource is the root class for all other podpac defined data sources,
including user defined data sources.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import warnings
from collections import OrderedDict

import numpy as np
import xarray as xr
import traitlets as tl

# Optional dependencies
try:
    import rasterio
    from rasterio import transform
    from rasterio.warp import reproject, Resampling
except:
    rasterio = None
try:
    import scipy
    from scipy.interpolate import (griddata, RectBivariateSpline, RegularGridInterpolator)
    from scipy.spatial import KDTree
except:
    scipy = None

# Internal imports
from podpac.core.units import UnitsDataArray
from podpac.core.coordinates import Coordinates
from podpac.core.coordinates import Coordinates1d, UniformCoordinates1d, ArrayCoordinates1d, StackedCoordinates
from podpac.core.node import Node
from podpac.core.utils import common_doc, trait_is_defined
from podpac.core.node import COMMON_NODE_DOC
from podpac.core.data.interpolate import (Interpolation, Interpolator, NearestNeighbor, INTERPOLATION_SHORTCUTS,
                                          INTERPOLATION_DEFAULT)

DATA_DOC = {
    'native_coordinates': 'The coordinates of the data source.',

    'get_data':
        """
        This method must be defined by the data source implementing the DataSource class.
        When data source nodes are evaluated, this method is called with request coordinates and coordinate indexes.
        The implementing method can choose which input provides the most efficient method of getting data
        (i.e via coordinates or via the index of the coordinates).
        
        Coordinates and coordinate indexes may be strided or subsets of the
        source data, but all coordinates and coordinate indexes will match 1:1 with the subset data.

        This method may return a numpy array, an xarray DaraArray, or a podpac UnitsDataArray.
        If a numpy array or xarray DataArray is returned, :meth:podpac.core.data.datasource.DataSource.evaluate will
        cast the data into a `UnitsDataArray` using the requested source coordinates.
        If a podpac UnitsDataArray is passed back, the :meth:podpac.core.data.datasource.DataSource.evaluate
        method will not do any further processing.
        The inherited Node method `create_output_array` can be used to generate the template UnitsDataArray
        in your DataSource.
        See :meth:podpac.core.node.Node.create_output_array for more details.
        
        Parameters
        ----------
        coordinates : Coordinates
            The coordinates that need to be retrieved from the data source using the coordinate system of the data
            source
        coordinates_index : List
            A list of slices or a boolean array that give the indices of the data that needs to be retrieved from
            the data source. The values in the coordinate_index will vary depending on the `coordinate_index_type`
            defined for the data source.
            
        Returns
        --------
        np.ndarray, xr.DataArray, podpac.core.units.UnitsDataArray
            A subset of the returned data. If a numpy array or xarray DataArray is returned,
            the data will be cast into  UnitsDataArray using the returned data to fill values
            at the requested source coordinates.
        """,
    
    'get_native_coordinates':
        """
        Returns a Coordinates object that describes the native coordinates of the data source.

        In most cases, this method is defined by the data source implementing the DataSource class.
        If method is not implemented by the data source, it will try to return `self.native_coordinates` 
        if `self.native_coordinates` is not None.

        Otherwise, this method will raise a NotImplementedError.

        Returns
        --------
        Coordinates
           The coordinates describing the data source array.

        Raises
        --------
        NotImplementedError
            Raised if get_native_coordinates is not implemented by data source subclass.

        Notes
        ------
        Need to pay attention to:
        - the order of the dimensions
        - the stacking of the dimension
        - the type of coordinates

        Coordinates should be non-nan and non-repeating for best compatibility
        """
    }

COMMON_DATA_DOC = COMMON_NODE_DOC.copy()
COMMON_DATA_DOC.update(DATA_DOC)      # inherit and overwrite with DATA_DOC


class DataSource(Node):
    """Base node for any data obtained directly from a single source.
    
    Attributes
    ----------
    source : Any
        The location of the source. Depending on the child node this can be a filepath,
        numpy array, or dictionary as a few examples.
    native_coordinates : Coordinates
        {native_coordinates} 
    coordinate_index_type : str, optional
        Type of index to use for data source. Possible values are ['list','numpy','xarray','pandas']
        Default is 'numpy'
    interpolation : str,
                    dict
                    optional
            Definition of interpolation methods for each dimension of the native coordinates.
            
            If input is a string, it must match one of the interpolation shortcuts defined in
            :ref:podpac.core.data.interpolate.INTERPOLATION_SHORTCUTS. The interpolation method associated
            with this string will be applied to all dimensions at the same time.

            If input is a dict, the dict must contain ordered set of keys defining dimensions and values
            defining the interpolation method to use with the dimensions.
            The key must be a string or tuple of dimension names (i.e. `'time'` or `('lat', 'lon')` ).
            The value can either be a string matching one of the interpolation shortcuts defined in
            :ref:podpac.core.data.interpolate.INTERPOLATION_SHORTCUTS or a dictionary.
            If the value is a dictionary, the dictionary must contain a key `'method'`
            defining the interpolation method name.
            If the interpolation method is not one of :ref:podpac.core.data.interpolate.INTERPOLATION_SHORTCUTS, a
            second key `'interpolators'` must be defined with a list of
            :ref:podpac.core.data.interpolate.Interpolator classes to use in order of uages.
            The dictionary may contain an option `'params'` key which contains a dict of parameters to pass along to
            the :ref:podpac.core.data.interpolate.Interpolator classes associated with the interpolation method.
            
            If the dictionary does not contain a key for all unstacked dimensions of the source coordinates, the
            :ref:podpac.core.data.interpolate.INTERPOLATION_DEFAULT value will be used.
            All dimension keys must be unstacked even if the underlying coordinate dimensions are stacked.
            Any extra dimensions included but not found in the source coordinates will be ignored.

            If input is a podpac.core.data.interpolate.Interpolation, this interpolation
            class will be used without modication.
            
            By default, the interpolation method is set to `'nearest'` for all dimensions.
    nan_vals : List, optional
        List of values from source data that should be interpreted as 'no data' or 'nans'

    Notes
    -----
    Developers of new DataSource nodes need to implement the `get_data` and `get_native_coordinates` methods.
    """
    
    source = tl.Any(help='Path to the raw data source')
    native_coordinates = tl.Instance(Coordinates)

    interpolation = tl.Union([
        tl.Dict(),
        tl.Enum(INTERPOLATION_SHORTCUTS)
    ], default_value=INTERPOLATION_DEFAULT)

    coordinate_index_type = tl.Enum(['list', 'numpy', 'xarray', 'pandas'], default_value='numpy')
    nan_vals = tl.List(allow_none=True)

    # TODO: remove in the 2nd stage of interpolation refactor
    # self.source_coordinates['time'].delta / 2
    interpolation_tolerance = tl.Union([tl.Float(default_value=1), tl.Instance(np.timedelta64)])

    # privates
    _interpolation = tl.Instance(Interpolation)
    
    # these correspond to _output_coordinates (_requested_coordinates with extra dimensions dropped)
    _requested_source_coordinates = tl.Instance(Coordinates)
    _requested_source_coordinates_index = tl.List()
    _requested_source_data = tl.Instance(UnitsDataArray)

    # when native_coordinates is not defined, default calls get_native_coordinates
    @tl.default('native_coordinates')
    def _default_native_coordinates(self):
        self.native_coordinates = self.get_native_coordinates()
        return self.native_coordinates

    # this adds a more helpful error message if user happens to try an inspect _interpolation before evaluate
    @tl.default('_interpolation')
    def _default_interpolation(self):
        self._set_interpolation()
        return self._interpolation

    @common_doc(COMMON_DATA_DOC)
    def eval(self, coordinates, output=None, method=None):
        """Evaluates this node using the supplied coordinates.

        The native coordinates are mapped to the requested coordinates, interpolated if necessary, and set to
        `_requested_source_coordinates` with associated index `_requested_source_coordinates_index`. The requested souce
        coordinates and index are passed to `get_data()` returning the source data at the native coordinatesset to
        `_requested_source_data`. Finally `_requested_source_data` is interpolated using the `interpolate` method and set
        to the `output` attribute of the node.


        Parameters
        ----------
        coordinates : Coordinates
            {requested_coordinates}
            Notes::
             * An exception is raised if the requested coordinates are missing dimensions in the DataSource.
             * Extra dimensions in the requested coordinates are dropped.
        output : podpac.core.units.UnitsDataArray, optional
            {eval_output}
        method : str, optional
            {eval_method}
        
        Returns
        -------
        {eval_return}

        Raises
        ------
        ValueError
            Cannot evaluate these coordinates
        """

        if self.coordinate_index_type != 'numpy':
            warnings.warn('Coordinates index type {} is not yet supported.'.format(self.coordinate_index_type) +
                          '`coordinate_index_type` is set to `numpy`', UserWarning)

        # store requested coordinates for debugging
        self._requested_coordinates = coordinates
        
        # check for missing dimensions
        for c in self.native_coordinates.values():
            if isinstance(c, Coordinates1d):
                if c.name not in coordinates.udims:
                    raise ValueError("Cannot evaluate these coordinates, missing dim '%s'" % c.name)
            elif isinstance(c, StackedCoordinates):
                stacked = [s for s in c if s.name in coordinates.udims]
                if not stacked:
                    raise ValueError("Cannot evaluate these coordinates, missing all dims in '%s'" % c.name)
                if any(s for s in stacked if not s.is_monotonic):
                    raise ValueError("Cannot evaluate these coordinates, cannot unambiguously map '%s' to %s" % (
                        coordinates.udims, self.native_coordinates.udims))
        
        # remove extra dimensions
        extra = []
        for c in coordinates.values():
            if isinstance(c, Coordinates1d):
                if c.name not in self.native_coordinates.udims:
                    extra.append(c.name)
            elif isinstance(c, StackedCoordinates):
                if all(dim not in self.native_coordinates.udims for dim in c.dims):
                    extra.append(c.name)
        self._output_coordinates = coordinates.drop(extra)

        # intersect the native coordinates with requested coordinates
        # to get native coordinates within requested coordinates bounds
        # TODO: support coordinate_index_type parameter to define other index types
        self._requested_source_coordinates, self._requested_source_coordinates_index = \
            self.native_coordinates.intersect(self._output_coordinates, outer=True, return_indices=True)

        # If requested coordinates and native coordinates do not intersect, shortcut with nan UnitsDataArary
        if np.prod(self._requested_source_coordinates.shape) == 0:
            if output is None:
                output = self.create_output_array(self._output_coordinates)
            else:
                output[:] = np.nan

            self._output = output
            return output
        
        # reset interpolation
        self._set_interpolation()

        # interpolate coordinates before getting data
        self._requested_source_coordinates, self._requested_source_coordinates_index = \
            self._interpolation.select_coordinates(self._output_coordinates,
                                                   self._requested_source_coordinates,
                                                   self._requested_source_coordinates_index)

        # get data from data source
        self._requested_source_data = self._get_data()

        # interpolate data into output
        if output is None:
            output = self.create_output_array(self._output_coordinates)
        output = self._interpolate(output)
        
        # set the order of dims to be the same as that of requested_coordinates
        # this is required in case the user supplied an output object with a different dims order
        output = output.transpose(*self._output_coordinates.dims)
        
        self._output = output
        return output

    def find_coordinates(self):
        """
        Get the available native coordinates for the Node. For a DataSource, this is just the native_coordinates.

        Returns
        -------
        coords_list : list
            singleton list containing the native_coordinates (Coordinates object)
        """

        return [self.native_coordinates]

    def get_interpolation_class(self):
        """Get the interpolation class currently set for this data source.
        
        The DataSource `interpolation` property is used to define the 
        :ref:podpac.core.data.interpolate.Interpolation class that will handle interpolation for requested coordinates.
        
        Returns
        -------
        podpac.core.data.interpolate.Interpolation
            Interpolation class defined by DataSource `interpolation` definition
        """

        return self._interpolation

    def _set_interpolation(self):
        """Update _interpolation property
        """

        # define interpolator with source coordinates dimensions
        if isinstance(self.interpolation, Interpolation):
            self._interpolation = self.interpolation
        else:
            self._interpolation = Interpolation(self.interpolation)



    def _get_data(self):
        """Wrapper for `self.get_data` with pre and post processing
        
        Returns
        -------
        podpac.core.units.UnitsDataArray
            Returns UnitsDataArray with coordinates defined by _requested_source_coordinates
        
        Raises
        ------
        ValueError
            Raised if unknown data is passed by from self.get_data
        NotImplementedError
            Raised if get_data is not implemented by data source subclass

        """
        # get data from data source at requested source coordinates and requested source coordinates index
        data = self.get_data(self._requested_source_coordinates, self._requested_source_coordinates_index)

        # convert data into UnitsDataArray depending on format
        # TODO: what other processing needs to happen here? 
        if isinstance(data, UnitsDataArray):
            udata_array = data
        elif isinstance(data, xr.DataArray):
            # TODO: check order of coordinates here
            udata_array = self.create_output_array(self._output_coordinates, data=data.data)
        elif isinstance(data, np.ndarray):
            udata_array = self.create_output_array(self._output_coordinates, data=data)
        else:
            raise ValueError('Unknown data type passed back from {}.get_data(): {}. '.format(type(self).__name__, type(data)) +
                             'Must be one of numpy.ndarray, xarray.DataArray, or podpac.UnitsDataArray')

        # fill nan_vals in data array
        if self.nan_vals:
            for nan_val in self.nan_vals:
                udata_array.data[udata_array.data == nan_val] = np.nan

        return udata_array


    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        
        Raises
        ------
        NotImplementedError
            This needs to be implemented by derived classes
        """
        raise NotImplementedError
        
    @common_doc(COMMON_DATA_DOC)
    def get_native_coordinates(self):
        """{get_native_coordinates}
        
        Raises
        ------
        NotImplementedError
            This needs to be implemented by derived classes
        """
        
        if trait_is_defined(self, 'native_coordinates'):
            return self.native_coordinates
        else:
            raise NotImplementedError('{0}.native_coordinates is not defined and '  \
                                      '{0}.get_native_coordinates() is not implemented'.format(self.__class__.__name__))
    
    def _interpolate(self, output):
        """Interpolates the source data to the destination using self.interpolation as the interpolation method.
        
        Returns
        -------
        UnitsDataArray
            Result of interpolating the source data to the destination coordinates
        """

        # return self._interpolation.interpolate(self._requested_source_coordinates,
        #                                       self._requested_source_data,
        #                                       self._output_coordinates,
        #                                       output)


        #### MOVE THIS TO INTERPOLATER
        # assign shortnames
        data_src = self._requested_source_data
        coords_src = self._requested_source_coordinates
        coords_dst = self._output_coordinates
        data_dst = output
        
        # This a big switch, funneling data to various interpolation routines
        if data_src.size == 1 and np.prod(coords_dst.shape) == 1:
            data_dst[:] = data_src
            return data_dst
        
        # Nearest preview of rasters
        if self._interpolation.definition == 'nearest_preview':
            crds = OrderedDict()
            tol = np.inf
            for c in data_dst.coords.keys():
                crds[c] = data_dst.coords[c].data.copy()
                if c is not 'time' and coords_dst[c].size > 0:
                    area_bounds = getattr(coords_dst[c], 'area_bounds', [-np.inf, np.inf])
                    delta = np.abs(area_bounds[1] - area_bounds[0]) / coords_dst[c].size
                    tol = min(tol, delta)
            crds_keys = list(crds.keys())
            if 'time' in crds:
                data_src = data_src.reindex(time=crds['time'], method=str('nearest'))
                del crds['time']
            data_dst.data = data_src.reindex(method=str('nearest'), tolerance=tol, **crds).transpose(*crds_keys)
            return data_dst
        
        # For now, we just do nearest-neighbor interpolation for time and alt
        # coordinates
        if 'time' in coords_src.dims and 'time' in coords_dst.dims:
            data_src = data_src.reindex(
                time=coords_dst.coords['time'], method='nearest', tolerance=self.interpolation_tolerance)
            coords_src['time'] = ArrayCoordinates1d.from_xarray(data_src['time'])
            if len(coords_dst.dims) == 1:
                return data_src

        if 'alt' in coords_src.dims and 'alt' in coords_dst.dims:
            data_src = data_src.reindex(alt=coords_dst.coords['alt'], method='nearest')
            coords_src['alt'] = ArrayCoordinates1d.from_xarray(data_src['alt'])
            if len(coords_dst.dims) == 1:
                return data_src

        # Raster to Raster interpolation from regular grids to regular grids
        
        if (rasterio is not None
                and self._interpolation.definition in INTERPOLATION_SHORTCUTS
                and 'lat' in coords_src.dims and 'lon' in coords_src.dims
                and 'lat' in coords_dst.dims and 'lon' in coords_dst.dims
                and coords_src['lat'].is_uniform and coords_src['lon'].is_uniform
                and coords_dst['lat'].is_uniform and coords_dst['lon'].is_uniform):
            return self.rasterio_interpolation(data_src, coords_src, data_dst, coords_dst)

        # Raster to Raster interpolation from irregular grids to arbitrary grids
        elif (('lat' in coords_src.dims and 'lon' in coords_src.dims)
                and ('lat' in coords_dst.dims and 'lon' in coords_dst.dims)):
            
            return self.interpolate_irregular_grid(data_src, coords_src, data_dst, coords_dst, grid=True)
        # Raster to lat_lon point interpolation
        elif (('lat' in coords_src.dims and 'lon' in coords_src.dims)
                and ('lat_lon' in coords_dst.dims or 'lon_lat' in coords_dst.dims)):
            coords_dst_us = coords_dst.unstack()
            return self.interpolate_irregular_grid(data_src, coords_src, data_dst, coords_dst_us, grid=False)

        elif 'lat_lon' in coords_src.dims or 'lon_lat' in coords_src.dims:
            return self.interpolate_point_data(data_src, coords_src, data_dst, coords_dst)
        
        raise NotImplementedError("The combination of source/destination coordinates has not been implemented.")
            
    def _loop_helper(self, func, keep_dims, data_src, coords_src, data_dst, coords_dst, **kwargs):
        """ Loop helper
        
        Parameters
        ----------
        func : TYPE
            Description
        keep_dims : TYPE
            Description
        data_src : TYPE
            Description
        coords_src : TYPE
            Description
        data_dst : TYPE
            Description
        coords_dst : TYPE
            Description
        **kwargs
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        loop_dims = [d for d in data_src.dims if d not in keep_dims]
        if len(loop_dims) > 0:
            for i in data_src.coords[loop_dims[0]]:
                ind = {loop_dims[0]: i}
                data_dst.loc[ind] = \
                    self._loop_helper(func, keep_dims,
                                      data_src.loc[ind], coords_src,
                                      data_dst.loc[ind], coords_dst, **kwargs)
        else:
            return func(data_src, coords_src, data_dst, coords_dst, **kwargs)
        return data_dst
        
    
    def rasterio_interpolation(self, data_src, coords_src, data_dst, coords_dst):
        """Summary
        
        Parameters
        ----------
        data_src : TYPE
            Description
        coords_src : TYPE
            Description
        data_dst : TYPE
            Description
        coords_dst : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        
        Raises
        ------
        ValueError
            Description
        """
        if len(data_src.dims) > 2:
            return self._loop_helper(self.rasterio_interpolation, ['lat', 'lon'],
                                     data_src, coords_src, data_dst, coords_dst)
        elif 'lat' not in data_src.dims or 'lon' not in data_src.dims:
            raise ValueError
        
        def get_rasterio_transform(c):
            """Summary
            
            Parameters
            ----------
            c : TYPE
                Description
            
            Returns
            -------
            TYPE
                Description
            """
            west, east = c['lon'].area_bounds
            south, north = c['lat'].area_bounds
            cols, rows = (c['lon'].size, c['lat'].size)
            #print (east, west, south, north)
            return transform.from_bounds(west, south, east, north, cols, rows)
        
        with rasterio.Env():
            src_transform = get_rasterio_transform(coords_src)
            src_crs = {'init': coords_src.gdal_crs}
            # Need to make sure array is c-contiguous
            if coords_src['lat'].is_descending:
                source = np.ascontiguousarray(data_src.data)
            else:
                source = np.ascontiguousarray(data_src.data[::-1, :])
        
            dst_transform = get_rasterio_transform(coords_dst)
            dst_crs = {'init': coords_dst.gdal_crs}
            # Need to make sure array is c-contiguous
            if not data_dst.data.flags['C_CONTIGUOUS']:
                destination = np.ascontiguousarray(data_dst.data) 
            else:
                destination = data_dst.data
        
            reproject(
                source,
                np.atleast_2d(destination.squeeze()),  # Needed for legacy compatibility
                src_transform=src_transform,
                src_crs=src_crs,
                src_nodata=np.nan,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=getattr(Resampling, self._interpolation.definition)
            )
            if coords_dst['lat'].is_descending:
                data_dst.data[:] = destination
            else:
                data_dst.data[:] = destination[::-1, :]
        return data_dst
            
    def interpolate_irregular_grid(self, data_src, coords_src,
                                   data_dst, coords_dst, grid=True):
        """Summary
        
        Parameters
        ----------
        data_src : TYPE
            Description
        coords_src : TYPE
            Description
        data_dst : TYPE
            Description
        coords_dst : TYPE
            Description
        grid : bool, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        
        Raises
        ------
        ValueError
            Description
        """
        if len(data_src.dims) > 2:
            keep_dims = ['lat', 'lon']
            return self._loop_helper(self.interpolate_irregular_grid, keep_dims,
                                     data_src, coords_src, data_dst, coords_dst,
                                     grid=grid)
        elif 'lat' not in data_src.dims or 'lon' not in data_src.dims:
            raise ValueError
        
        interp = self._interpolation.definition
        s = []
        if coords_src['lat'].is_descending:
            lat = coords_src['lat'].coordinates[::-1]
            s.append(slice(None, None, -1))
        else:
            lat = coords_src['lat'].coordinates
            s.append(slice(None, None))
        if coords_src['lon'].is_descending:
            lon = coords_src['lon'].coordinates[::-1]
            s.append(slice(None, None, -1))
        else:
            lon = coords_src['lon'].coordinates
            s.append(slice(None, None))
            
        data = data_src.data[s]
        
        # remove nan's
        I, J = np.isfinite(lat), np.isfinite(lon)
        coords_i = lat[I], lon[J]
        coords_i_dst = [coords_dst['lon'].coordinates,
                        coords_dst['lat'].coordinates]
        # Swap order in case datasource uses lon,lat ordering instead of lat,lon
        if coords_src.dims.index('lat') > coords_src.dims.index('lon'):
            I, J = J, I
            coords_i = coords_i[::-1]
            coords_i_dst = coords_i_dst[::-1]
        data = data[I, :][:, J]
        
        if interp in ['bilinear', 'nearest']:
            f = RegularGridInterpolator(
                coords_i, data, method=interp.replace('bi', ''), bounds_error=False, fill_value=np.nan)
            if grid:
                x, y = np.meshgrid(*coords_i_dst)
            else:
                x, y = coords_i_dst
            data_dst.data[:] = f((y.ravel(), x.ravel())).reshape(data_dst.shape)
        elif 'spline' in interp:
            if interp == 'cubic_spline':
                order = 3
            else:
                order = int(interp.split('_')[-1])
            f = RectBivariateSpline(coords_i[0], coords_i[1], data, kx=max(1, order), ky=max(1, order))
            data_dst.data[:] = f(coords_i_dst[1], coords_i_dst[0], grid=grid).reshape(data_dst.shape)
        return data_dst

    def interpolate_point_data(self, data_src, coords_src, data_dst, coords_dst, grid=True):
        """Summary
        
        Parameters
        ----------
        data_src : TYPE
            Description
        coords_src : TYPE
            Description
        data_dst : TYPE
            Description
        coords_dst : TYPE
            Description
        grid : bool, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        """

        order = 'lat_lon' if 'lat_lon' in coords_src.dims else 'lon_lat'
        
        # calculate tolerance
        if isinstance(coords_dst['lat'], UniformCoordinates1d):
            dlat = coords_dst['lat'].step
        else:
            dlat = (coords_dst['lat'].bounds[1] - coords_dst['lat'].bounds[0]) / (coords_dst['lat'].size-1)

        if isinstance(coords_dst['lon'], UniformCoordinates1d):
            dlon = coords_dst['lon'].step
        else:
            dlon = (coords_dst['lon'].bounds[1] - coords_dst['lon'].bounds[0]) / (coords_dst['lon'].size-1)
        
        tol = np.linalg.norm([dlat, dlon]) * 8

        if 'lat_lon' in coords_dst.dims or 'lon_lat' in coords_dst.dims:
            dst_order = 'lat_lon' if 'lat_lon' in coords_dst.dims else 'lon_lat'
            src_stacked = np.stack([coords_src[dim].coordinates for dim in coords_src[order].dims], axis=1)
            new_stacked = np.stack([coords_dst[dim].coordinates for dim in coords_src[order].dims], axis=1)
            pts = KDTree(src_stacked)
            dist, ind = pts.query(new_stacked, distance_upper_bound=tol)
            mask = ind == data_src[order].size
            ind[mask] = 0
            vals = data_src[{order: ind}]
            vals[{order: mask}] = np.nan
            dims = list(data_dst.dims)
            dims[dims.index(dst_order)] = order
            data_dst.data[:] = vals.transpose(*dims).data[:]
            return data_dst

        elif 'lat' in coords_dst.dims and 'lon' in coords_dst.dims:
            pts = np.stack([coords_src[dim].coordinates for dim in coords_src[order].dims], axis=1)
            if 'lat_lon' == order:
                pts = pts[:, ::-1]
            pts = KDTree(pts)
            lon, lat = np.meshgrid(coords_dst.coords['lon'], coords_dst.coords['lat'])
            dist, ind = pts.query(np.stack((lon.ravel(), lat.ravel()), axis=1), distance_upper_bound=tol)
            mask = ind == data_src[order].size
            ind[mask] = 0 # This is a hack to make the select on the next line work
                          # (the masked values are set to NaN on the following line)
            vals = data_src[{order: ind}]
            vals[mask] = np.nan
            # make sure 'lat_lon' or 'lon_lat' is the first dimension
            dims = [dim for dim in data_src.dims if dim != order]
            vals = vals.transpose(order, *dims).data
            shape = vals.shape
            coords = [coords_dst['lat'].coordinates, coords_dst['lon'].coordinates]
            coords += [coords_src[d].coordinates for d in dims]
            vals = vals.reshape(coords_dst['lat'].size, coords_dst['lon'].size, *shape[1:])
            vals = UnitsDataArray(vals, coords=coords, dims=['lat', 'lon'] + dims)
            # and transpose back to the destination order
            data_dst.data[:] = vals.transpose(*data_dst.dims).data[:]
            return data_dst

    @property
    @common_doc(COMMON_DATA_DOC)
    def base_definition(self):
        """Base node defintion for DataSource nodes. 
        
        Returns
        -------
        {definition_return}
        """
        d = super(DataSource, self).base_definition
        d['source'] = self.source

        # TODO: cast interpolation to string in way that can be recreated here
        # should this move to interpolation class? 
        # It causes issues when the _interpolation class has not been set up yet
        d['interpolation'] = self.interpolation
        return d
