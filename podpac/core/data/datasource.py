"""
Generic Data Source Class

DataSource is the root class for all other podpac defined data sources,
including user defined data sources.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import warnings
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import xarray as xr
import traitlets as tl

# Optional dependencies
try:
    import rasterio
    from rasterio import transform
    from rasterio.warp import reproject, Resampling
except ImportError:
    rasterio = None

try:
    from scipy.interpolate import (RectBivariateSpline, RegularGridInterpolator)
    from scipy.spatial import KDTree
except ImportError:
    scipy = None

# Internal imports
from podpac.core.units import UnitsDataArray
from podpac.core.coordinate.coordinate import Coordinate, UniformCoord
from podpac.core.node import Node
from podpac.core.utils import common_doc
from podpac.core.node import COMMON_NODE_DOC
from podpac.core.data.interpolate import Interpolator

DATA_DOC = {
    'get_data':
        """
        This method must be defined by the data source implementing the DataSource class.
        When data source nodes are executing, this method is called with request coordinates and coordinate indexes.
        The implementing method can choose which input provides the most efficient method of getting data
        (i.e via coordinates or via the index of the coordinates).
        
        Coordinates and coordinate indexes may be strided or subsets of the
        source data, but all coordinates and coordinate indexes will match 1:1 with the subset data.

        This method may return a numpy array, an xarray DaraArray, or a podpac UnitsDataArray.
        If a numpy array or xarray DataArray is returned, :meth:podpac.core.data.datasource.DataSource.evaluate will
        cast the data into a `UnitsDataArray` using the requested source coordinates.
        If a podpac UnitsDataArray is passed back, the :meth:podpac.core.data.datasource.DataSource.evaluate
        method will not do any further processing.
        The inherited Node method `initialize_coord_array` can be used to generate the template UnitsDataArray
        in your DataSource.
        See :meth:podpac.core.node.Node.initialize_coord_array for more details.
        
        Parameters
        ----------
        coordinates : podpac.core.coordinate.Coordinate
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
    'ds_native_coordinates': 'The coordinates of the data source.',
    'get_native_coordinates':
        """
        Returns a Coordinate object that describes the native coordinates of the data source.

        In most cases, this method is defined by the data source implementing the DataSource class.
        If this method is not implemented by the data source, this method will try to return `self.native_coordinates`,
        if they are defined and are an instance of a Coordinate class.
        Otherwise, this method will raise a NotImplementedError.

        Returns
        --------
        podpac.core.coordinate.Coordinate
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
    coordinate_index_type : str, optional
        Type of index to use for data source. Possible values are ['list','numpy','xarray','pandas']
        Default is 'numpy'
    interpolation : podpac.core.data.Interpolator, optional
        Type of interpolation to apply to each dimension
    nan_vals : List, optional
        List of values from source data that should be interpreted as 'no data' or 'nans'
    source : Any
        The location of the source. Depending on the child node this can be a filepath,
        numpy array, or dictionary as a few examples.
    
    Members
    -------
    requested_coordinates : podpac.core.coordinate.coordinate.Coordinates
        Coordinates requested by the data source when evalulating a node.
    requested_source_coordinates : podpac.core.coordinate.coordinate.Coordinates
        The `requested_coordinates` transformed into the native coordinate system
    requested_source_coordinates_index : list
        the index of the requested source coordinates based on `coordinate_index_type`
    requested_source_data : podpac.core.units.UnitsDataArray
        the data requested from the data source before being interpolated into self.output

    Notes
    -----
    Developers of new DataSource nodes need to implement the `get_data` and `get_native_coordinates` methods.
    """
    
    # TODO: which of these get tagged with attr?
    source = tl.Any(allow_none=False, help='Path to the raw data source')

    # TODO: decide how to handle once we implement Interpolation
    interpolation = tl.Union([
        tl.Instance(Interpolator, allow_none=True),
        tl.Enum(['nearest', 'nearest_preview', 'bilinear', 'cubic',
                 'cubic_spline', 'lanczos', 'average', 'mode',
                 'gauss', 'max', 'min', 'med', 'q1', 'q3']   # TODO: gauss is not supported by rasterio
               )], default_value='nearest').tag(attr=True)

    coordinate_index_type = tl.Enum(['list', 'numpy', 'xarray', 'pandas'], default_value='numpy')
    nan_vals = tl.List(allow_none=True)
    
    # TODO: remove when we have interpolation spec. This replaces interpolation_param for now.
    interpolation_tolerance = tl.Instance(np.timedelta64, allow_none=True)

    # TODO: include these attributes out here? How else do we document existence?
    requested_coordinates = tl.Instance(Coordinate, allow_none=True)
    requested_source_coordinates = tl.Instance(Coordinate)
    requested_source_coordinates_index = tl.List()
    requested_source_data = tl.Instance(UnitsDataArray)

    # default native_coordinates calls get_native_coordinates
    @tl.default('native_coordinates')
    def _native_coordinates_default(self):
        return self.get_native_coordinates()


    @common_doc(COMMON_DATA_DOC)
    def execute(self, coordinates, output=None, method=None):
        """Evaluates this node using the supplied coordinates.

        The evaluation process start by setting `requested_coordinates` to the supplied input coordinates.
        The native coordinates are mapped to the requested coordinates, interpolated if necessary, and set
        to `requested_source_coordinates` with associated index `requested_source_coordinates_index`.
        The requested souce coordinates and index are passed to `get_data()` returning the source data at
        the native coordinatesset to `requested_source_data`.
        Finally `requested_source_data` is interpolated using the `interpolate` method and set to 
        the `output` attribute of the node.


        Parameters
        ----------
        coordinates : podpac.core.coordinate.coordinate.Coordinates
            {requested_coordinates}
        output : podpac.core.units.UnitsDataArray, optional
            {execute_out}
        method : str, optional
            {execute_method}
        
        Returns
        -------
        {execute_return}
        """

        # initial checks
        if self.coordinate_index_type != 'numpy':
            warnings.warn('Coordinate index type {} is not yet supported.'.format(self.coordinate_index_type) +
                          '`coordinate_index_type` is set to `numpy`', UserWarning)
        
        # set input coordinates to requested_coordinates
        self.requested_coordinates = deepcopy(coordinates)

        # remove dimensions that don't exist in native coordinates
        for dim in self.requested_coordinates.dims_map.keys():
            if dim not in self.native_coordinates.dims_map.keys():
                self.requested_coordinates.drop_dims(dim)

        # initiate/reset output
        self.output = output

        # intersect the native coordinates with requested coordinates
        # to get native coordinates within requested coordinates bounds
        self.requested_source_coordinates = self.native_coordinates.intersect(self.requested_coordinates)

        # TODO: support coordinate_index_type parameter to define other index types
        self.requested_source_coordinates_index = self.native_coordinates.intersect(self.requested_coordinates, ind=True)

        # If requested coordinates and native coordinates do not intersect, shortcut with nan UnitsDataArary
        if np.prod(self.requested_source_coordinates.shape) == 0:
            udata_array = self.initialize_coord_array(self.requested_coordinates, init_type='nan')
            if self.output is None:
                self.output = udata_array
            else:
                self.output[:] = udata_array.transpose(*self.output.dims)

            return self.output
        
        # interpolate coordinates before getting data
        # TODO: extend when we edit interpolation methods
        if self.interpolation == 'nearest_preview':
            self._interpolate_requested_coordinates()

        # get data from data source
        self.requested_source_data = self._get_data()

        # interpolate data into self.output
        # TODO: streamline with interpolation methods
        o = self._interpolate()
        if o is not None:
            self.output = o  # should already be self.output

        # set the order of dims to be the same as that of requested_coordinates
        # + the dims that are missing from requested_coordinates.
        missing_dims = [dim for dim in self.native_coordinates.dims_map.keys() \
                        if dim not in self.requested_coordinates.dims_map.keys()]
        missing_dims = np.unique([self.native_coordinates.dims_map[md] for md in missing_dims]).tolist()
        transpose_dims = self.requested_coordinates.dims + missing_dims
        self.output = self.output.transpose(*transpose_dims)
        
        self.evaluated = True
        return self.output

    def _interpolate_requested_coordinates(self):
        """
        Interpolate the source coordinates based on the requested coordinates.
        Mutates `self.requested_source_coordinates` and `self.requested_source_coordinates_index`

        Returns
        -------
        None
        """

        # TODO: replace this with actual self.interpolation methods
        if self.interpolation == 'nearest_preview':
            # We can optimize a little
            new_coords = OrderedDict()
            new_coords_idx = []
            for i, d in enumerate(self.requested_source_coordinates.dims):
                if isinstance(self.requested_source_coordinates[d], UniformCoord):
                    if d in self.requested_coordinates.dims:
                        ndelta = np.round(self.requested_coordinates[d].delta / self.requested_source_coordinates[d].delta)
                        if ndelta <= 1:
                            ndelta = 1 # self.requested_source_coordinates[d].delta
                        coords = tuple(self.requested_source_coordinates[d].coords[:2]) + (ndelta * self.requested_source_coordinates[d].delta,)
                        new_coords[d] = coords
                        new_coords_idx.append(
                            slice(self.requested_source_coordinates_index[i].start,
                                  self.requested_source_coordinates_index[i].stop,
                                  int(ndelta))
                            )
                    else:
                        new_coords[d] = self.requested_source_coordinates[d]
                        new_coords_idx.append(self.requested_source_coordinates_index[i])
                else:
                    new_coords[d] = self.requested_source_coordinates[d]
                    new_coords_idx.append(self.requested_source_coordinates_index[i])

            # updates requested source coordinates and index
            self.requested_source_coordinates = Coordinate(new_coords)
            self.requested_source_coordinates_index = new_coords_idx
    

    def _get_data(self):
        """Wrapper for `self.get_data` with pre and post processing
        
        Returns
        -------
        podpac.core.units.UnitsDataArray
            Returns UnitsDataArray with coordinates defined by requested_source_coordinates
        
        Raises
        ------
        ValueError
            Raised if unknown data is passed by from self.get_data
        NotImplementedError
            Raised if get_data is not implemented by data source subclass

        """
        # get data from data source at requested source coordinates and requested source coordinates index
        data = self.get_data(self.requested_source_coordinates, self.requested_source_coordinates_index)

        # convert data into UnitsDataArray depending on format
        # TODO: what other processing needs to happen here? 
        if isinstance(data, UnitsDataArray):
            udata_array = data
        elif isinstance(data, xr.DataArray):
            # TODO: check order of coordinates here
            udata_array = self.initialize_coord_array(self.requested_coordinates, 'data', fillval=data)
        elif isinstance(data, np.ndarray):
            udata_array = self.initialize_coord_array(self.requested_coordinates, 'data', fillval=data)
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

        # TODO: This results in a recursive loop in the case of a default
        # if isinstance(self.native_coordinates, Coordinate):
        #     return self.native_coordinates

        raise NotImplementedError

    
    def _interpolate(self):
        """Interpolates the source data to the destination using self.interpolation as the interpolation method.
        
        Returns
        -------
        UnitsDataArray
            Result of interpolating the source data to the destination coordinates
        """

        # initialize output if not already input
        if self.output is None:
            self.output = self.initialize_output_array()
        else:
            # TODO: confirm that output is the right size ?
            pass

        # assign shortnames
        data_src = self.requested_source_data
        coords_src = self.requested_source_coordinates
        coords_dst = self.requested_coordinates
        data_dst = self.output
        
        # This a big switch, funneling data to various interpolation routines
        if data_src.size == 1 and np.prod(coords_dst.shape) == 1:
            data_dst[:] = data_src
            return data_dst
        
        # Nearest preview of rasters
        if self.interpolation == 'nearest_preview':
            crds = OrderedDict()
            tol = np.inf
            for c in data_dst.coords.keys():
                crds[c] = data_dst.coords[c].data.copy()
                if c is not 'time':
                    tol = min(tol, np.abs(getattr(coords_dst[c], 'delta', tol)))
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
            coords_src._coords['time'] = data_src['time'].data
            if len(coords_dst.dims) == 1:
                return data_src

        if 'alt' in coords_src.dims and 'alt' in coords_dst.dims:
            data_src = data_src.reindex(alt=coords_dst.coords['alt'], method='nearest')
            coords_src._coords['alt'] = data_src['alt'].data
            if len(coords_dst.dims) == 1:
                return data_src

        # Raster to Raster interpolation from regular grids to regular grids
        rasterio_interps = ['nearest', 'bilinear', 'cubic', 'cubic_spline',
                            'lanczos', 'average', 'mode', 'gauss', 'max', 'min',
                            'med', 'q1', 'q3']
        if rasterio is not None \
                and self.interpolation in rasterio_interps \
                and ('lat' in coords_src.dims and 'lon' in coords_src.dims) \
                and ('lat' in coords_dst.dims and 'lon' in coords_dst.dims) \
                and coords_src['lat'].rasterio_regularity \
                and coords_src['lon'].rasterio_regularity \
                and coords_dst['lat'].rasterio_regularity \
                and coords_dst['lon'].rasterio_regularity:
            return self.rasterio_interpolation(data_src, coords_src, data_dst, coords_dst)

        # Raster to Raster interpolation from irregular grids to arbitrary grids
        elif ('lat' in coords_src.dims and 'lon' in coords_src.dims) \
                and ('lat' in coords_dst.dims and 'lon' in coords_dst.dims)\
                and coords_src['lat'].scipy_regularity \
                and coords_src['lon'].scipy_regularity:
            
            return self.interpolate_irregular_grid(data_src, coords_src,
                                                   data_dst, coords_dst,
                                                   grid=True)
        # Raster to lat_lon point interpolation
        elif (('lat' in coords_src.dims and 'lon' in coords_src.dims)
              and coords_src['lat'].scipy_regularity
              and coords_src['lon'].scipy_regularity
              and ('lat_lon' in coords_dst.dims or 'lon_lat' in coords_dst.dims)):
            coords_dst_us = coords_dst.unstack() # TODO don't have to return
            return self.interpolate_irregular_grid(data_src, coords_src,
                                                   data_dst, coords_dst_us,
                                                   grid=False)

        elif 'lat_lon' in coords_src.dims or 'lon_lat' in coords_src.dims:
            return self.interpolate_point_data(data_src, coords_src,
                                               data_dst, coords_dst)
        
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
                resampling=getattr(Resampling, self.interpolation)
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
        
        interp = self.interpolation
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
            f = RegularGridInterpolator(coords_i, data,
                                        method=interp.replace('bi', ''),
                                        bounds_error=False, fill_value=np.nan)
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
            f = RectBivariateSpline(coords_i[0], coords_i[1],
                                    data,
                                    kx=max(1, order),
                                    ky=max(1, order))
            data_dst.data[:] = f(coords_i_dst[1],
                                 coords_i_dst[0],
                                 grid=grid).reshape(data_dst.shape)
        return data_dst

    def interpolate_point_data(self, data_src, coords_src,
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
        """
        if 'lat_lon' in coords_dst.dims or 'lon_lat' in coords_dst.dims:
            order = coords_src.dims_map['lat']
            dst_order = coords_dst.dims_map['lat']


            # there is a bug here that is not yet fixed
            if order != dst_order:
                raise NotImplementedError('%s -> %s interpolation not currently supported' % (
                    order, dst_order))

            i = list(coords_dst.dims).index(dst_order)
            new_crds = Coordinate(**{order: [coords_dst.unstack()[c].coordinates
                for c in order.split('_')]})
            tol = np.linalg.norm(coords_dst.delta[i]) * 8
            src_stacked = np.stack([c.coordinates for c in coords_src.stack_dict()[order]], axis=1)
            new_stacked = np.stack([c.coordinates for c in new_crds.stack_dict()[order]], axis=1) 
            pts = KDTree(src_stacked)
            dist, ind = pts.query(new_stacked, distance_upper_bound=tol)
            mask = ind == data_src[order].size
            ind[mask] = 0
            vals = data_src[{order: ind}]
            vals[{order: mask}] = np.nan
            dims = list(data_dst.dims)
            dims[i] = order
            data_dst.data[:] = vals.transpose(*dims).data[:]
            return data_dst
        elif 'lat' in coords_dst.dims and 'lon' in coords_dst.dims:
            order = coords_src.dims_map['lat']
            i = list(coords_dst.dims).index('lat')
            j = list(coords_dst.dims).index('lon')
            tol = np.linalg.norm([coords_dst.delta[i], coords_dst.delta[j]]) * 8
            pts = np.stack([c.coordinates for c in coords_src.stack_dict()[order]], axis=1)
            if 'lat_lon' == order:
                pts = pts[:, ::-1]
            pts = KDTree(pts)
            lon, lat = np.meshgrid(coords_dst.coords['lon'],
                    coords_dst.coords['lat'])
            dist, ind = pts.query(np.stack((lon.ravel(), lat.ravel()), axis=1),
                    distance_upper_bound=tol)
            mask = ind == data_src[order].size
            ind[mask] = 0 # This is a hack to make the select on the next line work
                          # (the masked values are set to NaN on the following line)
            vals = data_src[{order: ind}]
            vals[mask] = np.nan
            # make sure 'lat_lon' or 'lon_lat' is the first dimension
            dims = list(data_src.dims)
            dims.remove(order)
            vals = vals.transpose(order, *dims).data
            shape = vals.shape
            vals = vals.reshape(coords_dst['lat'].size, coords_dst['lon'].size,
                    *shape[1:])
            vals = UnitsDataArray(vals, dims=['lat', 'lon'] + dims,
                    coords=[coords_dst.coords['lat'], coords_dst.coords['lon']]
                    + [coords_src[d].coordinates for d in dims])
            data_dst.data[:] = vals.transpose(*data_dst.dims).data[:]
            return data_dst

    @property
    @common_doc(COMMON_DATA_DOC)
    def definition(self):
        """Pipeline node defintion for DataSource nodes. 
        
        Returns
        -------
        {definition_return}
        """
        d = self.base_definition()
        d['source'] = self.source
        return d
