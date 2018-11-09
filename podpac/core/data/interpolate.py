"""
Interpolation handling

Attributes
----------
INTERPOLATION_DEFAULT : str
    Default interpolation method used when creating a new :class:`Interpolation` class
INTERPOLATION_METHODS : dict
    Dictionary of string interpolation method strings and the associated interpolator classes that support
    the method (i.e. ``'nearest': [NearestNeighbor, Rasterio, Scipy]``)
INTERPOLATION_SHORTCUTS : list
    Keys of :attr:`INTERPOLATION_METHODS`

"""

from __future__ import division, unicode_literals, print_function, absolute_import
import warnings
from copy import deepcopy
from collections import OrderedDict
from six import string_types

import numpy as np
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
    from scipy.interpolate import (griddata, RectBivariateSpline,
                                   RegularGridInterpolator)
    from scipy.spatial import KDTree
except:
    scipy = None

# podac imports
from podpac.core.units import UnitsDataArray
from podpac.core.coordinates import Coordinates, UniformCoordinates1d, StackedCoordinates
from podpac.core.utils import common_doc

# common doc properties
INTERPOLATE_DOCS = {
    'interpolator_attributes':
        """
        method : str
            Current interpolation method to use in Interpolator (i.e. 'nearest').
            This attribute is set during node evaluation when a new :class:`Interpolation`
            class is constructed. See the :class:`podpac.data.DataSource` `interpolation` attribute for
            more information on specifying the interpolator method.
        dims_supported : list
            List of unstacked dimensions supported by the interpolator.
            This attribute should be defined by the implementing :class:`Interpolator`.
            Used by private convience method :meth:`_filter_udims_supported`.
        """,
    'nearest_neighbor_attributes':
        """
        Attributes
        ----------
        method : str
            Current interpolation method to use in Interpolator (i.e. 'nearest').
            This attribute is set during node evaluation when a new :class:`Interpolation`
            class is constructed. See the :class:`podpac.data.DataSource` `interpolation` attribute for
            more information on specifying the interpolator method.
        dims_supported : list
            List of unstacked dimensions supported by the interpolator.
            This attribute should be defined by the implementing :class:`Interpolator`.
            Used by private convience method :meth:`_filter_udims_supported`.
        spatial_tolerance : float
            Maximum distance to the nearest coordinate in space.
            Cooresponds to the unit of the space measurement.
        time_tolerance : float
            Maximum distance to the nearest coordinate in time coordinates.
            Accepts p.timedelta64() (i.e. np.timedelta64(1, 'D') for a 1-Day tolerance)
        """,
    'interpolator_can_select':
        """
        Evaluate if interpolator can downselect the source coordinates from the requested coordinates
        for the unstacked dims supplied.
        If not overwritten, this method returns an empty tuple (``tuple()``)
        
        Parameters
        ----------
        udims : tuple
            dimensions to select
        source_coordinates : :class:`podpac.Coordinates`
            Description
        eval_coordinates : :class:`podpac.Coordinates`
            Description
        
        Returns
        -------
        tuple
            Returns a tuple of dimensions that can be selected with this interpolator
            If no dimensions can be selected, method should return an emtpy tuple
        """,
    'interpolator_select':
        """
        Downselect coordinates with interpolator method
        
        Parameters
        ----------
        udims : tuple
            dimensions to select coordinates
        source_coordinates : :class:`podpac.Coordinates`
            Description
        source_coordinates_index : list
            Description
        eval_coordinates : :class:`podpac.Coordinates`
            Description
        
        Returns
        -------
        (:class:`podpac.Coordinates`, list)
            returns the new down selected coordinates and the new associated index. These coordinates must exist
            in the native coordinates of the source data

        Raises
        ------
        NotImplementedError
        """,
    'interpolator_can_interpolate':
        """
        Evaluate if this interpolation method can handle the requested coordinates and source_coordinates.
        If not overwritten, this method returns an empty tuple (`tuple()`)
        
        Parameters
        ----------
        udims : tuple
            dimensions to interpolate
        source_coordinates : :class:`podpac.Coordinates`
            Description
        eval_coordinates : :class:`podpac.Coordinates`
            Description
        
        Returns
        -------
        tuple
            Returns a tuple of dimensions that can be interpolated with this interpolator
            If no dimensions can be interpolated, method should return an emtpy tuple
        """,
    'interpolator_interpolate':
        """
        Interpolate data from requested coordinates to source coordinates.
        
        Parameters
        ----------
        udims : tuple
            dimensions to interpolate
        source_coordinates : :class:`podpac.Coordinates`
            Description
        source_data : podpac.core.units.UnitsDataArray
            Description
        eval_coordinates : :class:`podpac.Coordinates`
            Description
        output_data : podpac.core.units.UnitsDataArray
            Description
        
        Raises
        ------
        NotImplementedError
        
        Returns
        -------
        podpac.core.units.UnitDataArray
            returns the updated output of interpolated data
        """
}

class InterpolationException(Exception):
    """
    Custom label for interpolator exceptions
    """
    pass
    
@common_doc(INTERPOLATE_DOCS)
class Interpolator(tl.HasTraits):
    """Interpolation Method

    Attributes
    ----------
    {interpolator_attributes}
    
    """

    method = tl.Unicode(allow_none=False)
    dims_supported = tl.List(tl.Unicode(), allow_none=False)

    # Next are used for optimizing the interpolation pipeline
    # If -1, it's cost is assume the same as a competing interpolator in the
    # stack, and the determination is made based on the number of DOF before
    # and after each interpolation step.
    # cost_func = tl.CFloat(-1)  # The rough cost FLOPS/DOF to do interpolation
    # cost_setup = tl.CFloat(-1)  # The rough cost FLOPS/DOF to set up the interpolator

    def __init__(self, **kwargs):
        
        # Call traitlets constructor
        super(Interpolator, self).__init__(**kwargs)
        self.init()

    def init(self):
        """
        Overwrite this method if a Interpolator needs to do any
        additional initialization after the standard initialization.
        """
        pass

    def _filter_udims_supported(self, udims):
        
        # find the intersection between dims_supported and udims, return tuple of intersection
        return tuple(set(self.dims_supported) & set(udims))

    def _dim_in(self, dim, *coords, **kwargs):
        """Verify the dim exists on coordinates
        
        Parameters
        ----------
        dim : str, list of str
            Dimension or list of dimensions to verify
        *coords :class:`podpac.Coordinates`
            coordinates to evaluate
        unstacked : bool, optional
            True if you want to compare dimensions in unstacked form, otherwise compare dimensions however
            they are defined on the DataSource. Defaults to False.
        
        Returns
        -------
        Boolean
            True if the dim is in all input coordinates
        """

        unstacked = kwargs.pop('unstacked', False)

        if isinstance(dim, str):
            dim = [dim]
        elif not isinstance(dim, (list, tuple)):
            raise ValueError('`dim` input must be a str, list of str, or tuple of str')

        for coord in coords:
            for d in dim:
                if (unstacked and d not in coord.udims) or (not unstacked and d not in coord.dims):
                    return False
        
        return True

    def _loop_helper(self, func, keep_dims, udims, 
                     source_coordinates, source_data, eval_coordinates, output_data, **kwargs):
        """Loop helper
        
        Parameters
        ----------
        func : TYPE
            Description
        keep_dims : TYPE
            Description
        udims : TYPE
            Description
        source_coordinates : TYPE
            Description
        source_data : TYPE
            Description
        eval_coordinates : TYPE
            Description
        output_data : TYPE
            Description
        **kwargs
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        loop_dims = [d for d in source_data.dims if d not in keep_dims]
        if loop_dims:
            for i in source_data.coords[loop_dims[0]]:
                ind = {loop_dims[0]: i}
                output_data.loc[ind] = \
                    self._loop_helper(func, keep_dims, udims,
                                      source_coordinates, source_data.loc[ind],
                                      eval_coordinates, output_data.loc[ind], **kwargs)
        else:
            return func(udims, source_coordinates, source_data, eval_coordinates, output_data, **kwargs)

        return output_data

    @common_doc(INTERPOLATE_DOCS)
    def can_select(self, udims, source_coordinates, eval_coordinates):
        """
        {interpolator_can_select}
        """

        return tuple()

    @common_doc(INTERPOLATE_DOCS)
    def select_coordinates(self, udims, source_coordinates, source_coordinates_index, eval_coordinates):
        """
        {interpolator_select}
        """
        raise NotImplementedError

    @common_doc(INTERPOLATE_DOCS)
    def can_interpolate(self, udims, source_coordinates, eval_coordinates):
        """
        {interpolator_can_interpolate}
        """
        return tuple()

    @common_doc(INTERPOLATE_DOCS)
    def interpolate(self, udims, source_coordinates, source_data, eval_coordinates, output_data):
        """
        {interpolator_interpolate}
        """
        raise NotImplementedError

@common_doc(INTERPOLATE_DOCS)
class NearestNeighbor(Interpolator):
    """Nearest Neighbor Interpolation
    
    {nearest_neighbor_attributes}
    """
    dims_supported = ['lat', 'lon', 'alt', 'time']
    spatial_tolerance = tl.Float(default_value=np.inf)
    time_tolerance = tl.Instance(np.timedelta64, allow_none=True)

    @common_doc(INTERPOLATE_DOCS)
    def can_interpolate(self, udims, source_coordinates, eval_coordinates):
        """
        {interpolator_interpolate}
        """
        udims_subset = self._filter_udims_supported(udims)

        # confirm that udims are in both source and eval coordinates
        # TODO: handle stacked coordinates
        if self._dim_in(udims_subset, source_coordinates, eval_coordinates):
            return udims_subset
        else:
            return tuple()

    @common_doc(INTERPOLATE_DOCS)
    def interpolate(self, udims, source_coordinates, source_data, eval_coordinates, output_data):
        """
        {interpolator_interpolate}
        """
        indexers = []

        # select dimensions common to eval_coordinates and udims
        # TODO: this is sort of convoluted implementation
        for dim in eval_coordinates.dims:

            # TODO: handle stacked coordinates
            if isinstance(eval_coordinates[dim], StackedCoordinates):

                # udims within stacked dims that are in the input udims
                udims_in_stack = list(set(udims) & set(eval_coordinates[dim].dims))

                # TODO: how do we choose a dimension to use from the stacked coordinates?
                # For now, choose the first coordinate found in the udims definition
                if udims_in_stack:
                    raise InterpolationException('Nearest interpolation does not yet support stacked dimensions')
                    # dim = udims_in_stack[0]
                else:
                    continue

            # TODO: handle if the source coordinates contain `dim` within a stacked coordinate
            elif dim not in source_coordinates.dims:
                raise InterpolationException('Nearest interpolation does not yet support stacked dimensions')

            elif dim not in udims:
                continue

            # set tolerance value based on dim type
            tolerance = None
            if dim == 'time' and self.time_tolerance:
                tolerance = self.time_tolerance
            elif dim != 'time':
                # TODO: do we want this tolerance to always be calculated? or only when spatial_tolerance is specified?
                area_bounds = getattr(eval_coordinates[dim], 'area_bounds', [-np.inf, np.inf])
                delta = np.abs(area_bounds[1] - area_bounds[0]) / eval_coordinates[dim].size
                tolerance = min(self.spatial_tolerance, delta)

            # reindex using xarray
            indexer = {
                dim: eval_coordinates[dim].coordinates.copy()
            }
            indexers += [dim]
            source_data = source_data.reindex(method=str('nearest'), tolerance=tolerance, **indexer)

        # at this point, output_data and eval_coordinates have the same dim order
        # this transpose makes sure the source_data has the same dim order as the eval coordinates
        output_data.data = source_data.transpose(*indexers)

        return output_data


@common_doc(INTERPOLATE_DOCS)
class NearestPreview(NearestNeighbor):
    """Nearest Neighbor (Preview) Interpolation
    
    {nearest_neighbor_attributes}
    """

    @common_doc(INTERPOLATE_DOCS)
    def can_select(self, udims, source_coordinates, eval_coordinates):
        """
        {interpolator_can_select}
        """
        udims_subset = self._filter_udims_supported(udims)

        # confirm that udims are in both source and eval coordinates
        # TODO: handle stacked coordinates
        if self._dim_in(udims_subset, source_coordinates, eval_coordinates):
            return udims_subset
        else:
            return tuple()

    @common_doc(INTERPOLATE_DOCS)
    def select_coordinates(self, udims, source_coordinates, source_coordinates_index, eval_coordinates):
        """
        {interpolator_select}
        """
        new_coords = []
        new_coords_idx = []

        # iterate over the source coordinate dims in case they are stacked
        for src_dim, idx in zip(source_coordinates, source_coordinates_index):

            # TODO: handle stacked coordinates 
            if isinstance(source_coordinates[src_dim], StackedCoordinates):
                raise InterpolationException('NearestPreview select does not yet support stacked dimensions')

            if src_dim in eval_coordinates.dims:
                src_coords = source_coordinates[src_dim]
                dst_coords = eval_coordinates[src_dim]

                if isinstance(dst_coords, UniformCoordinates1d):
                    dst_start = dst_coords.start
                    dst_stop = dst_coords.stop
                    dst_delta = dst_coords.step
                else:
                    dst_start = dst_coords.coordinates[0]
                    dst_stop = dst_coords.coordinates[-1]
                    dst_delta = (dst_stop-dst_start) / (dst_coords.size - 1)

                if isinstance(src_coords, UniformCoordinates1d):
                    src_start = src_coords.start
                    src_stop = src_coords.stop
                    src_delta = src_coords.step
                else:
                    src_start = src_coords.coordinates[0]
                    src_stop = src_coords.coordinates[-1]
                    src_delta = (src_stop-src_start) / (src_coords.size - 1)

                ndelta = max(1, np.round(dst_delta / src_delta))
                if src_coords.size == 1:
                    c = src_coords.copy()
                else:
                    c = UniformCoordinates1d(src_start, src_stop, ndelta*src_delta, **src_coords.properties)
                
                if isinstance(idx, slice):
                    idx = slice(idx.start, idx.stop, int(ndelta))
                else:
                    idx = slice(idx[0], idx[-1], int(ndelta))
            else:
                c = source_coordinates[src_dim]

            new_coords.append(c)
            new_coords_idx.append(idx)

        return Coordinates(new_coords), new_coords_idx


@common_doc(INTERPOLATE_DOCS)
class Rasterio(Interpolator):
    """Rasterio Interpolation
    
    Attributes
    ----------
    {interpolator_attributes}
    rasterio_interpolators : list of str
        Interpolator methods available via rasterio
    """

    # TODO: implement these parameters for the method 'nearest'
    spatial_tolerance = tl.Float(default_value=np.inf)
    time_tolerance = tl.Instance(np.timedelta64, allow_none=True)

    # TODO: support 'gauss' method?

    @common_doc(INTERPOLATE_DOCS)
    def can_interpolate(self, udims, source_coordinates, eval_coordinates):
        """{interpolator_can_interpolate}"""

        # TODO: make this so we don't need to specify lat and lon together
        # or at least throw a warning
        if 'lat' in udims and 'lon' in udims and \
           self._dim_in(['lat', 'lon'], source_coordinates, eval_coordinates) and \
           source_coordinates['lat'].is_uniform and source_coordinates['lon'].is_uniform and \
           eval_coordinates['lat'].is_uniform and eval_coordinates['lon'].is_uniform:

            return udims
        
        # otherwise return no supported dims
        return tuple()

    @common_doc(INTERPOLATE_DOCS)
    def interpolate(self, udims, source_coordinates, source_data, eval_coordinates, output_data):
        """
        {interpolator_interpolate}
        """

        # TODO: handle when udims does not contain both lat and lon
        # if the source data has more dims than just lat/lon is asked, loop over those dims and run the interpolation
        # on those grids
        if len(source_data.dims) > 2:
            keep_dims = ['lat', 'lon']
            return self._loop_helper(self.interpolate, keep_dims,
                                     udims, source_data, source_coordinates, eval_coordinates, output_data)
        
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
            src_transform = get_rasterio_transform(source_coordinates)
            src_crs = {'init': source_coordinates.gdal_crs}
            # Need to make sure array is c-contiguous
            if source_coordinates['lat'].is_descending:
                source = np.ascontiguousarray(source_data.data)
            else:
                source = np.ascontiguousarray(source_data.data[::-1, :])
        
            dst_transform = get_rasterio_transform(eval_coordinates)
            dst_crs = {'init': eval_coordinates.gdal_crs}
            # Need to make sure array is c-contiguous
            if not output_data.data.flags['C_CONTIGUOUS']:
                destination = np.ascontiguousarray(output_data.data)
            else:
                destination = output_data.data
        
            reproject(
                source,
                np.atleast_2d(destination.squeeze()),  # Needed for legacy compatibility
                src_transform=src_transform,
                src_crs=src_crs,
                src_nodata=np.nan,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=getattr(Resampling, self.method)
            )
            if eval_coordinates['lat'].is_descending:
                output_data.data[:] = destination
            else:
                output_data.data[:] = destination[::-1, :]

        return output_data


@common_doc(INTERPOLATE_DOCS)
class ScipyPoint(Interpolator):
    """Scipy Point Interpolation
    
    Attributes
    ----------
    {interpolator_attributes}
    """

    # TODO: implement these parameters for the method 'nearest'
    spatial_tolerance = tl.Float(default_value=np.inf)
    time_tolerance = tl.Instance(np.timedelta64, allow_none=True)

    @common_doc(INTERPOLATE_DOCS)
    def can_interpolate(self, udims, source_coordinates, eval_coordinates):
        """
        {interpolator_can_interpolate}
        """

        # TODO: make this so we don't need to specify lat and lon together
        # or at least throw a warning
        if 'lat' in udims and 'lon' in udims and \
            not self._dim_in(['lat', 'lon'], source_coordinates) and \
            self._dim_in(['lat', 'lon'], source_coordinates, unstacked=True) and \
            self._dim_in(['lat', 'lon'], eval_coordinates, unstacked=True):

            return tuple(['lat', 'lon'])


        # otherwise return no supported dims
        return tuple()


    @common_doc(INTERPOLATE_DOCS)
    def interpolate(self, udims, source_coordinates, source_data, eval_coordinates, output_data):
        """
        {interpolator_interpolate}
        """

        order = 'lat_lon' if 'lat_lon' in source_coordinates.dims else 'lon_lat'
        
        # calculate tolerance
        if isinstance(eval_coordinates['lat'], UniformCoordinates1d):
            dlat = eval_coordinates['lat'].step
        else:
            dlat = (eval_coordinates['lat'].bounds[1] - eval_coordinates['lat'].bounds[0]) / (eval_coordinates['lat'].size-1)

        if isinstance(eval_coordinates['lon'], UniformCoordinates1d):
            dlon = eval_coordinates['lon'].step
        else:
            dlon = (eval_coordinates['lon'].bounds[1] - eval_coordinates['lon'].bounds[0]) / (eval_coordinates['lon'].size-1)
        
        tol = np.linalg.norm([dlat, dlon]) * 8

        if self._dim_in(['lat', 'lon'], eval_coordinates):
            pts = np.stack([source_coordinates[dim].coordinates for dim in source_coordinates[order].dims], axis=1)
            if order == 'lat_lon':
                pts = pts[:, ::-1]
            pts = KDTree(pts)
            lon, lat = np.meshgrid(eval_coordinates.coords['lon'], eval_coordinates.coords['lat'])
            dist, ind = pts.query(np.stack((lon.ravel(), lat.ravel()), axis=1), distance_upper_bound=tol)
            mask = ind == source_data[order].size
            ind[mask] = 0 # This is a hack to make the select on the next line work
                          # (the masked values are set to NaN on the following line)
            vals = source_data[{order: ind}]
            vals[mask] = np.nan
            # make sure 'lat_lon' or 'lon_lat' is the first dimension
            dims = [dim for dim in source_data.dims if dim != order]
            vals = vals.transpose(order, *dims).data
            shape = vals.shape
            coords = [eval_coordinates['lat'].coordinates, eval_coordinates['lon'].coordinates]
            coords += [source_coordinates[d].coordinates for d in dims]
            vals = vals.reshape(eval_coordinates['lat'].size, eval_coordinates['lon'].size, *shape[1:])
            vals = UnitsDataArray(vals, coords=coords, dims=['lat', 'lon'] + dims)
            # and transpose back to the destination order
            output_data.data[:] = vals.transpose(*output_data.dims).data[:]
            
            return output_data


        elif self._dim_in(['lat', 'lon'], eval_coordinates, unstacked=True):
            dst_order = 'lat_lon' if 'lat_lon' in eval_coordinates.dims else 'lon_lat'
            src_stacked = np.stack([source_coordinates[dim].coordinates for dim in source_coordinates[order].dims], axis=1)
            new_stacked = np.stack([eval_coordinates[dim].coordinates for dim in source_coordinates[order].dims], axis=1)
            pts = KDTree(src_stacked)
            dist, ind = pts.query(new_stacked, distance_upper_bound=tol)
            mask = ind == source_data[order].size
            ind[mask] = 0
            vals = source_data[{order: ind}]
            vals[{order: mask}] = np.nan
            dims = list(output_data.dims)
            dims[dims.index(dst_order)] = order
            output_data.data[:] = vals.transpose(*dims).data[:]
            
            return output_data


@common_doc(INTERPOLATE_DOCS)
class ScipyGrid(ScipyPoint):
    """Scipy Interpolation
    
    Attributes
    ----------
    {interpolator_attributes}
    """

    # TODO: implement these parameters for the method 'nearest'
    spatial_tolerance = tl.Float(default_value=np.inf)
    time_tolerance = tl.Instance(np.timedelta64, allow_none=True)

    @common_doc(INTERPOLATE_DOCS)
    def can_interpolate(self, udims, source_coordinates, eval_coordinates):
        """
        {interpolator_can_interpolate}
        """

        # TODO: make this so we don't need to specify lat and lon together
        # or at least throw a warning
        if 'lat' in udims and 'lon' in udims and \
            self._dim_in(['lat', 'lon'], source_coordinates) and \
            self._dim_in(['lat', 'lon'], eval_coordinates, unstacked=True):

            return udims

        # otherwise return no supported dims
        return tuple()

    @common_doc(INTERPOLATE_DOCS)
    def interpolate(self, udims, source_coordinates, source_data, eval_coordinates, output_data):
        """
        {interpolator_interpolate}
        """

        if self._dim_in(['lat', 'lon'], eval_coordinates):
            return self._interpolate_irregular_grid(udims, source_coordinates, source_data,
                                                    eval_coordinates, output_data, grid=True)

        elif self._dim_in(['lat', 'lon'], eval_coordinates, unstacked=True):
            eval_coordinates_us = eval_coordinates.unstack()
            return self._interpolate_irregular_grid(udims, source_coordinates, source_data,
                                                    eval_coordinates_us, output_data, grid=False)


    def _interpolate_irregular_grid(self, udims, source_coordinates, source_data,
                                    eval_coordinates, output_data, grid=True):

        if len(source_data.dims) > 2:
            keep_dims = ['lat', 'lon']
            return self._loop_helper(self._interpolate_irregular_grid, keep_dims,
                                     udims, source_coordinates, source_data, eval_coordinates, output_data, grid=grid)
        
        s = []
        if source_coordinates['lat'].is_descending:
            lat = source_coordinates['lat'].coordinates[::-1]
            s.append(slice(None, None, -1))
        else:
            lat = source_coordinates['lat'].coordinates
            s.append(slice(None, None))
        if source_coordinates['lon'].is_descending:
            lon = source_coordinates['lon'].coordinates[::-1]
            s.append(slice(None, None, -1))
        else:
            lon = source_coordinates['lon'].coordinates
            s.append(slice(None, None))
            
        data = source_data.data[s]
        
        # remove nan's
        I, J = np.isfinite(lat), np.isfinite(lon)
        coords_i = lat[I], lon[J]
        coords_i_dst = [eval_coordinates['lon'].coordinates,
                        eval_coordinates['lat'].coordinates]

        # Swap order in case datasource uses lon,lat ordering instead of lat,lon
        if source_coordinates.dims.index('lat') > source_coordinates.dims.index('lon'):
            I, J = J, I
            coords_i = coords_i[::-1]
            coords_i_dst = coords_i_dst[::-1]
        data = data[I, :][:, J]
        
        if self.method in ['bilinear', 'nearest']:
            f = RegularGridInterpolator(
                coords_i, data, method=self.method.replace('bi', ''), bounds_error=False, fill_value=np.nan)
            if grid:
                x, y = np.meshgrid(*coords_i_dst)
            else:
                x, y = coords_i_dst
            output_data.data[:] = f((y.ravel(), x.ravel())).reshape(output_data.shape)

        # TODO: what methods is 'spline' associated with?
        elif 'spline' in self.method:
            if self.method == 'cubic_spline':
                order = 3
            else:
                # TODO: make this a parameter
                order = int(self.method.split('_')[-1])

            f = RectBivariateSpline(coords_i[0], coords_i[1], data, kx=max(1, order), ky=max(1, order))
            output_data.data[:] = f(coords_i_dst[1], coords_i_dst[0], grid=grid).reshape(output_data.shape)

        return output_data

# List of available interpolators
INTERPOLATION_METHODS = {
    'nearest_preview': [NearestPreview],
    'nearest': [NearestNeighbor, Rasterio, ScipyGrid, ScipyPoint],
    'bilinear':[Rasterio, ScipyGrid],
    'cubic':[Rasterio],
    'cubic_spline':[Rasterio, ScipyGrid],
    'lanczos':[Rasterio],
    'average':[Rasterio],
    'mode':[Rasterio],
    'gauss':[Rasterio],
    'max':[Rasterio],
    'min':[Rasterio],
    'med':[Rasterio],
    'q1':[Rasterio],
    'q3': [Rasterio],
    'spline_2': [ScipyGrid],
    'spline_3': [ScipyGrid],
    'spline_4': [ScipyGrid]
}

# create shortcut list based on methods keys
INTERPOLATION_SHORTCUTS = INTERPOLATION_METHODS.keys()

# default interpolation
INTERPOLATION_DEFAULT = 'nearest'

class Interpolation():
    """Create an interpolation class to handle one interpolation method per unstacked dimension.
    Used to interpolate data within a datasource.
    
    Parameters
    ----------
    definition : str,
                 tuple (str, list of podpac.core.data.interpolate.Interpolator),
                 dict
        Interpolation definition used to define interpolation methods for each definiton.
        See :ref:podpac.core.data.datasource.DataSource.interpolation for more details.
    coordinates : :class:`podpac.Coordinates`
        source coordinates to be interpolated
    **kwargs :
        Keyword arguments passed on to each :ref:podpac.core.data.interpolate.Interpolator
    
    Raises
    ------
    InterpolationException
    TypeError
    
    """
 
    definition = None
    config = OrderedDict()             # container for interpolation methods for each dimension
    _last_interpolator_queue = None     # container for the last run interpolator queue - useful for debugging
    _last_select_queue = None           # container for the last run select queue - useful for debugging

    def __init__(self, definition=INTERPOLATION_DEFAULT):

        self.definition = definition
        self.config = OrderedDict()

        # set each dim to interpolator definition
        if isinstance(definition, dict):

            # covert input to an ordered dict to preserve order of dimensions
            definition = OrderedDict(definition)

            for key in iter(definition):

                # if dict is a default definition, skip the rest of the handling
                if not isinstance(key, tuple):
                    if key in ['method', 'params', 'interpolators']:
                        method = self._parse_interpolation_method(definition)
                        self._set_interpolation_method(('default',), method)
                        break

                # if key is not a tuple, convert it to one and set it to the udims key
                if not isinstance(key, tuple):
                    udims = (key,)
                else:
                    udims = key

                # make sure udims are not already specified in config
                for config_dims in iter(self.config):
                    if set(config_dims) & set(udims):
                        raise InterpolationException('Dimensions "{}" cannot be defined '.format(udims) +
                                                     'multiple times in interpolation definition {}'.format(definition))

                # get interpolation method
                method = self._parse_interpolation_method(definition[key])


                # add all udims to definition
                self._set_interpolation_method(udims, method)


            # set default if its not been specified in the dict
            if ('default',) not in self.config:

                default_method = self._parse_interpolation_method(INTERPOLATION_DEFAULT)
                self._set_interpolation_method(('default',), default_method)
            

        elif isinstance(definition, string_types):
            method = self._parse_interpolation_method(definition)
            self._set_interpolation_method(('default',), method)

        else:
            raise TypeError('"{}" is not a valid interpolation definition type. '.format(definition) +
                            'Interpolation definiton must be a string or dict')

        # make sure ('default',) is always the last entry in config dictionary
        default = self.config.pop(('default',))
        self.config[('default',)] = default

    def __repr__(self):
        rep = str(self.__class__.__name__)
        for udims in iter(self.config):
            # rep += '\n\t%s:\n\t\tmethod: %s\n\t\tinterpolators: %s\n\t\tparams: %s' % \
            rep += '\n\t%s: %s, %s, %s' % \
                (udims,
                 self.config[udims]['method'],
                 [i.__class__.__name__ for i in self.config[udims]['interpolators']],
                 self.config[udims]['params']
                )

        return rep

    def _parse_interpolation_method(self, definition):
        """parse interpolation definitions into a tuple of (method, Interpolator)
        
        Parameters
        ----------
        definition : str,
                     dict
            interpolation definition
            See :ref:podpac.core.data.datasource.DataSource.interpolation for more details.
        
        Returns
        -------
        dict
            dict with keys 'method', 'interpolators', and 'params'
        
        Raises
        ------
        InterpolationException
        TypeError
        """
        if isinstance(definition, string_types):
            if definition not in INTERPOLATION_SHORTCUTS:
                raise InterpolationException('"{}" is not a valid interpolation shortcut. '.format(definition) +
                                             'Valid interpolation shortcuts: {}'.format(INTERPOLATION_SHORTCUTS))
            return {
                'method': definition,
                'interpolators': INTERPOLATION_METHODS[definition],
                'params': {}
            }

        elif isinstance(definition, dict):

            # confirm method in dict
            if 'method' not in definition:
                raise InterpolationException('{} is not a valid interpolation definition. '.format(definition) +
                                             'Interpolation definition dict must contain key "method" string value')
            else:
                method_string = definition['method']

            # if specifying custom method, user must include interpolators
            if 'interpolators' not in definition and method_string not in INTERPOLATION_SHORTCUTS:
                raise InterpolationException('"{}" is not a valid interpolation shortcut. '.format(method_string) +
                                             'Specify list "interpolators" or change "method" ' +
                                             'to a valid interpolation shortcut: {}'.format(INTERPOLATION_SHORTCUTS))
            elif 'interpolators' not in definition:
                interpolators = INTERPOLATION_METHODS[method_string]
            else:
                interpolators = definition['interpolators']

            # default for params
            if 'params' in definition:
                params = definition['params']
            else:
                params = {}


            # confirm types
            if not isinstance(method_string, string_types):
                raise TypeError('{} is not a valid interpolation method. '.format(method_string) +
                                'Interpolation method must be a string')

            if not isinstance(interpolators, list):
                raise TypeError('{} is not a valid interpolator definition. '.format(interpolators) +
                                'Interpolator definition must be of type list containing Interpolator')

            if not isinstance(params, dict):
                raise TypeError('{} is not a valid interpolation params definition. '.format(params) +
                                'Interpolation params must be a dict')

            for interpolator in interpolators:
                self._validate_interpolator(interpolator)

            # if all checks pass, return the definition
            return {
                'method': method_string,
                'interpolators': interpolators,
                'params': params
            }

        else:
            raise TypeError('"{}" is not a valid Interpolator definition. '.format(definition) +
                            'Interpolation definiton must be a string or dict.')

    def _validate_interpolator(self, interpolator):
        """Make sure interpolator is a subclass of Interpolator
        
        Parameters
        ----------
        interpolator : any
            input definition to validate
        
        Raises
        ------
        TypeError
            Raises a type error if interpolator is not a subclass of Interpolator
        """
        try:
            valid = issubclass(interpolator, Interpolator)
            if not valid:
                raise TypeError()
        except TypeError:
            raise TypeError('{} is not a valid interpolator type. '.format(interpolator) +
                            'Interpolator must be of type {}'.format(Interpolator))

    def _set_interpolation_method(self, udims, definition):
        """Set the list of interpolation definitions to the input dimension
        
        Parameters
        ----------
        udims : tuple
            tuple of dimensiosn to assign definition to
        definition : dict
            dict definition returned from _parse_interpolation_method
        """

        method = deepcopy(definition['method'])
        interpolators = deepcopy(definition['interpolators'])
        params = deepcopy(definition['params'])

        # instantiate interpolators
        for (idx, interpolator) in enumerate(interpolators):
            interpolators[idx] = interpolator(method=method, **params)

        definition['interpolators'] = interpolators

        # set to interpolation configuration for dims
        self.config[udims] = definition

    def _select_interpolator_queue(self, source_coordinates, eval_coordinates, select_method, strict=False):
        """Create interpolator queue based on interpolation configuration and requested/native source_coordinates
        
        Parameters
        ----------
        source_coordinates : :class:`podpac.Coordinates`
            Description
        eval_coordinates : :class:`podpac.Coordinates`
            Description
        select_method : function
            method used to determine if interpolator can handle dimensions
        strict : bool, optional
            Raise an error if all dimensions can't be handled
        
        Returns
        -------
        OrderedDict
            Dict of (udims: Interpolator) to run in order
        
        Raises
        ------
        InterpolationException
            If `strict` is True, InterpolationException is raised when all dimensions cannot be handled
        """
        source_dims = set(source_coordinates.udims)
        handled_dims = set()

        interpolator_queue = OrderedDict()

        # go through all dims in config
        for key in iter(self.config):

            # if the key is set to (default,), it represents all the remaining dimensions that have not been handled
            # __init__ makes sure that (default,) will always be the last key in on
            if key == ('default',):
                udims = tuple(source_dims - handled_dims)
            else:
                udims = key

            # get configured list of interpolators for dim definition
            interpolators = self.config[key]['interpolators']

            # iterate through interpolators recording which dims they support
            for interpolator in interpolators:
                # if all dims have been handled already, skip the rest
                if not udims:
                    break

                # see which dims the interpolator can handle
                can_handle = getattr(interpolator, select_method)(udims, source_coordinates, eval_coordinates)

                # if interpolator can handle all udims
                if not set(udims) - set(can_handle):

                    # union of dims that can be handled by this interpolator and already supported dims
                    handled_dims = handled_dims | set(can_handle)

                    # set interpolator to work on that dimension in the interpolator_queue if dim has no interpolator
                    if udims not in interpolator_queue:
                        interpolator_queue[udims] = interpolator

        # throw error if the source_dims don't encompass all the supported dims
        # this should happen rarely because of default
        if len(source_dims) > len(handled_dims) and strict:
            missing_dims = list(source_dims - handled_dims)
            raise InterpolationException('Dimensions {} '.format(missing_dims) +
                                         'can\'t be handled by interpolation definition:\n {}'.format(self))

        # TODO: adjust by interpolation cost
        return interpolator_queue

    def select_coordinates(self, source_coordinates, source_coordinates_index, eval_coordinates):
        """
        Select a subset or coordinates if interpolator can downselect.
        
        At this point in the execution process, podpac has selected a subset of source_coordinates that intersects
        with the requested coordinates, dropped extra dimensions from requested coordinates, and confirmed
        source coordinates are not missing any dimensions.
        
        Parameters
        ----------
        source_coordinates : :class:`podpac.Coordinates`
            Intersected source coordinates
        source_coordinates_index : list
            Index of intersected source coordinates. See :ref:podpac.core.data.datasource.DataSource for
            more information about valid values for the source_coordinates_index
        eval_coordinates : :class:`podpac.Coordinates`
            Requested coordinates to evaluate
        
        Returns
        -------
        (:class:`podpac.Coordinates`, list)
            Returns tuple with the first element subset of selected coordinates and the second element the indicies
            of the selected coordinates
        """

        # TODO: short circuit if source_coordinates contains eval_coordinates
        # short circuit if source and eval coordinates are the same
        if source_coordinates == eval_coordinates:
            return source_coordinates, source_coordinates_index

        interpolator_queue = \
            self._select_interpolator_queue(source_coordinates, eval_coordinates, 'can_select')

        self._last_select_queue = interpolator_queue

        selected_coords = deepcopy(source_coordinates)
        selected_coords_idx = deepcopy(source_coordinates_index)

        for udims in interpolator_queue:
            interpolator = interpolator_queue[udims]

            # run interpolation. mutates selected coordinates and selected coordinates index
            selected_coords, selected_coords_idx = interpolator.select_coordinates(udims,
                                                                                   selected_coords,
                                                                                   selected_coords_idx,
                                                                                   eval_coordinates)

        return selected_coords, selected_coords_idx

    def interpolate(self, source_coordinates, source_data, eval_coordinates, output_data):
        """Interpolate data from requested coordinates to source coordinates
        
        Parameters
        ----------
        source_coordinates : :class:`podpac.Coordinates`
            Description
        source_data : podpac.core.units.UnitsDataArray
            Description
        eval_coordinates : :class:`podpac.Coordinates`
            Description
        output_data : podpac.core.units.UnitsDataArray
            Description
        
        Returns
        -------
        podpac.core.units.UnitDataArray
            returns the new output UnitDataArray of interpolated data
        
        Raises
        ------
        InterpolationException
            Raises InterpolationException when interpolator definition can't support all the dimensions
            of the requested coordinates
        """
        
        # short circuit if the source data and requested coordinates are of shape == 1
        if source_data.size == 1 and np.prod(eval_coordinates.shape) == 1:
            output_data[:] = source_data
            return output_data

        # TODO: short circuit if source_coordinates contains eval_coordinates
        # this has to be done better...
        # short circuit if source and eval coordinates are the same
        if not (set(source_coordinates.udims) - set(eval_coordinates.udims)):
            eq = True
            for udim in source_coordinates.udims:
                if not np.all(source_coordinates[udim].coordinates == eval_coordinates[udim].coordinates):
                    eq = False

            if eq:
                output_data.data = source_data.data
                return output_data

        interpolator_queue = \
            self._select_interpolator_queue(source_coordinates, eval_coordinates, 'can_interpolate', strict=True)

        # for debugging purposes, save the last defined interpolator queue
        self._last_interpolator_queue = interpolator_queue

        # iterate through each dim tuple in the queue
        for udims in interpolator_queue:
            interpolator = interpolator_queue[udims]

            # run interpolation
            output_data = interpolator.interpolate(udims,
                                                   source_coordinates,
                                                   source_data,
                                                   eval_coordinates,
                                                   output_data)

        return output_data
