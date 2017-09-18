from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
import numpy as np
import traitlets as tl
import matplotlib.colors, matplotlib.cm
import matplotlib.pyplot as plt
from pint import UnitRegistry
ureg = UnitRegistry()

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
except:
    scipy = None
    
    

# Internal imports
from podpac.core.coordinate import Coordinate
from podpac.core.node import Node, UnitsDataArray

class DataSource(Node):
    source = tl.Any(allow_none=False, help="Path to the raw data source")
    interpolation = tl.Enum(['nearest', 'nearest_preview', 'bilinear', 'cubic',
                             'cubic_spline', 'lanczos', 'average', 'mode',
                             'gauss', 'max', 'min', 'med', 'q1', 'q3'],
                            default_value='nearest')
    no_data_vals = tl.List(allow_none=True)
    
    def execute(self, coordinates, params=None, output=None):
        coords, params, out = \
            self._execute_common(coordinates, params, output)
        
        res = self.get_data_subset(coords)
        if len(res) == 1:
            self.output = res[0]
            return self.output
        
        data_subset, coords_subset = res
        if self.no_data_vals:
            for ndv in self.no_data_vals:
                data_subset.data[data_subset.data == ndv] = np.nan
        if output is None:
            res = self.interpolate_data(data_subset, coords_subset, coords)
            self.output = res  
        else:
            out[:] = self.interpolate_data(data_subset, coords_subset, coords)
            self.output = out
            
        self.evaluted = True        
        return self.output
        
    def get_data_subset(self, coordinates):
        """
        This should return an UnitsDataArray, and A Coordinate object, unless
        there is no intersection
        """
        pad = self.interpolation != 'nearest'
        coords_subset = self.native_coordinates.intersect(coordinates, pad=pad)
        coords_subset_slc = self.native_coordinates.intersect_ind_slice(coordinates, pad=pad)
        
        # If they do not intersect, we have a shortcut
        if np.prod(coords_subset.shape) == 0:
            return [self.initialize_coord_array(coordinates, init_type='nan')]

        if self.interpolation == 'nearest_preview':
            # We can optimize a little
            new_coords = OrderedDict()
            new_coords_slc = []
            for i, d in enumerate(coords_subset.dims):
                if coords_subset[d].regularity == 'regular':
                    if d in coordinates.dims:
                        ndelta = np.round(coordinates[d].delta /
                                          coords_subset[d].delta)
                        if ndelta <= 1:
                            ndelta = coords_subset[d].delta
                        coords = tuple(coords_subset[d].coords[:2]) \
                            + (ndelta * coords_subset[d].delta,)
                        new_coords[d] = coords
                        new_coords_slc.append(
                            slice(coords_subset_slc[i].start, 
                                  coords_subset_slc[i].stop,
                                  int(ndelta))
                            )
                    else:
                        new_coords[d] = coords_subset[d]
                        new_coords_slc.append(coords_subset_slc[i])
                else:
                    new_coords[d] = coords_subset[d]
                    new_coords_slc.append(coords_subset_slc[i])                    
            coords_subset = Coordinate(new_coords)
            coords_subset_slc = new_coords_slc
        
        data = self.get_data(coords_subset, coords_subset_slc)
        
        return data, coords_subset
        
    def get_data(self, coordinates, coodinates_slice):
        """
        This should return an UnitsDataArray
        coordinates and coordinates slice may be strided or subsets of the 
        source data, but all coordinates will match 1:1 with the subset data
        """
        raise NotImplementedError
    
    def interpolate_data(self, data_src, coords_src, coords_dst):
        # TODO: implement for all of the designed cases (points, etc)
        
        data_dst = self.output
        
        # This a big switch, funneling data to various interpolation routines
        if data_src.size == 1 and np.prod(coords_dst.shape) == 1:
            data_dst[:] = data_src
            return data_dst
        
        # Nearest preview of rasters
        if self.interpolation == 'nearest_preview':
            crd = OrderedDict()
            for c in data_src.coords.keys():
                crd[c] = data_dst.coords[c].sel(method=str('nearest'),
                                                **{c: data_src.coords[c]}
                                                )
            data_dst.loc[crd] = data_src.data[:]
            return data_dst
        
        # Raster to Raster interpolation from regular grids to regular grids
        rasterio_interps = ['nearest', 'bilinear', 'cubic', 'cubic_spline',
                            'lanczos', 'average', 'mode', 'gauss', 'max', 'min',
                            'med', 'q1', 'q3']         
        rasterio_regularity = ['single', 'regular', 'regular-rotated']
        if rasterio is not None \
                and self.interpolation in rasterio_interps \
                and ('lat' in coords_src.dims 
                     and 'lon' in coords_src.dims) \
                and ('lat' in coords_dst.dims and 'lon' in coords_dst.dims)\
                and coords_src['lat'].regularity in rasterio_regularity \
                and coords_src['lon'].regularity in rasterio_regularity \
                and coords_dst['lat'].regularity in rasterio_regularity \
                and coords_dst['lon'].regularity in rasterio_regularity:
            return self.rasterio_interpolation(data_src, coords_src,
                                               data_dst, coords_dst)
        # Raster to Raster interpolation from irregular grids to arbitrary grids
        elif ('lat' in coords_src.dims 
                and 'lon' in coords_src.dims) \
                and ('lat' in coords_dst.dims and 'lon' in coords_dst.dims)\
                and coords_src['lat'].regularity in ['irregular', 'regular'] \
                and coords_src['lon'].regularity in ['irregular', 'regular']:
            return self.interpolate_irregular_grid(data_src, coords_src,
                                                   data_dst, coords_dst,
                                                   grid=True)
        # Raster to lat_lon point interpolation
        elif ('lat' in coords_src.dims 
                and 'lon' in coords_src.dims) \
                and coords_src['lat'].regularity in ['irregular', 'regular'] \
                and coords_src['lon'].regularity in ['irregular', 'regular'] \
                and (np.any(['lat_lon' in d for d in coords_dst.dims]) or
                     np.any(['lon_lat' in d for d in coords_dst.dims])):
            coords_dst_us = coords_dst.unstack()
            return self.interpolate_irregular_grid(data_src, coords_src,
                                                   data_dst, coords_dst_us,
                                                   grid=False)            
            
    def _loop_helper(self, func, keep_dims, data_src, coords_src,
                     data_dst, coords_dst,
                     **kwargs):
        
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
        if len(data_src.dims) > 2:
            return self._loop_helper(self.rasterio_interpolation, ['lat', 'lon'], 
                                     data_src, coords_src, data_dst, coords_dst)
        elif 'lat' not in data_src.dims or 'lon' not in data_src.dims:
            raise ValueError
        
        def get_rasterio_transform(c):
            west, east = c['lon'].area_bounds
            south, north = c['lat'].area_bounds
            cols, rows = (c['lon'].size, c['lat'].size)
            #print (east, west, south, north)
            return transform.from_bounds(west, south, east, north, cols, rows)
        
        with rasterio.Env():
            src_transform = get_rasterio_transform(coords_src)
            src_crs = {'init': coords_src.gdal_crs}
            # Need to make sure array is c-contiguous
            if coords_src['lat'].is_max_to_min:
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
                destination,
                src_transform=src_transform,
                src_crs=src_crs,
                src_nodata=np.nan,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=getattr(Resampling, self.interpolation)
            )
            if coords_dst['lat'].is_max_to_min:
                data_dst.data[:] = destination
            else:
                data_dst.data[:] = destination[::-1, :]
        return data_dst
            
    def interpolate_irregular_grid(self, data_src, coords_src,
                                   data_dst, coords_dst, grid=True):
        if len(data_src.dims) > 2:
            keep_dims = ['lat', 'lon']
            return self._loop_helper(self.interpolate_irregular_grid, keep_dims, 
                                     data_src, coords_src, data_dst, coords_dst,
                                     grid=grid)
        elif 'lat' not in data_src.dims or 'lon' not in data_src.dims:
            raise ValueError
        
        interp = self.interpolation 
        s = []
        if coords_src['lat'].is_max_to_min:  
            lat = coords_src['lat'].coordinates[::-1]
            s.append(slice(None, None, -1))
        else:                  
            lat = coords_src['lat'].coordinates
            s.append(slice(None, None))
        if coords_src['lon'].is_max_to_min:  
            lon = coords_src['lon'].coordinates[::-1]
            s.append(slice(None, None, -1))
        else:                  
            lon = coords_src['lon'].coordinates
            s.append(slice(None, None))
            
        data = data_src.data[s]
        
        # remove nan's
        I, J = np.isfinite(lat), np.isfinite(lon)
        lat, lon = lat[I], lon[J]
        data = data[I, :][:, J]
        
        if interp in ['bilinear', 'nearest']:
            f = RegularGridInterpolator([lat, lon], data,
                                        method=interp.replace('bi', ''), 
                                        bounds_error=False, fill_value=np.nan)
            if grid:
                x, y = np.meshgrid(coords_dst['lon'].coordinates,
                                   coords_dst['lat'].coordinates)
            else:
                x = coords_dst['lon'].coordinates
                y = coords_dst['lat'].coordinates                
            data_dst.data[:] = f((y.ravel(), x.ravel())).reshape(data_dst.shape)
        elif 'spline' in interp:
            if interp == 'cubic_spline':
                order = 3
            else:
                order = int(interp.split('_')[-1])
            f = RectBivariateSpline(lat, lon,
                                    data, 
                                    kx=max(1, order), 
                                    ky=max(1, order))
            data_dst.data[:] = f(coords_dst.coords['lat'],
                                 coords_dst.coords['lon'],
                                 grid=grid).reshape(data_dst.shape)
        return data_dst
    
