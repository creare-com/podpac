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

# Internal imports
from podpac.core.coordinate import Coordinate
from podpac.core.node import Node, UnitsDataArray

class DataSource(Node):
    source = tl.Any(allow_none=False, help="Path to the raw data source")
    interpolation = tl.Enum(['nearest', 'bilinear', 'cubic', 'cubic_spline',
                            'lanczos', 'average', 'mode', 'gauss', 'max', 'min',
                            'med', 'q1', 'q3'],
                            default_value='nearest')
    
    def execute(self, coordinates, params=None, output=None):
        coords, params, out = \
            self._execute_common(coordinates, params, output)
        
        data_subset, coords_subset = self.get_data_subset(coords)

        if output is None:
            res = self.interpolate_data(data_subset, coords_subset, coords)
            self.output = res  
        else:
            out[:] = self.interpolate_data(data_subset,
                                                   coords_subset, coords)
            self.output = out
            
        self.evaluted = True        
        return self.output
        
    def get_data_subset(self, coordinates):
        """
        This should return an UnitsDataArray, and A Coordinate object
        """
        coords_subset = self.native_coordinates.intersect(coordinates)
        data = self.get_data(coords_subset)
        
        return data, coords_subset
        
    def get_data(self, coordinates):
        """
        This should return an UnitsDataArray
        """
        raise NotImplementedError
    
    def interpolate_data(self, data_subset, coords_subset, coords):
        # TODO: implement for all of the designed cases (points, etc)
        
        # This a big switch, funneling data to various interpolation routines
        
        # Raster to Raster interpolation from regular grids
        rasterio_interps = ['nearest', 'bilinear', 'cubic', 'cubic_spline',
                            'lanczos', 'average', 'mode', 'gauss', 'max', 'min',
                            'med', 'q1', 'q3']         
        rasterio_regularity = ['single', 'regular', 'regular-rotated']
        if rasterio is not None \
                and self.interpolation in rasterio_interps \
                and ('lat' in coords_subset.coords 
                     and 'lon' in coords_subset.coords) \
                and ('lat' in coords.coords and 'lon' in coords.coords)\
                and coords_subset['lat'].regularity in rasterio_regularity \
                and coords_subset['lon'].regularity in rasterio_regularity \
                and coords['lat'].regularity in rasterio_regularity \
                and coords['lon'].regularity in rasterio_regularity:
            return self.rasterio_interpolation(data_subset, coords_subset,
                                               self.output, coords)

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
        
    
    def rasterio_interpolation(self, data_subset, coords_subset, out, coords):
        if len(data_subset.dims) > 2:
            return self._loop_helper(self.rasterio_interpolation, ['lat', 'lon'], 
                                     data_subset, coords_subset, out, coords)
        elif 'lat' not in data_subset.dims or 'lon' not in data_subset.dims:
            raise ValueError
        
        def get_rasterio_transform(c):
            west, east = c['lon'].area_bounds
            south, north = c['lat'].area_bounds
            cols, rows = (c['lon'].size, c['lat'].size)
            #print (east, west, south, north)
            return transform.from_bounds(west, south, east, north, cols, rows)
        
        with rasterio.Env():
            src_transform = get_rasterio_transform(coords_subset)
            src_crs = {'init': coords_subset.gdal_crs}
            # Need to make sure array is c-contiguous
            if coords_subset['lat'].is_max_to_min:
                source = np.ascontiguousarray(data_subset.data)
            else:
                source = np.ascontiguousarray(data_subset.data[::-1, :])
        
            dst_transform = get_rasterio_transform(coords)
            dst_crs = {'init': coords.gdal_crs}
            # Need to make sure array is c-contiguous
            if not out.data.flags['C_CONTIGUOUS']:
                destination = np.ascontiguousarray(out.data) 
            else:
                destination = out.data
        
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
            if coords['lat'].is_max_to_min:
                out.data[:] = destination
            else:
                out.data[:] = destination[::-1, :]
        return out
            
    def resample_latlon_to_gc(latlon, data_src, gc_dst, order=0):
        if order < 2:
            f = RegularGridInterpolator((latlon[0][::-1].data, latlon[1].data),
                                        data_src[::-1, :].data,
                                        method=['nearest', 'linear'][order], 
                                        bounds_error=False, fill_value=np.nan)
            x, y = np.meshgrid(gc_dst.x_axis, gc_dst.y_axis)
            data_dst = f((y.ravel(), x.ravel())).reshape(gc_dst.y_axis.size, gc_dst.x_axis.size)
        else:
            f = RectBivariateSpline(latlon[0][::-1].data, latlon[1].data,
                                    data_src[::-1, :].data, 
                                    kx=max(1, order), 
                                    ky=max(1, order))
            data_dst = f(gc_dst.y_axis[::-1], gc_dst.x_axis, grid=True)[::-1, :]
        return data_dst
    
