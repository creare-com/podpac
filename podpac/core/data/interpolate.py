from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
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
    
import podpac

class InterpolationException(Exception):
    pass
    
class Interpolator(tl.HasTraits):
    eval_coords = tl.Instance(podpac.Coordinates)
    source_coords = tl.Instance(podpac.Coordinates)
    pad = tl.Int(1)
    interpolation = tl.Unicode()
    valid_interpolations = tl.Enum([])
    tolerance = tl.CFloat(np.inf)  # if any, or needed
    supported_dims = tl.List([])  # if empty, supports all of them
    extrapolation = tl.Bool(False)
    
    # Next are used for optimizing the interpolation pipeline
    # If -1, it's cost is assume the same as a competing interpolator in the 
    # stack, and the determination is made based on the number of DOF before
    # and after each interpolation step.
    cost_func = tl.CFloat(-1)  # The rough cost FLOPS/DOF to do interpolation
    cost_setup = tl.CFloat(-1)  # The rough cost FLOPS/DOF to set up the interpolator
    
    def validate(self):
        """ Should return two lists:
            valid_dims, (I can interpolated these) 
            invalid_dims (I cannot interpolate these)
        """
        raise NotImplementedError()
    
    def source_coords_subset(self, pad=None):
        """Returns the subset of coordinates needed from the source data
        to interpolate onto the destination data. 
        
        This implements the basic functionality. Specialized interpolators 
        likely want to optimize this, and should overwrite this function. 
        """
        if pad is None:
            pad = self.pad
        return [self.source_coords.intersect(self.eval_coords, pad=pad),
                self.source.intersect_ind_slice(self.eval_coords, pad=pad)]
    
    def __call__(self, source_data):
        """ should return evaluated data 
        """
        raise NotImplementedError()
        
    def __add__(self, other):
        if not isinstance(other, (Interpolator, InterpolationPipeline)):
            raise InterpolationException("Cannot add %s and %s" % (
                    self.__class__, other.__class__))
        if isinstance(other, InterpolationPipeline):
            other = other.interpolators
        else:
            other = [other]
        return InterpolationPipeline(interpolators=[self] + other)

class NearestNeighbor(Interpolator):
    pass

class Rasterio(Interpolator):
    pass

class ScipyGrid(Interpolator):
    pass

class ScipyPoint(Interpolator):
    pass

class Radial(Interpolator):
    pass

class OptimalInterpolation(Interpolator):
    """ I.E. Kriging """
    pass

AVAILABLE_INTERPOLATORS = [
        NearestNeighbor, 
        ScipyGrid, 
        ScipyPoint,
        Radial,
        OptimalInterpolation
]

class InterpolationPipeline(tl.HasTraits):
    """
    This class is supposed to do the interpolation in the most efficient
    manner.
    """
    interpolators = tl.List([])
    # in case the order of interpolation needs to be controlled 
    # (probably for accuracy of validation) instead of
    # optimized for performance
    fixed_order = tl.Bool(False)  
    # Because the cost calculations can be dubious, prefer the higher-ordered
    # interpolants over others when the costs are within this tolerance
    cost_tol = tl.CFloat()  
    
    def __add__(self, other):
        if not isinstance(other, (Interpolator, InterpolationPipeline)):
            raise InterpolationException("Cannot add %s and %s" % (
                    self.__class__, other.__class__))
        if isinstance(other, InterpolationPipeline):
            other = other.interpolators
        else:
            other = [other]
        return InterpolationPipeline(interpolators=self.interpolators + other,
                                     **self.kwargs)
        
    @property
    def kwargs(self):
        keep = ['fixed_order']
        return {k: getattr(self, k) for k in keep}
    
    def __call__(self, source_data):
        """ should return evaluated data 
        """
        out = source_data
        if not self.fixed_order:
            self.order_interpolators()
            
        for i in self.interpolators:
            out = i(out)
        return out
        
