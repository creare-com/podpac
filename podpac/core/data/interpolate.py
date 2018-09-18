"""
Interpolation handling

Attributes
----------
AVAILABLE_INTERPOLATORS : TYPE
Description
"""

from __future__ import division, unicode_literals, print_function, absolute_import
from copy import deepcopy

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
from podpac.core.coordinates import Coordinates


class InterpolationException(Exception):
    """
    Custom label for interpolator exceptions
    """
    pass
    

class InterpolationMethod(tl.HasTraits):
    """Summary
    
    Attributes
    ----------
    method : TYPE
        Description
    """
    
    method = tl.Unicode()
    tolerance = tl.CFloat(np.inf)

    # Next are used for optimizing the interpolation pipeline
    # If -1, it's cost is assume the same as a competing interpolator in the
    # stack, and the determination is made based on the number of DOF before
    # and after each interpolation step.
    cost_func = tl.CFloat(-1)  # The rough cost FLOPS/DOF to do interpolation
    cost_setup = tl.CFloat(-1)  # The rough cost FLOPS/DOF to set up the interpolator

    def interpolate_coordinates(self, requested_coordinates, source_coordinates, source_coordinates_index):
        """Summary
        """
        pass

    def interpolate(self, source_coordinates, source_data, requested_coordinates, output):
        """Summary
        """
        pass


class NearestNeighbor(InterpolationMethod):
    pass

class NearestPreview(InterpolationMethod):
    pass

class Rasterio(InterpolationMethod):
    pass

class ScipyGrid(InterpolationMethod):
    pass

class ScipyPoint(InterpolationMethod):
    pass

class Radial(InterpolationMethod):
    pass

class OptimalInterpolation(InterpolationMethod):
    """ I.E. Kriging """
    pass

# List of available interpolators
INTERPOLATION_METHODS = {
    'nearest': NearestNeighbor,
    'nearest_preview': NearestPreview,
    'rasterio': Rasterio,
    'scipygrid': ScipyGrid,
    'scipypoint': ScipyPoint,
    'radial': Radial,
    'optimal': OptimalInterpolation,
}


# TODO: how to handle these?
# = ['nearest', 'nearest_preview', 'bilinear', 'cubic',
#                       'cubic_spline', 'lanczos', 'average', 'mode',
#                       'gauss', 'max', 'min', 'med', 'q1', 'q3']   # TODO: gauss is not supported by rasterio
INTERPOLATION_SHORTCUTS = INTERPOLATION_METHODS.keys()

class Interpolator():
    """
    Meta class to handle interpolation across dimensions
    """
 
    _stacked = []     # container for stacked dims in interpolator
    _definition = {}  # container for interpolation methods for each dimension

    def __init__(self, definition, dims, default_method='nearest'):
        """Construct the Interpolator class to handle interpolations across all dimensions
        
        Parameters
        ----------
        definition : str, podpac.core.data.interpolate.InterpolationMethod, dict
            Definition object to construct the interpolator.
        dims : list<str>
            List of string dimension names
        
        Raises
        ------
        InterpolationException
            Description
        """

        # see if default method is in INTERPOLATION_METHODS
        if isinstance(default_method, str):
            if default_method not in INTERPOLATION_SHORTCUTS:
                raise InterpolationException('"{}" is not a valid interpolation method shortcut. '.format(default_method) +
                                             'Valid interpolation shortcuts: {}'.format(INTERPOLATION_SHORTCUTS))
        elif isinstance(default_method, InterpolationMethod):
            pass


        # set each dim to interpolator
        if isinstance(definition, (str, InterpolationMethod)):
            method = self._parse_interpolation_method(definition)

            for dim in dims:
                self._set_interpolation_method(dim, method)

        elif isinstance(definition, dict):
            for dim in dims:
                if dim not in definition.keys():
                    self._set_interpolation_method(dim, default_method)
                else:
                    method = self._parse_interpolation_method(definition[dim])
                    self._set_interpolation_method(dim, method)


    def _parse_interpolation_method(self, definition):
        
        if isinstance(definition, str):
            if definition not in INTERPOLATION_SHORTCUTS:
                raise InterpolationException('"{}" is not a valid interpolation shortcut. '.format(definition) +
                                             'Valid interpolation shortcuts: {}'.format(INTERPOLATION_SHORTCUTS))
            return INTERPOLATION_METHODS[definition]

        elif isinstance(definition, InterpolationMethod):
            return definition
        else:
            raise ValueError('{} is not a valid interpolation definition. '.format(type(definition)) +
                             'Interpolation definiton must be a string or InterpolationMethod.')

    def _set_interpolation_method(self, dim, Method):
        
        # keep track of stacked dims, but store the interpolator methods seperately by independent dimension
        if '_' in dim:
            stacked_dims = dim.split('_')
            for stacked_dim in stacked_dims:
                self._stacked += [stacked_dim]
                self._set_interpolation_method(stacked_dim, Method)

        # store the Method for the given dimension in the interpolator definition dictionary
        self._definition[dim] = Method


    def interpolate_coordinates(self, requested_coordinates, source_coordinates, source_coordinates_index):
        """
        Decide if we can interpolate coordinates
        
        Parameters
        ----------
        requested_coordinates : podpac.core.coordinates.Coordinates
            Requested coordinates to execute
        source_coordinates : podpac.core.coordinates.Coordinates
            Intersected source coordinates
        source_coordinates_index : List
            Index of intersected source coordinates. See :ref:podpac.core.data.datasource.DataSource for 
            more information about valid values for the source_coordinates_index
        """
        pass

    def interpolate(self, source_coordinates, source_data, requested_coordinates, output):
        pass

    def to_pipeline(self):
        pass

    def from_pipeline(self):
        pass
