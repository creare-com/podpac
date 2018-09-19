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


RASTERIO_INTERPS = ['nearest', 'bilinear', 'cubic', 'cubic_spline', 'lanczos', 'average', 'mode', 'gauss',
                    'max', 'min', 'med', 'q1', 'q3']


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
 
    _source_coordinates = None  # container for the source coordinates attached to the interpolator
                                # TODO: this could just be dims instead of the full coordinates
    _definition = {}            # container for interpolation methods for each dimension

    def __init__(self, definition, source_coordinates, default_method='nearest'):
        """Create an interpolator class to handle one interpolation method per unstacked dimension. 
        Used to interpolate data within a datasource. 
        
        Parameters
        ----------
        definition : str, podpac.core.data.interpolate.InterpolationMethod, dict
            Interpolator definition used to define interpolation methods for each definiton.
            See :ref:podpac.core.data.datasource.DataSource.interpolation for more details.
        source_coordinates : podpac.core.coordinates.Coordinates
            Coordinates where DataSource has existing values. Requested coordinates will be interpolated based on
                the source coordinates
        default_method : str, optional
            Interpolation method used for any dimensions not explicity defined in the interpolation definition.
            Only applies when a dictionary is used to define the interpolator.
        
        Raises
        ------
        InterpolationException
        ValueError

        """

        # check default method is in INTERPOLATION_METHODS
        if isinstance(default_method, str):
            if default_method not in INTERPOLATION_SHORTCUTS:
                raise InterpolationException('Default interpolation method "{}" is not a '.format(default_method) +
                                             'valid interpolation method shortcut. ' +
                                             'Valid interpolation shortcuts: {}'.format(INTERPOLATION_SHORTCUTS))
        elif isinstance(default_method, InterpolationMethod):
            pass


        # set each dim to interpolator
        if isinstance(definition, (str, InterpolationMethod)):
            method = self._parse_interpolation_method(definition)

            for dim in source_coordinates.udims:
                self._set_interpolation_method(dim, method)

        elif isinstance(definition, dict):
            for dim in source_coordinates.udims:

                # if coordinate dim is not included in definition, use default method
                if dim not in definition.keys():
                    self._set_interpolation_method(dim, INTERPOLATION_METHODS[default_method])

                # otherwise use the interpolation method specified in the definition
                else:
                    method = self._parse_interpolation_method(definition[dim])
                    self._set_interpolation_method(dim, method)
        else:
            raise ValueError('{} is not a valid interpolation definition. '.format(type(definition)) +
                             'Interpolation definiton must be a string, InterpolationMethod, or dict')

    def _parse_interpolation_method(self, definition):
        """Summary
        
        Parameters
        ----------
        definition : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        
        Raises
        ------
        InterpolationException
            Description
        ValueError
            Description
        """
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

    def _set_interpolation_method(self, dim, method):

        # store the method for the given dimension in the interpolator definition dictionary
        self._definition[dim] = method


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
