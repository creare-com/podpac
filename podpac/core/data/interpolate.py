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
from podpac.core.coordinates import Coordinates, UniformCoordinates1d


class InterpolationException(Exception):
    """
    Custom label for interpolator exceptions
    """
    pass
    

class InterpolationMethod(tl.HasTraits):
    """Interpolation Method 
    
    """

    # Next are used for optimizing the interpolation pipeline
    # If -1, it's cost is assume the same as a competing interpolator in the
    # stack, and the determination is made based on the number of DOF before
    # and after each interpolation step.
    cost_func = tl.CFloat(-1)  # The rough cost FLOPS/DOF to do interpolation
    cost_setup = tl.CFloat(-1)  # The rough cost FLOPS/DOF to set up the interpolator

    def validate(self, requested_coordinates, source_coordinates):
        """Validate that this interpolation method can handle the requested coordinates and source_coordinates
        
        Parameters
        ----------
        requested_coordinates : TYPE
            Description
        source_coordinates : TYPE
            Description
        
        Raises
        ------
        NotImplementedError
            Description
        """
        raise NotImplementedError

    def select_coordinates(self, requested_coordinates, source_coordinates, source_coordinates_index):
        """use interpolation method to downselect coordinates
        
        Parameters
        ----------
        requested_coordinates : TYPE
            Description
        source_coordinates : TYPE
            Description
        source_coordinates_index : TYPE
            Description

        Returns
        -------
        podpac.core.coordinates.Coordinates, list
            returns the new down selected coordinates and the new associated index
        
        Raises
        ------
        NotImplementedError
            Description
        """
        raise NotImplementedError

    def interpolate(self, source_coordinates, source_data, requested_coordinates, output):
        """interpolate data from requested coordinates to source coordinates
        
        Parameters
        ----------
        source_coordinates : TYPE
            Description
        source_data : TYPE
            Description
        requested_coordinates : TYPE
            Description
        output : TYPE
            Description
        
        Returns
        -------
        podpac.core.units.UnitDataArray
            returns the new output UnitDataArray of interpolated data

        Raises
        ------
        NotImplementedError
            Description
        """
        raise NotImplementedError


class NearestNeighbor(InterpolationMethod):
    pass

class NearestPreview(InterpolationMethod):

    def validate(self, requested_coordinates, source_coordinates):
        pass
    
    def select_coordinates(self, requested_coordinates, source_coordinates, source_coordinates_index):
        # We can optimize a little
        new_coords = []
        new_coords_idx = []

        for dim, idx in zip(source_coordinates, source_coordinates_index):
            if dim in requested_coordinates.dims:
                src_coords = source_coordinates[dim]
                dst_coords = requested_coordinates[dim]

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
                
                c = UniformCoordinates1d(src_start, src_stop, ndelta*src_delta, **src_coords.properties)
                
                if isinstance(idx, slice):
                    idx = slice(idx.start, idx.stop, int(ndelta))
                else:
                    idx = slice(idx[0], idx[-1], int(ndelta))
            else:
                c = source_coordinates[dim]

            new_coords.append(c)
            new_coords_idx.append(idx)

        # updates requested source coordinates and index
        new_source_coordinates = Coordinates(new_coords)
        new_source_coordinates_index = new_coords_idx

        return new_source_coordinates, new_source_coordinates_index
            
    def interpolate(self, source_coordinates, source_data, requested_coordinates, output):
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
    'optimal': [OptimalInterpolation],
    'nearest': [NearestNeighbor, NearestPreview, Rasterio, ScipyGrid],
    'bilinear':[Rasterio, ScipyGrid],
    'cubic':[Rasterio, ScipyGrid],
    'cubic_spline':[Rasterio, ScipyGrid],
    'lanczos':[Rasterio, ScipyGrid],
    'average':[Rasterio, ScipyGrid],
    'mode':[Rasterio, ScipyGrid],
    'gauss':[Rasterio, ScipyGrid],
    'max':[Rasterio, ScipyGrid],
    'min':[Rasterio, ScipyGrid],
    'med':[Rasterio, ScipyGrid],
    'q1':[Rasterio, ScipyGrid],
    'q3': [Rasterio, ScipyGrid],
    'radial': [Radial]
}

# create shortcut list based on methods keys
INTERPOLATION_SHORTCUTS = INTERPOLATION_METHODS.keys()


# default tolerance for each dimension
# TODO: units?
TOLERANCE_DEFAULTS = {
    'lat': 1,
    'lon': 1,
    'alt': 1,
    'time': 1
}

# TODO: do we want some kind of string preset?
TOLERANCE_PRESETS = {
    'high': {
        'lat': 1,
        'lon': 1,
        'alt': 1,
        'time': 1
    },
    'low': {
        'lat': 1,
        'lon': 1,
        'alt': 1,
        'time': 1
    }
}




class Interpolator():
    """Create an interpolator class to handle one interpolation method per unstacked dimension.
    Used to interpolate data within a datasource.
    
    Parameters
    ----------
    definition : str,
                 podpac.core.data.interpolate.InterpolationMethod,
                 dict,
                 list of podpac.core.data.interpolate.InterpolationMethod
        Interpolator definition used to define interpolation methods for each definiton.
        See :ref:podpac.core.data.datasource.DataSource.interpolation for more details.
    coordinates : podpac.core.coordinates.Coordinates
        source coordinates to be interpolated
    default_method : str, optional
        Interpolation method used for any dimensions not explicity defined in the interpolation definition.
        Only applies when a dictionary is used to define the interpolator.
    tolerance : int, float, dict, optional
        If tolerance is in an int, it will act for all dimensions.
        If tolerance is a dict, the keys specify the coordinate dimension and the value specicifies the tolerance.
        When using a dict, unspecified dimensions will get default tolerances.
    
    Raises
    ------
    InterpolationException
    TypeError
    
    """
 
    _tolerance = {}             # container for interpolation tolerance for each dimension
    _definition = {}            # container for interpolation methods for each dimension

    def __init__(self, definition, coordinates, default_method='nearest', tolerance=None):

        # parse definition
        self._parse_interpolation_definition(definition, coordinates, default_method)

        # parse tolerance definition
        self._parse_tolerance(tolerance, coordinates)


    def _parse_interpolation_definition(self, definition, coordinates, default_method):
        """parse interpolation definition object"""

        # set each dim to interpolator definition
        if isinstance(definition, dict):
            for dim in coordinates.udims:

                # if coordinate dim is not included in definition, use default method
                if dim not in definition.keys():
                    methods = self._parse_interpolation_methods(default_method)
                    self._set_interpolation_methods(dim, methods)

                # otherwise use the interpolation method specified in the definition
                else:
                    method = self._parse_interpolation_methods(definition[dim])
                    self._set_interpolation_methods(dim, method)

        elif isinstance(definition, (str, list, object)):
            methods = self._parse_interpolation_methods(definition)

            for dim in coordinates.udims:
                self._set_interpolation_methods(dim, methods)

        else:
            raise TypeError('{} is not a valid definition type. '.format(type(definition)) +
                            'Interpolation definiton must be a string, InterpolationMethod, dict, or list')

    def _parse_tolerance(self, tolerance, coordinates):
        """parse input tolerance  """

        # if tolerance is not defined, use the defaults for the known dims,
        # if there is no default tolerance for dim in coordinates, set to None
        if tolerance is None:
            for dim in coordinates.udims:
                if dim in TOLERANCE_DEFAULTS.keys():
                    self._set_interpolation_tolerance(dim, TOLERANCE_DEFAULTS[dim])
                else:
                    self._set_interpolation_tolerance(dim, None)

        # if tolerance is a number, set it to all dims
        elif isinstance(tolerance, (int, float)):
            for dim in coordinates.udims:
                self._set_interpolation_tolerance(dim, tolerance)
        
        # if tolerance is a dict, set dict keys to values, otherwise use default
        # if there is no default tolerance for dim in coordinates, set to None
        elif isinstance(tolerance, dict):
            for dim in coordinates.udims:

                # if coordinate dim is not included in tolerance, use default tolerance if it exists
                if dim not in tolerance.keys():
                    if dim in TOLERANCE_DEFAULTS.keys():
                        self._set_interpolation_tolerance(dim, TOLERANCE_DEFAULTS[dim])
                    else:
                        self._set_interpolation_tolerance(dim, None)
                else:
                    self._set_interpolation_tolerance(dim, tolerance[dim])
        
        else:
            raise TypeError('{} is not a valid tolerance. '.format(tolerance) +
                            'Tolerance must be a int, float, or a dict')


    def _parse_interpolation_methods(self, definition):
        """parse string, list, and InterpolationMethod definitions into a list of InterpolationMethod
        
        Parameters
        ----------
        definition : str,
                     podpac.core.data.interpolate.InterpolationMethod,
                     list of podpac.core.data.interpolate.InterpolationMethod
            interpolation definition for a single dimension.
            See :ref:podpac.core.data.datasource.DataSource.interpolation for more details.
        
        Returns
        -------
        list of InterpolatioMethod
            list of InterpolationMethod classes to parsed from input definition
        
        Raises
        ------
        InterpolationException
        TypeError
        """
        if isinstance(definition, str):
            if definition not in INTERPOLATION_SHORTCUTS:
                raise InterpolationException('"{}" is not a valid interpolation shortcut. '.format(definition) +
                                             'Valid interpolation shortcuts: {}'.format(INTERPOLATION_SHORTCUTS))
            return INTERPOLATION_METHODS[definition]

        elif isinstance(definition, list):

            # confirm that all items are InterpolationMethods
            for method in definition:
                self._validate_interpolation_method(method)

            return definition

        elif isinstance(definition, object):
            self._validate_interpolation_method(definition)
            return [definition]
        else:
            raise TypeError('{} is not a valid interpolation definition. '.format(type(definition)) +
                            'Interpolation definiton must be a string or InterpolationMethod.')

    def _validate_interpolation_method(self, method):
        """Make sure method is a subclass of InterpolationMethod
        
        Parameters
        ----------
        method : any
            input definition to validate
        
        Raises
        ------
        TypeError
            Raises a type error if method is not a subclass of InterpolationMethod
        """
        try:
            valid = issubclass(method, InterpolationMethod)
            if not valid:
                raise TypeError()
        except TypeError:
            raise TypeError('"{}" is not a vlidate input  '.format(method) +
                            'Interpolation items must be of type: ' +
                            '{}'.format(InterpolationMethod))

    def _set_interpolation_methods(self, dim, methods):
        """Set the list of interpolation methods to the input dimension
        
        Parameters
        ----------
        dim : string
            dimension to assign
        methods : list of podpac.core.data.interpolate.InterpolationMethod
            methods to assign to dimension
        """

        self._definition[dim] = methods

    def _set_interpolation_tolerance(self, dim, tolerance):
        """Set the prescribed tolerance to the input dimension
        
        Parameters
        ----------
        dim : string
            dimension to assign
        tolerance : int, float
            tolerance to assign to dimension
        """

        self._tolerance[dim] = tolerance




    def select_coordinates(self, requested_coordinates, source_coordinates, source_coordinates_index):
        """
        Decide if we can interpolate coordinates
        
        Parameters
        ----------
        requested_coordinates : podpac.core.coordinates.Coordinates
            Requested coordinates to execute
        source_coordinates : podpac.core.coordinates.Coordinates
            Intersected source coordinates
        source_coordinates_index : list
            Index of intersected source coordinates. See :ref:podpac.core.data.datasource.DataSource for
            more information about valid values for the source_coordinates_index
        """
        pass

        # for methods in each dimension, run validate. if true, then run select_coordinates in that interpolation method


    def interpolate(self, source_coordinates, source_data, requested_coordinates, output):
        pass

    def to_pipeline(self):
        pass

    def from_pipeline(self):
        pass
