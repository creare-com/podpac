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
    

class Interpolator(tl.HasTraits):
    """Interpolation Method
    
    Attributes
    ----------
    method : str
        current interpolation method to use in Interpolator (i.e. 'nearest')
    
    """

    method = tl.Unicode(allow_none=False)

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

    def validate(self, requested_coordinates, source_coordinates):
        """Validate that this interpolation method can handle the requested coordinates and source_coordinates
        
        Parameters
        ----------
        requested_coordinates : TYPE
            Description
        source_coordinates : TYPE
            Description
        
        Returns
        -------
        Bool
            True if the current interpolator can handle the requested coordinates and source coordinates for the
            currently defined method

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


class NearestNeighbor(Interpolator):
    tolerance = tl.Int()

    def validate(self, requested_coordinates, source_coordinates):
        return None

    def select_coordinates(self, requested_coordinates, source_coordinates, source_coordinates_index):
        return None

    def interpolate(self, source_coordinates, source_data, requested_coordinates, output):
        return None

class NearestPreview(Interpolator):
    tolerance = tl.Int()

    def validate(self, requested_coordinates, source_coordinates):
        return None
    
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
        return None


class Rasterio(Interpolator):

    def validate(self, requested_coordinates, source_coordinates):
        return None

    def select_coordinates(self, requested_coordinates, source_coordinates, source_coordinates_index):
        return None

    def interpolate(self, source_coordinates, source_data, requested_coordinates, output):
        return None


class ScipyGrid(Interpolator):
    
    def validate(self, requested_coordinates, source_coordinates):
        return None

    def select_coordinates(self, requested_coordinates, source_coordinates, source_coordinates_index):
        return None

    def interpolate(self, source_coordinates, source_data, requested_coordinates, output):
        return None


class ScipyPoint(Interpolator):
    
    def validate(self, requested_coordinates, source_coordinates):
        return None

    def select_coordinates(self, requested_coordinates, source_coordinates, source_coordinates_index):
        return None

    def interpolate(self, source_coordinates, source_data, requested_coordinates, output):
        return None


class Radial(Interpolator):
    
    def validate(self, requested_coordinates, source_coordinates):
        return None

    def select_coordinates(self, requested_coordinates, source_coordinates, source_coordinates_index):
        return None

    def interpolate(self, source_coordinates, source_data, requested_coordinates, output):
        return None


class OptimalInterpolation(Interpolator):
    """ I.E. Kriging """
    
    def validate(self, requested_coordinates, source_coordinates):
        return None

    def select_coordinates(self, requested_coordinates, source_coordinates, source_coordinates_index):
        return None

    def interpolate(self, source_coordinates, source_data, requested_coordinates, output):
        return None

# List of available interpolators
INTERPOLATION_METHODS = {
    'optimal': [OptimalInterpolation],
    'nearest': [NearestNeighbor, NearestPreview],
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

# default interoplation
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
    coordinates : podpac.core.coordinates.Coordinates
        source coordinates to be interpolated
    **kwargs :
        Keyword arguments passed on to each :ref:podpac.core.data.interpolate.Interpolator
    
    Raises
    ------
    InterpolationException
    TypeError
    
    """
 
    _definition = {}            # container for interpolation methods for each dimension

    def __init__(self, definition, coordinates, default=INTERPOLATION_DEFAULT, **kwargs):

        self._definition = {}

        # set each dim to interpolator definition
        if isinstance(definition, dict):

            defined_udims = []
            for udims in definition.keys():

                # get interpolation method
                method = self._parse_interpolation_method(definition[udims])

                # if udims is not a tuple, convert it to one
                if not isinstance(udims, tuple):
                    udims = (udims,)


                # add all udims to definition
                self._set_interpolation_method(udims, method, **kwargs)

                # create list of all udimss defined
                defined_udims += list(udims)

            # determine dims missing from definitions
            missing_udims = tuple(set(coordinates.udims) - set(defined_udims))

            # set default method to missing dims
            if missing_udims:
                default_method = self._parse_interpolation_method(default)
                self._set_interpolation_method(missing_udims, default_method, **kwargs)
            

        elif isinstance(definition, (str, tuple)):
            method = self._parse_interpolation_method(definition)
            self._set_interpolation_method(coordinates.udims, method, **kwargs)

        else:
            raise TypeError('{} is not a valid interpolation definition type. '.format(type(definition)) +
                            'Interpolation definiton must be a string, dict, or tuple')


    def __repr__(self):
        rep = str(self.__class__.__name__)
        for udims in self._definition.keys():
            rep += '\n\t%s: (%s, %s)' % (udims, self._definition[udims][0],
                                        [i.__class__.__name__ for i in self._definition[udims][1]])
        return rep

    def _parse_interpolation_method(self, definition):
        """parse interpolation definitions into a tuple of (method, Interpolator)
        
        Parameters
        ----------
        definition : str,
                     tuple (str, list of podpac.core.data.interpolate.Interpolator),
            interpolation definition for a single dimension.
            See :ref:podpac.core.data.datasource.DataSource.interpolation for more details.
        
        Returns
        -------
        tuple (str, list of podpac.core.data.interpolate.Interpolator)
            tuple with the first element the string method and second element a
            list of :ref:podpac.core.data.interpolate.Interpolator classes
        
        Raises
        ------
        InterpolationException
        TypeError
        """
        if isinstance(definition, str):
            if definition not in INTERPOLATION_SHORTCUTS:
                raise InterpolationException('"{}" is not a valid interpolation shortcut. '.format(definition) +
                                             'Valid interpolation shortcuts: {}'.format(INTERPOLATION_SHORTCUTS))
            return (definition, INTERPOLATION_METHODS[definition])

        elif isinstance(definition, tuple):
            method_string = definition[0]
            interpolators = definition[1]

            # confirm types
            if not isinstance(method_string, str):
                raise TypeError('{} is not a valid interpolation method. '.format(method_string) +
                                'Interpolation method must be a string')

            if not isinstance(interpolators, list):
                raise TypeError('{} is not a valid interpolator definition. '.format(interpolators) +
                                'Interpolator definition must be of type list containing Interpolator')

            for interpolator in interpolators:
                self._validate_interpolator(interpolator)

            # if all checks pass, return the definition
            return definition

        else:
            raise TypeError('{} is not a valid interpolation definition. '.format(definition) +
                            'Interpolation definiton must be a string or Interpolator.')

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

    def _set_interpolation_method(self, udims, method, **kwargs):
        """Set the list of interpolation methods to the input dimension
        
        Parameters
        ----------
        udims : tuple
            tuple of dimensiosn to assign method to
        method : tuple (str, list of podpac.core.data.interpolate.Interpolator)
            method and list of interpolators to assign to dimension
        **kwargs :
            keyword arguments passed into Interpolation class
        """

        method_string = method[0]
        interpolators = deepcopy(method[1])

        # instantiate interpolators
        for (idx, interpolator) in enumerate(interpolators):
            kwargs['method'] = method_string
            interpolators[idx] = interpolator(**kwargs)

        # set to interpolation dictionary
        self._definition[udims] = (method_string, interpolators)



    def select_coordinates(self, requested_coordinates, source_coordinates, source_coordinates_index):
        """
        Decide if we can interpolate coordinates.
        At this point, we have selected a subset of source_coordinates that intersects with the requested coordinates.
        We have dropped any extra dimensions from requested coordinates and we have confirmed that source coordinates
        are not missing any dimensions.
        
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
        
        for udims in self._definition.keys():
            method = self._definition[udims]

            method_string = method[0]
            interpolators = method[1]

            for interpolator in interpolators:
                cost = interpolator.evaluate_select(udims, 
                                                    requested_coordinates, 
                                                    source_coordinates, 
                                                    source_coordinates_index)
                if can_select_coordinates:
                    pass
        # for dim in requested_coordinates.udims:

        # for methods in each dimension, run validate. if true, then run select_coordinates in that interpolation method

    def _iterate_dims(self):
        pass

    def interpolate(self, source_coordinates, source_data, requested_coordinates, output):
        pass

    def to_pipeline(self):
        pass

    def from_pipeline(self):
        pass
