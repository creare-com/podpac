"""
Interpolation handling

Attributes
----------
AVAILABLE_INTERPOLATORS : TYPE
Description
"""

from __future__ import division, unicode_literals, print_function, absolute_import
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
from podpac.core.coordinates import Coordinates, UniformCoordinates1d, StackedCoordinates, ArrayCoordinates1d


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
    dims_supported : list
        list of supported dimensions by the interpolator. Used by default :ref:self.can_select 
        and self.can_interpolate methods if not overwritten by specific Interpolator
    
    """

    method = tl.Unicode(allow_none=False)
    dims_supported = tl.List(tl.Unicode(), allow_none=True)

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

    def _validate_udims(self, udims):
        
        # try each dim and return False if one of the dims is not supported
        for dim in udims:
            if dim not in self.dims_supported:
                return False
        
        # if all dims exist, return that the interpolator works
        return True

    def dim_exists(self, dim, source_coordinates, requested_coordinates):
        """Verify the dim exists on source and requested coordinates 
        
        Parameters
        ----------
        dim : str
            Dimension to verify
        source_coordinates : podpac.core.coordinates.Coordinates
            Description
        requested_coordinates : podpac.core.coordinates.Coordinates
            Description
        """

        return dim in source_coordinates.udims and dim in requested_coordinates.udims


    def can_select(self, udims, requested_coordinates, source_coordinates):
        """Evaluate if interpolator can downselect the source coordinates from the requested coordinates
        for the unstacked dims supplied
        
        Parameters
        ----------
        udims : tuple
            dimensions to select from
        requested_coordinates : podpac.core.coordinates.Coordinates
            Description
        source_coordinates : podpac.core.coordinates.Coordinates
            Description
        
        Raises
        ------
        NotImplementedError
        
        Returns
        -------
        Bool
            True if the current interpolator can handle the requested coordinates and source coordinates for the
            currently defined method
        """

        # if dims_supported is defined, then try each dim and return False if one of the dims is not allowed
        if self.dims_supported is not None:
            return self._validate_udims(udims)
        else:
            raise NotImplementedError

    def select_coordinates(self, udims, requested_coordinates, source_coordinates, source_coordinates_index):
        """use interpolation method to downselect coordinates
        
        Parameters
        ----------
        udims : tuple
            dimensions to select from
        requested_coordinates : podpac.core.coordinates.Coordinates
            Description
        source_coordinates : podpac.core.coordinates.Coordinates
            Description
        source_coordinates_index : list
            Description

        Returns
        -------
        podpac.core.coordinates.Coordinates, list
            returns the new down selected coordinates and the new associated index
        
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def can_interpolate(self, udims, requested_coordinates, source_coordinates):
        """Validate that this interpolation method can handle the requested coordinates and source_coordinates
        
        Parameters
        ----------
        udims : tuple
            dimensions to interpolate
        requested_coordinates : podpac.core.coordinates.Coordinates
            Description
        source_coordinates : podpac.core.coordinates.Coordinates
            Description
        
        Raises
        ------
        NotImplementedError
        
        Returns
        -------
        Bool
            True if the current interpolator can handle the requested coordinates and source coordinates for the
            currently defined method
        """

        # if dims_supported is defined, then try each dim and return False if one of the dims is not allowed
        if self.dims_supported is not None:
            return self._validate_udims(udims)
        else:
            raise NotImplementedError

    def interpolate(self, udims, source_coordinates, source_data, requested_coordinates, output_data):
        """Interpolate data from requested coordinates to source coordinates. 
        
        Parameters
        ----------
        udims : tuple
            dimensions to interpolate
        source_coordinates : podpac.core.coordinates.Coordinates
            Description
        source_data : podpac.core.units.UnitsDataArray
            Description
        requested_coordinates : podpac.core.coordinates.Coordinates
            Description
        output : podpac.core.units.UnitsDataArray
            Description
        
        Returns
        -------
        podpac.core.units.UnitDataArray
            returns the new output UnitDataArray of interpolated data

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError


class NearestNeighbor(Interpolator):

    dims_supported = ['lat', 'lon', 'alt', 'time']
    tolerance = tl.Int()

    # nearest neighbor can't select coordinates
    def can_select_coordinates(self, udims, requested_coordinates, source_coordinates):
        """NearestNeighbor can't select coordinates"""
        return False

    def interpolate(self, udims, source_coordinates, source_data, requested_coordinates, output_data):
        
        # first iterate through time and alt and interpolate those
        for dim in udims:

            if dim == 'time' and self.dim_exists(dim, source_coordinates, requested_coordinates):
                source_data = source_data.reindex(
                    time=requested_coordinates.coords['time'], method='nearest', tolerance=self.tolerance)
                
                source_coordinates['time'] = ArrayCoordinates1d.from_xarray(source_data['time'])


            if dim == 'alt' and self.dim_exists(dim, source_coordinates, requested_coordinates):
                source_data = source_data.reindex(alt=requested_coordinates.coords['alt'], method='nearest')
                source_coordinates['alt'] = ArrayCoordinates1d.from_xarray(source_data['alt'])

        # TODO: do other dimensions manually
        # for dim in udims:



class NearestPreview(Interpolator):
    
    dims_supported = ['lat', 'lon', 'alt', 'time']
    tolerance = tl.Int()

    def select_coordinates(self, udims, requested_coordinates, source_coordinates, source_coordinates_index):

        # TODO: currently ignoring udims - this will not work as expected with udims specified
        # this is the old implementation
        
        new_coords = []
        new_coords_idx = []

        # iterate over the source coordinate dims in case they are stacked
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


    def interpolate(self, udims, source_coordinates, source_data, requested_coordinates, output_data):
        
        crds = OrderedDict()
        tol = np.inf  # TODO: make property

        for c in udims:
            crds[c] = output_data.coords[c].data.copy()
            if c != 'time':
                # TODO: do we still use the `delta` attribute?
                tol = min(tol, np.abs(getattr(requested_coordinates[c], 'delta', tol)))
        
        if 'time' in crds:
            source_data = source_data.reindex(time=crds['time'], method=str('nearest'))
            del crds['time']

        crds_keys = list(crds.keys())
        output_data.data = source_data.reindex(method=str('nearest'), tolerance=tol, **crds).transpose(*crds_keys)

        return output_data


class Rasterio(Interpolator):

    dims_supported = ['lat', 'lon']
    rasterio_interpolators = ['nearest', 'bilinear', 'cubic', 'cubic_spline', 'lanczos', 'average', 'mode', 'gauss',
                              'max', 'min', 'med', 'q1', 'q3']

    def can_select_coordinates(self, udims, requested_coordinates, source_coordinates):
        """Rasterio can't select coordinates"""
        return False

    def can_interpolate(self, udims, requested_coordinates, source_coordinates):
        pass
        # return self.method in self.rasterio_interpolators 
        #        and self.dim_in('lat' in source_coordinates.dims and 'lon' in source_coordinates.dims
        #        and 'lat' in requested_coordinates.dims and 'lon' in requested_coordinates.dims
        #        and source_coordinates['lat'].is_uniform and source_coordinates['lon'].is_uniform
        #        and requested_coordinates['lat'].is_uniform and requested_coordinates['lon'].is_uniform

    def interpolate(self, udims, source_coordinates, source_data, requested_coordinates, output_data):
        return None


class ScipyGrid(Interpolator):
    
    def can_interpolate(self, udims, requested_coordinates, source_coordinates):
        return None

    def interpolate(self, udims, source_coordinates, source_data, requested_coordinates, output_data):
        return None


class ScipyPoint(Interpolator):
    
    def can_interpolate(self, udims, requested_coordinates, source_coordinates):
        return None

    def interpolate(self, udims, source_coordinates, source_data, requested_coordinates, output_data):
        return None


class Radial(Interpolator):
    
    def can_interpolate(self, udims, requested_coordinates, source_coordinates):
        return None

    def interpolate(self, udims, source_coordinates, source_data, requested_coordinates, output_data):
        return None


class OptimalInterpolation(Interpolator):
    """ I.E. Kriging """
    
    def can_interpolate(self, udims, requested_coordinates, source_coordinates):
        return None

    def interpolate(self, udims, source_coordinates, source_data, requested_coordinates, output_data):
        return None

# List of available interpolators
INTERPOLATION_METHODS = {
    'optimal': [OptimalInterpolation],
    'nearest_preview': [NearestPreview],
    'nearest': [NearestNeighbor],
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
 
    definition = None
    _config = {}            # container for interpolation methods for each dimension

    def __init__(self, definition=INTERPOLATION_DEFAULT, default=INTERPOLATION_DEFAULT, **kwargs):

        self.definition = definition
        self._config = {}

        # set each dim to interpolator definition
        if isinstance(definition, dict):

            for udims in iter(definition):

                # get interpolation method
                method = self._parse_interpolation_method(definition[udims])

                # if udims is not a tuple, convert it to one
                if not isinstance(udims, tuple):
                    udims = (udims,)


                # add all udims to definition
                self._set_interpolation_method(udims, method, **kwargs)



            # set default method to empty tuple
            default_method = self._parse_interpolation_method(default)
            self._set_interpolation_method(tuple(), default_method, **kwargs)
            

        elif isinstance(definition, (string_types, tuple)):
            method = self._parse_interpolation_method(definition)
            self._set_interpolation_method(tuple(), method, **kwargs)

        else:
            raise TypeError('"{}" is not a valid interpolation definition type. '.format(definition) +
                            'Interpolation definiton must be a string, dict, or tuple')



    def __repr__(self):
        rep = str(self.__class__.__name__)
        for udims in iter(self._config):
            rep += '\n\t%s: (%s, %s)' % (udims, self._config[udims][0],
                                         [i.__class__.__name__ for i in self._config[udims][1]])
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
        if isinstance(definition, string_types):
            if definition not in INTERPOLATION_SHORTCUTS:
                raise InterpolationException('"{}" is not a valid interpolation shortcut. '.format(definition) +
                                             'Valid interpolation shortcuts: {}'.format(INTERPOLATION_SHORTCUTS))
            return (definition, INTERPOLATION_METHODS[definition])

        elif isinstance(definition, tuple):
            method_string = definition[0]
            interpolators = definition[1]

            # confirm types
            if not isinstance(method_string, string_types):
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
            raise TypeError('"{}" is not a valid Interpolator definition. '.format(definition) +
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

        # set to interpolation dictionary with tuple of dims as key
        self._config[udims] = (method_string, interpolators)

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
        
        selected_coords = deepcopy(source_coordinates)
        selected_coords_idx = deepcopy(source_coordinates_index)

        # TODO: this does not yet work with udims - will not work correctly if udims specified
        for udims in iter(self._config):

            # iterate through interpolators until one can_select all udims
            interpolators = self._config[udims][1]
            interpolator_options = []
            for idx, interpolator in enumerate(interpolators):
                can_select = interpolator.can_select(udims, requested_coordinates, selected_coords)

                # can_select can be True or a cost evaluation (float)
                if can_select:
                    interpolator_options += [(idx, can_select)]


            # TODO: adjust `can_select` by interpolation cost
            # current just chooses the first interpolator interpolator_options
            if interpolator_options:
                best_option = interpolator_options[0]
                interpolator = interpolators[best_option[0]]
                selected_coords, selected_coords_idx = interpolator.select_coordinates(udims,
                                                                                       requested_coordinates,
                                                                                       selected_coords,
                                                                                       selected_coords_idx)

        return selected_coords, selected_coords_idx

    def interpolate(self, source_coordinates, source_data, requested_coordinates, output_data):
        """Interpolate data from requested coordinates to source coordinates
        
        Parameters
        ----------
        udims : tuple
            dimensions to interpolate
        source_coordinates : podpac.core.coordinates.Coordinates
            Description
        source_data : podpac.core.units.UnitsDataArray
            Description
        requested_coordinates : podpac.core.coordinates.Coordinates
            Description
        output : podpac.core.units.UnitsDataArray
            Description
        
        Returns
        -------
        podpac.core.units.UnitDataArray
            returns the new output UnitDataArray of interpolated data
        """
        
        # short circuit if the source data and requested coordinates are of shape == 1
        if source_data.size == 1 and np.prod(requested_coordinates.shape) == 1:
            output_data[:] = source_data
            return output_data

        # otherwise, iterate through the tuples of dimensions in the configuration
        for udims in iter(self._config):

            # iterate through interpolators until one can_select all udims
            interpolators = self._config[udims][1]
            interpolator_options = []
            for idx, interpolator in enumerate(interpolators):
                # should return dimensions that it can interpolate
                # {
                #  ('lat', 'lon'): Nearest,
                #  ('time'): Nearest
                #  }

                can_interpolate = interpolator.can_interpolate(udims, source_coordinates, requested_coordinates)

                if can_interpolate:
                    interpolator_options += [(idx, can_interpolate)]


            # TODO: adjust `can_interpolate` by interpolation cost
            # currently just chooses the first interpolator interpolator_options
            if interpolator_options:
                best_option = interpolator_options[0]
                interpolator = interpolators[best_option[0]]

                # run interpolation. mutates output.
                interpolator.interpolate(udims,
                                         source_coordinates,
                                         source_data,
                                         requested_coordinates,
                                         output_data)
