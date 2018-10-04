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
    dims_supported = tl.List(tl.Unicode())

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

    def _intersect_udims(self, udims):
        
        # find the intersection between dims_supported and udims, return tuple of intersection
        return tuple(set(self.dims_supported) & set(udims))

    def dim_available(self, dim, source_coordinates, requested_coordinates):
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
            dimensions to select
        requested_coordinates : podpac.core.coordinates.Coordinates
            Description
        source_coordinates : podpac.core.coordinates.Coordinates
            Description
        
        Raises
        ------
        NotImplementedError
        
        Returns
        -------
        tuple
            Returns a tuple of dimensions that can be selected with this interpolator
            If no dimensions can be selected, method should return an emtpy tuple
        """

        # if dims_supported is defined, then try each dim and return False if one of the dims is not allowed
        if self.dims_supported:
            return self._intersect_udims(udims)
        else:
            raise NotImplementedError

    def select_coordinates(self, udims, requested_coordinates, source_coordinates, source_coordinates_index):
        """Downselect coordinates with interpolator method
        
        Parameters
        ----------
        udims : tuple
            dimensions to select coordinates
        requested_coordinates : podpac.core.coordinates.Coordinates
            Description
        source_coordinates : podpac.core.coordinates.Coordinates
            Description
        source_coordinates_index : list
            Description

        Returns
        -------
        podpac.core.coordinates.Coordinates, list
            returns the new down selected coordinates and the new associated index. These coordinates must exist
            in the native coordinates of the source data
        
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def can_interpolate(self, udims, requested_coordinates, source_coordinates):
        """Evaluate if this interpolation method can handle the requested coordinates and source_coordinates
        
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
        tuple
            Returns a tuple of dimensions that can be interpolated with this interpolator
            If no dimensions can be interpolated, method should return an emtpy tuple
        """

        # if dims_supported is defined, then try each dim and return False if one of the dims is not allowed
        if self.dims_supported:
            return self._intersect_udims(udims)
        else:
            raise NotImplementedError

    def interpolate(self, udims, requested_coordinates, source_coordinates, source_data, output_data):
        """Interpolate data from requested coordinates to source coordinates.
        
        Parameters
        ----------
        udims : tuple
            dimensions to interpolate
        requested_coordinates : podpac.core.coordinates.Coordinates
            Description
        source_coordinates : podpac.core.coordinates.Coordinates
            Description
        source_data : podpac.core.units.UnitsDataArray
            Description
        output_data : TYPE
            Description
        
        Raises
        ------
        NotImplementedError

        Returns
        -------
        (podpac.core.coordinates.Coordinates, podpac.core.units.UnitDataArraym podpac.core.units.UnitDataArray)
            returns the updated (source_coordinates, source_data, and output_data) of interpolated data
        """
        raise NotImplementedError


class NearestNeighbor(Interpolator):

    dims_supported = ['lat', 'lon', 'alt', 'time']
    tolerance = tl.Int()

    # nearest neighbor can't select coordinates
    def can_select(self, udims, requested_coordinates, source_coordinates):
        """NearestNeighbor can't select coordinates"""
        return tuple()

    def interpolate(self, udims, requested_coordinates, source_coordinates, source_data, output_data):
        

        # if ... 


        # if dim == 'time' and self.dim_available(dim, source_coordinates, requested_coordinates):
        #     source_data = source_data.reindex(
        #         time=requested_coordinates.coords['time'], method='nearest', tolerance=self.tolerance)
            
        #     source_coordinates['time'] = ArrayCoordinates1d.from_xarray(source_data['time'])


        # if dim == 'alt' and self.dim_available(dim, source_coordinates, requested_coordinates):
        #     source_data = source_data.reindex(alt=requested_coordinates.coords['alt'], method='nearest')
        #     source_coordinates['alt'] = ArrayCoordinates1d.from_xarray(source_data['alt'])

        return source_coordinates, source_data, output_data


class NearestPreview(Interpolator):
    
    dims_supported = ['lat', 'lon', 'alt', 'time']
    tolerance = tl.Int()

    def select_coordinates(self, udims, requested_coordinates, source_coordinates, source_coordinates_index):

        # iterate over the source coordinate dims in case they are stacked
        for src_dim, idx in zip(source_coordinates, source_coordinates_index):
            if src_dim in requested_coordinates.dims:
                src_coords = source_coordinates[src_dim]
                dst_coords = requested_coordinates[src_dim]

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
                c = source_coordinates[src_dim]

            new_coords.append(c)
            new_coords_idx.append(idx)

        # updates requested source coordinates and index
        new_source_coordinates = Coordinates(new_coords)
        new_source_coordinates_index = new_coords_idx

        return new_source_coordinates, new_source_coordinates_index


    def interpolate(self, udims, requested_coordinates, source_coordinates, source_data, output_data):
        
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

        return source_coordinates, source_data, output_data


class Rasterio(Interpolator):

    dims_supported = ['lat', 'lon']
    rasterio_interpolators = ['nearest', 'bilinear', 'cubic', 'cubic_spline', 'lanczos', 'average', 'mode', 'gauss',
                              'max', 'min', 'med', 'q1', 'q3']

    def can_select(self, udims, requested_coordinates, source_coordinates):
        """Rasterio can't select coordinates"""
        return tuple()

    def can_interpolate(self, udims, requested_coordinates, source_coordinates):
        return tuple()
        # return self.method in self.rasterio_interpolators 
        #        and self.dim_in('lat' in source_coordinates.dims and 'lon' in source_coordinates.dims
        #        and 'lat' in requested_coordinates.dims and 'lon' in requested_coordinates.dims
        #        and source_coordinates['lat'].is_uniform and source_coordinates['lon'].is_uniform
        #        and requested_coordinates['lat'].is_uniform and requested_coordinates['lon'].is_uniform

    def interpolate(self, udims, requested_coordinates, source_coordinates, source_data, output_data):
        return None


class ScipyGrid(Interpolator):

    def interpolate(self, udims, requested_coordinates, source_coordinates, source_data, output_data):
        return None


class ScipyPoint(Interpolator):

    def interpolate(self, udims, requested_coordinates, source_coordinates, source_data, output_data):
        return None


class Radial(Interpolator):

    def interpolate(self, udims, requested_coordinates, source_coordinates, source_data, output_data):
        return None


class OptimalInterpolation(Interpolator):
    """ I.E. Kriging """

    def interpolate(self, udims, requested_coordinates, source_coordinates, source_data, output_data):
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

    def __init__(self, definition=INTERPOLATION_DEFAULT):

        self.definition = definition
        self._config = OrderedDict()

        # set each dim to interpolator definition
        if isinstance(definition, dict):

            # covert input to an ordered dict to preserve order of dimensions
            definition = OrderedDict(definition)

            for key in iter(definition):

                # if udims is not a tuple, convert it to one
                if not isinstance(key, tuple):
                    udims = (key,)
                else:
                    udims = key

                # make sure udims are not already specified in config
                for config_dims in iter(self._config):
                    if set(config_dims) & set(udims):
                        raise InterpolationException('Dimensions "{}" cannot be defined '.format(udims) +
                                                     'multiple times in interpolation definition {}'.format(definition))

                # get interpolation method
                method = self._parse_interpolation_method(definition[key])


                # add all udims to definition
                self._set_interpolation_method(udims, method)


            # set default if its not been specified in the dict
            if ('default',) not in self._config:

                default_method = self._parse_interpolation_method(INTERPOLATION_DEFAULT)
                self._set_interpolation_method(('default',), default_method)
            

        elif isinstance(definition, string_types):
            method = self._parse_interpolation_method(definition)
            self._set_interpolation_method(('default',), method)

        else:
            raise TypeError('"{}" is not a valid interpolation definition type. '.format(definition) +
                            'Interpolation definiton must be a string or dict')

    def __repr__(self):
        rep = str(self.__class__.__name__)
        for udims in iter(self._config):
            # rep += '\n\t%s:\n\t\tmethod: %s\n\t\tinterpolators: %s\n\t\tparams: %s' % \
            rep += '\n\t%s: %s, %s, %s' % \
                (udims,
                 self._config[udims]['method'],
                 [i.__class__.__name__ for i in self._config[udims]['interpolators']],
                 self._config[udims]['params']
                )

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

    def _set_interpolation_method(self, udims, method):
        """Set the list of interpolation methods to the input dimension
        
        Parameters
        ----------
        udims : tuple
            tuple of dimensiosn to assign method to
        method : dict
            dict method returned from _parse_interpolation_method
        """

        method_string = deepcopy(method['method'])
        interpolators = deepcopy(method['interpolators'])
        params = deepcopy(method['params'])

        # instantiate interpolators
        for (idx, interpolator) in enumerate(interpolators):
            interpolators[idx] = interpolator(method=method_string, **params)

        method['interpolators'] = interpolators

        # set to interpolation configuration for dims
        self._config[udims] = method

    def _select_interpolator_queue(self, requested_coordinates, source_coordinates, select_method, strict=False):
        """Create interpolator queue based on interpolation configuration and requested/native source_coordinates
        
        Parameters
        ----------
        requested_coordinates : podpac.core.coordinates.Coordinates
            Description
        source_coordinates : podpac.core.coordinates.Coordinates
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
        stacked_source_dims = set(source_coordinates.dims)
        handled_dims = set()

        interpolator_queue = OrderedDict()

        # go through all dims in config
        for key in iter(self._config):

            # if the key are set to (default,), it represents all the remaining dimensions that have not been handled
            # this will always happen as the only or last entry in a dictionary definition
            if key == ('default',):
                udims = tuple(source_dims - handled_dims)
            else:
                udims = key

            # get configured list of interpolators for dim definition
            interpolators = self._config[key]['interpolators']

            # iterate through interpolators recording which dims they support
            for interpolator in interpolators:
                # if all dims have been handled already, skip the rest
                if not udims:
                    break

                # see which dims the interpolator can handle
                can_handle = getattr(interpolator, select_method)(udims, requested_coordinates, source_coordinates)

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

    def _validate_interpolator_queue(self):
        pass

    def select_coordinates(self, requested_coordinates, source_coordinates, source_coordinates_index):
        """
        Select a subset or coordinates if interpolator can downselect.

        At this point in the execution process, podpac has selected a subset of source_coordinates that intersects
        with the requested coordinates, dropped extra dimensions from requested coordinates, and confirmed
        source coordinates are not missing any dimensions.
        
        Parameters
        ----------
        requested_coordinates : podpac.core.coordinates.Coordinates
            Requested coordinates to execute
        source_coordinates : podpac.core.coordinates.Coordinates
            Intersected source coordinates
        source_coordinates_index : list
            Index of intersected source coordinates. See :ref:podpac.core.data.datasource.DataSource for
            more information about valid values for the source_coordinates_index
        
        Returns
        -------
        podpac.core.coordinates.Coordinates, list
            Returns tuple with the first element subset of selected coordinates and the second element the indicies
            of the selected coordinates
        """
        
        interpolator_queue = \
            self._select_interpolator_queue(requested_coordinates, source_coordinates, 'can_select')

        selected_coords = deepcopy(source_coordinates)
        selected_coords_idx = deepcopy(source_coordinates_index)

        for udims in interpolator_queue:
            interpolator = interpolator_queue[udims]

            # run interpolation. mutates selected coordinates and selected coordinates index
            selected_coords, selected_coords_idx = interpolator.select_coordinates(udims,
                                                                                   requested_coordinates,
                                                                                   selected_coords,
                                                                                   selected_coords_idx)

        return selected_coords, selected_coords_idx

    def interpolate(self, source_coordinates, source_data, requested_coordinates, output_data):
        """Interpolate data from requested coordinates to source coordinates
        
        Parameters
        ----------
        source_coordinates : podpac.core.coordinates.Coordinates
            Description
        source_data : podpac.core.units.UnitsDataArray
            Description
        requested_coordinates : podpac.core.coordinates.Coordinates
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
        if source_data.size == 1 and np.prod(requested_coordinates.shape) == 1:
            output_data[:] = source_data
            return output_data

        interpolator_queue = \
            self._select_interpolator_queue(requested_coordinates, source_coordinates, 'can_interpolate')

        for udims, interpolator in interpolator_queue:

            # run interpolation. mutates output.
            source_coordinates, source_data, output_data = interpolator.interpolate(udims,
                                                                                    source_coordinates,
                                                                                    source_data,
                                                                                    requested_coordinates,
                                                                                    output_data)

        return output_data
