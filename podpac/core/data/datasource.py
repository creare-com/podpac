"""
Generic Data Source Class

DataSource is the root class for all other podpac defined data sources,
including user defined data sources.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import warnings

import numpy as np
import xarray as xr
import traitlets as tl

# Internal imports
from podpac.core.units import UnitsDataArray
from podpac.core.coordinates import Coordinates, Coordinates1d, UniformCoordinates1d, StackedCoordinates
from podpac.core.node import Node
from podpac.core.utils import common_doc, trait_is_defined
from podpac.core.node import COMMON_NODE_DOC
from podpac.core.data.interpolate import Interpolation, INTERPOLATION_SHORTCUTS, INTERPOLATION_DEFAULT

DATA_DOC = {
    'native_coordinates': 'The coordinates of the data source.',

    'get_data':
        """
        This method must be defined by the data source implementing the DataSource class.
        When data source nodes are evaluated, this method is called with request coordinates and coordinate indexes.
        The implementing method can choose which input provides the most efficient method of getting data
        (i.e via coordinates or via the index of the coordinates).
        
        Coordinates and coordinate indexes may be strided or subsets of the
        source data, but all coordinates and coordinate indexes will match 1:1 with the subset data.

        This method may return a numpy array, an xarray DaraArray, or a podpac UnitsDataArray.
        If a numpy array or xarray DataArray is returned, :meth:podpac.core.data.datasource.DataSource.evaluate will
        cast the data into a `UnitsDataArray` using the requested source coordinates.
        If a podpac UnitsDataArray is passed back, the :meth:podpac.core.data.datasource.DataSource.evaluate
        method will not do any further processing.
        The inherited Node method `create_output_array` can be used to generate the template UnitsDataArray
        in your DataSource.
        See :meth:podpac.core.node.Node.create_output_array for more details.
        
        Parameters
        ----------
        coordinates : Coordinates
            The coordinates that need to be retrieved from the data source using the coordinate system of the data
            source
        coordinates_index : List
            A list of slices or a boolean array that give the indices of the data that needs to be retrieved from
            the data source. The values in the coordinate_index will vary depending on the `coordinate_index_type`
            defined for the data source.
            
        Returns
        --------
        np.ndarray, xr.DataArray, podpac.core.units.UnitsDataArray
            A subset of the returned data. If a numpy array or xarray DataArray is returned,
            the data will be cast into  UnitsDataArray using the returned data to fill values
            at the requested source coordinates.
        """,
    
    'get_native_coordinates':
        """
        Returns a Coordinates object that describes the native coordinates of the data source.

        In most cases, this method is defined by the data source implementing the DataSource class.
        If method is not implemented by the data source, it will try to return `self.native_coordinates`
        if `self.native_coordinates` is not None.

        Otherwise, this method will raise a NotImplementedError.

        Returns
        --------
        Coordinates
           The coordinates describing the data source array.

        Raises
        --------
        NotImplementedError
            Raised if get_native_coordinates is not implemented by data source subclass.

        Notes
        ------
        Need to pay attention to:
        - the order of the dimensions
        - the stacking of the dimension
        - the type of coordinates

        Coordinates should be non-nan and non-repeating for best compatibility
        """
    }

COMMON_DATA_DOC = COMMON_NODE_DOC.copy()
COMMON_DATA_DOC.update(DATA_DOC)      # inherit and overwrite with DATA_DOC

@common_doc(COMMON_DATA_DOC)
class DataSource(Node):
    """Base node for any data obtained directly from a single source.
    
    Attributes
    ----------
    source : Any
        The location of the source. Depending on the child node this can be a filepath,
        numpy array, or dictionary as a few examples.
    native_coordinates : Coordinates
        {native_coordinates}
    coordinate_index_type : str, optional
        Type of index to use for data source. Possible values are ['list','numpy','xarray','pandas']
        Default is 'numpy'
    interpolation : str,
                    dict
                    optional
            Definition of interpolation methods for each dimension of the native coordinates.
            
            If input is a string, it must match one of the interpolation shortcuts defined in
            :ref:podpac.core.data.interpolate.INTERPOLATION_SHORTCUTS. The interpolation method associated
            with this string will be applied to all dimensions at the same time.

            If input is a dict, the dict must contain ordered set of keys defining dimensions and values
            defining the interpolation method to use with the dimensions.
            The key must be a string or tuple of dimension names (i.e. `'time'` or `('lat', 'lon')` ).
            The value can either be a string matching one of the interpolation shortcuts defined in
            :ref:podpac.core.data.interpolate.INTERPOLATION_SHORTCUTS or a dictionary.
            If the value is a dictionary, the dictionary must contain a key `'method'`
            defining the interpolation method name.
            If the interpolation method is not one of :ref:podpac.core.data.interpolate.INTERPOLATION_SHORTCUTS, a
            second key `'interpolators'` must be defined with a list of
            :ref:podpac.core.data.interpolate.Interpolator classes to use in order of uages.
            The dictionary may contain an option `'params'` key which contains a dict of parameters to pass along to
            the :ref:podpac.core.data.interpolate.Interpolator classes associated with the interpolation method.
            
            If the dictionary does not contain a key for all unstacked dimensions of the source coordinates, the
            :ref:podpac.core.data.interpolate.INTERPOLATION_DEFAULT value will be used.
            All dimension keys must be unstacked even if the underlying coordinate dimensions are stacked.
            Any extra dimensions included but not found in the source coordinates will be ignored.

            If input is a podpac.core.data.interpolate.Interpolation, this interpolation
            class will be used without modication.
            
            By default, the interpolation method is set to `'nearest'` for all dimensions.
    nan_vals : List, optional
        List of values from source data that should be interpreted as 'no data' or 'nans'

    Notes
    -----
    Developers of new DataSource nodes need to implement the `get_data` and `get_native_coordinates` methods.
    """
    
    source = tl.Any(help='Path to the raw data source')
    native_coordinates = tl.Instance(Coordinates)

    interpolation = tl.Union([
        tl.Dict(),
        tl.Enum(INTERPOLATION_SHORTCUTS),
        tl.Instance(Interpolation)
    ], default_value=INTERPOLATION_DEFAULT)

    coordinate_index_type = tl.Enum(['list', 'numpy', 'xarray', 'pandas'], default_value='numpy')
    nan_vals = tl.List(allow_none=True)

    # TODO: remove in the 2nd stage of interpolation refactor
    # self.source_coordinates['time'].delta / 2
    interpolation_tolerance = tl.Union([tl.Float(default_value=1), tl.Instance(np.timedelta64)])

    # privates
    _interpolation = tl.Instance(Interpolation)
    
    _evaluated_coordinates = tl.Instance(Coordinates, allow_none=True)
    _requested_source_coordinates = tl.Instance(Coordinates)
    _requested_source_coordinates_index = tl.List()
    _requested_source_data = tl.Instance(UnitsDataArray)

    # when native_coordinates is not defined, default calls get_native_coordinates
    @tl.default('native_coordinates')
    def _default_native_coordinates(self):
        self.native_coordinates = self.get_native_coordinates()
        return self.native_coordinates

    # this adds a more helpful error message if user happens to try an inspect _interpolation before evaluate
    @tl.default('_interpolation')
    def _default_interpolation(self):
        self._set_interpolation()
        return self._interpolation

    @common_doc(COMMON_DATA_DOC)
    def eval(self, coordinates, output=None):
        """Evaluates this node using the supplied coordinates.

        The native coordinates are mapped to the requested coordinates, interpolated if necessary, and set to
        `_requested_source_coordinates` with associated index `_requested_source_coordinates_index`. The requested
        source coordinates and index are passed to `get_data()` returning the source data at the
        native coordinatesset to `_requested_source_data`. Finally `_requested_source_data` is interpolated
        using the `interpolate` method and set to the `output` attribute of the node.


        Parameters
        ----------
        coordinates : Coordinates
            {requested_coordinates}
            Notes::
             * An exception is raised if the requested coordinates are missing dimensions in the DataSource.
             * Extra dimensions in the requested coordinates are dropped.
        output : podpac.core.units.UnitsDataArray, optional
            {eval_output}
        
        Returns
        -------
        {eval_return}

        Raises
        ------
        ValueError
            Cannot evaluate these coordinates
        """

        if self.coordinate_index_type != 'numpy':
            warnings.warn('Coordinates index type {} is not yet supported.'.format(self.coordinate_index_type) +
                          '`coordinate_index_type` is set to `numpy`', UserWarning)
        
        # check for missing dimensions
        for c in self.native_coordinates.values():
            if isinstance(c, Coordinates1d):
                if c.name not in coordinates.udims:
                    raise ValueError("Cannot evaluate these coordinates, missing dim '%s'" % c.name)
            elif isinstance(c, StackedCoordinates):
                if any(s.name not in coordinates.udims for s in c):
                    raise ValueError("Cannot evaluate these coordinates, missing at least one dim in '%s'" % c.name)
        
        # remove extra dimensions
        extra = []
        for c in coordinates.values():
            if isinstance(c, Coordinates1d):
                if c.name not in self.native_coordinates.udims:
                    extra.append(c.name)
            elif isinstance(c, StackedCoordinates):
                if all(dim not in self.native_coordinates.udims for dim in c.dims):
                    extra.append(c.name)
        coordinates = coordinates.drop(extra)

        # set input coordinates to evaluated coordinates
        # TODO move this if WCS can be updated to support
        self._evaluated_coordinates = coordinates

        # intersect the native coordinates with requested coordinates
        # to get native coordinates within requested coordinates bounds
        # TODO: support coordinate_index_type parameter to define other index types
        self._requested_source_coordinates, self._requested_source_coordinates_index = \
            self.native_coordinates.intersect(coordinates, outer=True, return_indices=True)

        # If requested coordinates and native coordinates do not intersect, shortcut with nan UnitsDataArary
        if np.prod(self._requested_source_coordinates.shape) == 0:
            if output is None:
                output = self.create_output_array(coordinates)
            else:
                output[:] = np.nan

            self._output = output
            return output
        
        # reset interpolation
        self._set_interpolation()

        # interpolate requested coordinates before getting data
        self._requested_source_coordinates, self._requested_source_coordinates_index = \
            self._interpolation.select_coordinates(self._requested_source_coordinates,
                                                   self._requested_source_coordinates_index,
                                                   coordinates)

        # get data from data source
        self._requested_source_data = self._get_data(coordinates)

        # if output is not input to evaluate, create it using the evaluated coordinates
        if output is None:
            output = self.create_output_array(coordinates)

        # interpolate data into output
        self._requested_source_coordinates, self._requested_source_data, output = \
            self._interpolation.interpolate(self._requested_source_coordinates,
                                            self._requested_source_data,
                                            coordinates,
                                            output)

        # set the order of dims to be the same as that of requested_coordinates
        # this is required in case the user supplied an output object with a different dims order
        output = output.transpose(*coordinates.dims)
        
        # save output to private for debugging
        self._output = output

        return output


    def _set_interpolation(self):
        """Update _interpolation property
        """

        # define interpolator with source coordinates dimensions
        if isinstance(self.interpolation, Interpolation):
            self._interpolation = self.interpolation
        else:
            self._interpolation = Interpolation(self.interpolation)



    def _get_data(self, coordinates):
        """Wrapper for `self.get_data` with pre and post processing
        
        Returns
        -------
        podpac.core.units.UnitsDataArray
            Returns UnitsDataArray with coordinates defined by _requested_source_coordinates
        
        Raises
        ------
        ValueError
            Raised if unknown data is passed by from self.get_data
        NotImplementedError
            Raised if get_data is not implemented by data source subclass

        """
        # get data from data source at requested source coordinates and requested source coordinates index
        data = self.get_data(self._requested_source_coordinates, self._requested_source_coordinates_index)

        # convert data into UnitsDataArray depending on format
        # TODO: what other processing needs to happen here?
        if isinstance(data, UnitsDataArray):
            udata_array = data
        elif isinstance(data, xr.DataArray):
            # TODO: check order of coordinates here
            udata_array = self.create_output_array(coordinates, data=data.data)
        elif isinstance(data, np.ndarray):
            udata_array = self.create_output_array(coordinates, data=data)
        else:
            raise ValueError('Unknown data type passed back from ' +
                             '{}.get_data(): {}. '.format(type(self).__name__, type(data)) +
                             'Must be one of numpy.ndarray, xarray.DataArray, or podpac.UnitsDataArray')

        # fill nan_vals in data array
        if self.nan_vals:
            for nan_val in self.nan_vals:
                udata_array.data[udata_array.data == nan_val] = np.nan

        return udata_array


    ########
    # Public DataSource Methods
    ########
    
    def find_coordinates(self):
        """
        Get the available native coordinates for the Node. For a DataSource, this is just the native_coordinates.

        Returns
        -------
        coords_list : list
            singleton list containing the native_coordinates (Coordinates object)
        """

        return [self.native_coordinates]

    def get_interpolation_class(self):
        """Get the interpolation class currently set for this data source.
        
        The DataSource `interpolation` property is used to define the
        :ref:podpac.core.data.interpolate.Interpolation class that will handle interpolation for requested coordinates.
        
        Returns
        -------
        podpac.core.data.interpolate.Interpolation
            Interpolation class defined by DataSource `interpolation` definition
        """

        return self._interpolation

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        
        Raises
        ------
        NotImplementedError
            This needs to be implemented by derived classes
        """
        raise NotImplementedError
        
    @common_doc(COMMON_DATA_DOC)
    def get_native_coordinates(self):
        """{get_native_coordinates}
        
        Raises
        ------
        NotImplementedError
            This needs to be implemented by derived classes
        """
        
        if trait_is_defined(self, 'native_coordinates'):
            return self.native_coordinates
        else:
            raise NotImplementedError('{0}.native_coordinates is not defined and '  \
                                      '{0}.get_native_coordinates() is not implemented'.format(self.__class__.__name__))

    @property
    @common_doc(COMMON_DATA_DOC)
    def base_definition(self):
        """Base node defintion for DataSource nodes. 
        
        Returns
        -------
        {definition_return}
        """
        d = super(DataSource, self).base_definition
        d['source'] = self.source

        # TODO: cast interpolation to string in way that can be recreated here
        # should this move to interpolation class? 
        # It causes issues when the _interpolation class has not been set up yet
        d['interpolation'] = self.interpolation
        return d
