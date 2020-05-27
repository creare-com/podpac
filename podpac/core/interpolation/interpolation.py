from __future__ import division, unicode_literals, print_function, absolute_import
from copy import deepcopy
from collections import OrderedDict
from six import string_types

import traitlets as tl
import numpy as np

# podpac imports
from podpac.core.interpolation.interpolator import Interpolator
from podpac.core.interpolation.interpolators import NearestNeighbor, NearestPreview, Rasterio, ScipyPoint, ScipyGrid

INTERPOLATION_DEFAULT = "nearest"
"""str : Default interpolation method used when creating a new :class:`Interpolation` class """

INTERPOLATORS = [NearestNeighbor, NearestPreview, Rasterio, ScipyPoint, ScipyGrid]
"""list : list of available interpolator classes"""

INTERPOLATORS_DICT = {}
"""dict : Dictionary of a string interpolator name and associated interpolator class"""

INTERPOLATION_METHODS = [
    "nearest_preview",
    "nearest",
    "bilinear",
    "cubic",
    "cubic_spline",
    "lanczos",
    "average",
    "mode",
    "gauss",
    "max",
    "min",
    "med",
    "q1",
    "q3",
    "spline_2",
    "spline_3",
    "spline_4",
]

INTERPOLATION_METHODS_DICT = {}
"""dict: Dictionary of string interpolation methods and associated interpolator classes
   (i.e. ``'nearest': [NearestNeighbor, Rasterio, Scipy]``) """


def load_interpolators():
    """Load interpolators from :list:`INTERPOLATORS`

    Defines :dict:`INTERPOLATORS_DICT`, and :dict:`INTERPOLATION_METHODS_DICT`
    """

    # create empty arrays in INTEPROLATOR_METHODS
    for method in INTERPOLATION_METHODS:
        INTERPOLATION_METHODS_DICT[method] = []

    # fill dictionaries with interpolator properties
    for interpolator_class in INTERPOLATORS:
        interpolator = interpolator_class()
        INTERPOLATORS_DICT[interpolator.name] = interpolator_class

        for method in INTERPOLATION_METHODS:
            if method in interpolator.methods_supported:
                INTERPOLATION_METHODS_DICT[method].append(interpolator_class)


# load interpolators when module is first loaded
# TODO does this really only load once?
# TODO maybe move this whole section?
load_interpolators()


class InterpolationException(Exception):
    """
    Custom label for interpolation exceptions
    """

    pass


class Interpolation(object):
    """Create an interpolation class to handle one interpolation method per unstacked dimension.
    Used to interpolate data within a datasource.
    
    Parameters
    ----------
    definition : str, tuple (str, list of podpac.core.data.interpolator.Interpolator), dict
        Interpolation definition used to define interpolation methods for each definiton.
        See :attr:`podpac.data.DataSource.interpolation` for more details.
    
    Raises
    ------
    InterpolationException
        Raised when definition parameter is improperly formatted
    
    """

    definition = None
    config = OrderedDict()  # container for interpolation methods for each dimension
    _last_interpolator_queue = None  # container for the last run interpolator queue - useful for debugging
    _last_select_queue = None  # container for the last run select queue - useful for debugging

    def __init__(self, definition=INTERPOLATION_DEFAULT):

        self.definition = deepcopy(definition)
        self.config = OrderedDict()

        # if definition is None, set to default
        # TODO: do we want to always have a default for interpolation?
        # Or should there be an option to turn off interpolation?
        if self.definition is None:
            self.definition = INTERPOLATION_DEFAULT

        # set each dim to interpolator definition
        if isinstance(definition, (dict, list)):

            # convert dict to list
            if isinstance(definition, dict):
                definition = [definition]

            for interp_definition in definition:

                # get interpolation method dict
                method = self._parse_interpolation_method(interp_definition)

                # specify dims
                if "dims" in interp_definition:
                    if isinstance(interp_definition["dims"], list):
                        udims = tuple(
                            sorted(interp_definition["dims"])
                        )  # make sure the dims are always in the same order
                    else:
                        raise TypeError('The "dims" key of an interpolation definition must be a list')
                else:
                    udims = ("default",)

                # make sure udims are not already specified in config
                for config_dims in iter(self.config):
                    if set(config_dims) & set(udims):
                        raise InterpolationException(
                            'Dimensions "{}" cannot be defined '.format(udims)
                            + "multiple times in interpolation definition {}".format(interp_definition)
                        )
                # add all udims to definition
                self._set_interpolation_method(udims, method)

            # set default if its not been specified in the dict
            if ("default",) not in self.config:

                default_method = self._parse_interpolation_method(INTERPOLATION_DEFAULT)
                self._set_interpolation_method(("default",), default_method)

        elif isinstance(definition, string_types):
            method = self._parse_interpolation_method(definition)
            self._set_interpolation_method(("default",), method)

        else:
            raise TypeError(
                '"{}" is not a valid interpolation definition type. '.format(definition)
                + "Interpolation definiton must be a string or list of dicts"
            )

        # make sure ('default',) is always the last entry in config dictionary
        default = self.config.pop(("default",))
        self.config[("default",)] = default

    def __repr__(self):
        rep = str(self.__class__.__name__)
        for udims in iter(self.config):
            # rep += '\n\t%s:\n\t\tmethod: %s\n\t\tinterpolators: %s\n\t\tparams: %s' % \
            rep += "\n\t%s: %s, %s, %s" % (
                udims,
                self.config[udims]["method"],
                [i.__class__.__name__ for i in self.config[udims]["interpolators"]],
                self.config[udims]["params"],
            )

        return rep

    def _parse_interpolation_method(self, definition):
        """parse interpolation definitions into a tuple of (method, Interpolator)
        
        Parameters
        ----------
        definition : str, dict
            interpolation definition
            See :attr:`podpac.data.DataSource.interpolation` for more details.
        
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
            if definition not in INTERPOLATION_METHODS:
                raise InterpolationException(
                    '"{}" is not a valid interpolation shortcut. '.format(definition)
                    + "Valid interpolation shortcuts: {}".format(INTERPOLATION_METHODS)
                )
            return {"method": definition, "interpolators": INTERPOLATION_METHODS_DICT[definition], "params": {}}

        elif isinstance(definition, dict):

            # confirm method in dict
            if "method" not in definition:
                raise InterpolationException(
                    "{} is not a valid interpolation definition. ".format(definition)
                    + 'Interpolation definition dict must contain key "method" string value'
                )
            else:
                method_string = definition["method"]

            # if specifying custom method, user must include interpolators
            if "interpolators" not in definition and method_string not in INTERPOLATION_METHODS:
                raise InterpolationException(
                    '"{}" is not a valid interpolation shortcut. '.format(method_string)
                    + 'Specify list "interpolators" or change "method" '
                    + "to a valid interpolation shortcut: {}".format(INTERPOLATION_METHODS)
                )
            elif "interpolators" not in definition:
                interpolators = INTERPOLATION_METHODS_DICT[method_string]
            else:
                interpolators = definition["interpolators"]

            # default for params
            if "params" in definition:
                params = definition["params"]
            else:
                params = {}

            # confirm types
            if not isinstance(method_string, string_types):
                raise TypeError(
                    "{} is not a valid interpolation method. ".format(method_string)
                    + "Interpolation method must be a string"
                )

            if not isinstance(interpolators, list):
                raise TypeError(
                    "{} is not a valid interpolator definition. ".format(interpolators)
                    + "Interpolator definition must be of type list containing Interpolator"
                )

            if not isinstance(params, dict):
                raise TypeError(
                    "{} is not a valid interpolation params definition. ".format(params)
                    + "Interpolation params must be a dict"
                )

            # handle when interpolator is a string (most commonly from a node definition)
            for idx, interpolator_class in enumerate(interpolators):
                if isinstance(interpolator_class, string_types):
                    if interpolator_class in INTERPOLATORS_DICT.keys():
                        interpolators[idx] = INTERPOLATORS_DICT[interpolator_class]
                    else:
                        raise TypeError(
                            'Interpolator "{}" is not in the dictionary of valid '.format(interpolator_class)
                            + "interpolators: {}".format(INTERPOLATORS_DICT)
                        )

            # validate interpolator class
            for interpolator in interpolators:
                self._validate_interpolator(interpolator)

            # if all checks pass, return the definition
            return {"method": method_string, "interpolators": interpolators, "params": params}

        else:
            raise TypeError(
                '"{}" is not a valid Interpolator definition. '.format(definition)
                + "Interpolation definiton must be a string or dict."
            )

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
            raise TypeError(
                "{} is not a valid interpolator type. ".format(interpolator)
                + "Interpolator must be of type {}".format(Interpolator)
            )

    def _set_interpolation_method(self, udims, definition):
        """Set the list of interpolation definitions to the input dimension
        
        Parameters
        ----------
        udims : tuple
            tuple of dimensiosn to assign definition to
        definition : dict
            dict definition returned from _parse_interpolation_method
        """

        method = deepcopy(definition["method"])
        interpolators = deepcopy(definition["interpolators"])
        params = deepcopy(definition["params"])

        # instantiate interpolators
        for (idx, interpolator) in enumerate(interpolators):
            interpolators[idx] = interpolator(method=method, **params)

        definition["interpolators"] = interpolators

        # set to interpolation configuration for dims
        self.config[udims] = definition

    def _select_interpolator_queue(self, source_coordinates, eval_coordinates, select_method, strict=False):
        """Create interpolator queue based on interpolation configuration and requested/native source_coordinates
        
        Parameters
        ----------
        source_coordinates : :class:`podpac.Coordinates`
            Description
        eval_coordinates : :class:`podpac.Coordinates`
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
        handled_dims = set()

        interpolator_queue = OrderedDict()

        # go through all dims in config
        for key in iter(self.config):

            # if the key is set to (default,), it represents all the remaining dimensions that have not been handled
            # __init__ makes sure that (default,) will always be the last key in on
            if key == ("default",):
                udims = tuple(sorted(source_dims - handled_dims))
            else:
                udims = key

            # get configured list of interpolators for dim definition
            interpolators = self.config[key]["interpolators"]

            # iterate through interpolators recording which dims they support
            for interpolator in interpolators:
                # if all dims have been handled already, skip the rest
                if not udims:
                    break

                # see which dims the interpolator can handle
                can_handle = getattr(interpolator, select_method)(udims, source_coordinates, eval_coordinates)

                # if interpolator can handle all udims
                if not set(udims) - set(can_handle):

                    # save union of dims that can be handled by this interpolator and already supported dims for next iteration
                    handled_dims = handled_dims | set(can_handle)

                    # set interpolator to work on that dimension in the interpolator_queue if dim has no interpolator
                    if udims not in interpolator_queue:

                        interpolator_queue[udims] = interpolator

        # throw error if the source_dims don't encompass all the supported dims
        # this should happen rarely because of default
        if len(source_dims) > len(handled_dims) and strict:
            missing_dims = list(source_dims - handled_dims)
            raise InterpolationException(
                "Dimensions {} ".format(missing_dims)
                + "can't be handled by interpolation definition:\n {}".format(self)
            )

        # TODO: adjust by interpolation cost
        return interpolator_queue

    def select_coordinates(self, source_coordinates, source_coordinates_index, eval_coordinates):
        """
        Select a subset or coordinates if interpolator can downselect.
        
        At this point in the execution process, podpac has selected a subset of source_coordinates that intersects
        with the requested coordinates, dropped extra dimensions from requested coordinates, and confirmed
        source coordinates are not missing any dimensions.
        
        Parameters
        ----------
        source_coordinates : :class:`podpac.Coordinates`
            Intersected source coordinates
        source_coordinates_index : list
            Index of intersected source coordinates. See :class:`podpac.data.DataSource` for
            more information about valid values for the source_coordinates_index
        eval_coordinates : :class:`podpac.Coordinates`
            Requested coordinates to evaluate
        
        Returns
        -------
        (:class:`podpac.Coordinates`, list)
            Returns tuple with the first element subset of selected coordinates and the second element the indicies
            of the selected coordinates
        """

        # TODO: short circuit if source_coordinates contains eval_coordinates
        # short circuit if source and eval coordinates are the same
        if source_coordinates == eval_coordinates:
            return source_coordinates, tuple(source_coordinates_index)

        interpolator_queue = self._select_interpolator_queue(source_coordinates, eval_coordinates, "can_select")

        self._last_select_queue = interpolator_queue

        selected_coords = deepcopy(source_coordinates)
        selected_coords_idx = deepcopy(source_coordinates_index)

        for udims in interpolator_queue:
            interpolator = interpolator_queue[udims]

            # run interpolation. mutates selected coordinates and selected coordinates index
            selected_coords, selected_coords_idx = interpolator.select_coordinates(
                udims, selected_coords, selected_coords_idx, eval_coordinates
            )

        return selected_coords, tuple(selected_coords_idx)

    def interpolate(self, source_coordinates, source_data, eval_coordinates, output_data):
        """Interpolate data from requested coordinates to source coordinates
        
        Parameters
        ----------
        source_coordinates : :class:`podpac.Coordinates`
            Description
        source_data : podpac.core.units.UnitsDataArray
            Description
        eval_coordinates : :class:`podpac.Coordinates`
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

        # TODO does this allow undesired extrapolation?
        # short circuit if the source data and requested coordinates are of shape == 1
        if source_data.size == 1 and eval_coordinates.size == 1:
            output_data[:] = source_data
            return output_data

        # TODO: short circuit if source_coordinates contains eval_coordinates
        # this has to be done better...
        # short circuit if source and eval coordinates are the same
        if all(udims in eval_coordinates.udims for udims in source_coordinates.udims):
            if all(source_coordinates[udim] == eval_coordinates[udim] for udim in source_coordinates.udims):
                output_data.data = source_data.transpose(*output_data.dims).data  # transpose and insert
                return output_data

        interpolator_queue = self._select_interpolator_queue(
            source_coordinates, eval_coordinates, "can_interpolate", strict=True
        )

        # for debugging purposes, save the last defined interpolator queue
        self._last_interpolator_queue = interpolator_queue

        # iterate through each dim tuple in the queue
        for udims in interpolator_queue:
            interpolator = interpolator_queue[udims]

            # run interpolation
            output_data = interpolator.interpolate(
                udims, source_coordinates, source_data, eval_coordinates, output_data
            )

        return output_data


class InterpolationTrait(tl.Union):
    default_value = INTERPOLATION_DEFAULT

    def __init__(
        self,
        trait_types=[tl.Dict(), tl.List(), tl.Enum(INTERPOLATION_METHODS), tl.Instance(Interpolation)],
        *args,
        **kwargs
    ):
        super(InterpolationTrait, self).__init__(trait_types=trait_types, *args, **kwargs)
