from __future__ import division, unicode_literals, print_function, absolute_import
import logging

import warnings
from copy import deepcopy
from collections import OrderedDict
from six import string_types
import numpy as np
import xarray as xr
import traitlets as tl

from podpac.core import settings
from podpac.core.units import UnitsDataArray
from podpac.core.coordinates import merge_dims, Coordinates, StackedCoordinates
from podpac.core.coordinates.utils import VALID_DIMENSION_NAMES
from podpac.core.interpolation.interpolator import Interpolator
from podpac.core.interpolation.nearest_neighbor_interpolator import NearestNeighbor, NearestPreview
from podpac.core.interpolation.rasterio_interpolator import RasterioInterpolator
from podpac.core.interpolation.scipy_interpolator import ScipyPoint, ScipyGrid
from podpac.core.interpolation.xarray_interpolator import XarrayInterpolator
from podpac.core.interpolation.none_interpolator import NoneInterpolator

_logger = logging.getLogger(__name__)


INTERPOLATION_DEFAULT = settings.settings.get("DEFAULT_INTERPOLATION", "nearest")
"""str : Default interpolation method used when creating a new :class:`Interpolation` class """

INTERPOLATORS = [
    NoneInterpolator,
    NearestNeighbor,
    XarrayInterpolator,
    RasterioInterpolator,
    ScipyPoint,
    ScipyGrid,
    NearestPreview,
]
"""list : list of available interpolator classes"""

INTERPOLATORS_DICT = {}
"""dict : Dictionary of a string interpolator name and associated interpolator class"""

INTERPOLATION_METHODS = [
    "none",
    "nearest_preview",
    "nearest",
    "linear",
    "bilinear",
    "quadratic",
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
    "slinear",  # Spline linear
    "splinef2d",
    "spline_2",
    "spline_3",
    "spline_4",
    "zero",
    "next",
    "previous",
]

INTERPOLATION_METHODS_DICT = {}
"""dict: Dictionary of string interpolation methods and associated interpolator classes
   (i.e. ``'nearest': [NearestNeighbor, RasterioInterpolator, ScipyGrid]``) """


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


class InterpolationManager(object):
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
    _interpolation_params = None

    def __init__(self, definition=INTERPOLATION_DEFAULT):

        self.definition = deepcopy(definition)
        self.config = OrderedDict()
        self._interpolation_params = {}

        # if definition is None, set to default
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
                self.config = self._set_interpolation_method(udims, method)

            # set default if its not been specified in the dict
            if ("default",) not in self.config:
                existing_dims = set(v for k in self.config.keys() for v in k)  # Default is NOT allowed to adjust these
                name = ("default",)
                if len(existing_dims) > 0:
                    valid_dims = set(VALID_DIMENSION_NAMES)
                    default_dims = valid_dims - existing_dims
                    name = tuple(default_dims)

                default_method = self._parse_interpolation_method(INTERPOLATION_DEFAULT)
                self.config = self._set_interpolation_method(name, default_method)

        elif isinstance(definition, string_types):
            method = self._parse_interpolation_method(definition)
            self.config = self._set_interpolation_method(("default",), method)

        else:
            raise TypeError(
                '"{}" is not a valid interpolation definition type. '.format(definition)
                + "Interpolation definiton must be a string or list of dicts"
            )

        # make sure ('default',) is always the last entry in config dictionary
        if ("default",) in self.config:
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
            parms = {k: v for k, v in params.items() if hasattr(interpolator, k)}
            interpolators[idx] = interpolator(method=method, **parms)

        definition["interpolators"] = interpolators

        # Record parameters to make sure they are being captured
        self._interpolation_params.update({k: False for k in params})

        # set to interpolation configuration for dims
        self.config[udims] = definition
        return self.config

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
                if self.config[key]["method"] not in interpolator.methods_supported:
                    can_handle = tuple()
                else:
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
        if len(source_dims - handled_dims) > 0 and strict:
            missing_dims = list(source_dims - handled_dims)
            raise InterpolationException(
                "Dimensions {} ".format(missing_dims)
                + "can't be handled by interpolation definition:\n {}".format(self)
            )

        # TODO: adjust by interpolation cost
        return interpolator_queue

    def select_coordinates(self, source_coordinates, eval_coordinates, index_type="numpy"):
        """
        Select a subset or coordinates if interpolator can downselect.

        At this point in the execution process, podpac has selected a subset of source_coordinates that intersects
        with the requested coordinates, dropped extra dimensions from requested coordinates, and confirmed
        source coordinates are not missing any dimensions.

        Parameters
        ----------
        source_coordinates : :class:`podpac.Coordinates`
            Intersected source coordinates
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
            return source_coordinates, tuple([slice(0, None)] * len(source_coordinates.shape))

        interpolator_queue = self._select_interpolator_queue(source_coordinates, eval_coordinates, "can_select")

        self._last_select_queue = interpolator_queue

        # For heterogeneous selections, we need to select and then recontruct each set of dimensions
        selected_coords = {}
        selected_coords_idx = {k: np.arange(source_coordinates[k].size) for k in source_coordinates.dims}
        for udims in interpolator_queue:
            interpolator = interpolator_queue[udims]
            extra_dims = [d for d in source_coordinates.udims if d not in udims]
            sc = source_coordinates.udrop(extra_dims)
            # run interpolation. mutates selected coordinates and selected coordinates index
            sel_coords, sel_coords_idx = interpolator.select_coordinates(
                udims, sc, eval_coordinates, index_type=index_type
            )
            # Save individual 1-D coordinates for later reconstruction
            for i, k in enumerate(sel_coords.dims):
                selected_coords[k] = sel_coords[k]
                selected_coords_idx[k] = sel_coords_idx[i]

        # Reconstruct dimensions
        for d in source_coordinates.dims:
            if d not in selected_coords:  # Some coordinates may not have a selector when heterogeneous
                selected_coords[d] = source_coordinates[d]
            # np.ix_ call doesn't work with slices, and fancy numpy indexing does not work well with mixed slice/index
            if isinstance(selected_coords_idx[d], slice) and index_type != "slice":
                selected_coords_idx[d] = np.arange(source_coordinates[d].size)[selected_coords_idx[d]]

        selected_coords = Coordinates(
            [selected_coords[k] for k in source_coordinates.dims],
            source_coordinates.dims,
            crs=source_coordinates.crs,
            validate_crs=False,
        )
        if index_type == "numpy":
            npcoords = []
            has_stacked = False
            for k in source_coordinates.dims:
                # Deal with nD stacked source coords (marked by coords being in tuple)
                if isinstance(selected_coords_idx[k], tuple):
                    has_stacked = True
                    npcoords.extend([sci for sci in selected_coords_idx[k]])
                else:
                    npcoords.append(selected_coords_idx[k])
            if has_stacked:
                # When stacked coordinates are nD we cannot use the catchall of the next branch
                selected_coords_idx2 = npcoords
            else:
                # This would not be needed if everything went as planned in
                # interpolator.select_coordinates, but this is a catchall that works
                # for 90% of the cases
                selected_coords_idx2 = np.ix_(*[np.ravel(npc) for npc in npcoords])
        elif index_type == "xarray":
            selected_coords_idx2 = []
            for i in selected_coords.dims:
                # Deal with nD stacked source coords (marked by coords being in tuple)
                if isinstance(selected_coords_idx[i], tuple):
                    selected_coords_idx2.extend([xr.DataArray(sci, dims=[i]) for sci in selected_coords_idx[i]])
                else:
                    selected_coords_idx2.append(selected_coords_idx[i])
            selected_coords_idx2 = tuple(selected_coords_idx2)
        elif index_type == "slice":
            selected_coords_idx2 = []
            for i in selected_coords.dims:
                # Deal with nD stacked source coords (marked by coords being in tuple)
                if isinstance(selected_coords_idx[i], tuple):
                    selected_coords_idx2.extend(selected_coords_idx[i])
                else:
                    if isinstance(selected_coords_idx[i], np.ndarray):
                        # This happens when the interpolator_queue is empty, so we have to turn the
                        # initialized coordinates into slices instead of numpy arrays
                        selected_coords_idx2.append(
                            slice(selected_coords_idx[i].min(), selected_coords_idx[i].max() + 1)
                        )
                    else:
                        selected_coords_idx2.append(selected_coords_idx[i])

            selected_coords_idx2 = tuple(selected_coords_idx2)
        else:
            raise ValueError("Unknown index_type '%s'" % index_type)
        return selected_coords, tuple(selected_coords_idx2)

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

        # loop through multiple outputs if necessary
        if "output" in output_data.dims:
            for output in output_data.coords["output"]:
                output_data.sel(output=output)[:] = self.interpolate(
                    source_coordinates,
                    source_data.sel(output=output).drop("output"),
                    eval_coordinates,
                    output_data.sel(output=output).drop("output"),
                )
            return output_data

        ## drop already-selected output variable
        # if "output" in output_data.coords:
        # source_data = source_data.drop("output")
        # output_data = output_data.drop("output")

        # short circuit if the source data and requested coordinates are of shape == 1
        if source_data.size == 1 and eval_coordinates.size == 1:
            output_data.data[:] = source_data.data.flatten()[0]
            return output_data

        # short circuit if source_coordinates contains eval_coordinates
        # TODO handle stacked issubset of unstacked case
        #      this case is currently skipped because of the set(eval_coordinates) == set(source_coordinates)))
        if eval_coordinates.issubset(source_coordinates) and set(eval_coordinates) == set(source_coordinates):
            if any(isinstance(c, StackedCoordinates) and c.ndim > 1 for c in eval_coordinates.values()):
                # TODO AFFINE
                # currently this is bypassing the short-circuit in the shaped stacked coordinates case
                pass
            else:
                try:
                    data = source_data.interp(output_data.coords, method="nearest")
                except (NotImplementedError, ValueError):
                    try:
                        data = source_data.sel(output_data.coords[output_data.dims])
                    except KeyError:
                        # Since the output is a subset of the original data,
                        # we can just rely on xarray's broadcasting capability
                        # to subselect data, as the final fallback
                        output_data[:] = 0
                        data = source_data + output_data

                output_data.data[:] = data.transpose(*output_data.dims)
                return output_data

        interpolator_queue = self._select_interpolator_queue(
            source_coordinates, eval_coordinates, "can_interpolate", strict=True
        )

        # for debugging purposes, save the last defined interpolator queue
        self._last_interpolator_queue = interpolator_queue

        # reset interpolation parameters
        for k in self._interpolation_params:
            self._interpolation_params[k] = False

        # iterate through each dim tuple in the queue
        dtype = output_data.dtype
        attrs = source_data.attrs
        for udims, interpolator in interpolator_queue.items():
            # TODO move the above short-circuits into this loop
            if all([ud not in source_coordinates.udims for ud in udims]):
                # Skip this udim if it's not part of the source coordinates (can happen with default)
                continue
            # Check if parameters are being used
            for k in self._interpolation_params:
                self._interpolation_params[k] = hasattr(interpolator, k) or self._interpolation_params[k]

            # interp_coordinates are essentially intermediate eval_coordinates
            interp_dims = [dim for dim, c in source_coordinates.items() if set(c.dims).issubset(udims)]
            other_dims = [dim for dim, c in eval_coordinates.items() if not set(c.dims).issubset(udims)]
            interp_coordinates = merge_dims(
                [source_coordinates.drop(interp_dims), eval_coordinates.drop(other_dims)], validate_crs=False
            )
            interp_data = UnitsDataArray.create(interp_coordinates, dtype=dtype)
            interp_data = interpolator.interpolate(
                udims, source_coordinates, source_data, interp_coordinates, interp_data
            )

            # prepare for the next iteration
            source_data = interp_data.transpose(*interp_coordinates.xdims)
            source_data.attrs = attrs
            source_coordinates = interp_coordinates

        output_data.data = interp_data.transpose(*output_data.dims)

        # Throw warnings for unused parameters
        for k in self._interpolation_params:
            if self._interpolation_params[k]:
                continue
            _logger.warning("The interpolation parameter '{}' was ignored during interpolation.".format(k))

        return output_data

    def _fix_coordinates_for_none_interp(self, eval_coordinates, source_coordinates):
        interpolator_queue = self._select_interpolator_queue(
            source_coordinates, eval_coordinates, "can_interpolate", strict=True
        )
        if not any([isinstance(interpolator_queue[k], NoneInterpolator) for k in interpolator_queue]):
            # Nothing to do, just return eval_coordinates
            return eval_coordinates

        # Likely need to fix the output, since the shape of output will
        # not match the eval coordinates in most cases
        new_dims = []
        new_coords = []
        covered_udims = []
        for k in interpolator_queue:
            if not isinstance(interpolator_queue[k], NoneInterpolator):
                # Keep the eval_coordinates for these dimensions
                for d in eval_coordinates.dims:
                    ud = d.split("_")
                    for u in ud:
                        if u in k:
                            new_dims.append(d)
                            new_coords.append(eval_coordinates[d])
                            covered_udims.extend(ud)
                            break
            else:
                for d in source_coordinates.dims:
                    ud = d.split("_")
                    for u in ud:
                        if u in k:
                            new_dims.append(d)
                            new_coords.append(source_coordinates[d])
                            covered_udims.extend(ud)
                            break
        new_coordinates = Coordinates(new_coords, new_dims)
        return new_coordinates


class InterpolationTrait(tl.Union):
    default_value = INTERPOLATION_DEFAULT

    # .tag(attr=True, required=True, default = "linear")
    def __init__(
        self,
        trait_types=[tl.Dict(), tl.List(), tl.Enum(INTERPOLATION_METHODS), tl.Instance(InterpolationManager)],
        *args,
        **kwargs
    ):
        super(InterpolationTrait, self).__init__(trait_types=trait_types, *args, **kwargs)
