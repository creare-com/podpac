"""
Abstract class for all Interpolator implementations

Attributes
----------
COMMON_INTERPOLATOR_DOCS : dict
    Documentation prototype for interpolators
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import logging

import numpy as np
import traitlets as tl
import six
from podpac.core.utils import common_doc
from podpac.core.interpolation.selector import Selector

# Set up logging
_log = logging.getLogger(__name__)


COMMON_INTERPOLATOR_DOCS = {
    "interpolator_attributes": """
        method : str
            Current interpolation method to use in Interpolator (i.e. 'nearest').
            This attribute is set during node evaluation when a new :class:`Interpolation`
            class is constructed. See the :class:`podpac.data.DataSource` `interpolation` attribute for
            more information on specifying the interpolator method.
        methods_supported : list
            List of methods supported by the interpolator.
            This attribute should be defined by the implementing :class:`Interpolator`.
            See :attr:`podpac.data.INTERPOLATION_METHODS` for list of available method strings.
        dims_supported : list
            List of unstacked dimensions supported by the interpolator.
            This attribute should be defined by the implementing :class:`Interpolator`.
            Used by private convience method :meth:`_filter_udims_supported`.
        """,
    "nearest_neighbor_attributes": """
        Attributes
        ----------
        method : str
            Current interpolation method to use in Interpolator (i.e. 'nearest').
            This attribute is set during node evaluation when a new :class:`Interpolation`
            class is constructed. See the :class:`podpac.data.DataSource` `interpolation` attribute for
            more information on specifying the interpolator method.
        dims_supported : list
            List of unstacked dimensions supported by the interpolator.
            This attribute should be defined by the implementing :class:`Interpolator`.
            Used by private convience method :meth:`_filter_udims_supported`.
        spatial_tolerance : float
            Default is inf. Maximum distance to the nearest coordinate in space.
            Cooresponds to the unit of the space measurement.
        time_tolerance : float
            Default is inf. Maximum distance to the nearest coordinate in time coordinates.
            Accepts p.timedelta64() (i.e. np.timedelta64(1, 'D') for a 1-Day tolerance)
        alt_tolerance : float
            Default is inf. Maximum distance to the nearest coordinate in altitude coordinates. Corresponds to the unit
            of the altitude as part of the requested coordinates
        spatial_scale : float
            Default is 1. This only applies when the source has stacked dimensions with different units.
            The spatial_scale defines the factor that lat, lon coordinates will be scaled by (coordinates are divided by spatial_scale)
            to output a valid distance for the combined set of dimensions.
        time_scale : float
            Default is 1. This only applies when the source has stacked dimensions with different units.
            The time_scale defines the factor that time coordinates will be scaled by (coordinates are divided by time_scale)
            to output a valid distance for the combined set of dimensions.
        alt_scale : float
            Default is 1. This only applies when the source has stacked dimensions with different units.
            The alt_scale defines the factor that alt coordinates will be scaled by (coordinates are divided by alt_scale)
            to output a valid distance for the combined set of dimensions.
        respect_bounds : bool
            Default is True. If True, any requested dimension OUTSIDE of the bounds will be interpolated as 'nan'.
            Otherwise, any point outside the bounds will have NN interpolation allowed.
        remove_nan: bool
            Default is False. If True, nan's in the source dataset will NOT be interpolated. This can be used if a value for the function
            is needed at every point of the request. It is not helpful when computing statistics, where nan values will be explicitly
            ignored. In that case, if remove_nan is True, nan values will take on the values of neighbors, skewing the statistical result.
        use_selector: bool
            Default is True. If True, a subset of the coordinates will be selected BEFORE the data of a dataset is retrieved. This
            reduces the number of data retrievals needed for large datasets. In cases where remove_nan = True, the selector may select
            only nan points, in which case the interpolation fails to produce non-nan data. This usually happens when requesting a single
            point from a dataset that contains nans. As such, in these cases set use_selector = False to get a non-nan value.

        """,
    "interpolator_can_select": """
        Evaluate if interpolator can downselect the source coordinates from the requested coordinates
        for the unstacked dims supplied.
        If not overwritten, this method returns an empty tuple (``tuple()``)

        Parameters
        ----------
        udims : tuple
            dimensions to select
        source_coordinates : :class:`podpac.Coordinates`
            Description
        eval_coordinates : :class:`podpac.Coordinates`
            Description

        Returns
        -------
        tuple
            Returns a tuple of dimensions that can be selected with this interpolator
            If no dimensions can be selected, method should return an emtpy tuple
        """,
    "interpolator_select": """
        Downselect coordinates with interpolator method

        Parameters
        ----------
        udims : tuple
            dimensions to select coordinates
        source_coordinates : :class:`podpac.Coordinates`
            Description
        source_coordinates_index : list
            Description
        eval_coordinates : :class:`podpac.Coordinates`
            Description

        Returns
        -------
        (:class:`podpac.Coordinates`, list)
            returns the new down selected coordinates and the new associated index. These coordinates must exist
            in the coordinates of the source data

        Raises
        ------
        NotImplementedError
        """,
    "interpolator_can_interpolate": """
        Evaluate if this interpolation method can handle the requested coordinates and source_coordinates.
        If not overwritten, this method returns an empty tuple (`tuple()`)

        Parameters
        ----------
        udims : tuple
            dimensions to interpolate
        source_coordinates : :class:`podpac.Coordinates`
            Description
        eval_coordinates : :class:`podpac.Coordinates`
            Description

        Returns
        -------
        tuple
            Returns a tuple of dimensions that can be interpolated with this interpolator
            If no dimensions can be interpolated, method should return an emtpy tuple
        """,
    "interpolator_interpolate": """
        Interpolate data from requested coordinates to source coordinates.

        Parameters
        ----------
        udims : tuple
            dimensions to interpolate
        source_coordinates : :class:`podpac.Coordinates`
            Description
        source_data : podpac.core.units.UnitsDataArray
            Description
        eval_coordinates : :class:`podpac.Coordinates`
            Description
        output_data : podpac.core.units.UnitsDataArray
            Description

        Raises
        ------
        NotImplementedError

        Returns
        -------
        podpac.core.units.UnitDataArray
            returns the updated output of interpolated data
        """,
}
"""dict : Common interpolate docs """


class InterpolatorException(Exception):
    """
    Custom label for interpolator exceptions
    """

    pass


@common_doc(COMMON_INTERPOLATOR_DOCS)
class Interpolator(tl.HasTraits):
    """Interpolation Method

    Attributes
    ----------
    {interpolator_attributes}

    """

    # defined by implementing Interpolator class
    methods_supported = tl.List(tl.Unicode())
    dims_supported = tl.List(tl.Unicode())

    # defined at instantiation
    method = tl.Unicode()

    # Next are used for optimizing the interpolation pipeline
    # If -1, it's cost is assume the same as a competing interpolator in the
    # stack, and the determination is made based on the number of DOF before
    # and after each interpolation step.
    # cost_func = tl.CFloat(-1)  # The rough cost FLOPS/DOF to do interpolation
    # cost_setup = tl.CFloat(-1)  # The rough cost FLOPS/DOF to set up the interpolator

    def __init__(self, **kwargs):

        # Call traitlets constructor
        super(Interpolator, self).__init__(**kwargs)

        # check method
        if len(self.methods_supported) and self.method not in self.methods_supported:
            raise InterpolatorException("Method {} is not supported by Interpolator {}".format(self.method, self.name))
        self.init()

    def __repr__(self):
        return "{} ({})".format(self.name, self.method)

    @property
    def name(self):
        """
        Interpolator definition

        Returns
        -------
        str
            String name of interpolator.
        """
        return str(self.__class__.__name__)

    @property
    def definition(self):
        """
        Interpolator definition

        Returns
        -------
        str
            String name of interpolator.
        """
        return self.name

    def init(self):
        """
        Overwrite this method if a Interpolator needs to do any
        additional initialization after the standard initialization.
        """
        pass

    def _filter_udims_supported(self, udims):

        # find the intersection between dims_supported and udims, return tuple of intersection
        return tuple(set(self.dims_supported) & set(udims))

    def _dim_in(self, dim, *coords, **kwargs):
        """Verify the dim exists on coordinates

        Parameters
        ----------
        dim : str, list of str
            Dimension or list of dimensions to verify
        *coords :class:`podpac.Coordinates`
            coordinates to evaluate
        unstacked : bool, optional
            True if you want to compare dimensions in unstacked form, otherwise compare dimensions however
            they are defined on the DataSource. Defaults to False.

        Returns
        -------
        Boolean
            True if the dim is in all input coordinates
        """

        unstacked = kwargs.pop("unstacked", False)

        if isinstance(dim, six.string_types):
            dim = [dim]
        elif not isinstance(dim, (list, tuple)):
            raise ValueError("`dim` input must be a str, list of str, or tuple of str")

        for coord in coords:
            for d in dim:
                if (unstacked and d not in coord.udims) or (not unstacked and d not in coord.dims):
                    return False

        return True

    def _loop_helper(
        self, func, interp_dims, udims, source_coordinates, source_data, eval_coordinates, output_data, **kwargs
    ):
        """In cases where the interpolator can only handle a limited number of dimensions, loop over the extra ones
        Parameters
        ----------
        func : callable
            The interpolation function that should be called on the data subset. Should have the following arguments:
            func(udims, source_coordinates, source_data, eval_coordinates, output_data)
        interp_dims: list(str)
            List of source dimensions that will be interpolator. The looped dimensions will be computed
        udims: list(str)
           The unstacked coordinates that this interpolator handles
        source_coordinates: podpac.Coordinates
            The coordinates of the source data
        eval_coordinates: podpac.Coordinates
            The user-requested or evaluated coordinates
        output_data: podpac.UnitsDataArray
            Container for the output of the interpolation function
        """
        loop_dims = [d for d in source_data.dims if d not in interp_dims]
        if not loop_dims:  # Do the actual interpolation
            return func(udims, source_coordinates, source_data, eval_coordinates, output_data, **kwargs)

        dim = loop_dims[0]
        for i in output_data.coords[dim]:
            idx = {dim: i}

            if not i.isin(source_data.coords[dim]):
                # This case should have been properly handled in the interpolation_manager
                raise InterpolatorException("Unexpected interpolation error")

            output_data.loc[idx] = self._loop_helper(
                func,
                interp_dims,
                udims,
                source_coordinates.drop(dim),
                source_data.loc[idx],
                eval_coordinates.drop(dim),
                output_data.loc[idx],
                **kwargs
            )
        return output_data

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def can_select(self, udims, source_coordinates, eval_coordinates):
        """
        {interpolator_can_select}
        """
        if not (self.method in Selector.supported_methods):
            return tuple()

        udims_subset = self._filter_udims_supported(udims)
        return udims_subset

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def select_coordinates(self, udims, source_coordinates, eval_coordinates, index_type="numpy"):
        """
        {interpolator_select}
        """
        selector = Selector(method=self.method)
        return selector.select(source_coordinates, eval_coordinates, index_type=index_type)

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def can_interpolate(self, udims, source_coordinates, eval_coordinates):
        """
        {interpolator_can_interpolate}
        """
        return tuple()

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def interpolate(self, udims, source_coordinates, source_data, eval_coordinates, output_data):
        """
        {interpolator_interpolate}
        """
        raise NotImplementedError
