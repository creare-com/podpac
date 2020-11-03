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

# Set up logging
_log = logging.getLogger(__name__)

from podpac.core.utils import common_doc

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
            Maximum distance to the nearest coordinate in space.
            Cooresponds to the unit of the space measurement.
        time_tolerance : float
            Maximum distance to the nearest coordinate in time coordinates.
            Accepts p.timedelta64() (i.e. np.timedelta64(1, 'D') for a 1-Day tolerance)
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
    spatial_tolerance = tl.Float(allow_none=True, default_value=np.inf)

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
        self, func, keep_dims, udims, source_coordinates, source_data, eval_coordinates, output_data, **kwargs
    ):
        loop_dims = [d for d in source_data.dims if d not in keep_dims]
        if loop_dims:
            dim = loop_dims[0]
            for i in output_data.coords[dim]:
                idx = {dim: i}

                # TODO: handle this using presecribed interpolation method instead of "nearest"
                if not i.isin(source_data.coords[dim]):
                    if self.method != "nearest":
                        _log.warning(
                            "Interpolation method {} is not supported yet in this context. Using 'nearest' for {}".format(
                                self.method, dim
                            )
                        )

                    # find the closest value
                    if dim == "time":
                        tol = self.time_tolerance
                    else:
                        tol = self.spatial_tolerance

                    diff = np.abs(source_data.coords[dim].values - i.values)
                    if tol == None or tol == "" or np.any(diff <= tol):
                        src_i = (diff).argmin()
                        src_idx = {dim: source_data.coords[dim][src_i]}
                    else:
                        src_idx = None  # There is no closest neighbor within the tolerance
                        continue

                else:
                    src_idx = idx

                output_data.loc[idx] = self._loop_helper(
                    func,
                    keep_dims,
                    udims,
                    source_coordinates.drop(dim),
                    source_data.loc[src_idx],
                    eval_coordinates.drop(dim),
                    output_data.loc[idx],
                    **kwargs
                )

        else:
            # TODO does this allow undesired extrapolation?
            # short circuit if the source data and requested coordinates are of size 1
            if source_data.size == 1 and eval_coordinates.size == 1:
                output_data.data[:] = source_data.data.flatten()[0]
                return output_data

            # short circuit if source_coordinates contains eval_coordinates
            if eval_coordinates.issubset(source_coordinates):
                # select/transpose, and copy
                d = {}
                for k, c in source_coordinates.items():
                    if isinstance(c, Coordinates1d):
                        d[k] = output_data[k].data
                    elif isinstance(c, StackedCoordinates):
                        bs = [np.isin(c[dim].coordinates, eval_coordinates[dim].coordinates) for dim in c.dims]
                        b = np.logical_and.reduce(bs)
                        d[k] = source_data[k].data[b]

                if all(isinstance(c, Coordinates1d) for c in source_coordinates.values()):
                    method = "nearest"
                else:
                    method = None

                output_data[:] = source_data.sel(output_data.coords, method=method)
                return output_data

            return func(udims, source_coordinates, source_data, eval_coordinates, output_data, **kwargs)

        return output_data

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def can_select(self, udims, source_coordinates, eval_coordinates):
        """
        {interpolator_can_select}
        """

        return tuple()

    @common_doc(COMMON_INTERPOLATOR_DOCS)
    def select_coordinates(self, udims, source_coordinates, source_coordinates_index, eval_coordinates):
        """
        {interpolator_select}
        """
        raise NotImplementedError

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
