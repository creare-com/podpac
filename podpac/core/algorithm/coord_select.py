"""
CoordSelect Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import traitlets as tl
import numpy as np

from podpac.core.settings import settings
from podpac.core.coordinates import Coordinates
from podpac.core.coordinates import UniformCoordinates1d, ArrayCoordinates1d
from podpac.core.coordinates import make_coord_value, make_coord_delta, add_coord
from podpac.core.node import Node, COMMON_NODE_DOC
from podpac.core.algorithm.algorithm import UnaryAlgorithm
from podpac.core.utils import common_doc, NodeTrait

COMMON_DOC = COMMON_NODE_DOC.copy()


class ModifyCoordinates(UnaryAlgorithm):
    """
    Base class for nodes that modify the requested coordinates before evaluation.

    Attributes
    ----------
    source : podpac.Node
        Source node that will be evaluated with the modified coordinates.
    coordinates_source : podpac.Node
        Node that supplies the available coordinates when necessary, optional. The source node is used by default.
    lat, lon, time, alt : List
        Modification parameters for given dimension. Varies by node.
    """

    coordinates_source = NodeTrait().tag(attr=True)
    lat = tl.List().tag(attr=True)
    lon = tl.List().tag(attr=True)
    time = tl.List().tag(attr=True)
    alt = tl.List().tag(attr=True)
    substitute_eval_coords = tl.Bool(False).tag(attr=True)

    _modified_coordinates = tl.Instance(Coordinates, allow_none=True)

    @tl.default("coordinates_source")
    def _default_coordinates_source(self):
        return self.source

    @common_doc(COMMON_DOC)
    def _eval(self, coordinates, output=None, _selector=None):
        """Evaluates this nodes using the supplied coordinates.

        Parameters
        ----------
        coordinates : podpac.Coordinates
            {requested_coordinates}
        output : podpac.UnitsDataArray, optional
            {eval_output}
        _selector: callable(coordinates, request_coordinates)
            {eval_selector}

        Returns
        -------
        {eval_return}

        Notes
        -------
        The input coordinates are modified and the passed to the base class implementation of eval.
        """

        self._requested_coordinates = coordinates
        self._modified_coordinates = Coordinates(
            [self.get_modified_coordinates1d(coordinates, dim) for dim in coordinates.dims],
            crs=coordinates.crs,
            validate_crs=False,
        )

        for dim in self._modified_coordinates.udims:
            if self._modified_coordinates[dim].size == 0:
                raise ValueError("Modified coordinates do not intersect with source data (dim '%s')" % dim)

        outputs = {}
        outputs["source"] = self.source.eval(self._modified_coordinates, output=output, _selector=_selector)

        if self.substitute_eval_coords:
            dims = outputs["source"].dims
            coords = self._requested_coordinates
            extra_dims = [d for d in coords.dims if d not in dims]
            coords = coords.drop(extra_dims)

            outputs["source"] = outputs["source"].assign_coords(**coords.xcoords)

        if output is None:
            output = outputs["source"]
        else:
            output[:] = outputs["source"]

        if settings["DEBUG"]:
            self._output = output
        return output


class ExpandCoordinates(ModifyCoordinates):
    """Evaluate a source node with expanded coordinates.

    This is normally used in conjunction with a reduce operation
    to calculate, for example, the average temperature over the last month. While this is simple to do when evaluating
    a single node (just provide the coordinates), this functionality is needed for nodes buried deeper in a pipeline.

    lat, lon, time, alt : List
        Expansion parameters for the given dimension: The options are::
         * [start_offset, end_offset, step] to expand uniformly around each input coordinate.
         * [start_offset, end_offset] to expand using the available source coordinates around each input coordinate.

    bounds_only: bool
        Default is False. If True, will only expand the bounds of the overall coordinates request. Otherwise, it will
        expand around EACH coordinate in the request. For example, with bounds_only == True, and an expansion of 3
        you may expand [5, 6, 8] to [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], whereas with bounds_only == False, it becomes
        [[2, 5, 8], [3, 6, 9], [5, 8, 11]] (brackets added for clarity, they will be concatenated).
    """

    substitute_eval_coords = tl.Bool(False, read_only=True)
    bounds_only = tl.Bool(False).tag(attr=True)

    def get_modified_coordinates1d(self, coords, dim):
        """Returns the expanded coordinates for the requested dimension, depending on the expansion parameter for the
        given dimension.

        Parameters
        ----------
        coords : Coordinates
            The requested input coordinates
        dim : str
            Dimension to expand

        Returns
        -------
        expanded : :class:`podpac.coordinates.Coordinates1d`
            Expanded coordinates
        """

        coords1d = coords[dim]
        expansion = getattr(self, dim)

        if not expansion:  # i.e. if list is empty
            # no expansion in this dimension
            return coords1d

        if len(expansion) == 2:
            # use available coordinates
            dstart = make_coord_delta(expansion[0])
            dstop = make_coord_delta(expansion[1])

            available_coordinates = self.coordinates_source.find_coordinates()
            if len(available_coordinates) != 1:
                raise ValueError("Cannot implicity expand coordinates; too many available coordinates")
            acoords = available_coordinates[0][dim]
            if self.bounds_only:
                cs = [
                    acoords.select(
                        add_coord(coords1d.coordinates[0], dstart), add_coord(coords1d.coordinates[-1], dstop)
                    )
                ]
            else:
                cs = [acoords.select((add_coord(x, dstart), add_coord(x, dstop))) for x in coords1d.coordinates]

        elif len(expansion) == 3:
            # use a explicit step size
            dstart = make_coord_delta(expansion[0])
            dstop = make_coord_delta(expansion[1])
            step = make_coord_delta(expansion[2])
            if self.bounds_only:
                cs = [
                    UniformCoordinates1d(
                        add_coord(coords1d.coordinates[0], dstart), add_coord(coords1d.coordinates[-1], dstop), step
                    )
                ]
            else:
                cs = [
                    UniformCoordinates1d(add_coord(x, dstart), add_coord(x, dstop), step) for x in coords1d.coordinates
                ]

        else:
            raise ValueError("Invalid expansion attrs for '%s'" % dim)

        return ArrayCoordinates1d(np.concatenate([c.coordinates for c in cs]), **coords1d.properties)


class SelectCoordinates(ModifyCoordinates):
    """Evaluate a source node with select coordinates.

    While this is simple to do when
    evaluating a single node (just provide the coordinates), this functionality is needed for nodes buried deeper in a
    pipeline. For example, if a single spatial reference point is used for a particular comparison, and this reference
    point is different than the requested coordinates, we need to explicitly select those coordinates using this Node.

    lat, lon, time, alt : List
        Selection parameters for the given dimension: The options are::
         * [value]: select this coordinate value
         * [start, stop]: select the available source coordinates within the given bounds
         * [start, stop, step]: select uniform coordinates defined by the given start, stop, and step
    """

    def get_modified_coordinates1d(self, coords, dim):
        """
        Get the desired 1d coordinates for the given dimension, depending on the selection attr for the given
        dimension::

        Parameters
        ----------
        coords : Coordinates
            The requested input coordinates
        dim : str
            Dimension for doing the selection

        Returns
        -------
        coords1d : ArrayCoordinates1d
            The selected coordinates for the given dimension.
        """

        coords1d = coords[dim]
        selection = getattr(self, dim)

        if not selection:
            # no selection in this dimension
            return coords1d

        if len(selection) == 1 or ((len(selection) == 2) and (selection[0] == selection[1])):
            # a single value
            coords1d = ArrayCoordinates1d(selection, **coords1d.properties)

        elif len(selection) == 2:
            # use available source coordinates within the selected bounds
            available_coordinates = self.coordinates_source.find_coordinates()
            if len(available_coordinates) != 1:
                raise ValueError(
                    "SelectCoordinates Node cannot determine the step size between bounds for dimension"
                    + "{} because source node (source.find_coordinates()) has {} different coordinates.".format(
                        dim, len(available_coordinates)
                    )
                    + "Please specify step-size for this dimension."
                )
            coords1d = available_coordinates[0][dim].select(selection)

        elif len(selection) == 3:
            # uniform coordinates using start, stop, and step
            coords1d = UniformCoordinates1d(*selection, **coords1d.properties)

        else:
            raise ValueError("Invalid selection attrs for '%s'" % dim)

        return coords1d


class YearSubstituteCoordinates(ModifyCoordinates):
    year = tl.Unicode().tag(attr=True)

    # Remove tags from attributes
    lat = tl.List()
    lon = tl.List()
    time = tl.List()
    alt = tl.List()
    coordinates_source = None

    def get_modified_coordinates1d(self, coord, dim):
        """
        Get the desired 1d coordinates for the given dimension, depending on the selection attr for the given
        dimension::

        Parameters
        ----------
        coords : Coordinates
            The requested input coordinates
        dim : str
            Dimension for doing the selection

        Returns
        -------
        coords1d : ArrayCoordinates1d
            The selected coordinates for the given dimension.
        """
        if dim != "time":
            return coord[dim]
        times = coord["time"]
        delta = np.datetime64(self.year)
        new_times = [add_coord(c, delta - c.astype("datetime64[Y]")) for c in times.coordinates]

        return ArrayCoordinates1d(new_times, name="time")


class TransformTimeUnits(ModifyCoordinates):
    time_units = tl.Enum(
        [
            "day",
            "dayofweek",
            "dayofyear",
            "daysinmonth",
            "microsecond",
            "minute",
            "month",
            "nanosecond",
            "quarter",
            "season",
            "second",
            "time",
            "week",
            "weekday",
            "weekday_name",
            "weekofyear",
            "year",
        ]
    ).tag(attr=True)

    # Remove tags from attributes
    lat = tl.List()
    lon = tl.List()
    time = tl.List()
    alt = tl.List()
    coordinates_source = None

    def get_modified_coordinates1d(self, coords, dim):
        """
        Get the desired 1d coordinates for the given dimension, depending on the selection attr for the given
        dimension::

        Parameters
        ----------
        coords : Coordinates
            The requested input coordinates
        dim : str
            Dimension for doing the selection

        Returns
        -------
        coords1d : ArrayCoordinates1d
            The selected coordinates for the given dimension.
        """
        if dim != "time":
            return coords[dim]

        return coords.transform_time(self.time_units)["time"]
