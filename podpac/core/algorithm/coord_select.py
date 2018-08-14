"""
Coord Select Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import traitlets as tl

from podpac.core.coordinate import Coordinate, UniformCoord, Coord
from podpac.core.coordinate import make_coord_value, make_coord_delta, add_coord
from podpac.core.node import Node, COMMON_NODE_DOC
from podpac.core.algorithm.algorithm import Algorithm
from podpac.core.utils import common_doc

COMMON_DOC = COMMON_NODE_DOC.copy()

class ExpandCoordinates(Algorithm):
    """Algorithm node used to expand requested coordinates. This is normally used in conjunction with a reduce operation
    to calculate, for example, the average temperature over the last month. While this is simple to do when evaluating
    a single node (just provide the coordinates), this functionality is needed for nodes buried deeper in a pipeline.
    
    Attributes
    ----------
    input_coordinates : podpac.Coordinate
        The coordinates that were used to execute the node
    native_coordinates_source : podpac.Coordinate
        The native coordinates of the source node whose coordinates are being expanded. This is needed in case the
        source doesn't have native coordinates (e.g. an Algorithm node).
    source : podpac.Node
        Source node that will be evaluated with the expanded coordinates.
    lat : List
        Expansion parameters for latitude. Format is ['start_offset', 'end_offset', 'step_size'].
    lon : List
        Expansion parameters for longitude. Format is ['start_offset', 'end_offset', 'step_size'].
    time : List
        Expansion parameters for time. Format is ['start_offset', 'end_offset', 'step_size'].
    alt : List
        Expansion parameters for altitude. Format is ['start_offset', 'end_offset', 'step_size'].
    """
    
    source = tl.Instance(Node)
    native_coordinates_source = tl.Instance(Node, allow_none=True)
    input_coordinates = tl.Instance(Coordinate, allow_none=True)
    lat = tl.List().tag(attr=True)
    lon = tl.List().tag(attr=True)
    time = tl.List().tag(attr=True)
    alt = tl.List().tag(attr=True)

    @property
    def native_coordinates(self):
        """Native coordinates of the source node, if available
        
        Returns
        -------
        podpac.Coordinate
            Native coordinates of the source node
        
        Raises
        ------
        Exception
            Is thrown if no suitable native_coordinates can be found
        """
        try:
            if self.native_coordinates_source:
                return self.native_coordinates_source.native_coordinates
            else:
                return self.source.native_coordinates
        except:
            raise Exception("no native coordinates found")

    def get_expanded_coord(self, dim):
        """Returns the expanded coordinates for the requested dimension
        
        Parameters
        ----------
        dim : str
            Dimension to expand
        
        Returns
        -------
        podpac.Coord
            Expanded coordinate
        
        Raises
        ------
        ValueError
            In case dimension is not in the parameters.
        """
        
        icoords = self.input_coordinates[dim]
        coords = getattr(self, dim)
        
        if not coords:  # i.e. if list is empty
            # no expansion in this dimension
            return icoords

        if len(coords) not in [2, 3]:
            raise ValueError("Invalid expansion params for '%s'" % dim)

        # get start and stop offsets
        dstart = make_coord_delta(coords[0])
        dstop = make_coord_delta(coords[1])

        if len(coords) == 2:
            # expand and use native coordinates
            ncoord = self.native_coordinates[dim]
            
            # TODO GroupCoord
            xcoords = [
                ncoord.select((add_coord(c, dstart), add_coord(c, dstop)))
                for c in icoords.coordinates
            ]
            xcoord = sum(xcoords[1:], xcoords[0])

        elif len(coords) == 3:
            # or expand explicitly
            delta = make_coord_delta(coords[2])
            
            # TODO GroupCoord
            xcoords = [
                UniformCoord(add_coord(c, dstart), add_coord(c, dstop), delta)
                for c in icoords.coordinates]
            xcoord = sum(xcoords[1:], xcoords[0])

        return xcoord

    @property
    def expanded_coordinates(self):
        """The expanded coordinates
        
        Returns
        -------
        podpac.Coordinate
            The expanded coordinates
        
        Raises
        ------
        ValueError
            Raised if expanded coordinates do not intersect with the source data. For example if a date in the future
            is selected.
        """
        kwargs = {}
        for dim in self.input_coordinates.dims:
            ec = self.get_expanded_coord(dim)
            if ec.size == 0:
                raise ValueError("Expanded/selected coordinates do not"
                                 " intersect with source data.")
            kwargs[dim] = ec
        kwargs['order'] = self.input_coordinates.dims
        return Coordinate(**kwargs)
   
    def algorithm(self):
        """Passthrough of the source data
        
        Returns
        -------
        UnitDataArray
            Source evaluated at the expanded coordinates
        """
        return self.source.output
 
    @common_doc(COMMON_DOC)
    def execute(self, coordinates, params=None, output=None, method=None):
        """Executes this nodes using the supplied coordinates and params. 

        Parameters
        ----------
        coordinates : podpac.Coordinate
            {evaluated_coordinates}
        params : dict, optional
            {execute_params} 
        output : podpac.UnitsDataArray, optional
            {execute_out}
        method : str, optional
            {execute_method}
            
        Returns
        -------
        {execute_return}
        
        Notes
        -------
        The input coordinates are modified and the passed to the base class implementation of execute.
        """
        self._params = self.get_params(params)
        self.input_coordinates = coordinates
        coordinates = self.expanded_coordinates

        return super(ExpandCoordinates, self).execute(coordinates, params, output, method)


class SelectCoordinates(ExpandCoordinates):
    """Algorithm node used to select coordinates different from the input coordinates. While this is simple to do when 
    evaluating a single node (just provide the coordinates), this functionality is needed for nodes buried deeper in a 
    pipeline. For example, if a single spatial reference point is used for a particular comparison, and this reference
    point is different than the requested coordinates, we need to explicitly select those coordinates using this Node. 
    
    """
    
    def get_expanded_coord(self, dim):
        """Function name is a misnomer -- should be get_selected_coord, but we are using a lot of the
        functionality of the ExpandCoordinates node. 
        
        Parameters
        ----------
        dim : str
            Dimension for doing the selection
        
        Returns
        -------
        podpac.Coord
            The selected coordinate
        
        Raises
        ------
        ValueError
            Description
        """
        icoords = self.input_coordinates[dim]
        coords = getattr(self, dim)
        
        if not coords:
            # no selection in this dimension
            return icoords

        if len(coords) not in [1, 2, 3]:
            raise ValueError("Invalid expansion params for '%s'" % dim)

        # get start offset
        start = make_coord_value(coords[0])
        
        if len(coords) == 1:
            xcoord = Coord(start)
            
        elif len(coords) == 2:
            # Get stop offset
            stop = make_coord_value(coords[1])
            # select and use native coordinates
            ncoord = self.native_coordinates[dim]
            xcoord = ncoord.select([start, stop])

        elif len(coords) == 3:
            # Get stop offset
            stop = make_coord_value(coords[1])            
            # or select explicitly
            delta = make_coord_delta(coords[2])
            xcoord = UniformCoord(start, stop, delta)

        return xcoord

