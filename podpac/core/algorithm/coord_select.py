"""
CoordSelect Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import traitlets as tl

from podpac.core.coordinates import Coordinates
from podpac.core.coordinates import UniformCoordinates1d, ArrayCoordinates1d
from podpac.core.coordinates import make_coord_value, make_coord_delta, add_coord
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
    input_coordinates : podpac.Coordinates
        The coordinates that were used to execute the node
    native_coordinates_source : podpac.Coordinates
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
    input_coordinates = tl.Instance(Coordinates, allow_none=True)
    lat = tl.List().tag(attr=True)
    lon = tl.List().tag(attr=True)
    time = tl.List().tag(attr=True)
    alt = tl.List().tag(attr=True)

    @property
    def native_coordinates(self):
        """Native coordinates of the source node, if available
        
        Returns
        -------
        podpac.Coordinates
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
        podpac.ArrayCoordinates1d
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
            raise ValueError("Invalid expansion attrs for '%s'" % dim)

        # get start and stop offsets
        dstart = make_coord_delta(coords[0])
        dstop = make_coord_delta(coords[1])

        if len(coords) == 2:
            # expand and use native coordinates
            ncoord = self.native_coordinates[dim]
            
            # TODO GroupCoord
            xcoords = [ncoord.select((add_coord(c, dstart), add_coord(c, dstop))) for c in icoords.coordinates]
            xcoord = sum(xcoords[1:], xcoords[0])

        elif len(coords) == 3:
            # or expand explicitly
            step = make_coord_delta(coords[2])
            
            # TODO GroupCoord
            xcoords = [
                UniformCoordinates1d(add_coord(c, dstart), add_coord(c, dstop), step, **c.properties)
                for c in icoords.coordinates]
            xcoord = sum(xcoords[1:], xcoords[0])

        assert xcoord.name is not None
        return xcoord

    def get_expanded_coordinates(self):
        """The expanded coordinates
        
        Returns
        -------
        podpac.Coordinates
            The expanded coordinates
        
        Raises
        ------
        ValueError
            Raised if expanded coordinates do not intersect with the source data. For example if a date in the future
            is selected.
        """
        coords = [self.get_expanded_coord(dim) for dim in self.input_coordinates.dims]
        return Coordinates(coords)
   
    def algorithm(self):
        """Passthrough of the source data
        
        Returns
        -------
        UnitDataArray
            Source evaluated at the expanded coordinates
        """
        return self.source.output
 
    @common_doc(COMMON_DOC)
    def execute(self, coordinates, output=None, method=None):
        """Executes this nodes using the supplied coordinates.

        Parameters
        ----------
        coordinates : podpac.Coordinates
            {requested_coordinates}
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
        self.input_coordinates = coordinates
        coordinates = self.get_expanded_coordinates()
        for dim, c in coordinates.unstacked.items():
            if c.size == 0:
                raise ValueError("Expanded/selected coordinates do not intersect with source data (dim '%s')" % dim)
        return super(ExpandCoordinates, self).execute(coordinates, output, method)


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
        podpac.ArrayCoordinates1d
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
            raise ValueError("Invalid expansion attrs for '%s'" % dim)

        # get start offset
        start = make_coord_value(coords[0])
        
        if len(coords) == 1:
            xcoord = ArrayCoordinates1d(start)
            
        elif len(coords) == 2:
            # Get stop offset
            stop = make_coord_value(coords[1])
            # select and use native coordinates
            ncoord = self.native_coordinates[dim]
            xcoord = ncoord.select([start, stop])

        elif len(coords) == 3:
            # Get stop offset
            stop = make_coord_value(coords[1])            
            # select explicitly
            step = make_coord_delta(coords[2])
            xcoord = UniformCoordinates1d(start, stop, step)

        return xcoord

