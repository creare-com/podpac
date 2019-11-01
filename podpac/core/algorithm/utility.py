"""
Utility Algorithm Nodes.
These nodes are mainly used for testing.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np
import traitlets as tl

# Internal dependencies
from podpac.core.coordinates import Coordinates
from podpac.core.algorithm.algorithm import Algorithm


class Arange(Algorithm):
    """A simple test node that gives each value in the output a number.
    """

    def algorithm(self, inputs):
        """Uses np.arange to give each value in output a unique number
        
        Arguments
        ---------
        inputs : dict
            Unused, should be empty for this algorithm.

        Returns
        -------
        UnitsDataArray
            A row-majored numbered array of the requested size. 
        """
        data = np.arange(self._requested_coordinates.size).reshape(self._requested_coordinates.shape)
        return self.create_output_array(self._requested_coordinates, data=data)


class CoordData(Algorithm):
    """Extracts the coordinates from a request and makes it available as a data
    
    Attributes
    ----------
    coord_name : str
        Name of coordinate to extract (one of lat, lon, time, alt)
    """

    coord_name = tl.Unicode("").tag(attr=True)

    def algorithm(self, inputs):
        """Extract coordinate from request and makes data available.
        
        Arguments
        ----------
        inputs : dict
            Unused, should be empty for this algorithm.

        Returns
        -------
        UnitsDataArray
            The coordinates as data for the requested coordinate.
        """

        if self.coord_name not in self._requested_coordinates.udims:
            raise ValueError("Coordinate name not in evaluated coordinates")

        c = self._requested_coordinates[self.coord_name]
        coords = Coordinates([c])
        return self.create_output_array(coords, data=c.coordinates)


class SinCoords(Algorithm):
    """A simple test node that creates a data based on coordinates and trigonometric (sin) functions. 
    """

    def algorithm(self, inputs):
        """Computes sinusoids of all the coordinates. 
        
        Arguments
        ----------
        inputs : dict
            Unused, should be empty for this algorithm.

        Returns
        -------
        UnitsDataArray
            Sinusoids of a certain period for all of the requested coordinates
        """
        out = self.create_output_array(self._requested_coordinates, data=1.0)
        crds = list(out.coords.values())
        try:
            i_time = list(out.coords.keys()).index("time")
            crds[i_time] = crds[i_time].astype("datetime64[h]").astype(float)
        except ValueError:
            pass

        crds = np.meshgrid(*crds, indexing="ij")
        for crd in crds:
            out *= np.sin(np.pi * crd / 90.0)
        return out
