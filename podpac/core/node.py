import xarray as xr
import numpy as np
import traitlets as tl

from coordinate import Coordinate

class Node(tl.HasTraits):

    output = tl.Instance(xr.Dataset, allow_none=True)
    native_coordinates = tl.Instance(Coordinate)
    evaluted = tl.Bool(default_value=False)
    evaluated_coordinates = tl.Instance(Coordinate)
    params = tl.Dict(default_value=None, allow_none=True)


    def __init__(self, *args, **kwargs):
        """ Do not overwrite me """
        targs, tkwargs = self._first_init(*args, **kwargs)
        super(Node, self).__init__(*targs, **tkwargs)
        self.init()

    def _first_init(*args, **kwargs):
        """ Only overwrite me if absolutely necessary """
        return args, kwargs

    def init(self):
        pass

    def execute(self, coordinates, params=None, output=None):
        """ This is the common interface used for ALL nodes. Pipelines only
        understand this and get_description. 
        """
        raise NotImplementedError

    def _execute_common(self, coordinates, params=None, output=None):
        """ 
        Common input sanatization etc for when executing a node 
        """
        if output is not None:
            # This should be a reference, not a copy
            # subselect if neccessary
            out = output[coordinates.get_coord] 

        return coordinates, params, out


    def get_description(self):
        """
        This is to get the pipeline lineage or provenance
        """
        raise NotImplementedError

    def get_intersecting_coordinates(self, evaluated=None, native=None):
        """ Helper function to get the reqions where the requested and
        native coordinates intersect.

        Parameters
        -------------
        evaluated: Coordinate
            Coordinates where the Node should be evaluated
        native: Coordinate
            The Node's native Coordinates

        Returns
        ---------
        en_intersect: Coordinate
            The coordinates of the overlap at the resolution/projection/scale
            of the evaluated Coordinate object
        ne_intersect: Coordinate
            Like en_intersect, but at the resolution/projection/scale of the
            native coordinates
        """
        if evaluate is None and self.evaluated:
            evaluated = self.evaluated_coordinates
        if native is None:
            native = self.native_coordinates

        en_intersect = evaluated.intersect(native)
        ne_intersect = native.intersect(evaluated)

        return en_intersect, ne_intersect

    
