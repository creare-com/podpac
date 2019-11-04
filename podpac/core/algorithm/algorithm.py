"""
Base class for Algorithm Nodes
"""

from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
import inspect

import numpy as np
import xarray as xr

# Internal dependencies
from podpac.core.coordinates import Coordinates, union
from podpac.core.units import UnitsDataArray
from podpac.core.node import Node
from podpac.core.node import NodeException
from podpac.core.node import COMMON_NODE_DOC
from podpac.core.node import node_eval
from podpac.core.utils import common_doc

COMMON_DOC = COMMON_NODE_DOC.copy()


class Algorithm(Node):
    """Base class for algorithm and computation nodes.
    
    Notes
    ------
    Developers of new Algorithm nodes need to implement the `algorithm` method. 
    """

    @property
    def _inputs(self):
        # this first version is nicer, but the gettattr(self, ref) can take a
        # a long time if it is has a default value or is a property

        # return = {
        #     ref:getattr(self, ref)
        #     for ref in self.trait_names()
        #     if isinstance(getattr(self, ref, None), Node)
        # }

        return {
            ref: getattr(self, ref)
            for ref, trait in self.traits().items()
            if hasattr(trait, "klass") and Node in inspect.getmro(trait.klass) and getattr(self, ref) is not None
        }

    @common_doc(COMMON_DOC)
    @node_eval
    def eval(self, coordinates, output=None):
        """Evalutes this nodes using the supplied coordinates. 
        
        Parameters
        ----------
        coordinates : podpac.Coordinates
            {requested_coordinates}
        output : podpac.UnitsDataArray, optional
            {eval_output}
        
        Returns
        -------
        {eval_return}
        """

        self._requested_coordinates = coordinates

        inputs = {}
        for key, node in self._inputs.items():
            inputs[key] = node.eval(coordinates)

        # accumulate output coordinates
        coords_list = [Coordinates.from_xarray(a.coords, crs=a.attrs.get("crs")) for a in inputs.values()]
        output_coordinates = union([coordinates] + coords_list)

        result = self.algorithm(inputs)
        if isinstance(result, np.ndarray):
            if output is None:
                output = self.create_output_array(output_coordinates, data=result)
            else:
                output.data[:] = result
        elif isinstance(result, xr.DataArray):
            if output is None:
                output = self.create_output_array(
                    Coordinates.from_xarray(result.coords, crs=result.attrs.get("crs")), data=result.data
                )
            else:
                output[:] = result.data
        elif isinstance(result, UnitsDataArray):
            if output is None:
                output = result
            else:
                output[:] = result
        else:
            raise NodeException

        return output

    def find_coordinates(self):
        """
        Get the available native coordinates for the inputs to the Node.

        Returns
        -------
        coords_list : list
            list of available coordinates (Coordinate objects)
        """

        return [c for node in self._inputs.values() for c in node.find_coordinates()]

    def algorithm(self, inputs):
        """
        Arguments
        ----------
        inputs : dict
            Evaluated outputs of the input nodes. The keys are the attribute names.
        
        Raises
        ------
        NotImplementedError
            Description
        """
        raise NotImplementedError

    @property
    def base_definition(self):
        """Base node definition. 

        Returns
        -------
        OrderedDict
            Extends base description by adding 'inputs'
        """

        d = super(Algorithm, self).base_definition
        inputs = self._inputs
        d["inputs"] = OrderedDict([(key, inputs[key]) for key in sorted(inputs.keys())])
        return d


class Mask(Algorithm):
    """ Masks the `source` based on a boolean expression involving the `mask` (i.e. source[mask <bool_op> <bool_val> ] = <masked_val>). For a normal boolean mask input, default values for `bool_op`, `bool_val` and `masked_val` can be used.
    
    Attributes
    ----------
    source : podpac.Node
        The source that will be masked
    mask : podpac.Node
        The data that will be used to compute the mask
    masked_val : float, optional
        Default value is np.nan. The value that will replace the masked items.
    bool_val : float, optional
        Default value is 1. The value used to compare the mask when creating the boolean expression
    bool_op : enum, optional
        Default value is '=='. One of ['==', '<', '<=', '>', '>=']
    in_place : bool, optional
        Default is False. If True, the source array will be changed in-place, which could affect the value of the source 
        in other parts of the pipeline. 
        
    Examples
    ----------
    # Mask data from a boolean data node using the default behavior.
    # Create a boolean masked Node (as an example)
    b = Arithmetic(A=SinCoords(), eqn='A>0)  
    # Create the source node
    a = Arange()  
    masked = Mask(source=a, mask=b)
    
    # Create a node that make the following substitution "a[b > 0] = np.nan"
    a = Arange()
    b = SinCoords()
    masked = Mask(source=a, mask=b,
                  masked_val=np.nan, 
                  bool_val=0, bool_op='>'
                  in_place=True)
    
    """

    source = NodeTrait()
    mask = NodeTrait()
    masked_val = tl.Float(np.nan).tag(attr=True)
    bool_val = tl.Float(1).tag(attr=True)
    bool_op = tl.Enum(["==", "<", "<=", ">", ">="], default_value="==").tag(attr=True)
    in_place = tl.Bool(False).tag(attr=True)

    def algorithm(self, inputs):
        """ Sets the values in inputs['source'] to self.masked_val using (inputs['mask'] <self.bool_op> <self.bool_val>)
        """
        # shorter names
        mask = inputs["mask"]
        source = inputs["source"]
        op = self.bool_op
        bv = self.bool_val

        # Make a copy if we don't want to change the source in-place
        if not self.in_place:
            source = source.copy()

        # Make the mask boolean
        if op == "==":
            mask = mask == bv
        elif op == "<":
            mask = mask < bv
        elif op == "<=":
            mask = mask <= bv
        elif op == ">":
            mask = mask > bv
        elif op == ">=":
            mask = mask >= bv

        # Mask the values and return
        source.set(self.masked_val, mask)

        return source
