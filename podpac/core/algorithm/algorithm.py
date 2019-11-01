"""
Algorithm Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
import inspect
import numpy as np
import xarray as xr
import traitlets as tl
from lazy_import import lazy_module

# Optional dependencies
ne = lazy_module("numexpr")

# Internal dependencies
from podpac.core.coordinates import Coordinates, union
from podpac.core.units import UnitsDataArray
from podpac.core.node import Node
from podpac.core.node import NodeException
from podpac.core.node import COMMON_NODE_DOC
from podpac.core.node import node_eval
from podpac.core.utils import common_doc
from podpac.core.utils import NodeTrait
from podpac.core.settings import settings

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


class Arithmetic(Algorithm):
    """Create a simple point-by-point computation of up to 7 different input nodes.
    
    Attributes
    ----------
    A : podpac.Node
        An input node that can be used in a computation. 
    B : podpac.Node
        An input node that can be used in a computation. 
    C : podpac.Node
        An input node that can be used in a computation. 
    D : podpac.Node
        An input node that can be used in a computation. 
    E : podpac.Node
        An input node that can be used in a computation. 
    F : podpac.Node
        An input node that can be used in a computation. 
    G : podpac.Node
        An input node that can be used in a computation. 
    eqn : str
        An equation stating how the datasources can be combined. 
        Parameters may be specified in {}'s
        
    Examples
    ----------
    a = SinCoords()
    b = Arange()
    arith = Arithmetic(A=a, B=b, eqn = 'A * B + {offset}', params={'offset': 1})
    """

    A = NodeTrait()
    B = NodeTrait(allow_none=True)
    C = NodeTrait(allow_none=True)
    D = NodeTrait(allow_none=True)
    E = NodeTrait(allow_none=True)
    F = NodeTrait(allow_none=True)
    G = NodeTrait(allow_none=True)
    eqn = tl.Unicode().tag(attr=True)
    params = tl.Dict().tag(attr=True)

    def _first_init(self, **kwargs):
        if not settings["ALLOW_PYTHON_EVAL_EXEC"]:
            raise PermissionError(
                "Insecure evaluation of Python code using Arithmetic node has not been allowed. If "
                "this is an error, use: `podpac.settings.set_allow_python_eval_exec(True)`. "
                "Alternatively create the file ALLOW_PYTHON_EVAL_EXEC in '{}' ".format(
                    settings._allow_python_eval_exec_paths[-1]
                )
                + "NOTE: making this setting True allows arbitrary execution of Python code through PODPAC "
                "Node definitions."
            )
        return kwargs

    def init(self):
        if self.eqn == "":
            raise ValueError("Arithmetic eqn cannot be empty")

    def algorithm(self, inputs):
        """ Compute the algorithms equation

        Attributes
        ----------
        inputs : dict
            Evaluated outputs of the input nodes. The keys are the attribute names.
        
        Returns
        -------
        UnitsDataArray
            Description
        """

        eqn = self.eqn.format(**self.params)

        fields = [f for f in "ABCDEFG" if getattr(self, f) is not None]
        res = xr.broadcast(*[inputs[f] for f in fields])
        f_locals = dict(zip(fields, res))

        try:
            result = ne.evaluate(eqn, f_locals)
        except (NotImplementedError, ImportError):
            result = eval(eqn, f_locals)
        res = res[0].copy()  # Make an xarray object with correct dimensions
        res[:] = result
        return res


class Generic(Algorithm):
    """
    Generic Algorithm Node that allows arbitrary Python code to be executed.
    
    Attributes
    ----------
    code : str
        The multi-line code that will be evaluated. This code should assign "output" to the desired result, and "output"
        needs to be a "numpy array" or "xarray DataArray"
    inputs : dict(str: podpac.Node)
        A dictionary of PODPAC nodes that will serve as the input data for the Python script

    Examples
    ----------
    a = SinCoords()
    b = Arange()
    code = '''import numpy as np
    output = np.minimum(a, b)
    '''
    generic = Generic(code=code, inputs={'a': a, 'b': b'})
    """

    code = tl.Unicode().tag(attr=True, readonly=True)
    inputs = tl.Dict().tag()

    def _first_init(self, **kwargs):
        if not settings["ALLOW_PYTHON_EVAL_EXEC"]:
            raise PermissionError(
                "Insecure evaluation of Python code using Generic node has not been allowed. If this "
                "this is an error, use: `podpac.settings.set_allow_python_eval_exec(True)`. "
                "Alternatively create the file ALLOW_PYTHON_EVAL_EXEC in '{}' ".format(
                    settings._allow_python_eval_exec_paths[-1]
                )
                + "NOTE: making this setting True allows arbitrary execution of Python code through PODPAC "
                "Node definitions."
            )
        return kwargs

    def algorithm(self, inputs):
        exec(self.code, inputs)
        return inputs["output"]

    @property
    def _inputs(self):
        return self.inputs


class Mask(Algorithm):
    """ Masks the `source` based on a boolean expression involving the `mask`.
    
    Attributes
    ----------
    source : podpac.Node
        The source that will be masked
    mask : podpac.Node
        The data that will be used to compute the mask
    masked_val : float, optional
        Default value is np.nan. The value that will replace the maked items.
    bool_val : float, optional
        Default value is 1. The value used to compare the mask when creating the boolean expression
    bool_op : enum, optional
        Default value is '=='. One of ['==', '<', '<=', '>', '>=']
    in_place : bool, optional
        Default is False. If True, the source array will be changed in placem which could affect the value of the source 
        in other parts of the pipeline. 
        
    Examples
    ----------
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
    masked_val = tl.Float(np.nan).tag(attr=True, readonly=True)
    bool_val = tl.Float(1).tag(attr=True, readonly=True)
    bool_op = tl.Enum(["==", "<", "<=", ">", ">="], default_value="==").tag(attr=True, readonly=True)
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
