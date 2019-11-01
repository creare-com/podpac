"""
General-purpose Algorithm Nodes.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np
import xarray as xr
import traitlets as tl

# Optional dependencies
from lazy_import import lazy_module

ne = lazy_module("numexpr")

# Internal dependencies
from podpac.core.node import Node
from podpac.core.utils import NodeTrait
from podpac.core.settings import settings
from podpac.core.algorithm.algorithm import Algorithm


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
                "Alternatively create the file ALLOW_PYTHON_EVAL_EXEC in {}".format(
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
    inputs = tl.Dict()

    def _first_init(self, **kwargs):
        if not settings["ALLOW_PYTHON_EVAL_EXEC"]:
            raise PermissionError(
                "Insecure evaluation of Python code using Generic node has not been allowed. If this "
                "this is an error, use: `podpac.settings.set_allow_python_eval_exec(True)`. "
                "Alternatively create the file ALLOW_PYTHON_EVAL_EXEC in {}".format(
                    settings._allow_python_eval_exec_paths[-1]
                )
                + "NOTE: making this setting True allows arbitrary execution of Python code through PODPAC "
                "Node definitions."
            )
        return kwargs

    def algorithm(self, inputs):
        exec (self.code, inputs)
        return inputs["output"]

    @property
    def _inputs(self):
        return self.inputs


class CombineOutputs(Algorithm):
    """ Combine nodes into a single node with multiple outputs."""

    inputs = tl.Dict()

    @property
    def _inputs(self):
        return self.inputs

    @tl.default("outputs")
    def _default_outputs(self):
        return list(self.inputs.keys())

    def algorithm(self, inputs):
        data = np.array([inputs[key].data for key in self.inputs])
        data = np.moveaxis(data, 0, -1)
        return data
