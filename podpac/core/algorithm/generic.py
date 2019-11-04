"""
General-purpose Algorithm Nodes.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import warnings

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


class GenericInputs(Algorithm):
    """Base class for Algorithms that accept generic named inputs."""

    inputs = tl.Dict()

    @property
    def _inputs(self):
        return self.inputs

    def _first_init(self, **kwargs):
        trait_names = self.trait_names()
        input_keys = [key for key in kwargs if key not in trait_names and isinstance(kwargs[key], Node)]
        inputs = {key: kwargs.pop(key) for key in input_keys}
        return super(GenericInputs, self)._first_init(inputs=inputs, **kwargs)


class Arithmetic(GenericInputs):
    """Create a simple point-by-point computation using named input nodes.
        
    Examples
    ----------
    a = SinCoords()
    b = Arange()
    arith = Arithmetic(A=a, B=b, eqn = 'A * B + {offset}', params={'offset': 1})
    """

    eqn = tl.Unicode().tag(attr=True)
    params = tl.Dict().tag(attr=True)

    def init(self):
        if not settings["ALLOW_PYTHON_EVAL_EXEC"]:
            warnings.warn(
                "Insecure evaluation of Python code using Arithmetic node has not been allowed. If "
                "this is an error, use: `podpac.settings.set_allow_python_eval_exec(True)`. "
                "Alternatively create the file ALLOW_PYTHON_EVAL_EXEC in {}".format(
                    settings._allow_python_eval_exec_paths[-1]
                )
                + "NOTE: making this setting True allows arbitrary execution of Python code through PODPAC "
                "Node definitions."
            )

        if self.eqn == "":
            raise ValueError("Arithmetic eqn cannot be empty")

        super(Arithmetic, self).init()

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

        eqn = self.eqn.format(**self.params)

        fields = self.inputs.keys()
        res = xr.broadcast(*[inputs[f] for f in fields])
        f_locals = dict(zip(fields, res))

        try:
            result = ne.evaluate(eqn, f_locals)
        except (NotImplementedError, ImportError):
            result = eval(eqn, f_locals)
        res = res[0].copy()  # Make an xarray object with correct dimensions
        res[:] = result
        return res


class Generic(GenericInputs):
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
    generic = Generic(code=code, a=a, b=b)
    """

    code = tl.Unicode().tag(attr=True, readonly=True)

    def init(self):
        if not settings["ALLOW_PYTHON_EVAL_EXEC"]:
            warnings.warn(
                "Insecure evaluation of Python code using Generic node has not been allowed. If this "
                "this is an error, use: `podpac.settings.set_allow_python_eval_exec(True)`. "
                "Alternatively create the file ALLOW_PYTHON_EVAL_EXEC in {}".format(
                    settings._allow_python_eval_exec_paths[-1]
                )
                + "NOTE: making this setting True allows arbitrary execution of Python code through PODPAC "
                "Node definitions."
            )
        super(Generic, self).init()

    def algorithm(self, inputs):
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
        exec (self.code, inputs)
        return inputs["output"]
