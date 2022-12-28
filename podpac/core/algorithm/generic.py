"""
General-purpose Algorithm Nodes.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import sys
import warnings

import numpy as np
import xarray as xr
import traitlets as tl

# Optional dependencies
from lazy_import import lazy_module

ne = lazy_module("numexpr")

from podpac import settings
from podpac import Coordinates
from podpac.core.node import Node, NodeException
from podpac.core.utils import NodeTrait
from podpac.core.algorithm.algorithm import Algorithm

if sys.version_info.major == 2:

    class PermissionError(OSError):
        pass


class GenericInputs(Algorithm):
    """Base class for Algorithms that accept generic named inputs."""

    inputs = tl.Dict(read_only=True, value_trait=NodeTrait(), key_trait=tl.Unicode()).tag(attr=True, required=True)

    _repr_keys = ["inputs"]

    def _first_init(self, **kwargs):
        trait_names = self.trait_names()
        for key in kwargs:
            if key in trait_names and isinstance(kwargs[key], Node):
                raise RuntimeError("Trait '%s' is reserved and cannot be used as an Generic Algorithm input" % key)
        input_keys = [key for key in kwargs if key not in trait_names and isinstance(kwargs[key], Node)]
        inputs = {key: kwargs.pop(key) for key in input_keys}
        self.set_trait("inputs", inputs)
        return super(GenericInputs, self)._first_init(**kwargs)

    @property
    def _base_definition(self):
        d = super(GenericInputs, self)._base_definition
        d["inputs"] = self.inputs
        return d


class Arithmetic(GenericInputs):
    """Create a simple point-by-point computation using named input nodes.

    Examples
    ----------
    a = SinCoords()
    b = Arange()
    arith = Arithmetic(A=a, B=b, eqn = 'A * B + {offset}', params={'offset': 1})
    """

    eqn = tl.Unicode().tag(attr=True, required=True)
    params = tl.Dict().tag(attr=True, required=True)

    _repr_keys = ["eqn"]

    def init(self):
        if not settings.allow_unsafe_eval:
            warnings.warn(
                "Insecure evaluation of Python code using Arithmetic node has not been allowed. If "
                "this is an error, use: `podpac.settings.allow_unrestricted_code_execution(True)`. "
                "NOTE: Allowing unsafe evaluation enables arbitrary execution of Python code through PODPAC "
                "Node definitions."
            )

        if self.eqn == "":
            raise ValueError("Arithmetic eqn cannot be empty")

        super(Arithmetic, self).init()

    def algorithm(self, inputs, coordinates):
        """Compute the algorithms equation

        Attributes
        ----------
        inputs : dict
            Evaluated outputs of the input nodes. The keys are the attribute names.
        coordinates : podpac.Coordinates
            Requested coordinates.
            Note that the ``inputs`` may contain with different coordinates.

        Returns
        -------
        result : UnitsDataArray
            Algorithm result.
        """

        if not settings.allow_unsafe_eval:
            raise PermissionError(
                "Insecure evaluation of Python code using Arithmetic node has not been allowed. If "
                "this is an error, use: `podpac.settings.allow_unrestricted_code_execution(True)`. "
                "NOTE: Allowing unsafe evaluation enables arbitrary execution of Python code through PODPAC "
                "Node definitions."
            )

        eqn = self.eqn.format(**self.params)

        fields = self.inputs.keys()
        res = xr.broadcast(*[inputs[f] for f in fields])
        f_locals = dict(zip(fields, res))

        try:
            from numexpr import evaluate  # Needed for some systems to get around lazy_module issues

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

    code = tl.Unicode().tag(attr=True, readonly=True, required=True)

    def init(self):
        if not settings.allow_unsafe_eval:
            warnings.warn(
                "Insecure evaluation of Python code using Generic node has not been allowed. If this "
                "this is an error, use: `podpac.settings.allow_unrestricted_code_execution(True)`. "
                "NOTE: Allowing unsafe evaluation enables arbitrary execution of Python code through PODPAC "
                "Node definitions."
            )
        super(Generic, self).init()

    def algorithm(self, inputs, coordinates):
        """
        Run the generic code.

        Attributes
        ----------
        inputs : dict
            Evaluated outputs of the input nodes. The keys are the attribute names.
        coordinates : podpac.Coordinates
            Requested coordinates.
            Note that the ``inputs`` may contain with different coordinates.

        Returns
        -------
        result : UnitsDataArray
            Algorithm result.
        """

        if not settings.allow_unsafe_eval:
            raise PermissionError(
                "Insecure evaluation of Python code using Generic node has not been allowed. If this "
                "this is an error, use: `podpac.settings.allow_unrestricted_code_execution(True)`. "
                "NOTE: Allowing unsafe evaluation enables arbitrary execution of Python code through PODPAC "
                "Node definitions."
            )
        exec(self.code, inputs)
        return inputs["output"]


class Mask(Algorithm):
    """
    Masks the `source` based on a boolean expression involving the `mask`
    (i.e. source[mask <bool_op> <bool_val> ] = <masked_val>).
    For a normal boolean mask input, default values for `bool_op`, `bool_val` and `masked_val` can be used.

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

    source = NodeTrait().tag(attr=True, required=True)
    mask = NodeTrait().tag(attr=True, required=True)
    masked_val = tl.Float(allow_none=True, default_value=None).tag(attr=True)
    bool_val = tl.Float(1).tag(attr=True)
    bool_op = tl.Enum(["==", "<", "<=", ">", ">="], default_value="==").tag(attr=True)
    in_place = tl.Bool(False).tag(attr=True)

    _repr_keys = ["source", "mask"]

    def algorithm(self, inputs, coordinates):
        """
        Sets the values in inputs['source'] to self.masked_val using (inputs['mask'] <self.bool_op> <self.bool_val>)

        Attributes
        ----------
        inputs : dict
            Evaluated outputs of the input nodes. The keys are the attribute names.
        coordinates : podpac.Coordinates
            Requested coordinates.
            Note that the ``inputs`` may contain with different coordinates.

        Returns
        -------
        result : UnitsDataArray
            Algorithm result.
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
        if self.masked_val is None:
            source.set(np.nan, mask)
        else:
            source.set(self.masked_val, mask)

        return source


class Combine(GenericInputs):
    """Combine multiple nodes into a single node with multiple outputs.

    If not output names are specified, the keyword argument names will be used.
    """

    @tl.default("outputs")
    def _default_outputs(self):
        input_keys = list(self.inputs.keys())
        return input_keys

    def algorithm(self, inputs, coordinates):
        cs = [Coordinates.from_xarray(x) for x in inputs.values()]
        if any(c != cs[0] for c in cs):
            raise NodeException("Cannot combine inputs with different coordinates")

        data = np.stack([inputs[key] for key in self.inputs], axis=-1)
        return self.create_output_array(cs[0], data=data)
