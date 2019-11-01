from __future__ import division, unicode_literals, print_function, absolute_import

import pytest
import numpy as np
import xarray as xr

import podpac
from podpac.core.algorithm.utility import Arange, SinCoords
from podpac.core.algorithm.general import Arithmetic, Generic, CombineOutputs


class TestArithmetic(object):
    def test_Arithmetic(self):
        coords = podpac.Coordinates([podpac.crange(-90, 90, 1.0), podpac.crange(-180, 180, 1.0)], dims=["lat", "lon"])
        sine_node = SinCoords()
        setting = podpac.settings["ALLOW_PYTHON_EVAL_EXEC"]
        podpac.settings.set_allow_python_eval_exec(True)
        node = Arithmetic(A=sine_node, B=sine_node, eqn="2*abs(A) - B + {offset}", params={"offset": 1})
        output = node.eval(coords)

        a = sine_node.eval(coords)
        b = sine_node.eval(coords)
        np.testing.assert_allclose(output, 2 * abs(a) - b + 1)
        podpac.settings.set_allow_python_eval_exec(setting)

    def test_missing_equation(self):
        setting = podpac.settings["ALLOW_PYTHON_EVAL_EXEC"]
        podpac.settings.set_allow_python_eval_exec(True)
        sine_node = SinCoords()
        with pytest.raises(ValueError):
            node = Arithmetic(A=sine_node, B=sine_node)
        podpac.settings.set_allow_python_eval_exec(setting)


class TestGeneric(object):
    def test_Generic_allowed(self):
        setting = podpac.settings["ALLOW_PYTHON_EVAL_EXEC"]
        podpac.settings.set_allow_python_eval_exec(True)
        coords = podpac.Coordinates([podpac.crange(-90, 90, 1.0), podpac.crange(-180, 180, 1.0)], dims=["lat", "lon"])
        a = SinCoords()
        b = Arange()
        node = Generic(code="import numpy as np\noutput = np.minimum(a,b)", inputs={"a": a, "b": b})
        output = node.eval(coords)

        a = node.eval(coords)
        b = node.eval(coords)
        np.testing.assert_allclose(output, np.minimum(a, b))
        podpac.settings.set_allow_python_eval_exec(setting)

    def test_Generic_not_allowed(self):
        setting = podpac.settings["ALLOW_PYTHON_EVAL_EXEC"]
        podpac.settings.set_allow_python_eval_exec(False)
        a = SinCoords()
        b = Arange()
        with pytest.raises(PermissionError):
            node = Generic(code="import numpy as np\noutput = np.minimum(a,b)", inputs={"a": a, "b": b})
        podpac.settings.set_allow_python_eval_exec(setting)


class TestCombineOutputs(object):
    def test_init(self):
        node = CombineOutputs(inputs={"a": Arange(), "b": SinCoords()})
        assert set(node.outputs) == {"a", "b"}

    def test_combine(self):
        coords = podpac.Coordinates([podpac.crange(-90, 90, 1.0), podpac.crange(-180, 180, 1.0)], dims=["lat", "lon"])

        a_node = Arange()
        b_node = SinCoords()
        ab_node = CombineOutputs(inputs={"a": Arange(), "b": SinCoords()})

        a = a_node.eval(coords)
        b = b_node.eval(coords)
        o = ab_node.eval(coords)

        assert o.dims == ("lat", "lon", "output")
        xr.testing.assert_equal(o["lat"], a["lat"])
        xr.testing.assert_equal(o["lon"], a["lon"])
        np.testing.assert_equal(o.sel(output="a").data, a.data)
        np.testing.assert_equal(o.sel(output="b").data, b.data)
