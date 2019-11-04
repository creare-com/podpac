from __future__ import division, unicode_literals, print_function, absolute_import

import copy

import pytest
import numpy as np
import xarray as xr

import podpac
from podpac.core.algorithm.utility import Arange, SinCoords
from podpac.core.algorithm.generic import GenericInputs, Arithmetic, Generic


class TestGenericInputs(object):
    def test_init(self):
        node = GenericInputs(a=Arange(), b=SinCoords())
        assert "a" in node.inputs
        assert "b" in node.inputs

    def test_serialization(self):
        node = GenericInputs(a=Arange(), b=SinCoords())
        d = node.definition
        assert d[node.base_ref]["inputs"]["a"] in d
        assert d[node.base_ref]["inputs"]["b"] in d

        node2 = node.from_definition(d)
        assert node2.hash == node.hash


class TestArithmetic(object):
    def setup_method(self):
        self.settings_orig = copy.deepcopy(podpac.settings)

    def teardown_method(self):
        for key in podpac.settings:
            podpac.settings[key] = self.settings_orig[key]

    def test_evaluate(self):
        podpac.settings.set_allow_python_eval_exec(True)

        coords = podpac.Coordinates([podpac.crange(-90, 90, 1.0), podpac.crange(-180, 180, 1.0)], dims=["lat", "lon"])
        sine_node = SinCoords()
        node = Arithmetic(A=sine_node, B=sine_node, eqn="2*abs(A) - B + {offset}", params={"offset": 1})
        output = node.eval(coords)

        a = sine_node.eval(coords)
        b = sine_node.eval(coords)
        np.testing.assert_allclose(output, 2 * abs(a) - b + 1)

    def test_evaluate_not_allowed(self):
        podpac.settings.set_allow_python_eval_exec(False)

        coords = podpac.Coordinates([podpac.crange(-90, 90, 1.0), podpac.crange(-180, 180, 1.0)], dims=["lat", "lon"])
        sine_node = SinCoords()

        with pytest.warns(UserWarning, match="Insecure evaluation"):
            node = Arithmetic(A=sine_node, B=sine_node, eqn="2*abs(A) - B + {offset}", params={"offset": 1})

        with pytest.raises(PermissionError):
            node.eval(coords)

    def test_missing_equation(self):
        sine_node = SinCoords()
        with pytest.raises(ValueError):
            node = Arithmetic(A=sine_node, B=sine_node)


class TestGeneric(object):
    def test_init_(self):
        a = SinCoords()
        b = Arange()

        podpac.settings.set_allow_python_eval_exec(True)
        node = Generic(code="import numpy as np\noutput = np.minimum(a,b)", a=a, b=b)

        podpac.settings.set_allow_python_eval_exec(False)
        with pytest.warns(UserWarning, match="Insecure evaluation"):
            node = Generic(code="import numpy as np\noutput = np.minimum(a,b)", a=a, b=b)

    def test_evaluate(self):
        podpac.settings.set_allow_python_eval_exec(True)

        coords = podpac.Coordinates([podpac.crange(-90, 90, 1.0), podpac.crange(-180, 180, 1.0)], dims=["lat", "lon"])
        a = SinCoords()
        b = Arange()
        node = Generic(code="import numpy as np\noutput = np.minimum(a,b)", a=a, b=b)
        output = node.eval(coords)

        a = node.eval(coords)
        b = node.eval(coords)
        np.testing.assert_allclose(output, np.minimum(a, b))

    def test_evaluate_not_allowed(self):
        podpac.settings.set_allow_python_eval_exec(False)

        coords = podpac.Coordinates([podpac.crange(-90, 90, 1.0), podpac.crange(-180, 180, 1.0)], dims=["lat", "lon"])
        a = SinCoords()
        b = Arange()

        with pytest.warns(UserWarning, match="Insecure evaluation"):
            node = Generic(code="import numpy as np\noutput = np.minimum(a,b)", a=a, b=b)

        with pytest.raises(PermissionError):
            node.eval(coords)
