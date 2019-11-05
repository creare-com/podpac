from __future__ import division, unicode_literals, print_function, absolute_import

import copy

import pytest
import numpy as np
import xarray as xr

import podpac
from podpac.core.algorithm.utility import Arange, SinCoords
from podpac.core.algorithm.generic import GenericInputs, Arithmetic, Generic, Mask


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
        podpac.settings.set_unsafe_eval(True)

        coords = podpac.Coordinates([podpac.crange(-90, 90, 1.0), podpac.crange(-180, 180, 1.0)], dims=["lat", "lon"])
        sine_node = SinCoords()
        node = Arithmetic(A=sine_node, B=sine_node, eqn="2*abs(A) - B + {offset}", params={"offset": 1})
        output = node.eval(coords)

        a = sine_node.eval(coords)
        b = sine_node.eval(coords)
        np.testing.assert_allclose(output, 2 * abs(a) - b + 1)

    def test_evaluate_not_allowed(self):
        podpac.settings.set_unsafe_eval(False)

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
    def test_init(self):
        a = SinCoords()
        b = Arange()

        podpac.settings.set_unsafe_eval(True)
        node = Generic(code="import numpy as np\noutput = np.minimum(a,b)", a=a, b=b)

        podpac.settings.set_unsafe_eval(False)
        with pytest.warns(UserWarning, match="Insecure evaluation"):
            node = Generic(code="import numpy as np\noutput = np.minimum(a,b)", a=a, b=b)

    def test_evaluate(self):
        podpac.settings.set_unsafe_eval(True)

        coords = podpac.Coordinates([podpac.crange(-90, 90, 1.0), podpac.crange(-180, 180, 1.0)], dims=["lat", "lon"])
        a = SinCoords()
        b = Arange()
        node = Generic(code="import numpy as np\noutput = np.minimum(a,b)", a=a, b=b)
        output = node.eval(coords)

        a = node.eval(coords)
        b = node.eval(coords)
        np.testing.assert_allclose(output, np.minimum(a, b))

    def test_evaluate_not_allowed(self):
        podpac.settings.set_unsafe_eval(False)

        coords = podpac.Coordinates([podpac.crange(-90, 90, 1.0), podpac.crange(-180, 180, 1.0)], dims=["lat", "lon"])
        a = SinCoords()
        b = Arange()

        with pytest.warns(UserWarning, match="Insecure evaluation"):
            node = Generic(code="import numpy as np\noutput = np.minimum(a,b)", a=a, b=b)

        node = Generic(
            code="import numpy as np\noutput = np.minimum(b,a)", a=b, b=a
        )  # needs to be different to avoid cache
        with pytest.raises(PermissionError):
            node.eval(coords)


class TestMask(object):
    def test_mask_defaults(self):
        coords = podpac.Coordinates([podpac.crange(-90, 90, 1.0), podpac.crange(-180, 180, 1.0)], dims=["lat", "lon"])
        sine_node = Arange()
        a = sine_node.eval(coords).copy()
        a.data[a.data == 1] = np.nan

        node = Mask(source=sine_node, mask=sine_node)
        output = node.eval(coords)

        np.testing.assert_allclose(output, a)

    def test_mask_defaults_bool_op(self):
        coords = podpac.Coordinates([podpac.clinspace(0, 1, 4), podpac.clinspace(0, 1, 3)], dims=["lat", "lon"])
        sine_node = Arange()
        a = sine_node.eval(coords).copy()

        # Less than
        node = Mask(source=sine_node, mask=sine_node, bool_op="<")
        output = node.eval(coords)
        b = a.copy()
        b.data[a.data < 1] = np.nan
        np.testing.assert_allclose(output, b)

        # Less than equal
        node = Mask(source=sine_node, mask=sine_node, bool_op="<=")
        output = node.eval(coords)
        b = a.copy()
        b.data[a.data <= 1] = np.nan
        np.testing.assert_allclose(output, b)

        # Greater than
        node = Mask(source=sine_node, mask=sine_node, bool_op=">")
        output = node.eval(coords)
        b = a.copy()
        b.data[a.data > 1] = np.nan
        np.testing.assert_allclose(output, b)

        # Greater than equal
        node = Mask(source=sine_node, mask=sine_node, bool_op=">=")
        output = node.eval(coords)
        b = a.copy()
        b.data[a.data >= 1] = np.nan
        np.testing.assert_allclose(output, b)

    def test_bool_val(self):
        coords = podpac.Coordinates([podpac.clinspace(0, 1, 4), podpac.clinspace(0, 1, 3)], dims=["lat", "lon"])
        sine_node = Arange()
        a = sine_node.eval(coords).copy()
        a.data[a.data == 2] = np.nan

        node = Mask(source=sine_node, mask=sine_node, bool_val=2)
        output = node.eval(coords)

        np.testing.assert_allclose(output, a)

    def test_masked_val(self):
        coords = podpac.Coordinates([podpac.clinspace(0, 1, 4), podpac.clinspace(0, 1, 3)], dims=["lat", "lon"])
        sine_node = Arange()
        a = sine_node.eval(coords).copy()
        a.data[a.data == 1] = -9999

        node = Mask(source=sine_node, mask=sine_node, masked_val=-9999)
        output = node.eval(coords)

        np.testing.assert_allclose(output, a)

    def test_in_place(self):
        coords = podpac.Coordinates([podpac.clinspace(0, 1, 4), podpac.clinspace(0, 1, 3)], dims=["lat", "lon"])
        sine_node = Arange()

        node = Mask(source=sine_node, mask=sine_node, in_place=True)
        output = node.eval(coords)
        a = sine_node.eval(coords)

        # In-place editing doesn't seem to work here
        # np.testing.assert_allclose(output, node.source._output)

        coords = podpac.Coordinates([podpac.clinspace(0, 1, 4), podpac.clinspace(0, 2, 3)], dims=["lat", "lon"])
        sine_node = Arange()
        node = Mask(source=sine_node, mask=sine_node, in_place=False)
        output = node.eval(coords)
        a = sine_node.eval(coords)

        assert not np.all(a == output)
