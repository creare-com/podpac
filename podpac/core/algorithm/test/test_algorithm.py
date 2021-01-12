from __future__ import division, unicode_literals, print_function, absolute_import

import warnings
from collections import OrderedDict

import pytest
import traitlets as tl
import numpy as np
import xarray as xr

import podpac
from podpac.core.utils import NodeTrait
from podpac.core.node import Node, NodeException
from podpac.core.data.array_source import Array
from podpac.core.algorithm.utility import Arange
from podpac.core.algorithm.generic import Arithmetic
from podpac.core.algorithm.algorithm import BaseAlgorithm, Algorithm, UnaryAlgorithm


class TestBaseAlgorithm(object):
    def test_eval_not_implemented(self):
        node = BaseAlgorithm()
        c = podpac.Coordinates([])
        with pytest.raises(NotImplementedError):
            node.eval(c)

    def test_find_coordinates(self):
        class MyAlgorithm(BaseAlgorithm):
            x = NodeTrait().tag(attr=True)
            y = NodeTrait().tag(attr=True)

        node = MyAlgorithm(
            x=Array(coordinates=podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"])),
            y=Array(coordinates=podpac.Coordinates([[0, 1, 2], [110, 120]], dims=["lat", "lon"])),
        )

        l = node.find_coordinates()
        assert isinstance(l, list)
        assert len(l) == 2
        assert node.x.coordinates in l
        assert node.y.coordinates in l


class TestAlgorithm(object):
    def test_algorithm_not_implemented(self):
        node = Algorithm()
        c = podpac.Coordinates([])
        with pytest.raises(NotImplementedError):
            node.eval(c)

    def test_multi_threading(self):
        coords = podpac.Coordinates([[1, 2, 3]], ["lat"])

        with podpac.settings:
            podpac.settings.set_unsafe_eval(True)
            podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"] = False
            podpac.settings["DEFAULT_CACHE"] = []
            podpac.settings["RAM_CACHE_ENABLED"] = False

            node1 = Arithmetic(A=Arange(), B=Arange(), eqn="A+B")
            node2 = Arithmetic(A=node1, B=Arange(), eqn="A+B")

            # multithreaded
            podpac.settings["MULTITHREADING"] = True
            podpac.settings["N_THREADS"] = 8

            omt = node2.eval(coords)

            # single threaded
            podpac.settings["MULTITHREADING"] = False

            ost = node2.eval(coords)

        np.testing.assert_array_equal(omt, ost)

    def test_multi_threading_cache_race(self):
        coords = podpac.Coordinates([np.linspace(0, 1, 1024)], ["lat"])
        with podpac.settings:
            podpac.settings["MULTITHREADING"] = True
            podpac.settings["N_THREADS"] = 3
            podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"] = True
            podpac.settings["DEFAULT_CACHE"] = ["ram"]
            podpac.settings["RAM_CACHE_ENABLED"] = True
            podpac.settings.set_unsafe_eval(True)
            A = Arithmetic(A=Arange(), eqn="A**2")
            B = Arithmetic(A=Arange(), eqn="A**2")
            C = Arithmetic(A=Arange(), eqn="A**2")
            D = Arithmetic(A=Arange(), eqn="A**2")
            E = Arithmetic(A=Arange(), eqn="A**2")
            F = Arithmetic(A=Arange(), eqn="A**2")

            node2 = Arithmetic(A=A, B=B, C=C, D=D, E=E, F=F, eqn="A+B+C+D+E+F")

            om = node2.eval(coords)

            from_cache = [n._from_cache for n in node2.inputs.values()]

            assert sum(from_cache) > 0

    def test_multi_threading_stress_nthreads(self):
        coords = podpac.Coordinates([np.linspace(0, 1, 4)], ["lat"])

        A = Arithmetic(A=Arange(), eqn="A**0")
        B = Arithmetic(A=Arange(), eqn="A**1")
        C = Arithmetic(A=Arange(), eqn="A**2")
        D = Arithmetic(A=Arange(), eqn="A**3")
        E = Arithmetic(A=Arange(), eqn="A**4")
        F = Arithmetic(A=Arange(), eqn="A**5")

        node2 = Arithmetic(A=A, B=B, C=C, D=D, E=E, F=F, eqn="A+B+C+D+E+F")
        node3 = Arithmetic(A=A, B=B, C=C, D=D, E=E, F=F, G=node2, eqn="A+B+C+D+E+F+G")

        with podpac.settings:
            podpac.settings["MULTITHREADING"] = True
            podpac.settings["N_THREADS"] = 8
            podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"] = False
            podpac.settings["DEFAULT_CACHE"] = []
            podpac.settings["RAM_CACHE_ENABLED"] = False
            podpac.settings.set_unsafe_eval(True)

            omt = node3.eval(coords)

        assert node3._multi_threaded
        assert not node2._multi_threaded

        with podpac.settings:
            podpac.settings["MULTITHREADING"] = True
            podpac.settings["N_THREADS"] = 9  # 2 threads available after first 7
            podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"] = False
            podpac.settings["DEFAULT_CACHE"] = []
            podpac.settings["RAM_CACHE_ENABLED"] = False
            podpac.settings.set_unsafe_eval(True)

            omt = node3.eval(coords)

        assert node3._multi_threaded
        assert node2._multi_threaded

    def test_algorithm_return_types(self):
        coords = podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"])

        # numpy array
        class ArrayAlgorithm(Algorithm):
            def algorithm(self, inputs):
                return np.ones(self._requested_coordinates.shape)

        node = ArrayAlgorithm()
        result = node.eval(coords)
        np.testing.assert_array_equal(result, np.ones((3, 2)))

        output = node.create_output_array(coords, data=0)
        result = node.eval(coords, output=output)
        np.testing.assert_array_equal(result, np.ones((3, 2)))
        np.testing.assert_array_equal(output, np.ones((3, 2)))

        # xarray DataArray
        class DataArrayAlgorithm(Algorithm):
            def algorithm(self, inputs):
                data = np.ones(self._requested_coordinates.shape)
                return self.create_output_array(self._requested_coordinates, data=data)

        node = DataArrayAlgorithm()
        result = node.eval(coords)
        np.testing.assert_array_equal(result, np.ones((3, 2)))

        output = node.create_output_array(coords, data=0)
        result = node.eval(coords, output=output)
        np.testing.assert_array_equal(result, np.ones((3, 2)))
        np.testing.assert_array_equal(output, np.ones((3, 2)))

        # podpac UnitsDataArray
        class UnitsDataArrayAlgorithm(Algorithm):
            def algorithm(self, inputs):
                data = np.ones(self._requested_coordinates.shape)
                return self.create_output_array(self._requested_coordinates, data=data)

        node = UnitsDataArrayAlgorithm()
        result = node.eval(coords)
        np.testing.assert_array_equal(result, np.ones((3, 2)))

        output = node.create_output_array(coords, data=0)
        result = node.eval(coords, output=output)
        np.testing.assert_array_equal(result, np.ones((3, 2)))
        np.testing.assert_array_equal(output, np.ones((3, 2)))

        # invalid
        class InvalidAlgorithm(Algorithm):
            def algorithm(self, inputs):
                return None

        node = InvalidAlgorithm()
        with pytest.raises(NodeException):
            node.eval(coords)

    def test_multiple_outputs(self):
        class MyAlgorithm(Algorithm):
            x = NodeTrait().tag(attr=True)
            y = NodeTrait().tag(attr=True)
            outputs = ["sum", "prod", "diff"]

            def algorithm(self, inputs):
                sum_ = inputs["x"] + inputs["y"]
                prod = inputs["x"] * inputs["y"]
                diff = inputs["x"] - inputs["y"]
                return np.stack([sum_, prod, diff], -1)

        coords = podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"])
        x = Arange()
        y = Array(source=np.full(coords.shape, 2), coordinates=coords)
        xout = np.arange(6).reshape(3, 2)

        # all outputs
        node = MyAlgorithm(x=x, y=y)
        result = node.eval(coords)
        assert result.dims == ("lat", "lon", "output")
        np.testing.assert_array_equal(result["output"], ["sum", "prod", "diff"])
        np.testing.assert_array_equal(result.sel(output="sum"), xout + 2)
        np.testing.assert_array_equal(result.sel(output="prod"), xout * 2)
        np.testing.assert_array_equal(result.sel(output="diff"), xout - 2)

        # extract an output
        node = MyAlgorithm(x=x, y=y, output="prod")
        result = node.eval(coords)
        assert result.dims == ("lat", "lon")
        np.testing.assert_array_equal(result, xout * 2)


class TestUnaryAlgorithm(object):
    source = Array(coordinates=podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"]))

    def test_outputs(self):
        node = UnaryAlgorithm(source=self.source)
        assert node.outputs == None

        node = UnaryAlgorithm(source=Array(outputs=["a", "b"]))
        assert node.outputs == ["a", "b"]
