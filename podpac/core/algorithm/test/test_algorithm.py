from __future__ import division, unicode_literals, print_function, absolute_import

import warnings

import pytest
from collections import OrderedDict
import numpy as np

import podpac
from podpac.core.algorithm.utility import Arange
from podpac.core.algorithm.generic import Arithmetic
from podpac.core.algorithm.algorithm import Algorithm


class TestAlgorithm(object):
    def test_not_implemented(self):
        node = Algorithm()
        c = podpac.Coordinates([])
        with pytest.raises(NotImplementedError):
            node.eval(c)

    def test_base_definition(self):
        # note: any algorithm node with attrs and inputs would be fine here
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Insecure evaluation.*")
            node = Arithmetic(A=Arange(), B=Arange(), eqn="A+B")

        d = node.base_definition
        assert isinstance(d, OrderedDict)
        assert "node" in d
        assert "attrs" in d

        # base (node, params)
        assert d["node"] == "core.algorithm.generic.Arithmetic"
        assert d["attrs"]["eqn"] == "A+B"

        # inputs
        assert "inputs" in d
        assert isinstance(d["inputs"], dict)
        assert "A" in d["inputs"]
        assert "B" in d["inputs"]

        # TODO value of d['inputs']['A'], etc

    def test_multi_threading(self):
        coords = podpac.Coordinates([[1, 2, 3]], ["lat"])
        node1 = Arithmetic(A=Arange(), B=Arange(), eqn="A+B")
        node2 = Arithmetic(A=node1, B=Arange(), eqn="A+B")

        with podpac.settings:
            podpac.settings["MULTITHREADING"] = True
            podpac.settings["N_THREADS"] = 8
            podpac.settings["CACHE_OUTPUT_DEFAULT"] = False
            podpac.settings["DEFAULT_CACHE"] = []
            podpac.settings["RAM_CACHE_ENABLED"] = False
            podpac.settings.set_unsafe_eval(True)

            omt = node2.eval(coords)

        with podpac.settings:
            podpac.settings["MULTITHREADING"] = False
            podpac.settings["CACHE_OUTPUT_DEFAULT"] = False
            podpac.settings["DEFAULT_CACHE"] = []
            podpac.settings["RAM_CACHE_ENABLED"] = False
            podpac.settings.set_unsafe_eval(True)

            ost = node2.eval(coords)

        np.testing.assert_array_equal(omt, ost)

    def test_multi_threading_cache_race(self):
        coords = podpac.Coordinates([np.linspace(0, 1, 1024)], ["lat"])
        with podpac.settings:
            podpac.settings["MULTITHREADING"] = True
            podpac.settings["N_THREADS"] = 3
            podpac.settings["CACHE_OUTPUT_DEFAULT"] = True
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
