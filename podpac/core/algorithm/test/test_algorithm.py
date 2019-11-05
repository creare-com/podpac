from __future__ import division, unicode_literals, print_function, absolute_import

import pytest
from collections import OrderedDict

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
        setting = podpac.settings.allow_unsafe_eval
        podpac.settings.set_unsafe_eval(True)
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
        podpac.settings.set_unsafe_eval(setting)
