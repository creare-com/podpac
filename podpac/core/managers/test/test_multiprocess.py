import numpy as np
import pytest

from podpac.core.coordinates import Coordinates
from podpac.core.algorithm.utility import Arange
from podpac.core.managers.multi_process import Process


class TestProcess(object):
    def test_mp_results_the_same(self):
        coords = Coordinates([[1, 2, 3, 4, 5]], ["time"])
        node = Arange()
        o_sp = node.eval(coords)

        node_mp = Process(source=node)
        o_mp = node_mp.eval(coords)

        np.testing.assert_array_equal(o_sp.data, o_mp.data)

    def test_mp_results_outputs(self):
        node = Arange(outputs=["a", "b"])
        node_mp = Process(source=node)
        assert node.outputs == node_mp.outputs

    def test_mp_results_the_same_set_output(self):
        coords = Coordinates([[1, 2, 3, 4, 5]], ["time"])
        node = Arange()
        o_sp = node.eval(coords)
        output = o_sp.copy()
        output[:] = np.nan

        node_mp = Process(source=node)
        o_mp = node_mp.eval(coords, output)

        np.testing.assert_array_equal(o_sp, output)
