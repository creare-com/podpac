import os
import sys
import time
import numpy as np
from threading import Thread

import pytest

from podpac import settings
from podpac.core.coordinates import Coordinates
from podpac.core.algorithm.utility import Arange
from podpac.core.managers.parallel import Parallel, ParallelOutputZarr
from podpac.core.managers.multi_process import Process


class TestParallel(object):
    def test_parallel_multi_thread_compute(self):
        node = Arange()
        coords = Coordinates([[1, 2, 3, 4, 5]], ["time"])
        node_p = Parallel(source=node, n_workers=2, chunks={"time": 2})
        o = node.eval(coords)
        o_p = node_p.eval(coords)

        np.testing.assert_array_equal(o, o_p)
