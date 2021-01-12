import os
import shutil
import sys
import time
import numpy as np
from threading import Thread
import tempfile
import logging

import pytest

from podpac import settings
from podpac.core.coordinates import Coordinates
from podpac.core.algorithm.utility import CoordData
from podpac.core.managers.parallel import Parallel, ParallelOutputZarr, ParallelAsync, ParallelAsyncOutputZarr
from podpac.core.managers.multi_process import Process

logger = logging.getLogger("podpac")
logger.setLevel(logging.DEBUG)


class TestParallel(object):
    def test_parallel_multi_thread_compute_fill_output(self):
        node = CoordData(coord_name="time")
        coords = Coordinates([[1, 2, 3, 4, 5]], ["time"])
        node_p = Parallel(source=node, number_of_workers=2, chunks={"time": 2})
        o = node.eval(coords)
        o_p = node_p.eval(coords)

        np.testing.assert_array_equal(o, o_p)

    def test_parallel_multi_thread_compute_fill_output2(self):
        node = CoordData(coord_name="time")
        coords = Coordinates([[1, 2, 3, 4, 5]], ["time"])
        node_p = Parallel(source=node, number_of_workers=2, chunks={"time": 2})
        o = node.eval(coords)
        o_p = o.copy()
        o_p[:] = np.nan
        node_p.eval(coords, output=o_p)

        np.testing.assert_array_equal(o, o_p)

    @pytest.mark.skipif(sys.version < "3.7", reason="python < 3.7 cannot handle processes launched from threads")
    def test_parallel_process(self):
        node = Process(source=CoordData(coord_name="time"))
        coords = Coordinates([[1, 2, 3, 4, 5]], ["time"])
        node_p = Parallel(source=node, number_of_workers=2, chunks={"time": 2})
        o = node.eval(coords)
        o_p = o.copy()
        o_p[:] = np.nan
        node_p.eval(coords, output=o_p)
        time.sleep(0.1)

        np.testing.assert_array_equal(o, o_p)


class TestParallelAsync(object):
    @pytest.mark.skipif(sys.version < "3.7", reason="python < 3.7 cannot handle processes launched from threads")
    def test_parallel_process_async(self):
        node = Process(source=CoordData(coord_name="time"))  # , block=False)
        coords = Coordinates([[1, 2, 3, 4, 5]], ["time"])
        node_p = ParallelAsync(source=node, number_of_workers=2, chunks={"time": 2}, fill_output=False)
        node_p.eval(coords)
        time.sleep(0.1)
        # Just try to make it run...


class TestParallelOutputZarr(object):
    @pytest.mark.skipif(sys.version < "3.7", reason="python < 3.7 cannot handle processes launched from threads")
    def test_parallel_process_zarr(self):
        # Can't use tempfile.TemporaryDirectory because multiple processess need access to dir
        tmpdir = os.path.join(tempfile.gettempdir(), "test_parallel_process_zarr.zarr")

        node = Process(source=CoordData(coord_name="time"))  # , block=False)
        coords = Coordinates([[1, 2, 3, 4, 5]], ["time"])
        node_p = ParallelOutputZarr(
            source=node, number_of_workers=2, chunks={"time": 2}, fill_output=False, zarr_file=tmpdir
        )
        o_zarr = node_p.eval(coords)
        time.sleep(0.1)
        # print(o_zarr.info)
        np.testing.assert_array_equal([1, 2, 3, 4, 5], o_zarr["data"][:])

        shutil.rmtree(tmpdir)

    @pytest.mark.skipif(sys.version < "3.7", reason="python < 3.7 cannot handle processes launched from threads")
    def test_parallel_process_zarr_async(self):
        # Can't use tempfile.TemporaryDirectory because multiple processess need access to dir
        tmpdir = os.path.join(tempfile.gettempdir(), "test_parallel_process_zarr_async.zarr")

        node = Process(source=CoordData(coord_name="time"))  # , block=False)
        coords = Coordinates([[1, 2, 3, 4, 5]], ["time"])
        node_p = ParallelAsyncOutputZarr(
            source=node, number_of_workers=5, chunks={"time": 2}, fill_output=False, zarr_file=tmpdir
        )
        o_zarr = node_p.eval(coords)
        # print(o_zarr.info)
        time.sleep(0.01)
        np.testing.assert_array_equal([1, 2, 3, 4, 5], o_zarr["data"][:])

        shutil.rmtree(tmpdir)

    @pytest.mark.skipif(sys.version < "3.7", reason="python < 3.7 cannot handle processes launched from threads")
    def test_parallel_process_zarr_async_starti(self):
        # Can't use tempfile.TemporaryDirectory because multiple processess need access to dir
        tmpdir = os.path.join(tempfile.gettempdir(), "test_parallel_process_zarr_async_starti.zarr")

        node = Process(source=CoordData(coord_name="time"))  # , block=False)
        coords = Coordinates([[1, 2, 3, 4, 5]], ["time"])
        node_p = ParallelAsyncOutputZarr(
            source=node, number_of_workers=5, chunks={"time": 2}, fill_output=False, zarr_file=tmpdir, start_i=1
        )
        o_zarr = node_p.eval(coords)
        # print(o_zarr.info)
        time.sleep(0.01)
        np.testing.assert_array_equal([np.nan, np.nan, 3, 4, 5], o_zarr["data"][:])

        node_p = ParallelAsyncOutputZarr(
            source=node, number_of_workers=5, chunks={"time": 2}, fill_output=False, zarr_file=tmpdir, start_i=0
        )
        o_zarr = node_p.eval(coords)
        np.testing.assert_array_equal([1, 2, 3, 4, 5], o_zarr["data"][:])

        shutil.rmtree(tmpdir)
